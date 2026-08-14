"""W1 acceptance: the ported code reproduces the screening task bit-for-bit.

The API changed (``verifiers=[(V, w)]`` became ``guidance_specs=[GuidanceSpec]``)
but the numerics must not have. Goldens were produced by running the screening
package once; see ``scripts/make_parity_goldens.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch import Tensor, nn

from diffusion.base.denoiser import MLPDenoiser
from diffusion.base.sampling.guided import (GuidanceSpec, gaussian_tilt_weight, sample_guided,
                                            constant_weight)
from diffusion.base.sampling.unguided import sample
from diffusion.base.schedule import NoiseSchedule
from diffusion.data.toy import ring_mode_centers
from verifiers.toy import HalfPlaneVerifier, TargetPointVerifier

GOLDEN_PATH = Path(__file__).parent / "data" / "screening_goldens.pt"


@pytest.fixture(scope="module")
def goldens() -> Dict[str, Any]:
    if not GOLDEN_PATH.exists():
        pytest.skip("goldens missing; run scripts/make_parity_goldens.py")
    payload: Dict[str, Any] = torch.load(GOLDEN_PATH, weights_only=False)
    return payload


@pytest.fixture(scope="module")
def setup() -> Tuple[nn.Module, NoiseSchedule, Tensor]:
    torch.manual_seed(0)
    denoiser = MLPDenoiser(input_dim=2, width=32, num_blocks=2, time_embed_dim=16)
    for parameter in denoiser.parameters():
        nn.init.normal_(parameter, std=0.05)
    denoiser.eval()
    return denoiser, NoiseSchedule(num_timesteps=100, schedule="cosine"), ring_mode_centers()


def _specs(names: List[str], schedule: NoiseSchedule, centers: Tensor) -> List[GuidanceSpec]:
    """Rebuild each golden's verifier set using the new GuidanceSpec API."""
    specs: List[GuidanceSpec] = []
    for name in names:
        if name == "halfplane_const":
            specs.append(GuidanceSpec(HalfPlaneVerifier(alpha=1.0), constant_weight(3.0),
                                      project_to_manifold=False))
        elif name == "halfplane_tilt":
            specs.append(GuidanceSpec(HalfPlaneVerifier(alpha=1.0),
                                      gaussian_tilt_weight(3.0, schedule, 2.0),
                                      project_to_manifold=False))
        elif name == "target_tilt":
            specs.append(GuidanceSpec(TargetPointVerifier(centers[4].clone(), sigma=1.0),
                                      gaussian_tilt_weight(1.0, schedule, 2.0),
                                      project_to_manifold=False))
        else:
            raise ValueError(name)
    return specs


@pytest.mark.parametrize("label", [
    "unguided_ddim", "unguided_ancestral", "empty_specs", "halfplane_const_w3",
    "halfplane_tilt_w3", "conflict_sum", "conflict_projected", "conflict_normalized",
    "conflict_alternating",
])
def test_matches_screening_bitwise(label: str, goldens: Dict[str, Any],
                                   setup: Tuple[nn.Module, NoiseSchedule, Tensor]) -> None:
    denoiser, schedule, centers = setup
    case = goldens["cases"][label]
    expected = goldens["samples"][label]
    shape = (64, 2)

    if case["kind"] == "unguided":
        actual = sample(denoiser, schedule, shape, num_steps=case["steps"], seed=case["seed"],
                        eta=case["eta"])
    else:
        actual = sample_guided(denoiser, schedule, shape,
                               guidance_specs=_specs(case["verifiers"], schedule, centers),
                               num_steps=case["steps"], seed=case["seed"], eta=case["eta"],
                               strategy=case.get("strategy", "sum"))
    assert isinstance(actual, Tensor)
    assert torch.equal(actual.cpu(), expected), (
        f"{label}: max abs diff {float((actual.cpu() - expected).abs().max()):.3e}")


def test_weight_fn_values_match_screening_schedule(
        setup: Tuple[nn.Module, NoiseSchedule, Tensor]) -> None:
    """The int-indexed weight_fn returns exactly what the tensor-indexed version did."""
    _, schedule, _ = setup
    weight_fn = gaussian_tilt_weight(3.0, schedule, 2.0)
    for t in (0, 1, 50, 99):
        alpha_bar = schedule.alphas_cumprod[t]
        v_t = alpha_bar * 2.0 + (1.0 - alpha_bar)
        expected = float(3.0 * alpha_bar.sqrt() * 2.0 / v_t)
        assert weight_fn(t) == expected
