"""Generate parity goldens from the screening-task package.

The W1 acceptance bar is that the ported code reproduces the screening task
bit-for-bit. Rather than vendoring the old package, we run it once here and
freeze its outputs into ``tests/data/screening_goldens.pt``; the parity test
then runs without the old code present.

Point ``PDIFF_SCREENING_REPO`` at a checkout of the screening task and run::

    PDIFF_SCREENING_REPO=/path/to/screening python -m scripts.make_parity_goldens

The denoiser is randomly initialised from a fixed seed rather than loaded from a
checkpoint, so the goldens test sampler numerics rather than model quality and
stay small enough to commit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parents[1]


def screening_repo() -> Path:
    """Locate the screening checkout from the environment."""
    raw = os.environ.get("PDIFF_SCREENING_REPO")
    if not raw:
        raise SystemExit("set PDIFF_SCREENING_REPO to the screening-task checkout")
    path = Path(raw).expanduser().resolve()
    if not (path / "guided_diffusion").is_dir():
        raise SystemExit(f"no guided_diffusion package under {path}")
    return path


def build_denoiser(module: Any, seed: int = 0) -> nn.Module:
    """Small denoiser with deterministic random weights, identical in both packages."""
    torch.manual_seed(seed)
    denoiser = module(input_dim=2, width=32, num_blocks=2, time_embed_dim=16)
    for parameter in denoiser.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    denoiser.eval()
    return denoiser


def cases() -> List[Tuple[str, Dict[str, Any]]]:
    """The sampler configurations frozen as goldens."""
    return [
        ("unguided_ddim", {"kind": "unguided", "eta": 0.0, "steps": 25, "seed": 7}),
        ("unguided_ancestral", {"kind": "unguided", "eta": 1.0, "steps": 20, "seed": 3}),
        ("empty_specs", {"kind": "guided", "verifiers": [], "eta": 1.0, "steps": 20, "seed": 3}),
        ("halfplane_const_w3", {"kind": "guided", "verifiers": ["halfplane_const"], "eta": 1.0,
                                "steps": 30, "seed": 5}),
        ("halfplane_tilt_w3", {"kind": "guided", "verifiers": ["halfplane_tilt"], "eta": 1.0,
                               "steps": 30, "seed": 5}),
        ("conflict_sum", {"kind": "guided", "verifiers": ["halfplane_tilt", "target_tilt"],
                          "eta": 1.0, "steps": 30, "seed": 11, "strategy": "sum"}),
        ("conflict_projected", {"kind": "guided", "verifiers": ["halfplane_tilt", "target_tilt"],
                                "eta": 1.0, "steps": 30, "seed": 11, "strategy": "projected"}),
        ("conflict_normalized", {"kind": "guided", "verifiers": ["halfplane_tilt", "target_tilt"],
                                 "eta": 1.0, "steps": 30, "seed": 11, "strategy": "normalized"}),
        ("conflict_alternating", {"kind": "guided", "verifiers": ["halfplane_tilt", "target_tilt"],
                                  "eta": 1.0, "steps": 30, "seed": 11, "strategy": "alternating"}),
    ]


def run_screening() -> Dict[str, Tensor]:
    """Execute every case against the screening package."""
    sys.path.insert(0, str(screening_repo()))
    from guided_diffusion.denoiser import MLPDenoiser
    from guided_diffusion.diffusion import sample
    from guided_diffusion.guidance import (gaussian_tilt_weight, sample_guided)
    from guided_diffusion.data import ring_mode_centers
    from guided_diffusion.schedule import NoiseSchedule
    from guided_diffusion.verifiers import HalfPlaneVerifier, TargetPointVerifier

    schedule = NoiseSchedule(num_timesteps=100, schedule="cosine")
    denoiser = build_denoiser(MLPDenoiser)
    centers = ring_mode_centers()
    shape = (64, 2)

    def make(name: str) -> Tuple[Any, Any]:
        if name == "halfplane_const":
            return HalfPlaneVerifier(alpha=1.0), 3.0
        if name == "halfplane_tilt":
            return HalfPlaneVerifier(alpha=1.0), gaussian_tilt_weight(3.0, schedule, 2.0)
        if name == "target_tilt":
            return (TargetPointVerifier(centers[4].clone(), sigma=1.0),
                    gaussian_tilt_weight(1.0, schedule, 2.0))
        raise ValueError(name)

    goldens: Dict[str, Tensor] = {}
    for label, spec in cases():
        if spec["kind"] == "unguided":
            out = sample(denoiser, schedule, shape, num_steps=spec["steps"], seed=spec["seed"],
                         eta=spec["eta"])
        else:
            verifiers = [make(n) for n in spec["verifiers"]]
            out = sample_guided(denoiser, schedule, shape, verifiers=verifiers,
                                num_steps=spec["steps"], seed=spec["seed"], eta=spec["eta"],
                                strategy=spec.get("strategy", "sum"))
        goldens[label] = out.detach().cpu()
    return goldens


def main() -> Path:
    """Write the golden file and return its path."""
    goldens = run_screening()
    payload = {"samples": goldens, "cases": dict(cases()),
               "note": "generated from the screening-task package; see scripts/make_parity_goldens.py"}
    path = REPO_ROOT / "tests" / "data" / "screening_goldens.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    for label, tensor in goldens.items():
        print(f"{label:24s} mean={tensor.mean():+.6f} norm={tensor.norm():.6f}")
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
