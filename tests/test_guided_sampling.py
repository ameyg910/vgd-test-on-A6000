"""Invariants of the GuidanceSpec API and the guided sampler.

The critical one is that an empty spec list recovers unguided sampling exactly;
it is carried over from the screening task and must never regress.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest
import torch
from torch import Tensor, nn

from diffusion.base.denoiser import MLPDenoiser
from diffusion.base.sampling.guided import (GuidanceSpec, compose_gradients, constant_weight,
                                            gaussian_tilt_weight, inverse_sigma_weight,
                                            late_start_weight, linear_ramp_weight, sample_guided,
                                            verifier_gradients)
from diffusion.base.sampling.unguided import sample
from diffusion.base.schedule import NoiseSchedule
from verifiers.base import Verifier
from verifiers.toy import HalfPlaneVerifier, TargetPointVerifier

Setup = Tuple[nn.Module, NoiseSchedule]


@pytest.fixture
def setup() -> Setup:
    torch.manual_seed(0)
    denoiser = MLPDenoiser(input_dim=2, width=32, num_blocks=2, time_embed_dim=16)
    for parameter in denoiser.parameters():
        nn.init.normal_(parameter, std=0.05)
    denoiser.eval()
    return denoiser, NoiseSchedule(num_timesteps=100, schedule="cosine")


def _as_tensor(value: Any) -> Tensor:
    assert isinstance(value, Tensor)
    return value


def test_empty_specs_recover_unguided_sampling(setup: Setup) -> None:
    denoiser, schedule = setup
    unguided = sample(denoiser, schedule, (64, 2), num_steps=25, seed=7)
    guided = _as_tensor(sample_guided(denoiser, schedule, (64, 2), guidance_specs=[],
                                      num_steps=25, seed=7))
    assert torch.equal(unguided, guided)


def test_empty_specs_recover_unguided_sampling_stochastic(setup: Setup) -> None:
    denoiser, schedule = setup
    unguided = sample(denoiser, schedule, (32, 2), num_steps=20, seed=3, eta=1.0)
    guided = _as_tensor(sample_guided(denoiser, schedule, (32, 2), guidance_specs=[],
                                      num_steps=20, seed=3, eta=1.0))
    assert torch.equal(unguided, guided)


def test_zero_weight_spec_matches_unguided(setup: Setup) -> None:
    denoiser, schedule = setup
    unguided = sample(denoiser, schedule, (32, 2), num_steps=20, seed=1)
    spec = GuidanceSpec(HalfPlaneVerifier(), constant_weight(0.0), project_to_manifold=False)
    guided = _as_tensor(sample_guided(denoiser, schedule, (32, 2), guidance_specs=[spec],
                                      num_steps=20, seed=1))
    assert torch.allclose(unguided, guided, atol=1e-6)


def test_guidance_moves_samples_in_the_expected_direction(setup: Setup) -> None:
    denoiser, schedule = setup
    unguided = _as_tensor(sample_guided(denoiser, schedule, (256, 2), guidance_specs=[],
                                        num_steps=30, seed=5))
    spec = GuidanceSpec(HalfPlaneVerifier(alpha=1.0), constant_weight(5.0),
                        project_to_manifold=False)
    guided = _as_tensor(sample_guided(denoiser, schedule, (256, 2), guidance_specs=[spec],
                                      num_steps=30, seed=5))
    assert float(guided[:, 0].mean()) > float(unguided[:, 0].mean())


def test_specs_keep_independent_weight_functions(setup: Setup) -> None:
    """Two specs, two schedules, no interaction: the anti-regression for the review finding."""
    _, schedule = setup
    specs = [
        GuidanceSpec(HalfPlaneVerifier(), constant_weight(2.0), project_to_manifold=False),
        GuidanceSpec(TargetPointVerifier(torch.tensor([-2.0, 0.0]), 0.5),
                     late_start_weight(5.0, start_t=40), project_to_manifold=False),
    ]
    x_t = torch.randn(8, 2)
    eps = torch.zeros(8, 2)

    t_high = torch.full((8,), 90, dtype=torch.long)
    grads = verifier_gradients(specs, x_t, t_high, schedule, eps)
    weights_high = [spec.weight_fn(90) for spec in specs]
    assert weights_high == [2.0, 0.0]
    assert torch.allclose(compose_gradients(grads, weights_high, "sum"), 2.0 * grads[0], atol=1e-6)

    weights_low = [spec.weight_fn(10) for spec in specs]
    assert weights_low == [2.0, 5.0]
    expected = 2.0 * grads[0] + 5.0 * grads[1]
    assert torch.allclose(compose_gradients(grads, weights_low, "sum"), expected, atol=1e-6)


def test_three_specs_with_distinct_schedules(setup: Setup) -> None:
    denoiser, schedule = setup
    specs = [
        GuidanceSpec(HalfPlaneVerifier(), constant_weight(1.0), project_to_manifold=False),
        GuidanceSpec(TargetPointVerifier(torch.tensor([0.0, 2.0]), 0.5),
                     linear_ramp_weight(3.0, 100), project_to_manifold=False),
        GuidanceSpec(TargetPointVerifier(torch.tensor([2.0, 0.0]), 0.5),
                     inverse_sigma_weight(2.0, schedule, lam=0.5), project_to_manifold=False),
    ]
    out = _as_tensor(sample_guided(denoiser, schedule, (16, 2), guidance_specs=specs,
                                   num_steps=20, seed=0))
    assert out.shape == (16, 2) and bool(torch.isfinite(out).all())


def test_weight_factories_are_time_dependent(setup: Setup) -> None:
    _, schedule = setup
    ramp = linear_ramp_weight(4.0, 100)
    late = late_start_weight(3.0, 40)
    tilt = gaussian_tilt_weight(3.0, schedule, 2.0)
    assert ramp(99) == pytest.approx(0.0, abs=1e-6)
    assert ramp(0) == pytest.approx(4.0, abs=1e-6)
    assert late(99) == 0.0 and late(0) == 3.0
    assert tilt(99) < tilt(50) < tilt(0)
    assert tilt(0) == pytest.approx(3.0, rel=1e-3)


@pytest.mark.parametrize("strategy", ["sum", "normalized", "projected", "alternating"])
def test_composition_strategies_produce_finite_samples(setup: Setup, strategy: str) -> None:
    denoiser, schedule = setup
    specs = [
        GuidanceSpec(HalfPlaneVerifier(), constant_weight(2.0), project_to_manifold=False),
        GuidanceSpec(TargetPointVerifier(torch.tensor([-2.0, 0.0]), 0.4), constant_weight(2.0),
                     project_to_manifold=False),
    ]
    out = _as_tensor(sample_guided(denoiser, schedule, (16, 2), guidance_specs=specs,
                                   num_steps=20, seed=0, strategy=strategy))
    assert out.shape == (16, 2) and bool(torch.isfinite(out).all())


def test_projected_composition_removes_conflict() -> None:
    grads = [torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]])]
    assert torch.allclose(compose_gradients(grads, [1.0, 1.0], "sum"), torch.zeros(1, 2))
    assert torch.allclose(compose_gradients(grads, [1.0, 1.0], "projected"), torch.zeros(1, 2))


def test_compose_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError):
        compose_gradients([torch.zeros(2, 2)], [1.0, 2.0], "sum")
    with pytest.raises(ValueError):
        compose_gradients([], [], "sum")


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError):
        compose_gradients([torch.zeros(2, 2)], [1.0], "nonsense")


def test_invalid_target_rejected() -> None:
    with pytest.raises(ValueError):
        GuidanceSpec(HalfPlaneVerifier(), constant_weight(1.0), target="x_middle")


def test_return_info_reports_cancellation(setup: Setup) -> None:
    """Opposed verifiers must show up as a negative pairwise cosine in diagnostics."""
    denoiser, schedule = setup
    specs = [
        GuidanceSpec(HalfPlaneVerifier(alpha=1.0), constant_weight(1.0), project_to_manifold=False,
                     name="right"),
        GuidanceSpec(TargetPointVerifier(torch.tensor([-2.0, 0.0]), 1.0), constant_weight(1.0),
                     project_to_manifold=False, name="left"),
    ]
    result = sample_guided(denoiser, schedule, (32, 2), guidance_specs=specs, num_steps=20,
                           seed=0, return_info=True)
    assert isinstance(result, tuple)
    _, info = result
    cosines = info["pairwise_cosine"]["right|left"]
    assert len(cosines) == 20
    assert min(cosines) < -0.5, "opposed verifiers must register as a conflict at some step"
    assert set(info["weights"]) == {"right", "left"}


def test_pairwise_cosine_is_negative_on_the_data_manifold(setup: Setup) -> None:
    """On the ring itself the two verifiers genuinely oppose, whatever the sampler does."""
    _, schedule = setup
    from diffusion.data.toy import ring_mode_centers

    centers = ring_mode_centers()
    specs = [
        GuidanceSpec(HalfPlaneVerifier(alpha=1.0), constant_weight(1.0), project_to_manifold=False),
        GuidanceSpec(TargetPointVerifier(centers[4].clone(), 1.0), constant_weight(1.0),
                     project_to_manifold=False),
    ]
    t = torch.zeros(centers.shape[0], dtype=torch.long)
    grads = verifier_gradients(specs, centers, t, schedule, torch.zeros_like(centers))
    cosine = torch.nn.functional.cosine_similarity(grads[0], grads[1], dim=-1)
    assert float(cosine.mean()) < 0.0


def test_projector_is_applied_when_supplied(setup: Setup) -> None:
    """A spec with project_to_manifold=True must route its gradient through the projector."""
    denoiser, schedule = setup
    calls: List[int] = []

    def projector(grad: Tensor, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        calls.append(1)
        return torch.zeros_like(grad)

    spec = GuidanceSpec(HalfPlaneVerifier(), constant_weight(5.0), project_to_manifold=True)
    guided = _as_tensor(sample_guided(denoiser, schedule, (16, 2), guidance_specs=[spec],
                                      num_steps=10, seed=2, projector=projector))
    unguided = sample(denoiser, schedule, (16, 2), num_steps=10, seed=2)
    assert len(calls) == 10
    assert torch.allclose(guided, unguided, atol=1e-6)


def test_projector_skipped_when_flag_false(setup: Setup) -> None:
    denoiser, schedule = setup

    def projector(grad: Tensor, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        raise AssertionError("projector must not be called when project_to_manifold=False")

    spec = GuidanceSpec(HalfPlaneVerifier(), constant_weight(1.0), project_to_manifold=False)
    sample_guided(denoiser, schedule, (8, 2), guidance_specs=[spec], num_steps=5, seed=0,
                  projector=projector)


def test_verifier_interface_is_dimension_agnostic() -> None:
    x = torch.randn(16, 768)
    verifier: Verifier = HalfPlaneVerifier(alpha=1.0, dim=3)
    assert verifier.log_value(x).shape == (16,)
    assert verifier.grad_log_value(x).shape == (16, 768)


def test_analytical_and_autograd_gradients_agree() -> None:
    x = torch.randn(64, 2)
    for verifier_cls, kwargs in [
        (HalfPlaneVerifier, {"alpha": 2.0}),
        (TargetPointVerifier, {"center": torch.tensor([-1.0, 0.5]), "sigma": 0.7}),
    ]:
        analytical = verifier_cls(**kwargs, use_autograd=False).grad_log_value(x)
        autograd = verifier_cls(**kwargs, use_autograd=True).grad_log_value(x)
        assert torch.allclose(analytical, autograd, atol=1e-5)


def test_sampling_is_reproducible_under_seed(setup: Setup) -> None:
    denoiser, schedule = setup
    spec = GuidanceSpec(HalfPlaneVerifier(), constant_weight(3.0), project_to_manifold=False)
    first = _as_tensor(sample_guided(denoiser, schedule, (32, 2), guidance_specs=[spec],
                                     num_steps=20, seed=11))
    second = _as_tensor(sample_guided(denoiser, schedule, (32, 2), guidance_specs=[spec],
                                      num_steps=20, seed=11))
    assert torch.equal(first, second)
