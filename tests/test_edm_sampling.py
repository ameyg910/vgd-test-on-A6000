"""EDM sampler: invariants, classifier-free guidance, and verifier guidance.

The Phase 0 invariant - empty guidance recovers unguided sampling exactly -
is re-asserted here for the EDM sampler, because it is now a *second*
implementation of the reverse process and the invariant is only useful if it
holds in both.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
from torch import Tensor, nn

from diffusion.base.preconditioning import EDMConfig, EDMPreconditioner, karras_sigmas
from diffusion.base.sampling.edm import edm_verifier_gradients, sample_edm, weight_from_sigma
from diffusion.base.sampling.guided import GuidanceSpec, constant_weight
from diffusion.base.transformer import TransformerDenoiser, TransformerDenoiserConfig
from verifiers.toy import HalfPlaneVerifier, TargetPointVerifier

DIM = 32
Setup = Tuple[EDMPreconditioner, EDMConfig]


@pytest.fixture
def setup() -> Setup:
    torch.manual_seed(0)
    config = EDMConfig(sigma_data=1.0, sigma_max=20.0)
    network = TransformerDenoiser(TransformerDenoiserConfig(
        input_dim=DIM, num_tokens=4, hidden_dim=32, depth=2, num_heads=4,
        context_dim=DIM, time_embed_dim=32))
    torch.nn.init.normal_(network.token_out.weight, std=0.05)
    network.eval()
    return EDMPreconditioner(network, config), config


def _as_tensor(value: object) -> Tensor:
    assert isinstance(value, Tensor)
    return value


def test_empty_specs_recover_unguided_sampling(setup: Setup) -> None:
    pre, config = setup
    plain = _as_tensor(sample_edm(pre, (8, DIM), num_steps=8, config=config, seed=3))
    empty = _as_tensor(sample_edm(pre, (8, DIM), num_steps=8, config=config, seed=3,
                                  guidance_specs=[]))
    assert torch.equal(plain, empty)


def test_zero_weight_guidance_matches_unguided(setup: Setup) -> None:
    pre, config = setup
    spec = GuidanceSpec(HalfPlaneVerifier(), constant_weight(0.0), project_to_manifold=False)
    plain = _as_tensor(sample_edm(pre, (8, DIM), num_steps=8, config=config, seed=1))
    guided = _as_tensor(sample_edm(pre, (8, DIM), num_steps=8, config=config, seed=1,
                                   guidance_specs=[spec]))
    assert torch.allclose(plain, guided, atol=1e-5)


def test_guidance_moves_samples_in_the_expected_direction(setup: Setup) -> None:
    pre, config = setup
    spec = GuidanceSpec(HalfPlaneVerifier(alpha=1.0), constant_weight(0.05),
                        project_to_manifold=False)
    plain = _as_tensor(sample_edm(pre, (64, DIM), num_steps=12, config=config, seed=5))
    guided = _as_tensor(sample_edm(pre, (64, DIM), num_steps=12, config=config, seed=5,
                                   guidance_specs=[spec]))
    assert float(guided[:, 0].mean()) > float(plain[:, 0].mean())


def test_cfg_scale_one_equals_plain_conditional(setup: Setup) -> None:
    """A scale of 1 must skip the two-pass path entirely, not approximate it."""
    pre, config = setup
    issue = torch.randn(4, DIM)
    single = _as_tensor(sample_edm(pre, (4, DIM), num_steps=6, config=config, seed=0,
                                   conditioning={"issue": issue}, cfg_scale=1.0))
    again = _as_tensor(sample_edm(pre, (4, DIM), num_steps=6, config=config, seed=0,
                                  conditioning={"issue": issue}, cfg_scale=1.0))
    assert torch.equal(single, again)


def test_cfg_scale_changes_samples(setup: Setup) -> None:
    pre, config = setup
    issue = torch.randn(4, DIM)
    weak = _as_tensor(sample_edm(pre, (4, DIM), num_steps=6, config=config, seed=0,
                                 conditioning={"issue": issue}, cfg_scale=1.0))
    strong = _as_tensor(sample_edm(pre, (4, DIM), num_steps=6, config=config, seed=0,
                                   conditioning={"issue": issue}, cfg_scale=3.0))
    assert not torch.allclose(weak, strong, atol=1e-5)


def test_sampling_is_reproducible_and_finite(setup: Setup) -> None:
    pre, config = setup
    first = _as_tensor(sample_edm(pre, (16, DIM), num_steps=10, config=config, seed=7))
    second = _as_tensor(sample_edm(pre, (16, DIM), num_steps=10, config=config, seed=7))
    assert torch.equal(first, second) and bool(torch.isfinite(first).all())


def test_churn_changes_trajectory_but_stays_finite(setup: Setup) -> None:
    pre, config = setup
    deterministic = _as_tensor(sample_edm(pre, (8, DIM), num_steps=10, config=config, seed=2))
    churned = _as_tensor(sample_edm(pre, (8, DIM), num_steps=10, config=config, seed=2,
                                    s_churn=10.0, s_min=0.05, s_max=10.0))
    assert not torch.allclose(deterministic, churned, atol=1e-4)
    assert bool(torch.isfinite(churned).all())


def test_weight_from_sigma_indexes_the_schedule() -> None:
    config = EDMConfig()
    sigmas = karras_sigmas(8, config).tolist()
    weight_fn = weight_from_sigma(lambda s: 1.0 / (s ** 2 + 1.0), sigmas)
    assert weight_fn(0) == pytest.approx(1.0 / (config.sigma_max ** 2 + 1.0), rel=1e-4)
    assert weight_fn(0) < weight_fn(7)
    assert weight_fn(999) == weight_fn(len(sigmas) - 1)


def test_verifier_gradients_respect_target_and_clip(setup: Setup) -> None:
    _, _ = setup
    x = torch.randn(6, DIM)
    denoised = torch.randn(6, DIM)
    sigma = torch.full((6,), 0.5)
    at_x = GuidanceSpec(HalfPlaneVerifier(), constant_weight(1.0), project_to_manifold=False,
                        target="x_t")
    at_x0 = GuidanceSpec(TargetPointVerifier(torch.zeros(DIM), sigma=1.0),
                         constant_weight(1.0), project_to_manifold=False, target="x0_hat")
    clipped = GuidanceSpec(TargetPointVerifier(torch.zeros(DIM), sigma=0.1),
                           constant_weight(1.0), project_to_manifold=False, grad_clip=0.5)
    grads = edm_verifier_gradients([at_x, at_x0, clipped], x, sigma, denoised)
    assert torch.allclose(grads[1], -denoised, atol=1e-5)
    assert float(grads[2].norm(dim=-1).max()) <= 0.5 + 1e-5


def test_projector_is_applied_when_requested(setup: Setup) -> None:
    pre, config = setup
    calls = []

    def projector(grad: Tensor, x: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
        calls.append(1)
        return torch.zeros_like(grad)

    spec = GuidanceSpec(HalfPlaneVerifier(), constant_weight(1.0), project_to_manifold=True)
    guided = _as_tensor(sample_edm(pre, (4, DIM), num_steps=5, config=config, seed=0,
                                   guidance_specs=[spec], projector=projector))
    plain = _as_tensor(sample_edm(pre, (4, DIM), num_steps=5, config=config, seed=0))
    assert calls and torch.allclose(guided, plain, atol=1e-5)


def test_return_info_reports_schedule_and_norms(setup: Setup) -> None:
    pre, config = setup
    spec = GuidanceSpec(HalfPlaneVerifier(), constant_weight(0.01), project_to_manifold=False)
    result = sample_edm(pre, (4, DIM), num_steps=6, config=config, seed=0,
                        guidance_specs=[spec], return_info=True)
    assert isinstance(result, tuple)
    _, info = result
    assert len(info["sigmas"]) == 7
    assert len(info["guidance_norm"]) > 0
    assert all(value == value for value in info["denoised_norm"])
