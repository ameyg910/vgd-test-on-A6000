"""EDM preconditioning: the identities Karras derives, asserted numerically.

These are the tests that catch a wrong preconditioning implementation *before*
50 GPU-hours are spent discovering that the loss went down and the samples are
noise.
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch
from torch import Tensor, nn

from diffusion.base.preconditioning import (EDMConfig, EDMPreconditioner, estimate_sigma_data,
                                            karras_sigmas, sample_training_sigmas,
                                            sigma_data_report)


class _Identity(nn.Module):
    """Network returning its input, for testing the wrapper in isolation."""

    def forward(self, x: Tensor, c_noise: Tensor) -> Tensor:
        return x


class _Linear(nn.Module):
    """Small deterministic network with a non-trivial map."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layer = nn.Linear(dim, dim)

    def forward(self, x: Tensor, c_noise: Tensor) -> Tensor:
        out: Tensor = self.layer(x) * (1.0 + c_noise.reshape(-1, 1))
        return out


@pytest.fixture
def config() -> EDMConfig:
    return EDMConfig(sigma_data=0.7)


def test_coefficients_match_karras_table(config: EDMConfig) -> None:
    pre = EDMPreconditioner(_Identity(), config)
    sigma = torch.tensor([1e-3, 0.1, 1.0, 10.0, 80.0])
    c_skip, c_out, c_in, c_noise = pre.coefficients(sigma, ndim=2)
    sd = config.sigma_data
    expected_denom = sigma ** 2 + sd ** 2
    assert torch.allclose(c_skip.reshape(-1), sd ** 2 / expected_denom, atol=1e-6)
    assert torch.allclose(c_out.reshape(-1), sigma * sd / expected_denom.sqrt(), atol=1e-6)
    assert torch.allclose(c_in.reshape(-1), 1.0 / expected_denom.sqrt(), atol=1e-6)
    assert torch.allclose(c_noise, sigma.log() / 4.0, atol=1e-6)


def test_network_input_has_unit_variance(config: EDMConfig) -> None:
    """c_in is chosen so the network sees unit-variance inputs at every sigma."""
    torch.manual_seed(0)
    pre = EDMPreconditioner(_Identity(), config)
    x_0 = torch.randn(20_000, 1) * config.sigma_data
    for sigma_value in (0.01, 0.5, 5.0, 50.0):
        sigma = torch.full((x_0.shape[0],), sigma_value)
        x = x_0 + sigma.reshape(-1, 1) * torch.randn_like(x_0)
        _, _, c_in, _ = pre.coefficients(sigma, ndim=2)
        assert float((c_in * x).std()) == pytest.approx(1.0, abs=0.05)


def test_training_target_has_unit_variance(config: EDMConfig) -> None:
    """c_out is chosen so the regression target also has unit variance."""
    torch.manual_seed(0)
    pre = EDMPreconditioner(_Identity(), config)
    x_0 = torch.randn(20_000, 1) * config.sigma_data
    for sigma_value in (0.01, 0.5, 5.0, 50.0):
        sigma = torch.full((x_0.shape[0],), sigma_value)
        _, target = pre.training_target(x_0, torch.randn_like(x_0), sigma)
        assert float(target.std()) == pytest.approx(1.0, abs=0.05)


def test_weighted_denoising_loss_equals_unit_variance_mse(config: EDMConfig) -> None:
    """lambda(sigma)||D - a_0||^2 is exactly ||F_theta(...) - target||^2."""
    torch.manual_seed(0)
    network = _Linear(16)
    pre = EDMPreconditioner(network, config)
    x_0 = torch.randn(64, 16) * config.sigma_data
    noise = torch.randn_like(x_0)
    sigma = torch.rand(64) * 5.0 + 0.05

    weighted = pre.edm_loss(x_0, sigma, noise)
    scaled_input, target = pre.training_target(x_0, noise, sigma)
    _, _, _, c_noise = pre.coefficients(sigma, ndim=2)
    raw = network(scaled_input, c_noise)
    direct = ((raw - target) ** 2).mean(dim=1)
    assert torch.allclose(weighted, direct, atol=1e-4, rtol=1e-4)


def test_denoiser_is_near_identity_at_zero_noise(config: EDMConfig) -> None:
    """As sigma -> 0, c_skip -> 1 and c_out -> 0: the denoiser returns its input."""
    pre = EDMPreconditioner(_Linear(16), config)
    x = torch.randn(8, 16)
    tiny = torch.full((8,), 1e-6)
    assert torch.allclose(pre(x, tiny), x, atol=1e-4)


def test_denoiser_ignores_input_at_huge_noise(config: EDMConfig) -> None:
    """As sigma -> infinity, c_skip -> 0: the prediction stops depending on x."""
    pre = EDMPreconditioner(_Linear(16), config)
    sigma = torch.full((8,), 1e4)
    x = torch.randn(8, 16) * 1e4
    contribution = pre.coefficients(sigma, 2)[0].reshape(-1)[0]
    assert float(contribution) < 1e-8


def test_karras_schedule_endpoints_and_monotonicity(config: EDMConfig) -> None:
    sigmas = karras_sigmas(32, config)
    assert sigmas.numel() == 33
    assert float(sigmas[0]) == pytest.approx(config.sigma_max, rel=1e-5)
    assert float(sigmas[-2]) == pytest.approx(config.sigma_min, rel=1e-4)
    assert float(sigmas[-1]) == 0.0
    assert bool((sigmas[1:-1] < sigmas[:-2]).all())


def test_training_sigma_distribution_is_lognormal(config: EDMConfig) -> None:
    torch.manual_seed(0)
    sigma = sample_training_sigmas(50_000, config, torch.device("cpu"))
    log_sigma = sigma.log()
    assert float(log_sigma.mean()) == pytest.approx(config.p_mean, abs=0.02)
    assert float(log_sigma.std()) == pytest.approx(config.p_std, abs=0.02)
    assert bool((sigma > 0).all())


def test_estimate_sigma_data_recovers_known_scale() -> None:
    torch.manual_seed(0)
    samples = torch.randn(10_000, 32) * 0.37
    assert estimate_sigma_data(samples) == pytest.approx(0.37, abs=0.01)


def test_sigma_data_report_flags_anisotropy() -> None:
    """The report must expose per-dimension spread, not just the pooled value."""
    torch.manual_seed(0)
    scale = torch.linspace(0.1, 2.0, 16)
    report = sigma_data_report(torch.randn(4_000, 16) * scale)
    assert report["per_dim_std_max"] / report["per_dim_std_min"] > 5.0
    assert report["dimension"] == 16 and report["count"] == 4_000


def test_config_with_sigma_data_is_immutable_update(config: EDMConfig) -> None:
    updated = config.with_sigma_data(1.25)
    assert updated.sigma_data == 1.25
    assert config.sigma_data == 0.7
    assert updated.rho == config.rho
