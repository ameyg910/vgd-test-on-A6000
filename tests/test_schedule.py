"""Tests for the noise schedule and the forward process."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from diffusion.base.schedule import NoiseSchedule


@pytest.fixture(params=["linear", "cosine"])
def schedule(request: pytest.FixtureRequest) -> NoiseSchedule:
    return NoiseSchedule(num_timesteps=200, schedule=request.param)


def test_alpha_beta_consistency(schedule: NoiseSchedule) -> None:
    assert torch.allclose(schedule.alphas, 1.0 - schedule.betas, atol=1e-6)
    assert torch.allclose(schedule.alphas_cumprod, torch.cumprod(schedule.alphas, dim=0), atol=1e-5)
    assert torch.all(schedule.betas > 0) and torch.all(schedule.betas < 1)


def test_alphas_cumprod_is_monotone_and_bounded(schedule: NoiseSchedule) -> None:
    diffs = schedule.alphas_cumprod[1:] - schedule.alphas_cumprod[:-1]
    assert torch.all(diffs <= 0)
    assert schedule.alphas_cumprod[0] < 1.0
    assert float(schedule.alphas_cumprod[-1]) < 0.05


def test_sigma_matches_definition(schedule: NoiseSchedule) -> None:
    assert torch.allclose(schedule.sigmas, (1.0 - schedule.alphas_cumprod).sqrt(), atol=1e-6)


def test_q_sample_moments(schedule: NoiseSchedule) -> None:
    torch.manual_seed(0)
    x_0 = torch.randn(20_000, 2)
    t = torch.full((20_000,), 100, dtype=torch.long)
    x_t = schedule.q_sample(x_0, t)
    expected_var = float(schedule.alphas_cumprod[100] + (1.0 - schedule.alphas_cumprod[100]))
    assert x_t.shape == x_0.shape
    assert abs(float(x_t.var()) - expected_var) < 0.05


def test_q_sample_is_deterministic_given_noise(schedule: NoiseSchedule) -> None:
    x_0 = torch.randn(8, 3)
    noise = torch.randn(8, 3)
    t = torch.randint(0, schedule.num_timesteps, (8,))
    assert torch.equal(schedule.q_sample(x_0, t, noise), schedule.q_sample(x_0, t, noise))


def test_predict_x0_inverts_forward_process(schedule: NoiseSchedule) -> None:
    x_0 = torch.randn(16, 4)
    noise = torch.randn(16, 4)
    t = torch.randint(0, schedule.num_timesteps // 2, (16,))
    x_t = schedule.q_sample(x_0, t, noise)
    assert torch.allclose(schedule.predict_x0_from_eps(x_t, t, noise), x_0, atol=1e-3)


def test_score_epsilon_roundtrip(schedule: NoiseSchedule) -> None:
    eps = torch.randn(10, 5)
    t = torch.randint(0, schedule.num_timesteps, (10,))
    score = schedule.score_from_eps(t, eps)
    assert torch.allclose(schedule.eps_from_score(t, score), eps, atol=1e-5)


def test_timesteps_are_descending_and_respaced(schedule: NoiseSchedule) -> None:
    full = schedule.timesteps()
    assert len(full) == schedule.num_timesteps
    assert full[0] == schedule.num_timesteps - 1 and full[-1] == 0
    sub = schedule.timesteps(50)
    assert len(sub) == 50
    assert torch.all(sub[1:] < sub[:-1])
    assert int(sub.max()) < schedule.num_timesteps
