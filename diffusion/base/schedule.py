"""Discrete-time variance-preserving (DDPM) noise schedule.

The schedule is the single source of truth for the forward process
``x_t = sqrt(alphas_cumprod[t]) * x_0 + sqrt(1 - alphas_cumprod[t]) * eps``
and therefore for the score/epsilon conversion used by the guided sampler.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch

__all__ = ["NoiseSchedule", "make_beta_schedule"]

BetaScheduleName = Literal["linear", "cosine"]


def make_beta_schedule(name: BetaScheduleName, num_timesteps: int,
                       beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Build the ``(num_timesteps,)`` beta vector for the requested schedule.

    ``cosine`` follows Nichol & Dhariwal (2021); ``linear`` follows Ho et al. (2020), with beta scaled by ``1000 / num_timesteps``
    so the terminal marginal is standard normal for any number of steps.
    """
    if name == "linear":
        scale = 1000.0 / num_timesteps
        return torch.linspace(scale * beta_start, scale * beta_end, num_timesteps,
                              dtype=torch.float64).clamp(max=0.999)
    if name == "cosine":
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
        offset = 0.008
        cumprod = torch.cos((steps + offset) / (1.0 + offset) * math.pi / 2.0) ** 2
        cumprod = cumprod / cumprod[0]
        betas = 1.0 - cumprod[1:] / cumprod[:-1]
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown beta schedule: {name!r}")


class NoiseSchedule:
    """Container for the VP/DDPM schedule tensors and the forward process.

    Attributes
    ----------
    betas, alphas, alphas_cumprod
        Standard DDPM quantities, shape ``(num_timesteps,)``.
    sigmas
        ``sqrt(1 - alphas_cumprod)``: the standard deviation of the noise
        component of ``x_t``. This is the quantity that converts a score into an
        epsilon-prediction, and the one used to scale verifier gradients.
    """

    def __init__(self, num_timesteps: int = 1000, schedule: BetaScheduleName = "cosine",
                 beta_start: float = 1e-4, beta_end: float = 2e-2,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32) -> None:
        self.num_timesteps = int(num_timesteps)
        self.schedule = schedule
        betas = make_beta_schedule(schedule, self.num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(dtype)
        self.alphas = alphas.to(dtype)
        self.alphas_cumprod = alphas_cumprod.to(dtype)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]]
        ).to(dtype)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sigmas = (1.0 - self.alphas_cumprod).sqrt()
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod / alphas) / (1.0 - alphas_cumprod)
        ).clamp(min=1e-20).to(dtype)
        self.device = torch.device("cpu")
        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "NoiseSchedule":
        """Move all schedule tensors to ``device`` (in place) and return ``self``."""
        for name, value in list(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        self.device = torch.device(device)
        return self

    def gather(self, tensor: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Index ``tensor`` at timesteps ``t`` and right-pad to ``ndim`` dims for broadcasting."""
        out = tensor.to(t.device).gather(0, t.reshape(-1))
        return out.reshape(-1, *((1,) * (ndim - 1)))

    def sqrt_alpha_bar(self, t: torch.Tensor, ndim: int = 2) -> torch.Tensor:
        """``sqrt(alphas_cumprod[t])``, broadcast-ready."""
        return self.gather(self.sqrt_alphas_cumprod, t, ndim)

    def sigma(self, t: torch.Tensor, ndim: int = 2) -> torch.Tensor:
        """``sqrt(1 - alphas_cumprod[t])``, broadcast-ready."""
        return self.gather(self.sigmas, t, ndim)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample ``x_t ~ q(x_t | x_0)`` for a batch of timesteps ``t``."""
        if noise is None:
            noise = torch.randn_like(x_0)
        ndim = x_0.dim()
        return self.sqrt_alpha_bar(t, ndim) * x_0 + self.sigma(t, ndim) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor,
                            eps: torch.Tensor) -> torch.Tensor:
        """Invert the forward process: ``x0_hat = (x_t - sigma_t * eps) / sqrt(alpha_bar_t)``."""
        ndim = x_t.dim()
        return (x_t - self.sigma(t, ndim) * eps) / self.sqrt_alpha_bar(t, ndim).clamp(min=1e-8)

    def score_from_eps(self, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Convert an epsilon-prediction to a score: ``score = -eps / sigma_t``."""
        return -eps / self.sigma(t, eps.dim()).clamp(min=1e-8)

    def eps_from_score(self, t: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`score_from_eps`."""
        return -self.sigma(t, score.dim()) * score

    def timesteps(self, num_steps: Optional[int] = None) -> torch.Tensor:
        """Descending timestep indices for the reverse loop, sub-sampled to ``num_steps``."""
        if num_steps is None or num_steps >= self.num_timesteps:
            return torch.arange(self.num_timesteps - 1, -1, -1, device=self.device)
        stride = self.num_timesteps / float(num_steps)
        idx = (torch.arange(num_steps, device=self.device) * stride).round().long()
        return idx.flip(0).clamp(max=self.num_timesteps - 1)
