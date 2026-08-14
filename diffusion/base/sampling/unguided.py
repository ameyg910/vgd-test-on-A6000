"""Unguided reverse sampling and the reverse-step primitive.

:func:`reverse_step` is the only place the update rule is written down. The
guided sampler in :mod:`diffusion.base.sampling.guided` reuses it verbatim and
changes only the epsilon fed in, which is what makes
``sample_guided(guidance_specs=[])`` identical to :func:`sample` rather than
merely close.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from diffusion.base.schedule import NoiseSchedule
from diffusion.utils import generator_from_seed

__all__ = ["reverse_step", "step_pairs", "sample"]

StepPair = Tuple[Tensor, Optional[Tensor]]


def reverse_step(schedule: NoiseSchedule, x_t: Tensor, t_cur: Tensor, t_prev: Optional[Tensor],
                 eps: Tensor, eta: float = 0.0, noise: Optional[Tensor] = None,
                 clip_x0: Optional[float] = None) -> Tensor:
    """One respaced DDIM/DDPM reverse step.

    ``eta = 0`` is deterministic DDIM; ``eta = 1`` reproduces the ancestral DDPM
    update on the (possibly sub-sampled) timestep grid.

    Parameters
    ----------
    t_prev:
        The next, smaller timestep, or ``None`` on the final step to ``x_0``.
    noise:
        Injected noise for ``eta > 0``. Passed in explicitly so that sampling is
        reproducible from a caller-owned generator.
    clip_x0:
        Optional symmetric clamp on the predicted clean sample. Left at ``None``
        for reported runs; see ``docs/debugging_log.md`` for why.
    """
    ndim = x_t.dim()
    alpha_bar_t = schedule.gather(schedule.alphas_cumprod, t_cur, ndim)
    if t_prev is None:
        alpha_bar_prev = torch.ones_like(alpha_bar_t)
    else:
        alpha_bar_prev = schedule.gather(schedule.alphas_cumprod, t_prev, ndim)

    x0_hat = (x_t - (1.0 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt().clamp(min=1e-8)
    if clip_x0 is not None:
        x0_hat = x0_hat.clamp(-clip_x0, clip_x0)

    sigma = eta * (
        ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp(min=1e-12))
        * (1.0 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-12))
    ).clamp(min=0.0).sqrt()

    direction = (1.0 - alpha_bar_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
    x_prev = alpha_bar_prev.sqrt() * x0_hat + direction
    if eta > 0.0:
        if noise is None:
            noise = torch.randn_like(x_t)
        x_prev = x_prev + sigma * noise
    return x_prev


def step_pairs(schedule: NoiseSchedule, num_steps: Optional[int] = None) -> Sequence[StepPair]:
    """The ``(t_cur, t_prev)`` pairs traversed by the reverse loop, high ``t`` first."""
    steps = schedule.timesteps(num_steps)
    pairs: List[StepPair] = []
    for index, t in enumerate(steps):
        pairs.append((t, steps[index + 1] if index + 1 < len(steps) else None))
    return pairs


@torch.no_grad()
def sample(denoiser: nn.Module, schedule: NoiseSchedule, shape: Tuple[int, ...],
           num_steps: Optional[int] = None, eta: float = 0.0, seed: Optional[int] = None,
           device: Optional[torch.device] = None, clip_x0: Optional[float] = None,
           x_T: Optional[Tensor] = None) -> Tensor:
    """Unguided reverse sampling; equivalent to guided sampling with no specs."""
    device = device or next(denoiser.parameters()).device
    schedule.to(device)
    generator = generator_from_seed(seed, device)
    x_t = torch.randn(shape, device=device, generator=generator) if x_T is None else x_T.to(device)

    denoiser.eval()
    for t_cur, t_prev in step_pairs(schedule, num_steps):
        t_batch = t_cur.reshape(1).expand(x_t.shape[0]).to(device)
        t_prev_batch = None if t_prev is None else t_prev.reshape(1).expand(x_t.shape[0]).to(device)
        eps = denoiser(x_t, t_batch)
        noise = (torch.randn(x_t.shape, device=device, generator=generator)
                 if eta > 0.0 and t_prev is not None else None)
        x_t = reverse_step(schedule, x_t, t_batch, t_prev_batch, eps, eta=eta, noise=noise,
                           clip_x0=clip_x0)
    return x_t
