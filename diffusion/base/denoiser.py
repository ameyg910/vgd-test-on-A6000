"""Epsilon-prediction network. Dimension-agnostic by construction."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn

__all__ = ["SinusoidalTimeEmbedding", "MLPDenoiser"]


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal embedding of the (integer) diffusion timestep."""

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """Map ``(B,)`` integer timesteps to a ``(B, dim)`` embedding."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
        embedding: Tensor = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class _ResidualBlock(nn.Module):
    """Pre-activation residual MLP block with FiLM-free additive time conditioning."""

    def __init__(self, width: int, time_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear_in = nn.Linear(width, width)
        self.time_proj = nn.Linear(time_dim, width)
        self.linear_out = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, h: Tensor, t_emb: Tensor) -> Tensor:
        residual = h
        h = self.act(self.linear_in(self.norm(h)) + self.time_proj(t_emb))
        out: Tensor = residual + self.linear_out(h)
        return out


class MLPDenoiser(nn.Module):
    """``eps_theta(x_t, t)`` for inputs of arbitrary dimension.

    ``input_dim`` is a constructor argument, so the same class serves the 2D toy
    problem and a 384- or 768-dimensional embedding space without modification.
    """

    def __init__(self, input_dim: int, width: int = 256, num_blocks: int = 3,
                 time_embed_dim: int = 128) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.width = int(width)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_proj = nn.Linear(self.input_dim, width)
        self.blocks = nn.ModuleList(_ResidualBlock(width, time_embed_dim) for _ in range(num_blocks))
        self.out_norm = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, self.input_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Predict the noise in ``x_t``.

        Parameters
        ----------
        x_t : Tensor
            ``(B, input_dim)`` noisy sample.
        t : Tensor
            ``(B,)`` integer timesteps (a 0-dim tensor is broadcast over the batch).
        """
        if x_t.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dim {self.input_dim}, got {x_t.shape[-1]}")
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        t_emb = self.time_embed(t)
        h = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, t_emb)
        out: Tensor = self.output_proj(self.act_out(h))
        return out

    def act_out(self, h: Tensor) -> Tensor:
        """Final normalisation applied before the output projection."""
        return torch.nn.functional.silu(self.out_norm(h))

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)

    def parameter_shapes(self) -> Sequence[Tuple[int, ...]]:
        """Shapes of all parameters, useful for quick sanity checks."""
        return tuple(tuple(p.shape) for p in self.parameters())
