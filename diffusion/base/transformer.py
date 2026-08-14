"""Transformer denoiser ``F_theta`` for 768-dim advice embeddings.

Design note (the one worth arguing about at review). A diffusion sample here is
a *single* 768-dim vector, so the naive reading of "stack transformer blocks
over the input" gives a sequence of length one, and self-attention over one
token is an expensive identity. This module therefore splits the embedding into
``num_tokens`` contiguous slices and attends over those, which is the same trick
latent-diffusion transformers use on latent patches: it gives the network an
internal sequence to route information through while leaving the input and
output spaces untouched. Conditioning (issue embedding, variable-length
episodic memory) enters through cross-attention, and the noise level enters
through adaptive layer-norm modulation rather than as an extra token, so that
every block is conditioned on ``sigma`` at every depth.

The alternative - a plain MLP over the 768 dims - is a legitimate baseline and
is what ``MLPDenoiser`` provides; ``experiments/`` can compare them, and the
comparison is cheap enough to be worth running before committing 50 GPU-hours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from diffusion.base.denoiser import SinusoidalTimeEmbedding

__all__ = ["TransformerDenoiserConfig", "TransformerDenoiser", "RMSNorm", "SwiGLU"]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalise the last dimension to unit RMS and rescale."""
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out: Tensor = (x.float() * norm).to(x.dtype) * self.weight
        return out


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: ``W2(silu(W1 x) * W3 x)``."""

    def __init__(self, dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden or int(2 * (4 * dim) / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Gated feed-forward."""
        out: Tensor = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return out


class _Attention(nn.Module):
    """Multi-head attention; self-attention when ``context`` is None."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def _split(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, context: Optional[Tensor] = None,
                context_mask: Optional[Tensor] = None) -> Tensor:
        """Attend ``x`` to itself, or to ``context`` when given.

        ``context_mask`` is ``True`` for valid positions, shape ``(B, M)``; it is
        what makes a variable-length episodic memory work.
        """
        source = x if context is None else context
        query, key, value = self._split(self.to_q(x)), self._split(self.to_k(source)), \
            self._split(self.to_v(source))
        attn_mask: Optional[Tensor] = None
        if context_mask is not None:
            attn_mask = context_mask[:, None, None, :].to(torch.bool)
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value,
                                                               attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(x.shape)
        projected: Tensor = self.proj(out)
        return projected


class _Block(nn.Module):
    """Pre-norm block: self-attention, cross-attention, SwiGLU; adaLN-modulated."""

    def __init__(self, dim: int, num_heads: int, ffn_hidden: Optional[int],
                 use_cross_attention: bool) -> None:
        super().__init__()
        self.norm_self = RMSNorm(dim)
        self.self_attn = _Attention(dim, num_heads)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.norm_cross = RMSNorm(dim)
            self.cross_attn = _Attention(dim, num_heads)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden)
        self.modulation = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, conditioning: Tensor, context: Optional[Tensor] = None,
                context_mask: Optional[Tensor] = None) -> Tensor:
        """One block. ``conditioning`` is the ``(B, dim)`` noise/label embedding."""
        shift_self, scale_self, gate_self, shift_ffn, scale_ffn, gate_ffn = \
            self.modulation(conditioning).unsqueeze(1).chunk(6, dim=-1)

        h = self.norm_self(x) * (1.0 + scale_self) + shift_self
        x = x + gate_self * self.self_attn(h)

        if self.use_cross_attention and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context, context_mask)

        h = self.norm_ffn(x) * (1.0 + scale_ffn) + shift_ffn
        out: Tensor = x + gate_ffn * self.ffn(h)
        return out


@dataclass(frozen=True)
class TransformerDenoiserConfig:
    """Architecture hyper-parameters; see ``docs/diffusion_architecture.md``."""

    input_dim: int = 768
    num_tokens: int = 12
    hidden_dim: int = 512
    depth: int = 8
    num_heads: int = 8
    ffn_hidden: Optional[int] = None
    time_embed_dim: int = 256
    context_dim: int = 768
    use_cross_attention: bool = True

    def __post_init__(self) -> None:
        if self.input_dim % self.num_tokens != 0:
            raise ValueError(f"input_dim {self.input_dim} not divisible by "
                             f"num_tokens {self.num_tokens}")

    @property
    def token_dim(self) -> int:
        """Width of each slice of the input vector."""
        return self.input_dim // self.num_tokens


class TransformerDenoiser(nn.Module):
    """``F_theta(x, c_noise, issue=..., episodic=...)`` over embedding vectors.

    Wrap this in :class:`diffusion.base.preconditioning.EDMPreconditioner` before
    training or sampling; on its own it predicts the EDM unit-variance target,
    not the clean sample.

    Classifier-free guidance: conditioning is replaced by a learned null token
    with probability ``cfg_dropout`` during training, and can be dropped
    explicitly at inference by passing ``drop_conditioning=True``.
    """

    def __init__(self, config: Optional[TransformerDenoiserConfig] = None) -> None:
        super().__init__()
        self.config = config or TransformerDenoiserConfig()
        cfg = self.config

        self.token_in = nn.Linear(cfg.token_dim, cfg.hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.num_tokens, cfg.hidden_dim) * 0.02)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(cfg.time_embed_dim),
            nn.Linear(cfg.time_embed_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.issue_proj = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.episodic_proj = nn.Linear(cfg.context_dim, cfg.hidden_dim)
        self.null_context = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        self.issue_to_conditioning = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.blocks = nn.ModuleList(
            _Block(cfg.hidden_dim, cfg.num_heads, cfg.ffn_hidden, cfg.use_cross_attention)
            for _ in range(cfg.depth)
        )
        self.norm_out = RMSNorm(cfg.hidden_dim)
        self.token_out = nn.Linear(cfg.hidden_dim, cfg.token_dim)
        nn.init.zeros_(self.token_out.weight)
        nn.init.zeros_(self.token_out.bias)

    @property
    def num_parameters(self) -> int:
        """Trainable parameter count."""
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)

    def _build_context(self, batch: int, device: torch.device, dtype: torch.dtype,
                       issue: Optional[Tensor], episodic: Optional[Tensor],
                       episodic_mask: Optional[Tensor],
                       drop: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Assemble conditioning tokens, their mask, and the pooled adaLN vector.

        ``drop`` is a ``(B,)`` boolean: rows marked True get the learned null
        context, which is how classifier-free guidance is trained.
        """
        tokens = [self.null_context.expand(batch, 1, -1).to(dtype)]
        mask = [torch.ones(batch, 1, device=device, dtype=torch.bool)]
        pooled = torch.zeros(batch, self.config.hidden_dim, device=device, dtype=dtype)

        if issue is not None:
            issue_token = self.issue_proj(issue).unsqueeze(1)
            tokens.append(issue_token)
            mask.append(torch.ones(batch, 1, device=device, dtype=torch.bool))
            pooled = pooled + self.issue_to_conditioning(issue_token.squeeze(1))
        if episodic is not None:
            tokens.append(self.episodic_proj(episodic))
            if episodic_mask is None:
                episodic_mask = torch.ones(batch, episodic.shape[1], device=device,
                                           dtype=torch.bool)
            mask.append(episodic_mask.to(torch.bool))

        context = torch.cat(tokens, dim=1)
        context_mask = torch.cat(mask, dim=1)

        if drop is not None:
            keep = (~drop).reshape(batch, 1, 1)
            null = self.null_context.expand(batch, context.shape[1], -1).to(dtype)
            context = torch.where(keep, context, null)
            context_mask = context_mask | drop.reshape(batch, 1)
            pooled = pooled * (~drop).reshape(batch, 1).to(pooled.dtype)
        return context, context_mask, pooled

    def forward(self, x: Tensor, c_noise: Tensor, issue: Optional[Tensor] = None,
                episodic: Optional[Tensor] = None, episodic_mask: Optional[Tensor] = None,
                drop_conditioning: Optional[Tensor] = None) -> Tensor:
        """Predict the EDM target for a batch of preconditioned inputs.

        Parameters
        ----------
        x:
            ``(B, input_dim)`` preconditioned noisy sample (``c_in * x_noisy``).
        c_noise:
            ``(B,)`` noise label from the preconditioner.
        issue, episodic:
            ``(B, context_dim)`` and ``(B, M, context_dim)`` conditioning.
        episodic_mask:
            ``(B, M)`` boolean, True for real memory slots.
        drop_conditioning:
            ``(B,)`` boolean; True rows are denoised unconditionally.
        """
        cfg = self.config
        if x.shape[-1] != cfg.input_dim:
            raise ValueError(f"expected last dim {cfg.input_dim}, got {x.shape[-1]}")
        batch = x.shape[0]
        if c_noise.dim() == 0:
            c_noise = c_noise.expand(batch)

        tokens = self.token_in(x.reshape(batch, cfg.num_tokens, cfg.token_dim)) + self.pos_embed
        conditioning = self.time_embed(c_noise)
        context, context_mask, pooled = self._build_context(
            batch, x.device, tokens.dtype, issue, episodic, episodic_mask, drop_conditioning)
        conditioning = conditioning + pooled

        for block in self.blocks:
            tokens = block(tokens, conditioning, context, context_mask)

        out: Tensor = self.token_out(self.norm_out(tokens))
        return out.reshape(batch, cfg.input_dim)
