"""Verifier-guided reverse sampling: the project's permanent guidance API.

Update rule. With the VP forward process ``x_t = sqrt(a_t) x_0 + sigma_t eps``
the network learns ``eps_theta = -sigma_t grad_{x_t} log p_t(x_t)``. Tilting the
marginal by a product of verifiers, ``p*(x) prop p(x) prod_k V_k(x)^{w_k}``, adds
``sum_k w_k grad log V_k`` to the score, so in epsilon-parameterisation

    eps_tilde = eps_theta - sigma_t * sum_k w_k(t) * grad_{x_t} log V_k(x_t, t).

Each verifier is described by one :class:`GuidanceSpec` carrying its own
``weight_fn`` of the timestep and its own projection flag. Weights are never
merged into a scalar and verifiers are never merged into a composite object, so
a third verifier with a different time-dependent schedule is one more list entry
and no code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from diffusion.base.sampling.unguided import reverse_step, step_pairs
from diffusion.base.schedule import NoiseSchedule
from diffusion.utils import generator_from_seed
from logging_config import get_logger
from verifiers.base import Verifier

__all__ = ["WeightFn", "GuidanceSpec", "ManifoldProjector", "CompositionStrategy",
           "constant_weight", "gaussian_tilt_weight", "linear_ramp_weight", "late_start_weight",
           "inverse_sigma_weight", "compose_gradients", "verifier_gradients", "sample_guided"]

logger = get_logger(__name__)

WeightFn = Callable[[int], float]
"""Guidance weight as a function of the integer timestep. See the factories below."""

CompositionStrategy = str
"""One of ``sum``, ``normalized``, ``projected``, ``alternating``."""


class ManifoldProjector(Protocol):
    """Projects a verifier gradient onto the tangent space of the data manifold.

    Implemented in W8 (``diffusion/base/manifold_projection.py``). The protocol
    is declared now so that :class:`GuidanceSpec` can carry the flag from W1
    without the sampler depending on the eventual implementation.
    """

    def __call__(self, grad: Tensor, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Return ``grad`` projected onto the tangent space at the Tweedie mean."""
        ...


@dataclass
class GuidanceSpec:
    """One verifier plus everything about how it should steer the sampler.

    Attributes
    ----------
    verifier:
        The external constraint. Only its ``grad_log_value`` is used here.
    weight_fn:
        Weight as a function of the integer timestep ``t``. Use
        :func:`constant_weight` for a scalar. Per-verifier by construction: two
        specs with different schedules coexist without interacting.
    project_to_manifold:
        Whether this verifier's gradient should be projected onto the data
        manifold's tangent space. Honoured once a projector is supplied to
        :func:`sample_guided`; until W8 lands, a spec that asks for projection
        without a projector logs a warning and is guided unprojected.
    target:
        ``"x_t"`` evaluates the verifier at the noisy iterate; ``"x0_hat"``
        evaluates at the Tweedie posterior mean and applies the chain-rule
        factor ``1 / sqrt(alpha_bar_t)`` (reconstruction guidance).
    grad_clip:
        Optional per-sample L2 clip on this verifier's gradient.
    name:
        Label used in diagnostics; defaults to the verifier's own name.
    """

    verifier: Verifier
    weight_fn: WeightFn = field(default_factory=lambda: constant_weight(1.0))
    project_to_manifold: bool = True
    target: str = "x_t"
    grad_clip: Optional[float] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.target not in ("x_t", "x0_hat"):
            raise ValueError(f"target must be 'x_t' or 'x0_hat', got {self.target!r}")
        if self.name is None:
            self.name = self.verifier.name

    @property
    def label(self) -> str:
        """Non-optional name, for use as a dictionary key."""
        return self.name or self.verifier.name


def constant_weight(w: float) -> WeightFn:
    """Constant guidance weight."""

    def weight_fn(t: int) -> float:
        return float(w)

    return weight_fn


def gaussian_tilt_weight(w: float, schedule: NoiseSchedule, data_var: float = 1.0) -> WeightFn:
    """Noise-aware weight ``w * sqrt(a_t) * s^2 / (a_t s^2 + 1 - a_t)``.

    Guidance should reproduce the family obtained by tilting at ``t = 0`` and
    *then* noising. For data ``N(0, s^2 I)`` and a linear ``log V`` that family
    has score ``score_p + w sqrt(a_t) s^2 / v_t * grad log V`` with
    ``v_t = a_t s^2 + 1 - a_t``. A constant weight instead applies the full tilt
    at every noise level and over-guides at large ``t``; this factor is 1 at
    ``t = 0`` and decays to 0 at ``t = T``. Validated on the 2D toy problem,
    where the constant-weight sampler diverges from ``w = 2`` upward.
    """
    alphas_cumprod = schedule.alphas_cumprod.detach().cpu()

    def weight_fn(t: int) -> float:
        alpha_bar = alphas_cumprod[int(t)]
        v_t = alpha_bar * data_var + (1.0 - alpha_bar)
        return float(w * alpha_bar.sqrt() * data_var / v_t.clamp(min=1e-8))

    return weight_fn


def linear_ramp_weight(w_max: float, num_timesteps: int, power: float = 1.0) -> WeightFn:
    """Weight growing from 0 at ``t = T`` to ``w_max`` at ``t = 0``."""

    def weight_fn(t: int) -> float:
        frac = 1.0 - float(t) / float(max(1, num_timesteps - 1))
        return float(w_max * min(max(frac, 0.0), 1.0) ** power)

    return weight_fn


def late_start_weight(w_max: float, start_t: int) -> WeightFn:
    """Zero while ``t > start_t``, then ``w_max``."""

    def weight_fn(t: int) -> float:
        return float(w_max) if int(t) <= start_t else 0.0

    return weight_fn


def inverse_sigma_weight(w: float, schedule: NoiseSchedule, lam: float = 1.0) -> WeightFn:
    """Adaptive schedule ``w / (sigma_t^2 + lambda)`` from the W8 workplan item.

    Provided here so the API shape is fixed early; the budget normalisation and
    the per-verifier ``lambda_k`` tuning are W8/W9 work.
    """
    sigmas = schedule.sigmas.detach().cpu()

    def weight_fn(t: int) -> float:
        sigma_sq = float(sigmas[int(t)] ** 2)
        return float(w / (sigma_sq + lam))

    return weight_fn


def _clip_rows(grad: Tensor, max_norm: float) -> Tensor:
    """Clip each row of ``grad`` to L2 norm ``max_norm``."""
    norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    clipped: Tensor = grad * (max_norm / norm).clamp(max=1.0)
    return clipped


def verifier_gradients(specs: Sequence[GuidanceSpec], x_t: Tensor, t: Tensor,
                       schedule: NoiseSchedule, eps: Tensor, context: Optional[Any] = None,
                       projector: Optional[ManifoldProjector] = None) -> List[Tensor]:
    """Per-verifier ``grad_{x_t} log V_k``, clipped and optionally projected, not yet weighted."""
    grads: List[Tensor] = []
    x0_hat: Optional[Tensor] = None
    for spec in specs:
        if spec.target == "x_t":
            grad = spec.verifier.grad_log_value(x_t, t, context)
        else:
            if x0_hat is None:
                x0_hat = schedule.predict_x0_from_eps(x_t, t, eps).detach()
            grad = spec.verifier.grad_log_value(x0_hat, t, context)
            grad = grad / schedule.sqrt_alpha_bar(t, x_t.dim()).clamp(min=1e-8)
        grad = grad.to(x_t.dtype)
        if spec.grad_clip is not None:
            grad = _clip_rows(grad, spec.grad_clip)
        if spec.project_to_manifold and projector is not None:
            grad = projector(grad, x_t, t, eps)
        grads.append(grad)
    return grads


def compose_gradients(grads: Sequence[Tensor], weights: Sequence[float],
                      strategy: CompositionStrategy = "sum", step_index: int = 0) -> Tensor:
    """Combine per-verifier gradients into a single guidance direction.

    ``sum``
        The additive rule implied by the product-of-experts tilt.
    ``normalized``
        Each gradient is scaled to unit per-sample norm first, so that a
        verifier cannot dominate through gradient magnitude alone - an artefact
        of how each verifier happens to be parameterised.
    ``projected``
        PCGrad-style conflict resolution (Yu et al., 2020): when two weighted
        gradients have a negative inner product, each is projected onto the
        normal plane of the other before summing.
    ``alternating``
        Sequential guidance: only verifier ``step_index % K`` acts this step.
    """
    if not grads:
        raise ValueError("compose_gradients requires at least one gradient")
    if len(grads) != len(weights):
        raise ValueError(f"got {len(grads)} gradients and {len(weights)} weights")

    if strategy == "normalized":
        grads = [g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8) for g in grads]
    weighted = [w * g for w, g in zip(weights, grads)]

    if strategy in ("sum", "normalized"):
        summed: Tensor = torch.stack(weighted, dim=0).sum(dim=0)
        return summed
    if strategy == "alternating":
        return weighted[step_index % len(weighted)] * float(len(weighted))
    if strategy == "projected":
        adjusted = [g.clone() for g in weighted]
        for i in range(len(adjusted)):
            for j in range(len(weighted)):
                if i == j:
                    continue
                other = weighted[j]
                dot = (adjusted[i] * other).sum(dim=-1, keepdim=True)
                denom = (other * other).sum(dim=-1, keepdim=True).clamp(min=1e-12)
                adjusted[i] = adjusted[i] - (dot.clamp(max=0.0) / denom) * other
        return torch.stack(adjusted, dim=0).sum(dim=0)
    raise ValueError(f"unknown composition strategy: {strategy!r}")


def _warn_missing_projector(specs: Sequence[GuidanceSpec],
                            projector: Optional[ManifoldProjector]) -> None:
    """Log once when specs request projection but no projector was supplied."""
    if projector is not None:
        return
    requested = [s.label for s in specs if s.project_to_manifold]
    if requested:
        logger.warning("manifold_projection_requested_but_unavailable", verifiers=requested,
                       detail="guiding unprojected; projector lands in W8")


@torch.no_grad()
def sample_guided(
    denoiser: nn.Module,
    schedule: NoiseSchedule,
    shape: Tuple[int, ...],
    guidance_specs: Sequence[GuidanceSpec] = (),
    context: Optional[Any] = None,
    num_steps: Optional[int] = None,
    seed: Optional[int] = None,
    eta: float = 0.0,
    strategy: CompositionStrategy = "sum",
    projector: Optional[ManifoldProjector] = None,
    device: Optional[torch.device] = None,
    clip_x0: Optional[float] = None,
    x_T: Optional[Tensor] = None,
    return_info: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
    """Sample with multi-verifier guidance applied at inference time.

    Parameters
    ----------
    guidance_specs:
        One :class:`GuidanceSpec` per verifier. An empty sequence recovers
        unguided sampling *exactly*: the guidance branch is never entered and
        the reverse update is the same code path as
        :func:`diffusion.base.sampling.unguided.sample`. Asserted in
        ``tests/test_guided_sampling.py``.
    projector:
        Optional manifold projector honoured by specs with
        ``project_to_manifold=True`` (W8).
    return_info:
        Also return per-step diagnostics: the guidance norm and the weighted
        gradient norm per verifier, plus the pairwise cosine similarity between
        verifier gradients. The cosines are how gradient cancellation is
        detected rather than silently accepted.
    """
    device = device or next(denoiser.parameters()).device
    schedule.to(device)
    specs = list(guidance_specs)
    _warn_missing_projector(specs, projector)
    generator = generator_from_seed(seed, device)
    x_t = torch.randn(shape, device=device, generator=generator) if x_T is None else x_T.to(device)

    info: Dict[str, Any] = {
        "guidance_norm": [],
        "per_verifier_norm": {spec.label: [] for spec in specs},
        "pairwise_cosine": {},
        "weights": {spec.label: [] for spec in specs},
    }
    denoiser.eval()

    for step_index, (t_cur, t_prev) in enumerate(step_pairs(schedule, num_steps)):
        t_batch = t_cur.reshape(1).expand(x_t.shape[0]).to(device)
        t_prev_batch = None if t_prev is None else t_prev.reshape(1).expand(x_t.shape[0]).to(device)
        eps = denoiser(x_t, t_batch)

        if specs:
            t_scalar = int(t_cur.item())
            grads = verifier_gradients(specs, x_t, t_batch, schedule, eps, context, projector)
            weights = [spec.weight_fn(t_scalar) for spec in specs]
            guidance = compose_gradients(grads, weights, strategy=strategy, step_index=step_index)
            sigma_t = schedule.sigma(t_batch, x_t.dim())
            eps = eps - sigma_t * guidance
            if return_info:
                _record(info, specs, grads, weights, guidance)

        noise = (torch.randn(x_t.shape, device=device, generator=generator)
                 if eta > 0.0 and t_prev is not None else None)
        x_t = reverse_step(schedule, x_t, t_batch, t_prev_batch, eps, eta=eta, noise=noise,
                           clip_x0=clip_x0)

    return (x_t, info) if return_info else x_t


def _record(info: Dict[str, Any], specs: Sequence[GuidanceSpec], grads: Sequence[Tensor],
            weights: Sequence[float], guidance: Tensor) -> None:
    """Append one step of guidance diagnostics to ``info``."""
    info["guidance_norm"].append(float(guidance.norm(dim=-1).mean()))
    for spec, grad, weight in zip(specs, grads, weights):
        info["per_verifier_norm"][spec.label].append(float((weight * grad).norm(dim=-1).mean()))
        info["weights"][spec.label].append(float(weight))
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            key = f"{specs[i].label}|{specs[j].label}"
            cosine = torch.nn.functional.cosine_similarity(grads[i], grads[j], dim=-1).mean()
            info["pairwise_cosine"].setdefault(key, []).append(float(cosine))
