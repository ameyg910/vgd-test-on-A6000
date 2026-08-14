"""EDM sampling (Karras Algorithm 2) with classifier-free and verifier guidance.

Guidance derivation for the EDM/VE parameterisation. With ``x = a_0 + sigma eps``
the score of the noisy marginal is

    grad_x log p(x; sigma) = (D_theta(x; sigma) - x) / sigma^2,

so tilting the marginal by ``prod_k V_k^{w_k}`` - which adds
``sum_k w_k grad log V_k`` to the score (Q1 of the screening writeup, unchanged)
- is equivalent to shifting the *denoiser output*:

    D_tilde(x; sigma) = D_theta(x; sigma) + sigma^2 * sum_k w_k grad_x log V_k(x).

This is the EDM analogue of ``eps_tilde = eps_theta - sigma_t sum_k w_k grad log V_k``
from the toy project, and the ``sigma^2`` prefactor plays the role the
``sigma_t`` prefactor played there. Note it grows with the noise level, so the
noise-aware weight shaping that was necessary in the toy problem is necessary
here too - see ``docs/debugging_log.md``.

Classifier-free guidance composes on top, applied to the denoiser before the
verifier term so that the two are not entangled:

    D_cfg = D_uncond + s (D_cond - D_uncond).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from diffusion.base.preconditioning import EDMConfig, EDMPreconditioner, karras_sigmas
from diffusion.base.sampling.guided import (CompositionStrategy, GuidanceSpec, ManifoldProjector,
                                            WeightFn, compose_gradients)
from diffusion.utils import generator_from_seed
from logging_config import get_logger

__all__ = ["sample_edm", "edm_verifier_gradients", "weight_from_sigma", "DenoiseFn"]

logger = get_logger(__name__)

DenoiseFn = Callable[[Tensor, Tensor], Tensor]


def weight_from_sigma(fn: Callable[[float], float], sigmas: Sequence[float]) -> WeightFn:
    """Adapt a ``sigma``-parameterised weight into the step-indexed ``WeightFn``.

    ``GuidanceSpec.weight_fn`` takes an integer because that is what the toy
    (discrete-time) sampler provides. For EDM the natural variable is the
    continuous noise level, so close over the schedule here rather than
    changing the API that Phase 0 fixed.
    """
    table = [float(s) for s in sigmas]

    def weight_fn(step: int) -> float:
        index = min(max(int(step), 0), len(table) - 1)
        return float(fn(table[index]))

    return weight_fn


def edm_verifier_gradients(specs: Sequence[GuidanceSpec], x: Tensor, sigma: Tensor,
                           denoised: Tensor, context: Optional[Any] = None,
                           projector: Optional[ManifoldProjector] = None) -> List[Tensor]:
    """Per-verifier ``grad_x log V_k`` for the EDM parameterisation.

    ``target="x0_hat"`` evaluates the verifier at the denoiser output - the
    Tweedie posterior mean, which in EDM *is* ``D_theta(x; sigma)`` - and uses
    the identity-Jacobian approximation, exactly as DPS does. That is the
    setting we expect to use in embedding space, where evaluating a policy
    verifier at a noisy point is meaningless.
    """
    grads: List[Tensor] = []
    for spec in specs:
        point = x if spec.target == "x_t" else denoised.detach()
        grad = spec.verifier.grad_log_value(point, sigma, context).to(x.dtype)
        if spec.grad_clip is not None:
            norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            grad = grad * (spec.grad_clip / norm).clamp(max=1.0)
        if spec.project_to_manifold and projector is not None:
            grad = projector(grad, x, sigma, denoised)
        grads.append(grad)
    return grads


def _denoise(preconditioner: EDMPreconditioner, x: Tensor, sigma: Tensor,
             conditioning: Optional[Dict[str, Any]], cfg_scale: float) -> Tensor:
    """Denoise with optional classifier-free guidance."""
    kwargs = dict(conditioning or {})
    if cfg_scale == 1.0 or not kwargs:
        plain: Tensor = preconditioner(x, sigma, **kwargs)
        return plain
    batch = x.shape[0]
    drop_off = torch.zeros(batch, dtype=torch.bool, device=x.device)
    drop_on = torch.ones(batch, dtype=torch.bool, device=x.device)
    conditional = preconditioner(x, sigma, **kwargs, drop_conditioning=drop_off)
    unconditional = preconditioner(x, sigma, **kwargs, drop_conditioning=drop_on)
    guided: Tensor = unconditional + cfg_scale * (conditional - unconditional)
    return guided


@torch.no_grad()
def sample_edm(
    preconditioner: EDMPreconditioner,
    shape: Tuple[int, ...],
    num_steps: int = 32,
    config: Optional[EDMConfig] = None,
    conditioning: Optional[Dict[str, Any]] = None,
    cfg_scale: float = 1.0,
    guidance_specs: Sequence[GuidanceSpec] = (),
    guidance_context: Optional[Any] = None,
    strategy: CompositionStrategy = "sum",
    projector: Optional[ManifoldProjector] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    s_churn: float = 0.0,
    s_min: float = 0.0,
    s_max: float = float("inf"),
    s_noise: float = 1.0,
    return_info: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
    """Deterministic (or churned) Heun sampler, EDM Algorithm 2.

    ``guidance_specs=[]`` and ``cfg_scale=1.0`` give the plain unconditional
    sampler; the guidance branch is not entered at all, which keeps the Phase 0
    invariant (empty specs == unguided) true in this sampler too.

    Parameters
    ----------
    s_churn, s_min, s_max, s_noise:
        Stochasticity controls from Karras §5. ``s_churn = 0`` is the
        deterministic probability-flow ODE and is the default; churn is worth
        trying if samples collapse onto too few modes.
    """
    config = config or preconditioner.config
    device = device or next(preconditioner.parameters()).device
    generator = generator_from_seed(seed, device)
    specs = list(guidance_specs)

    sigmas = karras_sigmas(num_steps, config, device=device)
    x = torch.randn(shape, device=device, generator=generator) * sigmas[0]
    info: Dict[str, Any] = {"sigmas": sigmas.tolist(), "guidance_norm": [],
                            "denoised_norm": []}

    def guided_denoise(x_in: Tensor, sigma_scalar: Tensor, step: int) -> Tensor:
        sigma_batch = sigma_scalar.expand(x_in.shape[0])
        denoised = _denoise(preconditioner, x_in, sigma_batch, conditioning, cfg_scale)
        if specs:
            grads = edm_verifier_gradients(specs, x_in, sigma_batch, denoised,
                                           guidance_context, projector)
            weights = [spec.weight_fn(step) for spec in specs]
            guidance = compose_gradients(grads, weights, strategy=strategy, step_index=step)
            denoised = denoised + (sigma_scalar ** 2) * guidance
            if return_info:
                info["guidance_norm"].append(float(guidance.norm(dim=-1).mean()))
        if return_info:
            info["denoised_norm"].append(float(denoised.norm(dim=-1).mean()))
        return denoised

    for step in range(num_steps):
        sigma_cur, sigma_next = sigmas[step], sigmas[step + 1]

        gamma = 0.0
        if s_churn > 0.0 and s_min <= float(sigma_cur) <= s_max:
            gamma = min(s_churn / num_steps, 2.0 ** 0.5 - 1.0)
        sigma_hat = sigma_cur * (1.0 + gamma)
        if gamma > 0.0:
            noise = torch.randn(x.shape, device=device, generator=generator) * s_noise
            x = x + (sigma_hat ** 2 - sigma_cur ** 2).clamp(min=0.0).sqrt() * noise

        denoised = guided_denoise(x, sigma_hat, step)
        derivative = (x - denoised) / sigma_hat
        x_next = x + (sigma_next - sigma_hat) * derivative

        if float(sigma_next) > 0.0:
            denoised_next = guided_denoise(x_next, sigma_next, step)
            derivative_next = (x_next - denoised_next) / sigma_next
            x_next = x + (sigma_next - sigma_hat) * 0.5 * (derivative + derivative_next)
        x = x_next

    return (x, info) if return_info else x
