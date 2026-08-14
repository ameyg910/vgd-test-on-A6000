"""EDM preconditioning (Karras et al., 2022), implemented exactly as specified.

The network ``F_theta`` never sees raw ``a`` or raw ``sigma``. The denoiser the
rest of the codebase talks to is

    D_theta(a; sigma) = c_skip(sigma) a + c_out(sigma) F_theta(c_in(sigma) a, c_noise(sigma))

with the four coefficients chosen (Karras §5, Table 1) so that the network's
input has unit variance for every noise level, its training target has unit
variance, and errors it makes are amplified as little as possible:

    c_in    = 1 / sqrt(sigma^2 + sigma_data^2)
    c_skip  = sigma_data^2 / (sigma^2 + sigma_data^2)
    c_out   = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
    c_noise = ln(sigma) / 4

The loss weight ``lambda(sigma) = 1 / c_out(sigma)^2`` then makes the effective
training objective uniform across noise levels: minimising
``lambda ||D - a_0||^2`` is exactly minimising ``||F_theta(...) - target||^2``
with a unit-variance target. Both forms are implemented below and
``tests/test_preconditioning.py`` asserts they agree.

``sigma_data`` is *not* a free constant: it is the standard deviation of the
data, and for 768-dim sentence embeddings it must be measured, not assumed.
See :func:`estimate_sigma_data`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = ["EDMConfig", "EDMPreconditioner", "estimate_sigma_data", "karras_sigmas",
           "sample_training_sigmas"]


@dataclass(frozen=True)
class EDMConfig:
    """EDM hyper-parameters. Defaults are Karras et al. (2022) Table 1 / §5.

    Attributes
    ----------
    sigma_data:
        Standard deviation of the (normalised) data. Measure it with
        :func:`estimate_sigma_data`; the 0.5 default is the image-domain value
        and is almost certainly wrong for embeddings.
    sigma_min, sigma_max:
        Range of noise levels used at sampling time.
    rho:
        Curvature of the sampling schedule; 7 is Karras' tuned value.
    p_mean, p_std:
        Log-normal training noise distribution: ``ln sigma ~ N(p_mean, p_std^2)``.
    """

    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    p_mean: float = -1.2
    p_std: float = 1.2

    def with_sigma_data(self, sigma_data: float) -> "EDMConfig":
        """Return a copy with ``sigma_data`` replaced by a measured value."""
        return EDMConfig(sigma_data=float(sigma_data), sigma_min=self.sigma_min,
                         sigma_max=self.sigma_max, rho=self.rho, p_mean=self.p_mean,
                         p_std=self.p_std)


def estimate_sigma_data(samples: Tensor) -> float:
    """Standard deviation of the data, pooled over all elements.

    EDM assumes a single scalar ``sigma_data``; for embeddings whose per-
    dimension variances differ this is the pooled value, and the per-dimension
    spread should be removed by whitening the data first (see
    ``diffusion/data/embeddings.py``).
    """
    return float(samples.detach().float().std(unbiased=True))


def _broadcast(sigma: Tensor, ndim: int) -> Tensor:
    """Reshape a ``(B,)`` sigma vector to broadcast against a ``ndim``-dim batch."""
    return sigma.reshape(-1, *((1,) * (ndim - 1)))


class EDMPreconditioner(nn.Module):
    """Wraps a raw network ``F_theta`` into the preconditioned denoiser ``D_theta``.

    The wrapped network is called with an already-scaled input and a scaled
    noise label, so it never has to cope with inputs whose magnitude varies over
    four orders of magnitude. Everything outside this class - samplers,
    guidance, metrics - sees only ``D_theta``, which maps a noisy sample to a
    prediction of the clean one.
    """

    def __init__(self, network: nn.Module, config: Optional[EDMConfig] = None) -> None:
        super().__init__()
        self.network = network
        self.config = config or EDMConfig()

    def coefficients(self, sigma: Tensor, ndim: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(c_skip, c_out, c_in, c_noise)`` broadcast to ``ndim`` dims.

        ``c_noise`` is returned with shape ``(B,)`` because it is a label for
        the time embedding, not a multiplier on the data.
        """
        sigma_data = self.config.sigma_data
        sigma_flat = sigma.reshape(-1)
        denom = sigma_flat ** 2 + sigma_data ** 2
        c_skip = sigma_data ** 2 / denom
        c_out = sigma_flat * sigma_data / denom.sqrt()
        c_in = 1.0 / denom.sqrt()
        c_noise = sigma_flat.log() / 4.0
        return (_broadcast(c_skip, ndim), _broadcast(c_out, ndim),
                _broadcast(c_in, ndim), c_noise)

    def loss_weight(self, sigma: Tensor, ndim: int = 2) -> Tensor:
        """``lambda(sigma) = 1 / c_out(sigma)^2``, the weight that flattens the objective."""
        sigma_data = self.config.sigma_data
        sigma_flat = sigma.reshape(-1)
        weight = (sigma_flat ** 2 + sigma_data ** 2) / (sigma_flat * sigma_data) ** 2
        return _broadcast(weight, ndim)

    def forward(self, x: Tensor, sigma: Tensor, **kwargs: object) -> Tensor:
        """Denoise: predict the clean sample ``a_0`` given ``x = a_0 + sigma * eps``.

        Extra keyword arguments (conditioning tensors, dropout flags) are passed
        straight through to the wrapped network.
        """
        c_skip, c_out, c_in, c_noise = self.coefficients(sigma, x.dim())
        raw = self.network(c_in * x, c_noise, **kwargs)
        denoised: Tensor = c_skip * x + c_out * raw
        return denoised

    def training_target(self, x_0: Tensor, noise: Tensor, sigma: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(network_input, network_target)`` for the unit-variance form.

        The network is asked to predict ``(a_0 - c_skip * x) / c_out``, whose
        variance is 1 at every noise level. Training on this with a plain MSE is
        equivalent to the weighted denoising loss; see
        :meth:`edm_loss` and the parity test.
        """
        c_skip, c_out, c_in, _ = self.coefficients(sigma, x_0.dim())
        x = x_0 + _broadcast(sigma.reshape(-1), x_0.dim()) * noise
        target = (x_0 - c_skip * x) / c_out
        return c_in * x, target

    def edm_loss(self, x_0: Tensor, sigma: Tensor, noise: Optional[Tensor] = None,
                 **kwargs: object) -> Tensor:
        """Weighted denoising loss ``lambda(sigma) ||D_theta(x; sigma) - a_0||^2``.

        Returned per-sample (shape ``(B,)``) so callers can log its dependence on
        ``sigma``, which is the first thing to inspect when training stalls.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        x = x_0 + _broadcast(sigma.reshape(-1), x_0.dim()) * noise
        denoised = self.forward(x, sigma, **kwargs)
        weight = self.loss_weight(sigma, x_0.dim())
        per_sample: Tensor = (weight * (denoised - x_0) ** 2).flatten(1).mean(dim=1)
        return per_sample


def sample_training_sigmas(batch_size: int, config: EDMConfig, device: torch.device,
                           generator: Optional[torch.Generator] = None) -> Tensor:
    """Draw ``sigma ~ exp(N(p_mean, p_std^2))``, the EDM training distribution.

    The log-normal concentrates samples where the denoising problem is neither
    trivial nor hopeless, which is where the network actually learns.
    """
    normal = torch.randn(batch_size, device=device, generator=generator)
    return (config.p_mean + config.p_std * normal).exp()


def karras_sigmas(num_steps: int, config: EDMConfig,
                  device: Optional[torch.device] = None) -> Tensor:
    """Sampling schedule from Karras §5, eq. 5, with a trailing zero.

    ``sigma_i = (sigma_max^(1/rho) + i/(N-1) (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho``
    for ``i < N``, followed by ``sigma_N = 0`` so the final step lands exactly on
    the data manifold.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    ramp = torch.linspace(0, 1, num_steps, device=device, dtype=torch.float64)
    min_inv = config.sigma_min ** (1.0 / config.rho)
    max_inv = config.sigma_max ** (1.0 / config.rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** config.rho
    return torch.cat([sigmas, torch.zeros(1, device=device, dtype=torch.float64)]).float()


def sigma_data_report(samples: Tensor) -> dict[str, float]:
    """Diagnostics needed before choosing ``sigma_data`` for a new corpus."""
    flat = samples.detach().float()
    return {
        "pooled_std": float(flat.std(unbiased=True)),
        "mean_norm": float(flat.norm(dim=-1).mean()),
        "per_dim_std_min": float(flat.std(dim=0, unbiased=True).min()),
        "per_dim_std_max": float(flat.std(dim=0, unbiased=True).max()),
        "per_dim_std_mean": float(flat.std(dim=0, unbiased=True).mean()),
        "mean_abs_dim_mean": float(flat.mean(dim=0).abs().mean()),
        "dimension": int(flat.shape[-1]),
        "count": int(flat.shape[0]),
    }
