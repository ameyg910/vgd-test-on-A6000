"""Analytical toy verifiers used by the 2D validation and the W9 conflict study.

These are kept in the project because the toy experiments remain the cleanest
setting for ablating composition strategies and manifold projection. The
production verifiers live in ``verifiers/policy.py``, ``verifiers/tool.py`` and
``verifiers/episodic.py`` from Phase 2.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor

from verifiers.base import Verifier

__all__ = ["HalfPlaneVerifier", "TargetPointVerifier", "RegionIndicatorVerifier"]


class HalfPlaneVerifier(Verifier):
    """Linear preference for one side of a hyperplane: ``log V(x) = alpha * <x, n>``.

    The default normal ``e_0`` and ``alpha = 1`` reproduce the toy specification
    ``log V(x) = alpha * x_1``. Defined in any dimension.
    """

    name = "half_plane"

    def __init__(self, alpha: float = 1.0, dim: int = 0, normal: Optional[Tensor] = None,
                 use_autograd: bool = False, name: Optional[str] = None) -> None:
        super().__init__(use_autograd=use_autograd, name=name)
        self.alpha = float(alpha)
        self.dim = int(dim)
        self.normal = None if normal is None else normal.reshape(1, -1)

    def _direction(self, x: Tensor) -> Tensor:
        if self.normal is not None:
            normal = self.normal.to(device=x.device, dtype=x.dtype)
            unit: Tensor = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return unit
        direction = torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)
        direction[0, self.dim] = 1.0
        return direction

    def log_value(self, x: Tensor, t: Optional[Tensor] = None,
                  context: Optional[Any] = None) -> Tensor:
        value: Tensor = self.alpha * (x * self._direction(x)).sum(dim=-1)
        return value

    def analytical_grad(self, x: Tensor, t: Optional[Tensor] = None,
                        context: Optional[Any] = None) -> Optional[Tensor]:
        return self.alpha * self._direction(x).expand_as(x)


class TargetPointVerifier(Verifier):
    """Gaussian preference for a target point: ``log V(x) = -||x - c||^2 / (2 sigma^2)``."""

    name = "target_point"

    def __init__(self, center: Tensor, sigma: float = 0.5, use_autograd: bool = False,
                 name: Optional[str] = None) -> None:
        super().__init__(use_autograd=use_autograd, name=name)
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        self.center = center.reshape(1, -1)
        self.sigma = float(sigma)

    def _center(self, x: Tensor) -> Tensor:
        return self.center.to(device=x.device, dtype=x.dtype)

    def log_value(self, x: Tensor, t: Optional[Tensor] = None,
                  context: Optional[Any] = None) -> Tensor:
        delta = x - self._center(x)
        return -(delta ** 2).sum(dim=-1) / (2.0 * self.sigma ** 2)

    def analytical_grad(self, x: Tensor, t: Optional[Tensor] = None,
                        context: Optional[Any] = None) -> Optional[Tensor]:
        return -(x - self._center(x)) / (self.sigma ** 2)


class RegionIndicatorVerifier(Verifier):
    """Smooth (sigmoid-relaxed) membership in a region defined by ``constraint(x) > 0``.

    Kept as a template for tool-style verifiers whose exact value is a hard
    predicate: the relaxation is what makes the gradient informative.
    """

    name = "region"

    def __init__(self, constraint: Callable[[Tensor], Tensor], temperature: float = 0.1,
                 name: Optional[str] = None) -> None:
        super().__init__(use_autograd=True, name=name)
        self.constraint = constraint
        self.temperature = float(temperature)

    def log_value(self, x: Tensor, t: Optional[Tensor] = None,
                  context: Optional[Any] = None) -> Tensor:
        return torch.nn.functional.logsigmoid(self.constraint(x) / self.temperature)
