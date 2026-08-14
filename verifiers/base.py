"""The ``Verifier`` interface shared by every external constraint in the project.

Concrete verifiers arriving in Phase 2 - ``PolicyVerifier`` (V_P),
``ToolVerifier`` (V_tau) and ``EpisodicVerifier`` (V_E) - subclass this and are
consumed by the sampler through ``grad_log_value`` alone. The base class
implements that method with ``torch.autograd.grad`` so a neural verifier needs
no extra code; subclasses may override :meth:`analytical_grad` for a closed
form, and ``use_autograd=True`` forces the generic path for cross-checking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor

__all__ = ["Verifier"]


class Verifier(ABC):
    """Abstract external constraint scoring a batch of samples.

    Implementations must be batched: ``log_value`` maps ``(B, D)`` to ``(B,)``
    and ``grad_log_value`` maps ``(B, D)`` to ``(B, D)``, for any ``D``.
    """

    name: str = "verifier"

    def __init__(self, use_autograd: bool = False, name: Optional[str] = None) -> None:
        self.use_autograd = bool(use_autograd)
        if name is not None:
            self.name = name

    @abstractmethod
    def log_value(self, x: Tensor, t: Optional[Tensor] = None,
                  context: Optional[Any] = None) -> Tensor:
        """Unnormalised log-score ``log V(x)``, shape ``(B,)``."""

    def analytical_grad(self, x: Tensor, t: Optional[Tensor] = None,
                        context: Optional[Any] = None) -> Optional[Tensor]:
        """Closed-form ``grad_x log V(x)``, or ``None`` when unavailable."""
        return None

    def grad_log_value(self, x: Tensor, t: Optional[Tensor] = None,
                       context: Optional[Any] = None) -> Tensor:
        """Gradient of :meth:`log_value` w.r.t. ``x``, shape ``(B, D)``."""
        if not self.use_autograd:
            grad = self.analytical_grad(x, t, context)
            if grad is not None:
                return grad
        return self.autograd_grad(x, t, context)

    def autograd_grad(self, x: Tensor, t: Optional[Tensor] = None,
                      context: Optional[Any] = None) -> Tensor:
        """Generic autograd fallback; works inside ``torch.no_grad`` sampling loops."""
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            value = self.log_value(x_req, t, context)
            grad, = torch.autograd.grad(value.sum(), x_req)
        return grad.detach()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, use_autograd={self.use_autograd})"
