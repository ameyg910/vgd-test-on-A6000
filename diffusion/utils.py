"""Seeding, RNG and determinism helpers.

Plot helpers deliberately live in ``experiments/``: the library must import
without pulling in matplotlib, so that training jobs on the cluster do not
depend on a display stack.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

__all__ = ["set_seed", "generator_from_seed", "enable_deterministic_algorithms"]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generator_from_seed(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    """Build a device-local generator, or ``None`` when ``seed`` is ``None``.

    Samplers take a local generator rather than seeding globally, so that a
    sampling call is reproducible without mutating the caller's RNG state.
    """
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def enable_deterministic_algorithms(warn_only: bool = True) -> None:
    """Request deterministic kernels; ``warn_only`` keeps non-deterministic ops usable.

    Standard 15 asks for determinism where possible and honesty where not: with
    ``warn_only=True`` PyTorch warns instead of raising for ops that have no
    deterministic implementation, and those warnings belong in the writeup.
    """
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False
