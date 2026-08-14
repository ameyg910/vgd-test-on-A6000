"""Plotting helpers for the 2D toy experiments.

Kept out of the library so that training jobs never import matplotlib.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

__all__ = ["scatter_2d"]


def scatter_2d(
    samples: torch.Tensor,
    ax: Optional[Axes] = None,
    title: str = "",
    mode_centers: Optional[torch.Tensor] = None,
    limit: float = 3.5,
    color: str = "#2b6cb0",
    alpha: float = 0.25,
    size: float = 4.0,
) -> Axes:
    """Scatter-plot the first two coordinates of ``samples`` on a fixed square frame."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.2))
    points = samples.detach().cpu().numpy()
    ax.scatter(points[:, 0], points[:, 1], s=size, alpha=alpha, c=color, linewidths=0)
    if mode_centers is not None:
        centers = mode_centers.detach().cpu().numpy()
        ax.scatter(centers[:, 0], centers[:, 1], s=28, c="#c53030", marker="x", linewidths=1.4)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return ax
