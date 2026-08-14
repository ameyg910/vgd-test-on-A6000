"""Evaluation metrics. Nothing here is specific to 2D or to eight modes."""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
import numpy.typing as npt
import torch

from verifiers.base import Verifier
from torch import Tensor

__all__ = ["compliance_rate", "assign_to_references", "mode_coverage", "modes_covered",
           "mean_log_value", "wasserstein_distance"]

RegionFn = Callable[[Tensor], Tensor]


def compliance_rate(samples: Tensor, region_fn: RegionFn) -> float:
    """Fraction of ``samples`` for which ``region_fn`` is true.

    ``region_fn`` maps ``(N, D)`` to a boolean or 0/1 tensor of shape ``(N,)``.
    """
    mask = region_fn(samples)
    return float(mask.to(torch.float32).mean())


def assign_to_references(samples: Tensor, references: Tensor) -> Tensor:
    """Index of the nearest reference point for each sample, shape ``(N,)``."""
    distances = torch.cdist(samples.to(torch.float32), references.to(torch.float32).to(samples.device))
    return distances.argmin(dim=-1)


def mode_coverage(samples: Tensor, mode_centers: Tensor, normalized: bool = True) -> float:
    """Shannon entropy of the histogram of nearest-reference assignments.

    With ``normalized=True`` the entropy is divided by ``log(K)``, giving 1.0 for
    a uniform spread over the ``K`` references and 0.0 for total collapse.
    """
    num_refs = int(mode_centers.shape[0])
    counts = torch.bincount(assign_to_references(samples, mode_centers), minlength=num_refs)
    probs = counts.to(torch.float64) / counts.sum().clamp(min=1)
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())
    if normalized and num_refs > 1:
        entropy /= math.log(num_refs)
    return entropy


def modes_covered(samples: Tensor, mode_centers: Tensor, min_fraction: float = 0.01,
                  max_distance: Optional[float] = None) -> int:
    """Number of references holding at least ``min_fraction`` of the samples.

    When ``max_distance`` is given, samples further than that from their nearest
    reference are discarded first (they belong to no mode).
    """
    assignments = assign_to_references(samples, mode_centers)
    if max_distance is not None:
        distances = torch.cdist(samples.to(torch.float32),
                                mode_centers.to(torch.float32).to(samples.device))
        keep = distances.min(dim=-1).values <= max_distance
        assignments = assignments[keep]
    if assignments.numel() == 0:
        return 0
    counts = torch.bincount(assignments, minlength=int(mode_centers.shape[0]))
    return int((counts.to(torch.float64) / counts.sum() >= min_fraction).sum())


def mean_log_value(samples: Tensor, verifier: "Verifier") -> float:
    """Average ``log V(x)`` over ``samples`` for any object exposing ``log_value``."""
    return float(verifier.log_value(samples).mean())


def wasserstein_distance(samples_a: Tensor, samples_b: Tensor,
                         max_points: int = 2000, seed: int = 0) -> float:
    """2-Wasserstein distance between two point clouds of equal dimension.

    Uses ``scipy.stats.wasserstein_distance_nd`` when available and falls back to
    a sliced (random-projection) estimate, which is the practical choice in high
    dimension.
    """
    a = _subsample(samples_a, max_points, seed)
    b = _subsample(samples_b, max_points, seed + 1)
    try:
        from scipy.stats import wasserstein_distance_nd

        return float(wasserstein_distance_nd(a.cpu().numpy(), b.cpu().numpy()))
    except Exception:
        return sliced_wasserstein(a, b, num_projections=256, seed=seed)


def sliced_wasserstein(samples_a: Tensor, samples_b: Tensor, num_projections: int = 256,
                       seed: int = 0) -> float:
    """Sliced 1-Wasserstein estimate; dimension-agnostic and cheap."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dim = samples_a.shape[-1]
    directions = torch.randn(num_projections, dim, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    proj_a = (samples_a.cpu() @ directions.T).sort(dim=0).values
    proj_b = (samples_b.cpu() @ directions.T).sort(dim=0).values
    size = min(proj_a.shape[0], proj_b.shape[0])
    idx_a = torch.linspace(0, proj_a.shape[0] - 1, size).round().long()
    idx_b = torch.linspace(0, proj_b.shape[0] - 1, size).round().long()
    return float((proj_a[idx_a] - proj_b[idx_b]).abs().mean())


def _subsample(samples: Tensor, max_points: int, seed: int) -> Tensor:
    """Deterministically subsample rows when a point cloud is larger than ``max_points``."""
    if samples.shape[0] <= max_points:
        return samples.detach()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(samples.shape[0], generator=generator)[:max_points]
    return samples.detach().cpu()[idx]


def pareto_front(objectives: "npt.ArrayLike", maximize: bool = True) -> "npt.NDArray[np.bool_]":
    """Boolean mask of Pareto-optimal rows of an ``(N, M)`` objective matrix."""
    values = np.asarray(objectives, dtype=float)
    if not maximize:
        values = -values
    mask = np.ones(values.shape[0], dtype=bool)
    for i, point in enumerate(values):
        if not mask[i]:
            continue
        dominated = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        if dominated.any():
            mask[i] = False
    return mask
