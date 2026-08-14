"""Evaluation for embedding-space diffusion: the W3 and W4 acceptance checks.

The workplan's checks translate into four measurable quantities:

* **Do samples look like advice embeddings?** :func:`distribution_report`
  compares norm and per-dimension variance, and adds a sliced-Wasserstein
  distance so "similar statistics" is a number rather than an impression. The
  moment-matching checks alone are weak - a model that memorised the data mean
  passes them - so the shuffled-baseline comparison in
  :func:`retrieval_report` is what actually discriminates.
* **Do decoded samples read as plausible advice?** Nearest-neighbour retrieval
  against the advice bank is the mechanical half; reading the text is the human
  half and cannot be automated away.
* **Does conditioning carry information?** :func:`conditional_report` measures
  recall of the true advice given its issue, against an unconditional control.
* **Does the conditional beat the unconditional on held-out reconstruction?**
  :func:`reconstruction_error` on issue-disjoint held-out pairs.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from diffusion.metrics import sliced_wasserstein

__all__ = ["distribution_report", "retrieval_report", "conditional_report",
           "reconstruction_error", "nearest_neighbour_indices"]


def _cosine(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise cosine similarity between rows of ``a`` and rows of ``b``."""
    a_norm = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b_norm = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    similarity: Tensor = a_norm @ b_norm.T
    return similarity


def nearest_neighbour_indices(queries: Tensor, bank: Tensor, k: int = 1) -> Tensor:
    """Indices of the ``k`` most cosine-similar bank rows for each query."""
    return _cosine(queries, bank).topk(k=min(k, bank.shape[0]), dim=-1).indices


def distribution_report(samples: Tensor, reference: Tensor, seed: int = 0) -> Dict[str, float]:
    """Compare generated and real embeddings on norm, per-dimension spread and SWD.

    ``per_dim_std_corr`` is the correlation between the per-dimension standard
    deviations of the two sets: it catches a model that matched the pooled scale
    while flattening the anisotropy, which is exactly what happens when the data
    was not whitened before training.
    """
    samples, reference = samples.detach().float(), reference.detach().float()
    sample_std = samples.std(dim=0, unbiased=True)
    reference_std = reference.std(dim=0, unbiased=True)
    stacked = torch.stack([sample_std, reference_std])
    correlation = float(torch.corrcoef(stacked)[0, 1]) if stacked.shape[1] > 1 else float("nan")
    return {
        "sample_norm_mean": float(samples.norm(dim=-1).mean()),
        "reference_norm_mean": float(reference.norm(dim=-1).mean()),
        "sample_pooled_std": float(samples.std(unbiased=True)),
        "reference_pooled_std": float(reference.std(unbiased=True)),
        "per_dim_std_corr": correlation,
        "per_dim_std_ratio": float((sample_std.mean() / reference_std.mean().clamp(min=1e-8))),
        "mean_shift": float((samples.mean(dim=0) - reference.mean(dim=0)).norm()),
        "sliced_wasserstein": sliced_wasserstein(samples, reference, num_projections=256,
                                                 seed=seed),
    }


def retrieval_report(samples: Tensor, bank: Tensor, seed: int = 0) -> Dict[str, float]:
    """Nearest-neighbour statistics against the advice bank.

    ``coverage`` is the fraction of bank entries that are some sample's nearest
    neighbour - the mode-collapse detector. ``gaussian_baseline_similarity``
    is the same statistic for matched Gaussian noise, so a high similarity can
    be read as evidence rather than assumed to be good.
    """
    similarity = _cosine(samples, bank)
    best = similarity.max(dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(samples.shape, generator=generator) * samples.std()
    baseline = _cosine(noise, bank).max(dim=-1).values
    return {
        "nn_similarity_mean": float(best.values.mean()),
        "nn_similarity_p10": float(best.values.quantile(0.1)),
        "coverage": float(torch.unique(best.indices).numel() / bank.shape[0]),
        "gaussian_baseline_similarity": float(baseline.mean()),
    }


def conditional_report(samples: Tensor, target_advice: Tensor, bank: Tensor,
                       ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """Recall@k of each sample's own target advice within the bank.

    ``samples[i]`` must be generated from the issue whose advice is
    ``target_advice[i]``, and ``target_advice[i]`` must appear in ``bank``.
    A conditional model should beat an unconditional one here; if it does not,
    the conditioning pathway is not carrying information regardless of what the
    loss curve says.
    """
    similarity = _cosine(samples, bank)
    target_index = _cosine(target_advice, bank).argmax(dim=-1)
    ranking = similarity.argsort(dim=-1, descending=True)
    hits = (ranking == target_index.unsqueeze(1))
    rank_of_target = hits.float().argmax(dim=-1)
    report = {f"recall@{k}": float((rank_of_target < k).float().mean()) for k in ks}
    report["median_rank"] = float(rank_of_target.median())
    report["target_similarity"] = float(
        torch.nn.functional.cosine_similarity(samples, target_advice, dim=-1).mean())
    return report


def reconstruction_error(samples: Tensor, target_advice: Tensor) -> Dict[str, float]:
    """Distance from generated advice to the true held-out advice."""
    cosine = torch.nn.functional.cosine_similarity(samples, target_advice, dim=-1)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_p10": float(cosine.quantile(0.1)),
        "l2_mean": float((samples - target_advice).norm(dim=-1).mean()),
    }
