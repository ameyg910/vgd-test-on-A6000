"""(issue, advice) embedding pairs: loading, normalisation, and a synthetic stand-in.

The real corpus (~30k pairs from the Yelp pipeline, embedded with
``all-mpnet-base-v2``) is the Infra Student's W2 deliverable. Everything here is
written against the file contract below so the modelling work can proceed and be
tested before that lands, and so swapping in the real file changes no code:

    <data_dir>/advice_pairs.npz
        issue:   float32 (N, 768)
        advice:  float32 (N, 768)
        issue_id: int64 (N,)        # groups paraphrases of the same issue

:class:`SyntheticEmbeddingCorpus` generates data with the statistics that matter
for diffusion - near-unit norm, anisotropic per-dimension variance, clustered
rather than Gaussian - so that a failure on synthetic data is informative about
a failure on real data.

Normalisation is not cosmetic here. EDM assumes a single scalar ``sigma_data``;
sentence embeddings are unit-norm with per-dimension standard deviations that
vary by an order of magnitude, so without whitening the noise level that is
"medium" for one coordinate is "already destroyed" for another. That is the
fourth failure mode listed in the workplan ("samples don't converge to data
marginals"), and :class:`EmbeddingNormalizer` is the fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

__all__ = ["EmbeddingNormalizer", "EmbeddingPairDataset", "SyntheticEmbeddingCorpus",
           "build_pair_dataloader", "load_pairs"]


@dataclass
class EmbeddingNormalizer:
    """Per-dimension standardisation with an optional global rescale.

    ``mode="whiten"`` divides each dimension by its own standard deviation,
    which is what makes a single scalar ``sigma_data`` meaningful.
    ``mode="global"`` keeps the shape of the distribution and only centres and
    rescales it; use it for the ablation that shows whitening was necessary.
    """

    mean: Tensor
    std: Tensor
    mode: str = "whiten"
    eps: float = 1e-6

    @classmethod
    def fit(cls, samples: Tensor, mode: str = "whiten") -> "EmbeddingNormalizer":
        """Estimate normalisation statistics from a sample of embeddings."""
        if mode not in ("whiten", "global", "none"):
            raise ValueError(f"unknown normaliser mode {mode!r}")
        samples = samples.detach().float()
        mean = samples.mean(dim=0)
        if mode == "whiten":
            std = samples.std(dim=0, unbiased=True)
        elif mode == "global":
            std = samples.std(unbiased=True).expand(samples.shape[-1]).clone()
        else:
            mean = torch.zeros_like(mean)
            std = torch.ones_like(mean)
        return cls(mean=mean, std=std, mode=mode)

    def to(self, device: torch.device) -> "EmbeddingNormalizer":
        """Move statistics to ``device`` (in place) and return self."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, x: Tensor) -> Tensor:
        """Map raw embeddings into the normalised space the model trains in."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device).clamp(min=self.eps)

    def decode(self, x: Tensor) -> Tensor:
        """Inverse of :meth:`encode`; use before nearest-neighbour retrieval."""
        return x * self.std.to(x.device).clamp(min=self.eps) + self.mean.to(x.device)

    def state_dict(self) -> Dict[str, object]:
        """Serialisable statistics, stored alongside every checkpoint."""
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "mode": self.mode,
                "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "EmbeddingNormalizer":
        """Rebuild from :meth:`state_dict`."""
        mean = state["mean"]
        std = state["std"]
        assert isinstance(mean, Tensor) and isinstance(std, Tensor)
        eps = state["eps"]
        assert isinstance(eps, (int, float))
        return cls(mean=mean, std=std, mode=str(state["mode"]), eps=float(eps))


def load_pairs(path: Path) -> Dict[str, Tensor]:
    """Load an ``advice_pairs.npz`` file into tensors, validating the contract."""
    with np.load(path) as handle:
        missing = {"issue", "advice"} - set(handle.files)
        if missing:
            raise KeyError(f"{path} missing arrays: {sorted(missing)}")
        issue = torch.from_numpy(handle["issue"]).float()
        advice = torch.from_numpy(handle["advice"]).float()
        issue_id = (torch.from_numpy(handle["issue_id"]).long() if "issue_id" in handle.files
                    else torch.arange(issue.shape[0], dtype=torch.long))
    if issue.shape != advice.shape:
        raise ValueError(f"issue {tuple(issue.shape)} and advice {tuple(advice.shape)} differ")
    return {"issue": issue, "advice": advice, "issue_id": issue_id}


class EmbeddingPairDataset(Dataset[Dict[str, Tensor]]):
    """(issue, advice) embedding pairs, normalised on construction.

    Items are dictionaries so the conditional and unconditional training loops
    can share one dataset: the unconditional run simply ignores ``issue``.
    """

    def __init__(self, issue: Tensor, advice: Tensor,
                 issue_id: Optional[Tensor] = None,
                 normalizer: Optional[EmbeddingNormalizer] = None,
                 fit_on: str = "advice") -> None:
        if normalizer is None:
            source = advice if fit_on == "advice" else torch.cat([advice, issue], dim=0)
            normalizer = EmbeddingNormalizer.fit(source)
        self.normalizer = normalizer
        self.advice = normalizer.encode(advice)
        self.issue = normalizer.encode(issue)
        self.issue_id = (issue_id if issue_id is not None
                         else torch.arange(advice.shape[0], dtype=torch.long))

    @classmethod
    def from_npz(cls, path: Path, normalizer: Optional[EmbeddingNormalizer] = None
                 ) -> "EmbeddingPairDataset":
        """Build from the corpus file contract documented at module level."""
        arrays = load_pairs(path)
        return cls(arrays["issue"], arrays["advice"], arrays["issue_id"], normalizer)

    @property
    def embedding_dim(self) -> int:
        """Dimension of a single embedding."""
        return int(self.advice.shape[-1])

    def __len__(self) -> int:
        return int(self.advice.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {"advice": self.advice[index], "issue": self.issue[index],
                "issue_id": self.issue_id[index]}

    def split(self, holdout: int, seed: int = 0
              ) -> Tuple["EmbeddingPairDataset", "EmbeddingPairDataset"]:
        """Split by ``issue_id`` so paraphrases never straddle train and test.

        Splitting by row would leak: four paraphrases of one advice would put
        near-duplicates on both sides and make the held-out reconstruction check
        meaningless.
        """
        unique = torch.unique(self.issue_id)
        generator = torch.Generator().manual_seed(seed)
        order = unique[torch.randperm(unique.numel(), generator=generator)]
        held = set(order[:holdout].tolist())
        mask = torch.tensor([int(i) in held for i in self.issue_id.tolist()], dtype=torch.bool)
        return (self._subset(~mask), self._subset(mask))

    def _subset(self, mask: Tensor) -> "EmbeddingPairDataset":
        """Rows selected by ``mask``, sharing this dataset's normaliser."""
        subset = EmbeddingPairDataset.__new__(EmbeddingPairDataset)
        subset.normalizer = self.normalizer
        subset.advice = self.advice[mask]
        subset.issue = self.issue[mask]
        subset.issue_id = self.issue_id[mask]
        return subset


class SyntheticEmbeddingCorpus:
    """Stand-in corpus with embedding-like statistics, for tests and smoke runs.

    Advice embeddings are drawn from ``num_clusters`` anisotropic Gaussians on a
    sphere; the issue embedding for a pair is a noisy rotation of its advice, so
    conditioning carries real information and a conditional model can be shown
    to beat an unconditional one. This is a stand-in for the pipeline output,
    not a claim about real advice geometry.
    """

    def __init__(self, num_pairs: int = 4096, dim: int = 768, num_clusters: int = 16,
                 paraphrases: int = 4, cluster_std: float = 0.25,
                 issue_noise: float = 0.6, seed: int = 0) -> None:
        self.dim = dim
        generator = torch.Generator().manual_seed(seed)
        centers = torch.randn(num_clusters, dim, generator=generator)
        centers = centers / centers.norm(dim=-1, keepdim=True)
        scale = torch.rand(dim, generator=generator) * 0.9 + 0.1

        groups = max(1, num_pairs // paraphrases)
        assign = torch.randint(0, num_clusters, (groups,), generator=generator)
        base = centers[assign] + cluster_std * scale * torch.randn(groups, dim,
                                                                  generator=generator)
        advice = base.repeat_interleave(paraphrases, dim=0)
        advice = advice + 0.1 * cluster_std * scale * torch.randn(advice.shape,
                                                                  generator=generator)
        issue_base = base + issue_noise * scale * torch.randn(groups, dim, generator=generator)
        issue = issue_base.repeat_interleave(paraphrases, dim=0)

        self.advice = advice / advice.norm(dim=-1, keepdim=True)
        self.issue = issue / issue.norm(dim=-1, keepdim=True)
        self.issue_id = torch.arange(groups).repeat_interleave(paraphrases)

    def dataset(self, normalizer: Optional[EmbeddingNormalizer] = None) -> EmbeddingPairDataset:
        """Wrap the synthetic arrays in the same dataset the real corpus uses."""
        return EmbeddingPairDataset(self.issue, self.advice, self.issue_id, normalizer)

    def save_npz(self, path: Path) -> Path:
        """Write to the corpus file contract, for end-to-end pipeline tests."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, issue=self.issue.numpy(), advice=self.advice.numpy(),
                 issue_id=self.issue_id.numpy())
        return path


def build_pair_dataloader(dataset: Dataset[Dict[str, Tensor]], batch_size: int = 256,
                          shuffle: bool = True, seed: Optional[int] = None,
                          num_workers: int = 0) -> DataLoader[Dict[str, Tensor]]:
    """DataLoader with a seeded shuffling generator."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      drop_last=True, num_workers=num_workers)
