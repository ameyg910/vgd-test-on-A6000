"""Toy datasets. The real project swaps this module for sentence embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

__all__ = ["RingMixtureConfig", "ring_mode_centers", "sample_ring_mixture", "GaussianMixtureDataset",
           "build_dataloader"]


@dataclass(frozen=True)
class RingMixtureConfig:
    """Configuration of a mixture of isotropic Gaussians on a circle."""

    num_modes: int = 8
    radius: float = 2.0
    std: float = 0.1
    num_samples: int = 10_000
    seed: int = 0


def ring_mode_centers(num_modes: int = 8, radius: float = 2.0,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the ``(num_modes, 2)`` mode centres, evenly spaced on a circle."""
    angles = torch.arange(num_modes, dtype=dtype) * (2.0 * math.pi / num_modes)
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)


def sample_ring_mixture(config: RingMixtureConfig,
                        generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw ``config.num_samples`` points from the mixture.

    Returns
    -------
    (samples, labels)
        ``samples`` has shape ``(N, 2)``; ``labels`` holds the generating mode index.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(config.seed)
    centers = ring_mode_centers(config.num_modes, config.radius)
    labels = torch.randint(0, config.num_modes, (config.num_samples,), generator=generator)
    noise = torch.randn(config.num_samples, 2, generator=generator) * config.std
    return centers[labels] + noise, labels


class GaussianMixtureDataset(Dataset[Tuple[Tensor, Tensor]]):
    """``torch.utils.data.Dataset`` wrapper around the ring mixture.

    Items are ``(x, label)`` pairs. Labels are unused by the diffusion model but
    are needed by the learned-classifier verifier in the bonus experiment.
    """

    def __init__(self, config: Optional[RingMixtureConfig] = None) -> None:
        self.config = config or RingMixtureConfig()
        self.samples, self.labels = sample_ring_mixture(self.config)
        self.mode_centers = ring_mode_centers(self.config.num_modes, self.config.radius)

    @property
    def input_dim(self) -> int:
        """Dimension of a single datum."""
        return int(self.samples.shape[-1])

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index], self.labels[index]


def build_dataloader(dataset: Dataset[Tuple[Tensor, Tensor]], batch_size: int = 512,
                     shuffle: bool = True,
                     seed: Optional[int] = None) -> DataLoader[Tuple[Tensor, Tensor]]:
    """Create a ``DataLoader`` with an optionally seeded shuffling generator."""
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, drop_last=True)
