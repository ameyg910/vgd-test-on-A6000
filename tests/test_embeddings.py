"""Embedding corpus: normalisation, the file contract, and leak-free splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from diffusion.base.preconditioning import estimate_sigma_data, sigma_data_report
from diffusion.data.embeddings import (EmbeddingNormalizer, EmbeddingPairDataset,
                                       SyntheticEmbeddingCorpus, build_pair_dataloader,
                                       load_pairs)


def test_whitening_makes_per_dimension_scale_uniform() -> None:
    """The point of whitening: one scalar sigma_data becomes meaningful."""
    torch.manual_seed(0)
    raw = torch.randn(2_000, 16) * torch.linspace(0.05, 2.0, 16)
    before = sigma_data_report(raw)
    assert before["per_dim_std_max"] / before["per_dim_std_min"] > 10.0

    normalizer = EmbeddingNormalizer.fit(raw)
    after = sigma_data_report(normalizer.encode(raw))
    assert after["per_dim_std_max"] / after["per_dim_std_min"] < 1.2
    assert estimate_sigma_data(normalizer.encode(raw)) == pytest.approx(1.0, abs=0.05)


def test_normalizer_roundtrip_is_lossless() -> None:
    torch.manual_seed(0)
    raw = torch.randn(500, 32) * 0.3 + 1.5
    normalizer = EmbeddingNormalizer.fit(raw)
    assert torch.allclose(normalizer.decode(normalizer.encode(raw)), raw, atol=1e-4)


def test_global_mode_preserves_anisotropy() -> None:
    """The ablation baseline must *not* equalise dimensions."""
    torch.manual_seed(0)
    raw = torch.randn(2_000, 16) * torch.linspace(0.05, 2.0, 16)
    report = sigma_data_report(EmbeddingNormalizer.fit(raw, mode="global").encode(raw))
    assert report["per_dim_std_max"] / report["per_dim_std_min"] > 10.0


def test_normalizer_state_dict_roundtrip() -> None:
    torch.manual_seed(0)
    normalizer = EmbeddingNormalizer.fit(torch.randn(100, 8) * 2.0)
    restored = EmbeddingNormalizer.from_state_dict(normalizer.state_dict())
    x = torch.randn(4, 8)
    assert torch.allclose(normalizer.encode(x), restored.encode(x), atol=1e-6)
    assert restored.mode == normalizer.mode


def test_dataset_items_and_shapes() -> None:
    corpus = SyntheticEmbeddingCorpus(num_pairs=64, dim=32, num_clusters=4, seed=0)
    dataset = corpus.dataset()
    item = dataset[0]
    assert set(item) == {"advice", "issue", "issue_id"}
    assert item["advice"].shape == (32,) and dataset.embedding_dim == 32
    assert len(dataset) == 64


def test_split_never_leaks_paraphrases_across_sides() -> None:
    """Splitting by row would put near-duplicates on both sides; split by issue."""
    corpus = SyntheticEmbeddingCorpus(num_pairs=200, dim=16, num_clusters=5,
                                      paraphrases=4, seed=0)
    dataset = corpus.dataset()
    train, test = dataset.split(holdout=10, seed=0)
    assert len(train) + len(test) == len(dataset)
    overlap = set(train.issue_id.tolist()) & set(test.issue_id.tolist())
    assert not overlap
    assert train.normalizer is dataset.normalizer


def test_npz_contract_roundtrip(tmp_path: Path) -> None:
    corpus = SyntheticEmbeddingCorpus(num_pairs=32, dim=16, num_clusters=2, seed=1)
    path = corpus.save_npz(tmp_path / "advice_pairs.npz")
    arrays = load_pairs(path)
    assert set(arrays) == {"issue", "advice", "issue_id"}
    assert arrays["advice"].shape == (32, 16)
    dataset = EmbeddingPairDataset.from_npz(path)
    assert len(dataset) == 32


def test_load_pairs_rejects_missing_arrays(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, advice=np.zeros((4, 8), dtype=np.float32))
    with pytest.raises(KeyError):
        load_pairs(path)


def test_synthetic_conditioning_is_informative() -> None:
    """Issue must predict its advice better than a random other advice.

    If this fails the synthetic corpus cannot distinguish a working conditional
    model from a broken one, and the W4 sanity checks would be vacuous.
    """
    corpus = SyntheticEmbeddingCorpus(num_pairs=256, dim=64, num_clusters=8, seed=0)
    matched = torch.nn.functional.cosine_similarity(corpus.issue, corpus.advice, dim=-1)
    shuffled = torch.nn.functional.cosine_similarity(
        corpus.issue, corpus.advice[torch.randperm(corpus.advice.shape[0])], dim=-1)
    assert float(matched.mean()) > float(shuffled.mean()) + 0.2


def test_dataloader_is_seeded_and_drops_last() -> None:
    dataset = SyntheticEmbeddingCorpus(num_pairs=70, dim=16, seed=0).dataset()
    first = next(iter(build_pair_dataloader(dataset, batch_size=32, seed=0)))
    second = next(iter(build_pair_dataloader(dataset, batch_size=32, seed=0)))
    assert torch.equal(first["advice"], second["advice"])
    assert first["advice"].shape == (32, 16)
