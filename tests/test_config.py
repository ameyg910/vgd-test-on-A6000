"""Configuration, path resolution and experiment provenance.

Regression guard for the screening-task review finding: no module may hardcode
an absolute path, and every path must be overridable from the environment.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import pytest

from config import Settings, config_hash, get_settings, repo_root, resolve_device

SOURCE_DIRS = ("diffusion", "verifiers", "experiments", "scripts")
ABSOLUTE_PATH = re.compile(r"""["'](/home/|/Users/|/mnt/|[A-Z]:\\\\)""")


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_resolve_under_repo_root() -> None:
    settings = Settings()
    assert settings.artifact_dir == repo_root() / "artifacts"
    assert settings.figure_dir == repo_root() / "figures"
    assert settings.embedding_dim == 768


def test_environment_overrides_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDIFF_ARTIFACT_DIR", str(tmp_path / "scratch"))
    settings = Settings()
    assert settings.artifact_dir == tmp_path / "scratch"


def test_ensure_dirs_creates_everything(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATA_DIR", "ARTIFACT_DIR", "CHECKPOINT_DIR", "FIGURE_DIR", "CACHE_DIR",
                 "LOG_DIR"):
        monkeypatch.setenv(f"PDIFF_{name}", str(tmp_path / name.lower()))
    settings = Settings().ensure_dirs()
    for path in (settings.data_dir, settings.artifact_dir, settings.checkpoint_dir,
                 settings.figure_dir, settings.cache_dir, settings.log_dir):
        assert path.is_dir()


def test_path_settings_expand_user_and_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDIFF_SCRATCH_TEST", "/tmp/pdiff-scratch")
    monkeypatch.setenv("PDIFF_DATA_DIR", "$PDIFF_SCRATCH_TEST/data")
    assert Settings().data_dir == Path("/tmp/pdiff-scratch/data")


def test_repo_root_is_independent_of_cwd(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    expected = repo_root()
    monkeypatch.chdir(tmp_path)
    assert repo_root() == expected


def test_resolve_device_accepts_explicit_names() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_config_hash_is_stable_and_order_independent() -> None:
    first = config_hash({"lr": 1e-3, "seed": 0})
    second = config_hash({"seed": 0, "lr": 1e-3})
    assert first == second and len(first) == 12
    assert config_hash({"lr": 2e-3, "seed": 0}) != first


def test_no_hardcoded_absolute_paths_in_source() -> None:
    offenders = []
    for directory in SOURCE_DIRS:
        for path in (repo_root() / directory).rglob("*.py"):
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if ABSOLUTE_PATH.search(line):
                    offenders.append(f"{path.relative_to(repo_root())}:{number}: {line.strip()}")
    assert not offenders, "hardcoded absolute paths:\n" + "\n".join(offenders)


def test_secrets_are_not_settings_fields() -> None:
    """API keys come from the ambient environment, never from a settings field on disk."""
    forbidden = {"wandb_api_key", "hf_token", "openai_api_key", "api_key"}
    assert not forbidden & set(Settings.model_fields)
