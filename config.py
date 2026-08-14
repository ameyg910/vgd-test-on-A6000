"""Single source of truth for paths, devices and run-level settings.

Nothing in this repository constructs an absolute path itself. Everything goes
through :func:`get_settings`, which resolves in this order:

1. an explicit constructor argument,
2. an environment variable (prefix ``PDIFF_``, or a ``.env`` file),
3. a default derived from :func:`repo_root`, which is located relative to this
   file and therefore works from any working directory or machine.

Secrets (``WANDB_API_KEY``, HF tokens) are read from the environment only and
are never written to disk or logged.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings", "repo_root", "resolve_device", "config_hash"]

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def repo_root() -> Path:
    """Repository root, located relative to this file rather than the CWD."""
    return Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Run-level configuration.

    Every field can be overridden with an environment variable, e.g.
    ``PDIFF_ARTIFACT_DIR=/scratch/$USER/artifacts``. On the shared A6000 boxes
    that is how per-user scratch space is selected without touching the code.
    """

    model_config = SettingsConfigDict(env_prefix="PDIFF_", env_file=".env",
                                      env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Field(default_factory=lambda: repo_root() / "data")
    artifact_dir: Path = Field(default_factory=lambda: repo_root() / "artifacts")
    checkpoint_dir: Path = Field(default_factory=lambda: repo_root() / "artifacts" / "checkpoints")
    figure_dir: Path = Field(default_factory=lambda: repo_root() / "figures")
    cache_dir: Path = Field(default_factory=lambda: repo_root() / ".cache")
    log_dir: Path = Field(default_factory=lambda: repo_root() / "logs")

    device: DeviceName = "auto"
    seed: int = 0
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768

    wandb_project: str = "prescriptive-diffusion"
    wandb_entity: Optional[str] = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("data_dir", "artifact_dir", "checkpoint_dir", "figure_dir", "cache_dir",
                     "log_dir", mode="after")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        """Expand ``~`` and environment variables in path-valued settings."""
        return Path(os.path.expandvars(str(value))).expanduser()

    def ensure_dirs(self) -> "Settings":
        """Create every configured directory; safe to call repeatedly."""
        for path in (self.data_dir, self.artifact_dir, self.checkpoint_dir, self.figure_dir,
                     self.cache_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def torch_device(self) -> torch.device:
        """Resolve the configured device name to a concrete ``torch.device``."""
        return resolve_device(self.device)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached; call ``get_settings.cache_clear()`` in tests)."""
    return Settings()


def resolve_device(name: DeviceName = "auto") -> torch.device:
    """Map a device name to a device, falling back through CUDA, MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def config_hash(payload: Dict[str, Any], length: int = 12) -> str:
    """Stable short hash of a config dict, used to tag experiment provenance."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]
