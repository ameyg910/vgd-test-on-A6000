"""Training loop for EDM-preconditioned embedding diffusion.

Covers the cross-cutting standards that Phase 1 turns on: bf16 autocast,
wandb logging of loss curves and gradient norms, EMA weights for sampling,
resumable checkpoints, and a GPU-hours summary printed at exit.

The loss is logged per noise-level bucket as well as in aggregate. That is not
decoration: when embedding diffusion fails, the aggregate loss usually still
goes down, and the diagnostic signal is a bucket - typically the high-``sigma``
one - that flatlines while the others improve.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from diffusion.base.preconditioning import EDMConfig, EDMPreconditioner, sample_training_sigmas
from logging_config import get_logger

__all__ = ["EDMTrainConfig", "TrainState", "train_edm", "save_edm_checkpoint",
           "load_edm_checkpoint", "loss_by_sigma_bucket"]

logger = get_logger(__name__)


@dataclass
class EDMTrainConfig:
    """Optimisation hyper-parameters for :func:`train_edm`."""

    steps: int = 20_000
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 500
    grad_clip: Optional[float] = 1.0
    ema_decay: float = 0.9995
    cfg_dropout: float = 0.1
    batch_size: int = 256
    log_every: int = 100
    checkpoint_every: int = 2_000
    seed: int = 0
    amp_dtype: str = "bf16"
    conditional: bool = True
    num_sigma_buckets: int = 4


@dataclass
class TrainState:
    """Everything needed to resume a run and to report what it cost."""

    step: int = 0
    gpu_seconds: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    best_loss: float = float("inf")

    def summary(self) -> Dict[str, float]:
        """Compact end-of-run report (standard 9: GPU-hours accounting)."""
        return {"steps": float(self.step), "gpu_hours": self.gpu_seconds / 3600.0,
                "final_loss": self.loss_history[-1] if self.loss_history else float("nan"),
                "best_loss": self.best_loss}


class EMA:
    """Exponential moving average of parameters, with warmup-aware decay."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, Tensor] = {k: v.detach().clone()
                                          for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Blend current weights into the shadow copy.

        Early in training the shadow is warmed up with a smaller effective decay
        so the first sampled checkpoints are not dominated by initialisation.
        """
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self.shadow[key] = value.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        """Load averaged weights into ``model``."""
        model.load_state_dict(self.shadow)


def _autocast_dtype(name: str) -> Optional[torch.dtype]:
    """Map a config string to a torch dtype, or ``None`` for full precision."""
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None,
            "none": None}.get(name, None)


def _lr_at(step: int, config: EDMTrainConfig) -> float:
    """Linear warmup then cosine decay to 5% of the peak."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.learning_rate * (0.05 + 0.95 * cosine)


def loss_by_sigma_bucket(sigma: Tensor, loss: Tensor, config: EDMConfig,
                         num_buckets: int = 4) -> Dict[str, float]:
    """Mean loss per log-sigma bucket, the main training diagnostic."""
    log_sigma = sigma.detach().float().log()
    low, high = math.log(config.sigma_min), math.log(config.sigma_max)
    edges = torch.linspace(low, high, num_buckets + 1, device=log_sigma.device)
    out: Dict[str, float] = {}
    for index in range(num_buckets):
        mask = (log_sigma >= edges[index]) & (log_sigma < edges[index + 1])
        if bool(mask.any()):
            out[f"loss/sigma_bucket_{index}"] = float(loss.detach()[mask].mean())
    return out


def _cycle(loader: DataLoader[Dict[str, Tensor]]) -> Iterator[Dict[str, Tensor]]:
    """Infinite iterator over a finite dataloader."""
    while True:
        for batch in loader:
            yield batch


def train_edm(
    preconditioner: EDMPreconditioner,
    loader: DataLoader[Dict[str, Tensor]],
    config: Optional[EDMTrainConfig] = None,
    device: Optional[torch.device] = None,
    state: Optional[TrainState] = None,
    wandb_run: Optional[Any] = None,
    checkpoint_path: Optional[Path] = None,
    extra_checkpoint: Optional[Dict[str, Any]] = None,
) -> TrainState:
    """Train ``preconditioner.network`` with the EDM objective.

    Returns the :class:`TrainState`, and leaves EMA weights loaded into the
    model so the caller can sample immediately. Pass ``state`` from
    :func:`load_edm_checkpoint` to resume.
    """
    config = config or EDMTrainConfig()
    state = state or TrainState()
    device = device or next(preconditioner.parameters()).device
    preconditioner.to(device)

    optimizer = torch.optim.AdamW(preconditioner.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, betas=(0.9, 0.99))
    ema = EMA(preconditioner, config.ema_decay)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    amp_dtype = _autocast_dtype(config.amp_dtype)
    use_amp = amp_dtype is not None and device.type in ("cuda", "cpu")

    preconditioner.train()
    batches = _cycle(loader)
    start = time.time()

    for step in range(state.step, config.steps):
        batch = next(batches)
        advice = batch["advice"].to(device, non_blocking=True)
        kwargs: Dict[str, Any] = {}
        if config.conditional and "issue" in batch:
            kwargs["issue"] = batch["issue"].to(device, non_blocking=True)
            kwargs["drop_conditioning"] = (
                torch.rand(advice.shape[0], device=device, generator=generator)
                < config.cfg_dropout)

        sigma = sample_training_sigmas(advice.shape[0], preconditioner.config, device, generator)
        for group in optimizer.param_groups:
            group["lr"] = _lr_at(step, config)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            per_sample = preconditioner.edm_loss(advice, sigma, **kwargs)
            loss = per_sample.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        grad_norm = float("nan")
        if config.grad_clip is not None:
            grad_norm = float(nn.utils.clip_grad_norm_(preconditioner.parameters(),
                                                       config.grad_clip))
        optimizer.step()
        ema.update(preconditioner, step)

        loss_value = float(loss.detach())
        state.step = step + 1
        state.loss_history.append(loss_value)
        state.best_loss = min(state.best_loss, loss_value)

        if (step + 1) % config.log_every == 0:
            metrics: Dict[str, float] = {
                "loss": loss_value, "lr": _lr_at(step, config), "grad_norm": grad_norm,
                **loss_by_sigma_bucket(sigma, per_sample, preconditioner.config,
                                       config.num_sigma_buckets),
            }
            logger.info("train_step", step=step + 1, **{k: round(v, 5) for k, v in
                                                        metrics.items()})
            if wandb_run is not None:
                wandb_run.log(metrics, step=step + 1)

        if checkpoint_path is not None and (step + 1) % config.checkpoint_every == 0:
            state.gpu_seconds = time.time() - start
            save_edm_checkpoint(checkpoint_path, preconditioner, ema, config, state,
                                extra_checkpoint)

    state.gpu_seconds = time.time() - start
    ema.copy_to(preconditioner)
    preconditioner.eval()
    if checkpoint_path is not None:
        save_edm_checkpoint(checkpoint_path, preconditioner, ema, config, state,
                            extra_checkpoint)
    logger.info("train_done", **{k: round(v, 4) for k, v in state.summary().items()})
    return state


def save_edm_checkpoint(path: Path, preconditioner: EDMPreconditioner, ema: Optional[EMA],
                        config: EDMTrainConfig, state: TrainState,
                        extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write a resumable checkpoint (weights, EMA, optimiser-free)."""
    payload: Dict[str, Any] = {
        "network": preconditioner.network.state_dict(),
        "ema": ema.shadow if ema is not None else None,
        "edm_config": asdict(preconditioner.config),
        "train_config": asdict(config),
        "state": asdict(state),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_edm_checkpoint(path: Path, network: nn.Module,
                        map_location: Optional[Any] = None,
                        use_ema: bool = True) -> Dict[str, Any]:
    """Restore a checkpoint into ``network``; returns the full payload.

    With ``use_ema`` the averaged weights are loaded, which is what should be
    used for every reported sample.
    """
    payload: Dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    if use_ema and payload.get("ema"):
        preconditioner_state = {k[len("network."):]: v for k, v in payload["ema"].items()
                                if k.startswith("network.")}
        network.load_state_dict(preconditioner_state or payload["network"])
    else:
        network.load_state_dict(payload["network"])
    return payload
