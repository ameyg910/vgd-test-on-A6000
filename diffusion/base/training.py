"""Denoising score-matching training loop, EMA and checkpointing.

Scaled up in Phase 1 with EDM preconditioning, bf16 mixed precision, wandb
logging and resumable checkpoints; the loss and EMA logic here is the part that
carries over unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from diffusion.base.schedule import NoiseSchedule

__all__ = ["TrainConfig", "EMA", "diffusion_loss", "train_denoiser", "save_checkpoint",
           "load_checkpoint"]


@dataclass
class TrainConfig:
    """Hyper-parameters for :func:`train_denoiser`."""

    epochs: int = 300
    learning_rate: float = 2e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    ema_decay: Optional[float] = 0.999
    log_every: int = 50
    seed: int = 0
    lr_min_factor: float = 0.05
    history: List[float] = field(default_factory=list)


class EMA:
    """Exponential moving average of model parameters, used for sampling."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the current model parameters into the shadow copy."""
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key] = value.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        """Load the averaged parameters into ``model``."""
        model.load_state_dict(self.shadow)


def diffusion_loss(denoiser: nn.Module, schedule: NoiseSchedule, x_0: Tensor,
                   generator: Optional[torch.Generator] = None) -> Tensor:
    """Simple (unweighted) denoising score-matching loss ``E||eps - eps_theta||^2``."""
    batch = x_0.shape[0]
    t = torch.randint(0, schedule.num_timesteps, (batch,), device=x_0.device, generator=generator)
    noise = torch.randn(x_0.shape, device=x_0.device, generator=generator, dtype=x_0.dtype)
    x_t = schedule.q_sample(x_0, t, noise)
    return torch.mean((noise - denoiser(x_t, t)) ** 2)


def train_denoiser(denoiser: nn.Module, schedule: NoiseSchedule,
                   dataloader: DataLoader[Any],
                   config: Optional[TrainConfig] = None,
                   device: Optional[torch.device] = None,
                   progress: Optional[Callable[[int, float], None]] = None) -> TrainConfig:
    """Train ``denoiser`` in place; returns the config with its ``history`` filled in.

    An EMA copy of the weights is written back into ``denoiser`` at the end when
    ``config.ema_decay`` is set, since EMA weights sample noticeably better.
    """
    config = config or TrainConfig()
    device = device or next(denoiser.parameters()).device
    denoiser.to(device)
    schedule.to(device)

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs), eta_min=config.learning_rate * config.lr_min_factor
    )
    ema = EMA(denoiser, config.ema_decay) if config.ema_decay else None
    generator = torch.Generator(device=device).manual_seed(config.seed)

    denoiser.train()
    for epoch in range(config.epochs):
        epoch_loss, batches = 0.0, 0
        for batch in dataloader:
            x_0 = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            loss = diffusion_loss(denoiser, schedule, x_0, generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(denoiser.parameters(), config.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(denoiser)
            epoch_loss += float(loss.detach())
            batches += 1
        scheduler.step()
        mean_loss = epoch_loss / max(1, batches)
        config.history.append(mean_loss)
        if progress is not None and (epoch + 1) % config.log_every == 0:
            progress(epoch + 1, mean_loss)

    if ema is not None:
        ema.copy_to(denoiser)
    denoiser.eval()
    return config

def save_checkpoint(path: Path, denoiser: nn.Module, config: TrainConfig,
                    extra: Optional[Dict[str, Any]] = None) -> "Path":
    """Write a resumable checkpoint containing weights, config and provenance."""
    payload: Dict[str, Any] = {
        "state_dict": denoiser.state_dict(),
        "loss_history": list(config.history),
        "train_config": vars(config),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, denoiser: nn.Module,
                    map_location: Optional[Any] = None) -> Dict[str, Any]:
    """Load weights into ``denoiser`` and return the full checkpoint payload."""
    payload: Dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    denoiser.load_state_dict(payload["state_dict"])
    denoiser.eval()
    return payload
