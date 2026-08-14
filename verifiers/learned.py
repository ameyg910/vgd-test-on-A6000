"""Learned verifiers: a neural network scoring samples, differentiated by autograd.

This is the pattern every production verifier follows from Phase 2 onward - the
policy and tool verifiers are MLPs whose gradients reach the sampler through
``Verifier.autograd_grad``. The mode classifier below is the 2D stand-in used to
validate that path end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from verifiers.base import Verifier

__all__ = ["ModeClassifier", "ClassifierTrainConfig", "train_classifier", "classifier_accuracy",
           "ClassifierVerifier"]


class ModeClassifier(nn.Module):
    """MLP mapping a sample to logits over ``num_classes`` reference modes."""

    def __init__(self, input_dim: int, num_classes: int, width: int = 128,
                 num_layers: int = 3) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, width), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(B, num_classes)`` logits."""
        logits: Tensor = self.net(x)
        return logits


@dataclass
class ClassifierTrainConfig:
    """Hyper-parameters for :func:`train_classifier`."""

    epochs: int = 30
    learning_rate: float = 3e-3
    noise_std: float = 0.0
    seed: int = 0
    history: List[float] = field(default_factory=list)


def train_classifier(model: ModeClassifier, dataloader: DataLoader[Tuple[Tensor, Tensor]],
                     config: Optional[ClassifierTrainConfig] = None,
                     device: Optional[torch.device] = None) -> ClassifierTrainConfig:
    """Train the classifier on ``(x, label)`` batches; returns the filled config.

    ``config.noise_std`` optionally augments inputs with Gaussian noise, which
    widens the region where the learned gradient is informative.
    """
    config = config or ClassifierTrainConfig()
    device = device or next(model.parameters()).device
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device).manual_seed(config.seed)

    for _ in range(config.epochs):
        total, batches = 0.0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if config.noise_std > 0.0:
                x = x + config.noise_std * torch.randn(x.shape, device=device, generator=generator)
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        config.history.append(total / max(1, batches))
    model.eval()
    return config


@torch.no_grad()
def classifier_accuracy(model: ModeClassifier, x: Tensor, y: Tensor) -> float:
    """Top-1 accuracy of ``model`` on a labelled batch."""
    return float((model(x).argmax(dim=-1) == y).to(torch.float32).mean())


class ClassifierVerifier(Verifier):
    """Learned verifier: log-probability that ``x`` belongs to a preferred class set.

    ``log V(x) = logsumexp_{k in preferred} logits_k - logsumexp_k logits_k``.
    Gradients go through the generic autograd path, exactly as a policy or
    reward model would in the real project.
    """

    name = "classifier"

    def __init__(self, classifier: nn.Module, preferred_classes: Sequence[int],
                 name: Optional[str] = None) -> None:
        super().__init__(use_autograd=True, name=name)
        self.classifier = classifier
        self.preferred = tuple(int(c) for c in preferred_classes)

    def log_value(self, x: Tensor, t: Optional[Tensor] = None,
                  context: Optional[Any] = None) -> Tensor:
        logits = self.classifier(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        index = torch.tensor(self.preferred, device=x.device, dtype=torch.long)
        return torch.logsumexp(log_probs.index_select(-1, index), dim=-1)
