"""Train and evaluate the embedding-space diffusion model.

    python -m scripts.train_embedding_diffusion --preset smoke     # CPU sanity run
    python -m scripts.train_embedding_diffusion --unconditional    # W3
    python -m scripts.train_embedding_diffusion                    # W4

The W3 instruction is to train unconditionally *first* and stop if it does not
converge; ``--unconditional`` is that run, and its report is what gates moving
on. Both runs share this script so the only difference between them is the flag,
not a second code path that could diverge.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from pydantic import BaseModel, Field

from config import Settings, config_hash, get_settings
from diffusion.base.edm_training import EDMTrainConfig, TrainState, train_edm
from diffusion.base.preconditioning import (EDMConfig, EDMPreconditioner, estimate_sigma_data,
                                            sigma_data_report)
from diffusion.base.sampling.edm import sample_edm
from diffusion.base.transformer import TransformerDenoiser, TransformerDenoiserConfig
from diffusion.data.embeddings import (EmbeddingPairDataset, SyntheticEmbeddingCorpus,
                                       build_pair_dataloader)
from diffusion.eval.embedding_metrics import (conditional_report, distribution_report,
                                              reconstruction_error, retrieval_report)
from diffusion.utils import set_seed
from logging_config import configure_logging, get_logger

logger = get_logger(__name__)


class DataConfig(BaseModel):
    corpus: str = "advice_pairs.npz"
    synthetic: bool = True
    synthetic_pairs: int = 30_000
    dim: int = 768
    normalize: str = "whiten"
    holdout_issues: int = 200


class ModelConfig(BaseModel):
    num_tokens: int = 12
    hidden_dim: int = 512
    depth: int = 8
    num_heads: int = 8
    time_embed_dim: int = 256


class EDMSection(BaseModel):
    sigma_data: Optional[float] = None
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    p_mean: float = -1.2
    p_std: float = 1.2


class TrainSection(BaseModel):
    steps: int = 60_000
    batch_size: int = 256
    learning_rate: float = 2e-4
    warmup_steps: int = 500
    ema_decay: float = 0.9995
    cfg_dropout: float = 0.1
    grad_clip: Optional[float] = 1.0
    amp_dtype: str = "bf16"
    conditional: bool = True
    log_every: int = 100
    checkpoint_every: int = 2_000


class SampleSection(BaseModel):
    num_steps: int = 32
    num_samples: int = 1_000
    cfg_scale: float = 2.0


class RunConfig(BaseModel):
    """Validated top-level config; hashed into the artifact path."""

    name: str = "embedding_diffusion"
    seed: int = 0
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    edm: EDMSection = Field(default_factory=EDMSection)
    train: TrainSection = Field(default_factory=TrainSection)
    sample: SampleSection = Field(default_factory=SampleSection)

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        """Parse and validate the YAML config."""
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def apply_smoke_preset(self) -> "RunConfig":
        """Shrink everything so the full pipeline runs on a laptop in minutes.

        This exists so the pipeline is exercised end to end before it is trusted
        with GPU-hours; it is not a substitute for the real run.
        """
        updated = self.model_copy(deep=True)
        updated.name = f"{self.name}_smoke"
        updated.data.synthetic = True
        updated.data.synthetic_pairs = 2_048
        updated.data.dim = 128
        updated.data.holdout_issues = 32
        updated.model = ModelConfig(num_tokens=8, hidden_dim=128, depth=4, num_heads=4,
                                    time_embed_dim=64)
        updated.train = TrainSection(steps=1_500, batch_size=128, learning_rate=1e-3,
                                     warmup_steps=100, ema_decay=0.995,
                                     cfg_dropout=self.train.cfg_dropout, amp_dtype="fp32",
                                     conditional=self.train.conditional, log_every=250,
                                     checkpoint_every=1_000)
        updated.sample = SampleSection(num_steps=24, num_samples=512,
                                       cfg_scale=self.sample.cfg_scale)
        return updated

    def provenance(self) -> str:
        """Short hash identifying this configuration."""
        return config_hash(self.model_dump())


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preset", choices=("full", "smoke"), default="full")
    parser.add_argument("--unconditional", action="store_true",
                        help="W3: train on advice embeddings only")
    parser.add_argument("--steps", type=int, default=None, help="override train.steps")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_dataset(config: RunConfig, settings: Settings) -> EmbeddingPairDataset:
    """Load the real corpus if present, else fall back to the synthetic stand-in."""
    path = settings.data_dir / config.data.corpus
    if not config.data.synthetic and path.exists():
        logger.info("corpus_loaded", path=str(path))
        return EmbeddingPairDataset.from_npz(path)
    if not config.data.synthetic:
        logger.warning("corpus_missing_using_synthetic", expected=str(path))
    corpus = SyntheticEmbeddingCorpus(num_pairs=config.data.synthetic_pairs,
                                      dim=config.data.dim, seed=config.seed)
    return corpus.dataset()


def build_model(config: RunConfig, dim: int, edm_config: EDMConfig) -> EDMPreconditioner:
    """Instantiate the denoiser and wrap it in the EDM preconditioner."""
    model_config = TransformerDenoiserConfig(
        input_dim=dim, num_tokens=config.model.num_tokens, hidden_dim=config.model.hidden_dim,
        depth=config.model.depth, num_heads=config.model.num_heads,
        time_embed_dim=config.model.time_embed_dim, context_dim=dim)
    network = TransformerDenoiser(model_config)
    logger.info("model_built", parameters=network.num_parameters,
                tokens=model_config.num_tokens, token_dim=model_config.token_dim)
    return EDMPreconditioner(network, edm_config)


def evaluate(preconditioner: EDMPreconditioner, train_set: EmbeddingPairDataset,
             held_out: EmbeddingPairDataset, config: RunConfig,
             device: torch.device) -> Dict[str, Any]:
    """Run the W3/W4 checks and return a JSON-serialisable report."""
    dim = train_set.embedding_dim
    count = min(config.sample.num_samples, 1024)
    report: Dict[str, Any] = {}

    unconditional = sample_edm(preconditioner, (count, dim),
                               num_steps=config.sample.num_steps, seed=config.seed,
                               device=device)
    assert isinstance(unconditional, torch.Tensor)
    bank = train_set.advice.to(device)
    # Distribution statistics are computed in *raw* embedding space: after
    # whitening every reference dimension has unit variance by construction, so
    # per_dim_std_corr measured in normalised space is comparing against a
    # constant and carries no signal.
    normalizer = train_set.normalizer
    report["distribution"] = distribution_report(normalizer.decode(unconditional.cpu()),
                                                 normalizer.decode(bank.cpu()))
    report["retrieval"] = retrieval_report(unconditional.cpu(), bank.cpu())

    if len(held_out) > 0:
        issues = held_out.issue[:count].to(device)
        targets = held_out.advice[:count].to(device)
        full_bank = torch.cat([bank, targets], dim=0)
        conditional = sample_edm(preconditioner, (issues.shape[0], dim),
                                 num_steps=config.sample.num_steps,
                                 conditioning={"issue": issues},
                                 cfg_scale=config.sample.cfg_scale, seed=config.seed,
                                 device=device)
        assert isinstance(conditional, torch.Tensor)
        report["conditional"] = conditional_report(conditional.cpu(), targets.cpu(),
                                                   full_bank.cpu())
        report["conditional_reconstruction"] = reconstruction_error(conditional.cpu(),
                                                                    targets.cpu())
        control = unconditional[:issues.shape[0]]
        report["unconditional_control"] = conditional_report(control.cpu(), targets.cpu(),
                                                             full_bank.cpu())
        report["unconditional_reconstruction"] = reconstruction_error(control.cpu(),
                                                                       targets.cpu())
    return report


def main(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Train, evaluate, and write the report; returns the report."""
    args = args or parse_args()
    configure_logging()
    settings = get_settings().ensure_dirs()

    config_path = args.config or (Path(__file__).resolve().parents[1] / "configs" /
                                  "embedding_diffusion.yaml")
    config = RunConfig.load(config_path)
    if args.preset == "smoke":
        config = config.apply_smoke_preset()
    if args.unconditional:
        config.train.conditional = False
        config.name = f"{config.name}_uncond"
    if args.steps is not None:
        config.train.steps = args.steps

    set_seed(config.seed)
    device = torch.device(args.device) if args.device else settings.torch_device()
    provenance = config.provenance()
    out_dir = settings.artifact_dir / config.name / provenance
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run_start", name=config.name, provenance=provenance, device=str(device),
                conditional=config.train.conditional)

    dataset = build_dataset(config, settings)
    train_set, held_out = dataset.split(holdout=config.data.holdout_issues, seed=config.seed)
    stats = sigma_data_report(train_set.advice)
    sigma_data = config.edm.sigma_data or estimate_sigma_data(train_set.advice)
    logger.info("data_ready", train=len(train_set), held_out=len(held_out),
                sigma_data=round(sigma_data, 4),
                anisotropy=round(stats["per_dim_std_max"] / stats["per_dim_std_min"], 3))

    edm_config = EDMConfig(sigma_data=sigma_data, sigma_min=config.edm.sigma_min,
                           sigma_max=config.edm.sigma_max, rho=config.edm.rho,
                           p_mean=config.edm.p_mean, p_std=config.edm.p_std)
    preconditioner = build_model(config, train_set.embedding_dim, edm_config)

    wandb_run = None
    if args.wandb:
        import wandb  # imported lazily so the dependency stays optional

        wandb_run = wandb.init(project=settings.wandb_project, entity=settings.wandb_entity,
                               mode=settings.wandb_mode, name=f"{config.name}-{provenance}",
                               config=config.model_dump())

    train_config = EDMTrainConfig(
        steps=config.train.steps, learning_rate=config.train.learning_rate,
        warmup_steps=config.train.warmup_steps, grad_clip=config.train.grad_clip,
        ema_decay=config.train.ema_decay, cfg_dropout=config.train.cfg_dropout,
        batch_size=config.train.batch_size, log_every=config.train.log_every,
        checkpoint_every=config.train.checkpoint_every, seed=config.seed,
        amp_dtype=config.train.amp_dtype, conditional=config.train.conditional)
    loader = build_pair_dataloader(train_set, batch_size=config.train.batch_size,
                                   seed=config.seed)

    state: TrainState = train_edm(
        preconditioner, loader, train_config, device=device,
        checkpoint_path=settings.checkpoint_dir / f"{config.name}_{provenance}.pt",
        extra_checkpoint={"normalizer": train_set.normalizer.state_dict(),
                          "model_config": asdict(preconditioner.network.config),
                          "run_config": config.model_dump()},
        wandb_run=wandb_run)

    report: Dict[str, Any] = {
        "config": config.model_dump(), "provenance": provenance,
        "parameters": preconditioner.network.num_parameters,
        "sigma_data": sigma_data, "data_stats": stats, "train": state.summary(),
        "eval": evaluate(preconditioner, train_set, held_out, config, device),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    if wandb_run is not None:
        wandb_run.finish()
    logger.info("run_done", output=str(out_dir / "report.json"))
    print(json.dumps(report["eval"], indent=2, default=float))
    return report


if __name__ == "__main__":
    main()
