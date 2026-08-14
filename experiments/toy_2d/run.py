"""Run the 2D toy experiments on the ported GuidanceSpec API.

    python -m experiments.toy_2d.run --stage all

Config comes from a pydantic-validated YAML file and is hashed into every
artifact name component, so a result can always be traced to the config that
produced it. Paths come from :mod:`config`; nothing here is absolute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import matplotlib
import torch
import yaml
from pydantic import BaseModel, Field

matplotlib.use("Agg")

from config import Settings, config_hash, get_settings
from diffusion.base.denoiser import MLPDenoiser
from diffusion.base.schedule import NoiseSchedule
from diffusion.data.toy import GaussianMixtureDataset
from experiments.toy_2d.pipeline import (ExperimentContext, cached, plot_learned_verifier,
                                         plot_pareto, plot_single_verifier_panels,
                                         plot_strategy_comparison, plot_sweep,
                                         plot_three_objective_pareto, plot_two_verifier_grid,
                                         plot_two_verifier_heatmaps,
                                         plot_weight_shaping_ablation, run_learned_verifier,
                                         run_single_verifier_panels, run_strategy_comparison,
                                         run_two_verifier_grid, run_weight_shaping_ablation,
                                         run_weight_sweep, save_all)
from logging_config import configure_logging, get_logger
from verifiers.learned import ClassifierVerifier, ModeClassifier
from verifiers.toy import HalfPlaneVerifier

logger = get_logger(__name__)

STAGES = ("panels", "sweep", "grid", "strategies", "ablation", "learned")


class SamplerConfig(BaseModel):
    """Sampler settings shared by every stage."""

    num_samples: int = 1500
    num_steps: int = 150
    eta: float = 1.0
    data_var: float = 2.0
    shaped_weights: bool = True


class SweepConfig(BaseModel):
    weights: List[float]
    seeds: List[int] = Field(default_factory=lambda: [0, 1, 2])


class PanelConfig(BaseModel):
    weights: List[float]


class GridConfig(BaseModel):
    w1: List[float]
    w2: List[float]
    target_mode_index: int = 4
    target_sigma: float = 1.0
    seeds: List[int] = Field(default_factory=lambda: [0])


class StrategyConfig(BaseModel):
    w1: float
    w2_values: List[float]
    names: List[str]


class AblationConfig(BaseModel):
    weights: List[float]


class LearnedConfig(BaseModel):
    weights: List[float]
    preferred_modes: List[int]


class ToyConfig(BaseModel):
    """Full validated config for the toy experiments."""

    name: str = "toy_2d"
    seed: int = 0
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    sweep: SweepConfig
    panels: PanelConfig
    grid: GridConfig
    strategies: StrategyConfig
    ablation: AblationConfig
    learned: LearnedConfig

    @classmethod
    def load(cls, path: Path) -> "ToyConfig":
        """Parse and validate a YAML config."""
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def provenance(self) -> str:
        """Short hash identifying this exact configuration."""
        return config_hash(self.model_dump())


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=("all",) + STAGES)
    parser.add_argument("--config", type=Path, default=None,
                        help="defaults to <repo>/configs/toy_2d.yaml")
    parser.add_argument("--force", action="store_true", help="ignore cached stage results")
    return parser.parse_args()


def build_context(config: ToyConfig, settings: Settings) -> ExperimentContext:
    """Load the trained toy denoiser and assemble the shared context."""
    checkpoint = torch.load(settings.checkpoint_dir / "denoiser_toy2d.pt", weights_only=False)
    denoiser = MLPDenoiser(checkpoint["input_dim"], checkpoint["width"], checkpoint["num_blocks"])
    denoiser.load_state_dict(checkpoint["state_dict"])
    denoiser.eval()
    return ExperimentContext(
        denoiser=denoiser,
        schedule=NoiseSchedule(checkpoint["timesteps"], "cosine"),
        dataset=GaussianMixtureDataset(),
        num_samples=config.sampler.num_samples,
        num_steps=config.sampler.num_steps,
        eta=config.sampler.eta,
        data_var=config.sampler.data_var,
        base_seed=config.seed,
        shaped_weights=config.sampler.shaped_weights,
    )


def load_learned_verifier(settings: Settings, preferred: Sequence[int]) -> ClassifierVerifier:
    """Rebuild the trained mode classifier and wrap it as a learned verifier."""
    checkpoint = torch.load(settings.checkpoint_dir / "mode_classifier.pt", weights_only=False)
    classifier = ModeClassifier(checkpoint["input_dim"], checkpoint["num_classes"])
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return ClassifierVerifier(classifier, preferred, name="learned_modes")


def _scalars(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop tensors and round floats, for the JSON summary."""
    out: List[Dict[str, Any]] = []
    for record in records:
        row: Dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, torch.Tensor):
                continue
            row[key] = round(float(value), 4) if isinstance(value, (int, float)) else value
        out.append(row)
    return out


def main(args: Optional[argparse.Namespace] = None) -> List[Path]:
    """Run the requested stages; return the figure paths written."""
    args = args or parse_args()
    configure_logging()
    settings = get_settings().ensure_dirs()
    config_path = args.config or (Path(__file__).resolve().parents[2] / "configs" / "toy_2d.yaml")
    config = ToyConfig.load(config_path)
    provenance = config.provenance()
    logger.info("toy_experiments_start", stage=args.stage, config=str(config_path),
                provenance=provenance)

    ctx = build_context(config, settings)
    cache_dir = settings.artifact_dir / "toy_2d" / provenance
    stages = set(STAGES) if args.stage == "all" else {args.stage}
    figures: Dict[str, Any] = {}
    summaries: Dict[str, List[Dict[str, Any]]] = {}

    def stage_cache(name: str, compute: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        return cached(cache_dir / f"res_{name}.pt", compute, args.force)

    if "panels" in stages:
        result = stage_cache("panels",
                             lambda: run_single_verifier_panels(ctx, config.panels.weights))
        figures["fig1_single_verifier_panels"] = plot_single_verifier_panels(result,
                                                                            ctx.mode_centers)
        summaries["panels"] = _scalars(result["panels"])

    if "sweep" in stages:
        result = stage_cache("sweep", lambda: run_weight_sweep(ctx, config.sweep.weights,
                                                               config.sweep.seeds))
        figures["fig2_compliance_vs_w"] = plot_sweep(result, "compliance_half_plane",
                                                     "compliance with $V_1$")
        figures["fig3_diversity_vs_w"] = plot_sweep(result, "entropy",
                                                    "normalised mode entropy", color="#2f855a")
        figures["fig4_pareto"] = plot_pareto(result)
        summaries["sweep"] = _scalars(result["records"])

    if "grid" in stages:
        result = stage_cache("grid", lambda: run_two_verifier_grid(
            ctx, config.grid.w1, config.grid.w2, target_mode_index=config.grid.target_mode_index,
            target_sigma=config.grid.target_sigma, seeds=config.grid.seeds))
        figures["fig5_two_verifier_grid"] = plot_two_verifier_grid(result, ctx.mode_centers)
        figures["fig6_two_verifier_heatmaps"] = plot_two_verifier_heatmaps(result)
        figures["fig7_three_objective_pareto"] = plot_three_objective_pareto(result)
        summaries["grid"] = _scalars(result["records"])

    if "strategies" in stages:
        result = stage_cache("strategies", lambda: run_strategy_comparison(
            ctx, w1=config.strategies.w1, w2_values=config.strategies.w2_values,
            strategies=config.strategies.names,
            target_mode_index=config.grid.target_mode_index,
            target_sigma=config.grid.target_sigma))
        figures["fig8_composition_strategies"] = plot_strategy_comparison(result)
        summaries["strategies"] = _scalars(result["records"])

    if "ablation" in stages:
        result = stage_cache("ablation",
                             lambda: run_weight_shaping_ablation(ctx, config.ablation.weights))
        figures["fig9_weight_shaping_ablation"] = plot_weight_shaping_ablation(result)
        summaries["ablation"] = _scalars(result["records"])

    if "learned" in stages:
        verifier = load_learned_verifier(settings, config.learned.preferred_modes)
        preferred_centers = ctx.mode_centers[list(config.learned.preferred_modes)]

        def preferred_region(x: torch.Tensor) -> torch.Tensor:
            distances = torch.cdist(x, preferred_centers.to(x.device)).min(dim=-1).values
            return distances < 0.6

        result = stage_cache("learned", lambda: run_learned_verifier(
            ctx, verifier, config.learned.weights, preferred_region,
            reference_verifier=HalfPlaneVerifier()))
        figures["fig10_learned_verifier"] = plot_learned_verifier(result, ctx.mode_centers)
        summaries["learned"] = _scalars(result["records"])

    figure_dir = settings.figure_dir / "toy_2d"
    paths = save_all(figures, figure_dir)
    summary_dir = settings.artifact_dir / "toy_2d" / provenance
    summary_dir.mkdir(parents=True, exist_ok=True)
    for stage_name, payload in summaries.items():
        (summary_dir / f"summary_{stage_name}.json").write_text(json.dumps(payload, indent=2))
    logger.info("toy_experiments_done", figures=[p.name for p in paths],
                summaries=sorted(summaries), output=str(summary_dir))
    return paths


if __name__ == "__main__":
    main()
