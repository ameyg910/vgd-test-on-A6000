"""Experiment drivers for the toy validation.

Each ``run_*`` function returns a plain dictionary of results and is kept apart
from the corresponding ``plot_*`` function, so the notebook can cache expensive
sampling and re-plot cheaply. Nothing in the core package imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor, nn

from config import Settings, get_settings
from diffusion.base.sampling.guided import (GuidanceSpec, WeightFn, constant_weight,
                                            gaussian_tilt_weight, sample_guided)
from diffusion.base.schedule import NoiseSchedule
from diffusion.data.toy import GaussianMixtureDataset
from diffusion.metrics import compliance_rate, mode_coverage, modes_covered, pareto_front
from experiments.toy_2d.plotting import scatter_2d
from verifiers.base import Verifier
from verifiers.toy import HalfPlaneVerifier, TargetPointVerifier

__all__ = ["ExperimentContext", "right_half_region", "ball_region", "evaluate_samples", "cached",
           "run_single_verifier_panels", "run_weight_sweep", "run_two_verifier_grid",
           "run_strategy_comparison", "run_weight_shaping_ablation", "run_learned_verifier",
           "plot_single_verifier_panels", "plot_sweep", "plot_pareto", "plot_two_verifier_grid",
           "plot_two_verifier_heatmaps", "plot_three_objective_pareto", "plot_strategy_comparison",
           "plot_weight_shaping_ablation", "plot_learned_verifier", "save_all"]

RegionFn = Callable[[Tensor], Tensor]


@dataclass
class ExperimentContext:
    """Everything the drivers need, so no experiment function reaches for globals."""

    denoiser: nn.Module
    schedule: NoiseSchedule
    dataset: GaussianMixtureDataset
    num_samples: int = 1500
    num_steps: int = 150
    eta: float = 1.0
    data_var: float = 2.0
    base_seed: int = 0
    shaped_weights: bool = True

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of one sample batch."""
        return (self.num_samples, self.dataset.input_dim)

    @property
    def mode_centers(self) -> Tensor:
        """Reference points used by the coverage metrics."""
        return self.dataset.mode_centers

    def weight_fn(self, w: float) -> WeightFn:
        """Turn a nominal weight into the weight function handed to a GuidanceSpec."""
        if not self.shaped_weights:
            return constant_weight(float(w))
        return gaussian_tilt_weight(float(w), self.schedule, data_var=self.data_var)

    def spec(self, verifier: Verifier, w: float, name: Optional[str] = None) -> GuidanceSpec:
        """Build a GuidanceSpec for this context.

        ``project_to_manifold`` is False here: manifold projection lands in W8,
        and the toy baselines must stay comparable to the screening results
        until it does. The W8 ablation flips this flag rather than the API.
        """
        return GuidanceSpec(verifier=verifier, weight_fn=self.weight_fn(w),
                            project_to_manifold=False, name=name)

    def sample(self, specs: Sequence[GuidanceSpec], seed: int, strategy: str = "sum") -> Tensor:
        """Draw one guided batch with this context's sampler settings."""
        out = sample_guided(self.denoiser, self.schedule, self.shape, guidance_specs=specs,
                            num_steps=self.num_steps, seed=seed, eta=self.eta, strategy=strategy)
        assert isinstance(out, Tensor)
        return out


def right_half_region(x: Tensor) -> Tensor:
    """Compliance region of the half-plane verifier: first coordinate positive."""
    return x[:, 0] > 0


def ball_region(center: Tensor, radius: float) -> RegionFn:
    """Compliance region of a target-point verifier: inside a ball of given radius."""

    def region(x: Tensor) -> Tensor:
        inside: Tensor = (x - center.to(x.device)).norm(dim=-1) < radius
        return inside

    return region


def evaluate_samples(samples: Tensor, mode_centers: Tensor,
                     regions: Dict[str, RegionFn]) -> Dict[str, float]:
    """Compliance for each named region, plus diversity and on-manifold diagnostics."""
    result: Dict[str, float] = {f"compliance_{name}": compliance_rate(samples, fn)
                                for name, fn in regions.items()}
    result["entropy"] = mode_coverage(samples, mode_centers)
    result["modes_covered"] = float(modes_covered(samples, mode_centers, max_distance=0.6))
    distances = torch.cdist(samples, mode_centers.to(samples.device)).min(dim=-1).values
    result["off_manifold_frac"] = float((distances > 0.6).to(torch.float32).mean())
    result["mean_radius"] = float(samples.norm(dim=-1).mean())
    return result


def cached(path: Path, compute: Callable[[], Dict[str, Any]], force: bool = False) -> Dict[str, Any]:
    """Return ``compute()``, memoised to ``path``. Delete the file to recompute."""
    path = Path(path)
    if path.exists() and not force:
        cachedresult: Dict[str, Any] = torch.load(path, weights_only=False)
        return cachedresult
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


def run_single_verifier_panels(ctx: ExperimentContext,
                               weights: Sequence[float] = (0.0, 1.0, 3.0, 10.0)) -> Dict[str, Any]:
    """Guided samples at a few nominal weights, for the qualitative scatter panels."""
    panels: List[Dict[str, Any]] = []
    for w in weights:
        specs = [] if w == 0.0 else [ctx.spec(HalfPlaneVerifier(alpha=1.0), w)]
        samples = ctx.sample(specs, seed=ctx.base_seed)
        stats = evaluate_samples(samples, ctx.mode_centers, {"half_plane": right_half_region})
        panels.append({"w": float(w), "samples": samples.cpu(), **stats})
    return {"panels": panels}


def run_weight_sweep(ctx: ExperimentContext, weights: Sequence[float],
                     seeds: Sequence[int] = (0, 1, 2)) -> Dict[str, Any]:
    """Compliance and diversity versus nominal weight, repeated over seeds."""
    records: List[Dict[str, float]] = []
    for w in weights:
        for seed in seeds:
            specs = [] if w == 0.0 else [ctx.spec(HalfPlaneVerifier(alpha=1.0), w)]
            samples = ctx.sample(specs, seed=1000 + seed)
            stats = evaluate_samples(samples, ctx.mode_centers, {"half_plane": right_half_region})
            records.append({"w": float(w), "seed": float(seed), **stats})
    return {"records": records, "weights": [float(w) for w in weights],
            "seeds": [int(s) for s in seeds]}


def run_two_verifier_grid(ctx: ExperimentContext, w1_grid: Sequence[float],
                          w2_grid: Sequence[float], target_mode_index: int = 4,
                          target_sigma: float = 0.5, strategy: str = "sum",
                          seeds: Sequence[int] = (0,)) -> Dict[str, Any]:
    """Sweep a 2D weight grid: half-plane (right) against a target point (left mode)."""
    center = ctx.mode_centers[target_mode_index].clone()
    regions = {"half_plane": right_half_region, "target": ball_region(center, 0.6)}
    records: List[Dict[str, Any]] = []
    for w1 in w1_grid:
        for w2 in w2_grid:
            per_seed: List[Dict[str, float]] = []
            samples: Optional[Tensor] = None
            for seed in seeds:
                specs: List[GuidanceSpec] = []
                if w1 > 0:
                    specs.append(ctx.spec(HalfPlaneVerifier(alpha=1.0), w1, name="half_plane"))
                if w2 > 0:
                    specs.append(ctx.spec(TargetPointVerifier(center, target_sigma), w2,
                                          name="target_point"))
                samples = ctx.sample(specs, seed=2000 + seed, strategy=strategy)
                per_seed.append(evaluate_samples(samples, ctx.mode_centers, regions))
            averaged = {key: float(np.mean([r[key] for r in per_seed])) for key in per_seed[0]}
            assert samples is not None
            records.append({"w1": float(w1), "w2": float(w2), "samples": samples.cpu(), **averaged})
    return {"records": records, "w1_grid": [float(w) for w in w1_grid],
            "w2_grid": [float(w) for w in w2_grid], "center": center.cpu(),
            "target_mode_index": int(target_mode_index), "strategy": strategy}


def run_strategy_comparison(ctx: ExperimentContext, w1: float, w2_values: Sequence[float],
                            strategies: Sequence[str] = ("sum", "normalized", "projected",
                                                         "alternating"),
                            target_mode_index: int = 4,
                            target_sigma: float = 0.5) -> Dict[str, Any]:
    """Compare composition strategies along a slice of the two-verifier grid."""
    center = ctx.mode_centers[target_mode_index].clone()
    regions = {"half_plane": right_half_region, "target": ball_region(center, 0.6)}
    records: List[Dict[str, Any]] = []
    for strategy in strategies:
        for w2 in w2_values:
            specs = [ctx.spec(HalfPlaneVerifier(alpha=1.0), w1, name="half_plane")]
            if w2 > 0:
                specs.append(ctx.spec(TargetPointVerifier(center, target_sigma), w2,
                                      name="target_point"))
            samples = ctx.sample(specs, seed=3000, strategy=strategy)
            records.append({"strategy": strategy, "w2": float(w2),
                            **evaluate_samples(samples, ctx.mode_centers, regions)})
    return {"records": records, "w1": float(w1), "w2_values": [float(w) for w in w2_values],
            "strategies": list(strategies)}


def run_weight_shaping_ablation(ctx: ExperimentContext,
                                weights: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
                                ) -> Dict[str, Any]:
    """Constant weight versus the derived noise-aware weight, at equal nominal ``w``."""
    records: List[Dict[str, Any]] = []
    for shaped in (True, False):
        for w in weights:
            weight_fn: WeightFn = (gaussian_tilt_weight(float(w), ctx.schedule, ctx.data_var)
                                   if shaped else constant_weight(float(w)))
            spec = GuidanceSpec(HalfPlaneVerifier(alpha=1.0), weight_fn,
                                project_to_manifold=False)
            samples = ctx.sample([spec], seed=4000)
            records.append({"shaped": bool(shaped), "w": float(w),
                            **evaluate_samples(samples, ctx.mode_centers,
                                               {"half_plane": right_half_region})})
    return {"records": records, "weights": [float(w) for w in weights]}


def run_learned_verifier(ctx: ExperimentContext, verifier: Verifier, weights: Sequence[float],
                         preferred_region: RegionFn,
                         reference_verifier: Optional[Verifier] = None) -> Dict[str, Any]:
    """Bonus experiment: guidance from a trained classifier, with gradient diagnostics."""
    records: List[Dict[str, Any]] = []
    for w in weights:
        specs = [] if w == 0.0 else [ctx.spec(verifier, w)]
        samples = ctx.sample(specs, seed=5000)
        stats = evaluate_samples(samples, ctx.mode_centers, {"preferred": preferred_region})
        records.append({"w": float(w), "samples": samples.cpu(), **stats})

    radii = torch.linspace(0.5, 8.0, 40)
    probe = radii.reshape(-1, 1) * torch.tensor([[1.0, 0.0]])
    result: Dict[str, Any] = {"records": records, "probe_radii": radii,
                              "probe_grad_norm": verifier.grad_log_value(probe).norm(dim=-1).cpu()}
    if reference_verifier is not None:
        result["reference_grad_norm"] = (reference_verifier.grad_log_value(probe)
                                         .norm(dim=-1).cpu())
    return result


def _aggregate(records: Sequence[Dict[str, float]], key: str,
               value: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group ``records`` by ``key``; return ``(keys, means, stds)`` of ``value``."""
    keys = sorted({float(r[key]) for r in records})
    means, stds = [], []
    for k in keys:
        values = [float(r[value]) for r in records if float(r[key]) == k]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
    return np.asarray(keys), np.asarray(means), np.asarray(stds)


def plot_single_verifier_panels(result: Dict[str, Any], mode_centers: Tensor) -> Figure:
    """Scatter panels of guided samples at increasing weight."""
    panels = result["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.3))
    for ax, panel in zip(np.atleast_1d(axes), panels):
        limit = 3.5 if panel["mean_radius"] < 3.0 else float(panel["samples"].abs().max()) * 1.05
        scatter_2d(panel["samples"], ax=ax, mode_centers=mode_centers, limit=limit,
                   title=f"w = {panel['w']:g}   compliance {panel['compliance_half_plane']:.2f}\n"
                         f"entropy {panel['entropy']:.2f}   mean |x| {panel['mean_radius']:.2f}")
        ax.axvline(0.0, color="#718096", lw=0.8, ls="--")
    fig.tight_layout()
    return fig


def plot_sweep(result: Dict[str, Any], value: str, ylabel: str, color: str = "#2b6cb0") -> Figure:
    """Mean and standard deviation of a metric against the nominal weight."""
    keys, means, stds = _aggregate(result["records"], "w", value)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.errorbar(keys, means, yerr=stds, marker="o", capsize=3, color=color, lw=1.4, ms=4)
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pareto(result: Dict[str, Any]) -> Figure:
    """(compliance, diversity) scatter with the Pareto front highlighted."""
    keys, compliance, _ = _aggregate(result["records"], "w", "compliance_half_plane")
    _, entropy, _ = _aggregate(result["records"], "w", "entropy")
    mask = pareto_front(np.stack([compliance, entropy], axis=1), maximize=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.plot(compliance, entropy, color="#a0aec0", lw=0.8, zorder=1)
    ax.scatter(compliance[~mask], entropy[~mask], c="#a0aec0", s=30, label="dominated", zorder=2)
    ax.scatter(compliance[mask], entropy[mask], c="#c53030", s=48, label="Pareto-optimal", zorder=3)
    for w, c, e in zip(keys, compliance, entropy):
        ax.annotate(f"{w:g}", (c, e), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("compliance with $V_1$")
    ax.set_ylabel("normalised mode entropy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_two_verifier_grid(result: Dict[str, Any], mode_centers: Tensor) -> Figure:
    """Scatter panels over the ``(w1, w2)`` grid."""
    w1_grid, w2_grid = result["w1_grid"], result["w2_grid"]
    lookup = {(r["w1"], r["w2"]): r for r in result["records"]}
    fig, axes = plt.subplots(len(w1_grid), len(w2_grid),
                             figsize=(2.1 * len(w2_grid), 2.1 * len(w1_grid)))
    for i, w1 in enumerate(w1_grid):
        for j, w2 in enumerate(w2_grid):
            record = lookup[(w1, w2)]
            scatter_2d(record["samples"], ax=axes[i][j], mode_centers=mode_centers, size=2.5,
                       alpha=0.2,
                       title=f"$w_1$={w1:g}, $w_2$={w2:g}\n"
                             f"$c_1$={record['compliance_half_plane']:.2f}  "
                             f"$c_2$={record['compliance_target']:.2f}  "
                             f"H={record['entropy']:.2f}")
    fig.tight_layout()
    return fig


def plot_two_verifier_heatmaps(result: Dict[str, Any]) -> Figure:
    """Heatmaps of both compliance rates and of mode entropy over the grid."""
    w1_grid, w2_grid = result["w1_grid"], result["w2_grid"]
    lookup = {(r["w1"], r["w2"]): r for r in result["records"]}
    fields = [("compliance_half_plane", "compliance with $V_1$ (right half)"),
              ("compliance_target", "compliance with $V_2$ (target mode)"),
              ("entropy", "normalised mode entropy")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for ax, (field, title) in zip(axes, fields):
        matrix = np.array([[lookup[(w1, w2)][field] for w2 in w2_grid] for w1 in w1_grid])
        image = ax.imshow(matrix, origin="lower", cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(w2_grid)), [f"{w:g}" for w in w2_grid], fontsize=8)
        ax.set_yticks(range(len(w1_grid)), [f"{w:g}" for w in w1_grid], fontsize=8)
        ax.set_xlabel("$w_2$ (target point)")
        ax.set_ylabel("$w_1$ (half plane)")
        ax.set_title(title, fontsize=9)
        threshold = matrix.min() + 0.6 * (matrix.max() - matrix.min())
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] < threshold else "black")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_three_objective_pareto(result: Dict[str, Any]) -> Figure:
    """Q3: Pareto front over (compliance $V_1$, compliance $V_2$, diversity)."""
    records = result["records"]
    objectives = np.array([[r["compliance_half_plane"], r["compliance_target"], r["entropy"]]
                           for r in records])
    mask = pareto_front(objectives, maximize=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    scatter = axes[0].scatter(objectives[:, 0], objectives[:, 1], c=objectives[:, 2],
                              cmap="viridis", s=60)
    axes[0].scatter(objectives[mask, 0], objectives[mask, 1], facecolors="none",
                    edgecolors="#c53030", s=150, linewidths=1.6, label="Pareto-optimal")
    for record, point in zip(records, objectives):
        axes[0].annotate(f"({record['w1']:g},{record['w2']:g})", point[:2], fontsize=6,
                         xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("compliance with $V_1$")
    axes[0].set_ylabel("compliance with $V_2$")
    axes[0].legend(fontsize=8)
    fig.colorbar(scatter, ax=axes[0], label="mode entropy", fraction=0.046)

    w1_grid, w2_grid = result["w1_grid"], result["w2_grid"]
    index = {(r["w1"], r["w2"]): i for i, r in enumerate(records)}
    matrix = np.array([[float(mask[index[(w1, w2)]]) for w2 in w2_grid] for w1 in w1_grid])
    axes[1].imshow(matrix, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(range(len(w2_grid)), [f"{w:g}" for w in w2_grid], fontsize=8)
    axes[1].set_yticks(range(len(w1_grid)), [f"{w:g}" for w in w1_grid], fontsize=8)
    axes[1].set_xlabel("$w_2$")
    axes[1].set_ylabel("$w_1$")
    axes[1].set_title("green = Pareto-optimal, red = dominated", fontsize=9)
    fig.tight_layout()
    return fig


def plot_strategy_comparison(result: Dict[str, Any]) -> Figure:
    """Q4: compliance with each verifier as $w_2$ grows, per composition strategy."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    fields = [("compliance_target", "compliance with $V_2$"),
              ("compliance_half_plane", "compliance with $V_1$"),
              ("entropy", "normalised mode entropy")]
    for ax, (field, ylabel) in zip(axes, fields):
        for strategy in result["strategies"]:
            records = [r for r in result["records"] if r["strategy"] == strategy]
            ax.plot([r["w2"] for r in records], [r[field] for r in records], marker="o", ms=4,
                    lw=1.3, label=strategy)
        ax.set_xlabel("$w_2$ (target point)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"composition strategies at fixed $w_1$ = {result['w1']:g}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_weight_shaping_ablation(result: Dict[str, Any]) -> Figure:
    """Constant versus noise-aware weight: compliance, diversity and off-manifold drift."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    fields = [("compliance_half_plane", "compliance"), ("entropy", "normalised mode entropy"),
              ("mean_radius", "mean $\\|x\\|$")]
    for ax, (field, ylabel) in zip(axes, fields):
        for shaped, label, color in [(True, "noise-aware $w(t)$", "#2b6cb0"),
                                     (False, "constant $w$", "#c53030")]:
            records = [r for r in result["records"] if r["shaped"] == shaped]
            ax.plot([r["w"] for r in records], [r[field] for r in records], marker="o", ms=4,
                    lw=1.3, color=color, label=label)
        ax.set_xlabel("nominal weight $w$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[2].set_yscale("log")
    axes[2].axhline(2.0, color="#718096", ls="--", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_learned_verifier(result: Dict[str, Any], mode_centers: Tensor) -> Figure:
    """Bonus: samples under classifier guidance plus the off-support gradient probe."""
    records = result["records"]
    fig, axes = plt.subplots(1, len(records) + 1, figsize=(2.7 * (len(records) + 1), 3.0))
    for ax, record in zip(axes, records):
        scatter_2d(record["samples"], ax=ax, mode_centers=mode_centers, size=2.5, alpha=0.2,
                   title=f"w = {record['w']:g}\ncompliance {record['compliance_preferred']:.2f}, "
                         f"H {record['entropy']:.2f}")
    ax = axes[-1]
    ax.plot(result["probe_radii"], result["probe_grad_norm"], color="#c53030",
            label="learned verifier")
    if "reference_grad_norm" in result:
        ax.plot(result["probe_radii"], result["reference_grad_norm"], color="#2b6cb0",
                label="analytical verifier")
    ax.axvspan(1.4, 2.6, color="#a0aec0", alpha=0.3, label="data support")
    ax.set_xlabel("$\\|x\\|$ along $e_1$")
    ax.set_ylabel("$\\|\\nabla \\log V\\|$")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("gradient magnitude off-support", fontsize=9)
    fig.tight_layout()
    return fig


def save_all(figures: Dict[str, Figure], output_dir: Optional[Path] = None) -> List[Path]:
    """Write every figure in ``figures`` as ``<name>.png`` and return the paths."""
    directory = Path(output_dir) if output_dir is not None else get_settings().figure_dir
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for name, figure in figures.items():
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        paths.append(path)
    return paths
