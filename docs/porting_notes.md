# W1 porting notes: screening task → project monorepo

What changed, what deliberately did not, and how the "bit-for-bit identical" bar
was verified.

## Review findings and how each is addressed

| Finding | Status | Where |
|---|---|---|
| Hardcoded paths | Fixed | All paths flow through `config.Settings`; `tests/test_config.py::test_no_hardcoded_absolute_paths_in_source` scans `diffusion/`, `verifiers/`, `experiments/`, `scripts/` and fails the suite on any absolute path literal. |
| Single global `w` + inline `_Combo` | Fixed | `GuidanceSpec` carries a per-verifier `weight_fn`. Gradients are computed per spec in `verifier_gradients` and combined in `compose_gradients`; no composite verifier object exists anywhere. |
| Two-verifier experiment missed the conflict case | Fixed | The target verifier sits on mode 4 = `(-2, 0)`, opposite the right-half verifier. `configs/toy_2d.yaml` records the choice with a comment so it cannot silently drift back. |
| LLM-first workflow | Process change | Not a code artifact. See "Working notes" below. |

The conflict-case fix predates this port — it was corrected in the rebuilt
screening submission, and the cancellation result it produced (the anti-diagonal
band where both compliances revert to unguided values) is what motivated the
composition strategies now in `compose_gradients`.

## The new API

```python
@dataclass
class GuidanceSpec:
    verifier: Verifier
    weight_fn: Callable[[int], float]
    project_to_manifold: bool = True
    target: str = "x_t"          # or "x0_hat" for reconstruction guidance
    grad_clip: float | None = None
    name: str | None = None
```

`sample_guided(denoiser, schedule, shape, guidance_specs=[...])`. Three verifiers
on three different schedules is three list entries, no code change —
`tests/test_guided_sampling.py::test_three_specs_with_distinct_schedules`.

**Weight functions.** `weight_fn` takes an *integer* timestep and returns a
float, per the workplan spec. The screening code passed a tensor of timesteps;
since every sample in a batch shares `t` at a given reverse step, the scalar form
is equivalent and simpler. Factories provided: `constant_weight`,
`gaussian_tilt_weight` (the derived noise-aware schedule), `linear_ramp_weight`,
`late_start_weight`, and `inverse_sigma_weight` (the `w / (σ_t² + λ)` form from
the W8 workplan item, included now so the API shape is settled early — budget
normalisation and per-verifier `λ_k` tuning are W8/W9 work).

**`project_to_manifold` defaults to `True`** as specified, but the projector
itself is W8. Until then `sample_guided` logs a structured warning
(`manifold_projection_requested_but_unavailable`) naming the affected verifiers
and guides unprojected. A `ManifoldProjector` protocol is declared so W8 plugs in
without touching the spec dataclass; `test_projector_is_applied_when_supplied`
already pins the wiring using a stub projector. The toy experiments pass
`project_to_manifold=False` explicitly so their baselines stay comparable to the
screening numbers — the W8 ablation flips that flag rather than changing the API.

## Verified parity

Two independent levels, both green:

1. **Sampler numerics.** `scripts/make_parity_goldens.py` runs the screening
   package once (path from `PDIFF_SCREENING_REPO`, never hardcoded) and freezes
   outputs for nine configurations into `tests/data/screening_goldens.pt`:
   unguided DDIM, unguided ancestral, empty spec list, constant-weight guidance,
   noise-aware guidance, and the conflict case under all four composition
   strategies. `tests/test_screening_parity.py` asserts `torch.equal` — not
   `allclose` — against each.

2. **Experiment-level.** All six toy stages were re-run through the new API and
   their summary JSONs compared field-by-field to the screening artifacts.
   Identical across panels (4 rows), sweep (30), grid (16), strategies (20),
   ablation (12) and learned (4).

Reproduce with:

```bash
PDIFF_SCREENING_REPO=/path/to/screening python -m scripts.make_parity_goldens
pytest
python -m experiments.toy_2d.run --stage all --force
```

## Structural changes

- `guided_diffusion/diffusion.py` split into `diffusion/base/training.py` (loss,
  EMA, checkpoint save/load) and `diffusion/base/sampling/unguided.py`
  (`reverse_step`, `sample`). The reverse step remains the single shared
  primitive, which is what makes empty-spec sampling identical rather than close.
- `guided_diffusion/verifiers.py` split into `verifiers/base.py` (the ABC that
  V_P, V_τ and V_E will implement) and `verifiers/toy.py`.
- `guided_diffusion/classifier.py` + `ClassifierVerifier` merged into
  `verifiers/learned.py`, since they demonstrate one thing: a neural verifier
  reaching the sampler through the autograd path.
- Plotting moved to `experiments/toy_2d/plotting.py` so that library imports
  never pull in matplotlib on the cluster.
- `MLPDenoiser` is retained as the toy denoiser; the transformer denoiser is W2
  and will live beside it in `diffusion/base/denoiser.py`.

## Deferred to later phases, deliberately

- EDM preconditioning (`diffusion/base/preconditioning.py`) — W2. The current
  schedule is discrete-time VP/DDPM, correct for the toy problem; the 768-dim
  model needs Karras preconditioning and this port does not pretend otherwise.
- Tweedie helpers and manifold projection — W8.
- `wandb` logging, bf16, `torch.compile`, resumable multi-GPU training — Phase 1.
  `save_checkpoint`/`load_checkpoint` exist now and store the loss history and
  train config, so resume has somewhere to land.
- Pydantic schemas for *data records* — Phase 1, when there are records. Config
  schemas are already pydantic (`experiments/toy_2d/run.py`).

## Working notes

Per the workplan's Friday-demo expectation, the one mathematical choice from this
week worth defending: `gaussian_tilt_weight` is not a heuristic. Tilting the
noisy marginal at every `t` does not produce the family you get by tilting at
`t = 0` and then noising. For data `N(0, s²I)` and a linear `log V` the latter
has score `∇log p_t + w·√ᾱ_t·s²/v_t·∇log V` with `v_t = ᾱ_t s² + 1 − ᾱ_t`, so the
correct weight carries a factor equal to 1 at `t = 0` and decaying to 0 at
`t = T`. Constant weights over-guide at high noise and the toy sampler diverges
from `w = 2` upward (mean ‖x‖ 19.5 at `w = 2`, 132 at `w = 3`, against a data
radius of 2). This is the same reasoning that makes the W8 adaptive schedule
`w/(σ_t² + λ)` the right shape, and it is why `weight_fn` is a function of `t`
rather than a scalar.
