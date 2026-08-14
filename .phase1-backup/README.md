# Prescriptive Diffusion

Policy-guided diffusion for prescriptive business advice: a conditional diffusion
model over `all-mpnet-base-v2` advice embeddings, steered at inference time by
gradients from external verifiers (policy compliance, tool feasibility, episodic
similarity). The base model is never retrained when a new policy arrives — the
policy enters as another verifier gradient.

**Status: Phase 0 complete (W1).** Screening-task code ported onto the project's
permanent guidance API, with parity verified. Phase 1 (768-dim transformer
denoiser + EDM preconditioning) is next.

## Layout

```
config.py               all paths and run settings; nothing else builds a path
logging_config.py       structlog setup
diffusion/
  base/schedule.py      VP/DDPM schedule, q_sample, score <-> epsilon
  base/denoiser.py      MLP denoiser (toy); transformer lands in W2
  base/training.py      loss, EMA, checkpoint save/load
  base/sampling/        unguided.py (reverse_step, sample), guided.py (GuidanceSpec)
  data/toy.py           2D ring dataset
  metrics.py            compliance, mode entropy, Wasserstein, Pareto
verifiers/
  base.py               Verifier ABC — V_P, V_tau, V_E implement this in Phase 2
  toy.py                analytical toy verifiers
  learned.py            neural verifier via the autograd path
experiments/toy_2d/     ported 2D validation (pipeline, plots, CLI)
configs/toy_2d.yaml     pydantic-validated, hashed into artifact paths
scripts/                parity golden generation
tests/                  schedule, guidance invariants, config, screening parity
docs/                   porting notes, debugging log
```

## Setup

```bash
pip install -e ".[dev,viz]"
pytest                       # 57 tests
mypy                         # strict, clean
```

Every path is overridable from the environment, which is how per-user scratch is
selected on the shared A6000 boxes:

```bash
export PDIFF_ARTIFACT_DIR=/scratch/$USER/prescriptive-diffusion/artifacts
export PDIFF_CHECKPOINT_DIR=/scratch/$USER/prescriptive-diffusion/checkpoints
```

## The guidance API

```python
from diffusion import GuidanceSpec, sample_guided
from diffusion.base.sampling.guided import constant_weight, gaussian_tilt_weight

samples = sample_guided(
    denoiser, schedule, shape=(1000, 768),
    guidance_specs=[
        GuidanceSpec(policy_verifier, gaussian_tilt_weight(3.0, schedule, data_var)),
        GuidanceSpec(tool_verifier, constant_weight(1.0), grad_clip=5.0),
        GuidanceSpec(episodic_verifier, inverse_sigma_weight(2.0, schedule, lam=0.5)),
    ],
    strategy="sum", seed=0,
)
```

Each verifier keeps its own weight function of `t`. `guidance_specs=[]` is
bitwise identical to unguided sampling — an invariant with a test.

## Reproducing the toy validation

```bash
python -m experiments.toy_2d.run --stage all
```

Results are cached under `artifacts/toy_2d/<config-hash>/`; the hash changes with
the config, so results are always traceable to what produced them.
