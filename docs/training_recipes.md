# Training recipes (Phase 1)

Config: `configs/embedding_diffusion.yaml`, validated by pydantic and hashed
into the artifact path, so every report traces to the exact settings that
produced it. Entry point: `scripts/train_embedding_diffusion.py`.

## Runs

```bash
# CPU sanity run: 128-dim synthetic corpus, 1.5M-param model, ~4 minutes.
python -m scripts.train_embedding_diffusion --preset smoke

# W3: unconditional first. Stop here if it does not converge.
python -m scripts.train_embedding_diffusion --unconditional --wandb

# W4: conditional, with 10% CFG dropout.
python -m scripts.train_embedding_diffusion --wandb
```

`--unconditional` sets `train.conditional=False` and changes nothing else, so
the two runs cannot silently diverge in anything but the conditioning pathway.

## Optimisation

| setting | value | why |
|---|---|---|
| optimiser | AdamW, β=(0.9, 0.99) | β₂=0.99 is the diffusion convention; 0.999 is slow to adapt when the loss is noisy across σ |
| lr | 2e-4, 500-step linear warmup, cosine to 5% | warmup matters because adaLN gates start at zero and gradients are initially concentrated in the output head |
| grad clip | 1.0 | reported per step; a rising clip fraction is the early sign of σ-schedule mismatch |
| EMA | 0.9995, warmup-aware | **sample only from EMA weights**; live weights look worse and will mislead a go/no-go |
| precision | bf16 autocast | fp16 is not safe here — `c_out` spans orders of magnitude and underflows |
| batch | 256 | |

## What to watch, in priority order

1. **Per-σ-bucket loss, not aggregate loss.** The failure mode this project
   should fear is an aggregate loss that falls while the high-σ bucket
   flatlines: the model learns to denoise easy samples and never learns the
   global structure, and samples come out as noise. `loss_by_sigma_bucket` logs
   all four buckets every `log_every` steps.
2. **`sigma_data` and anisotropy at startup.** Logged as `data_ready`. If the
   anisotropy ratio is far from 1 after whitening, the normaliser did not fit.
3. **Sample norm vs. reference norm.** Cheap, and catches a mis-scaled schedule
   immediately.
4. **Conditional vs. unconditional control.** The report always evaluates both
   on the same held-out issues; the conditional model must beat the control or
   the conditioning is decorative.

## Evaluation

`diffusion/eval/embedding_metrics.py`, run automatically at the end of training
and written to `report.json`:

- `distribution` — norm, pooled std, per-dimension std correlation, mean shift,
  sliced Wasserstein. Computed in **raw** embedding space (decoded), because
  after whitening every reference dimension has unit variance by construction
  and the correlation would be measured against a constant.
- `retrieval` — nearest-neighbour similarity and coverage against the advice
  bank, with a matched-Gaussian baseline alongside. See the caveat in
  `docs/debugging_log.md`: in high dimension this baseline is nearly as high as
  the model's, so **this metric alone cannot establish that samples are good**.
- `conditional` / `unconditional_control` — recall@k and median rank of the true
  advice given its issue, versus the unconditional control on the same targets.
- `*_reconstruction` — cosine and L2 to the true held-out advice.

Held-out pairs are split **by `issue_id`**, so the four paraphrases of one advice
never straddle the split. Splitting by row would put near-duplicates on both
sides and inflate every held-out number.

## Smoke-run reference numbers

1,500 steps, 128-dim synthetic corpus, 1.5M params, 3.7 min CPU
(`artifacts/embedding_diffusion_smoke/9b05165b53f8/report.json`):

| metric | conditional | unconditional control |
|---|---|---|
| median rank of true advice | 49 | 1015 |
| recall@10 | 0.156 | 0.000 |
| cosine to true advice | 0.373 | 0.010 |

Distribution: sample norm 11.36 vs reference 11.30, pooled std 1.005 vs 0.999,
sliced Wasserstein 0.121.

These numbers say the *pipeline* works — preconditioning, training, sampling,
CFG and the evaluation harness all agree — on a synthetic corpus with
deliberately informative conditioning. They say nothing about whether real
advice embeddings will train. That is the W3 question and it needs the real
corpus.
