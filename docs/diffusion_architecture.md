# Diffusion architecture (Phase 1, W2)

For review before training. Implementation: `diffusion/base/transformer.py`,
`diffusion/base/preconditioning.py`. Everything below is reproduced by
`python -m scripts.train_embedding_diffusion --preset smoke`.

## The problem with "stack transformer blocks over the input"

A sample here is one 768-dim vector, so the literal reading of the proposal
gives a sequence of length one, and self-attention over a single token is an
identity map with 4·h² parameters attached. Three options:

1. **Drop self-attention**, keep cross-attention + FFN. Cheapest, but the model
   can then only mix the 768 coordinates through the FFN.
2. **Condition-as-sequence**: treat `[a_t, i, Ē]` as a 3-token sequence. Makes
   self-attention meaningful but conflates the sample with its conditioning,
   and the sequence length still doesn't grow with the episodic memory.
3. **Token-split the sample** (chosen): reshape `a_t ∈ R^768` into
   `num_tokens = 12` slices of 64 dims, attend over those, and cross-attend into
   the conditioning. This is what latent-diffusion transformers do with latent
   patches.

I chose (3). It gives the network an internal sequence to route information
through, keeps the sample and its conditioning in separate attention paths, and
leaves the input/output spaces untouched. The cost is an arbitrary choice of
`num_tokens`, and the slicing is not semantically meaningful — dimensions 0–63
of an mpnet embedding are not a "patch". **This is the weakest assumption in the
design and the first thing I'd ablate**; `num_tokens ∈ {1, 4, 12, 48}` is a
cheap sweep, and `num_tokens = 1` degenerates to option (1), which makes the
ablation a direct test of whether the internal sequence buys anything.

`MLPDenoiser` (Phase 0) remains as the honest baseline. If a 50M-parameter
transformer does not beat a well-tuned MLP on the same data, we should say so.

## Layout

```
a_t (B, 768) ──split──> (B, 12, 64) ──linear──> (B, 12, 512) + learned pos
                                                      │
c_noise ──sinusoidal──> MLP ──> (B, 512) ─┐           │  ×8 blocks:
issue ──linear──> pooled ─────────────────┴─> adaLN ──┤    RMSNorm → self-attn   (gated)
                                                      │    RMSNorm → cross-attn  (to context)
issue, episodic ──linear──> context tokens ───────────┘    RMSNorm → SwiGLU      (gated)
                                                      │
                                          RMSNorm → linear ──> (B, 12, 64) ──> (B, 768)
```

- **Pre-norm, RMSNorm, SwiGLU** as specified. No biases in attention or FFN.
- **Noise conditioning by adaLN, not an extra token.** The block produces
  shift/scale/gate for the attention and FFN sub-layers from the `c_noise`
  embedding. A denoiser's behaviour must change qualitatively across four
  orders of magnitude of σ; modulating every block is a stronger lever than one
  token the model may learn to ignore. The modulation projection is
  zero-initialised, so each block starts as the identity.
- **Cross-attention** into `[null, issue, episodic₁..M]` with a key-padding
  mask, which is what makes the variable-length episodic memory work.
  `tests/test_transformer.py` asserts that padded slots cannot influence the
  output even when filled with ±10³.
- **Zero-initialised output projection**, so at step 0 the preconditioned
  denoiser is exactly the identity `D(x;σ) ≈ x` rather than noise.
- **Classifier-free guidance** via a learned null-context parameter with
  per-row dropout. The test that a dropped row is *bitwise* the unconditional
  forward pass matters: if dropping only partly removes conditioning, the CFG
  extrapolation `D_u + s(D_c − D_u)` is meaningless.

**Parameters: 47.7M** at `depth=8, hidden=512, heads=8, num_tokens=12` — inside
the ~50M budget. The adaLN projections are 3.1M of the 6.0M per block, i.e. more
than half the model; sharing them across blocks would free ~22M for depth if we
turn out to be depth-limited.

## EDM preconditioning

Implemented exactly as Karras Table 1, with `sigma_data` **measured, not
assumed**:

| coefficient | value |
|---|---|
| `c_in` | `1/√(σ² + σ_data²)` |
| `c_skip` | `σ_data²/(σ² + σ_data²)` |
| `c_out` | `σ·σ_data/√(σ² + σ_data²)` |
| `c_noise` | `ln(σ)/4` |
| `λ(σ)` | `1/c_out(σ)²` |

Training noise `ln σ ~ N(−1.2, 1.2²)`; sampling schedule Karras eq. 5 with
ρ = 7 and a trailing σ = 0; Heun 2nd-order sampler (`s_churn = 0` by default).
`tests/test_preconditioning.py` asserts the two properties the coefficients
exist for — network input variance ≈ 1 and target variance ≈ 1 at
σ ∈ {0.01, 0.5, 5, 50} — and that the weighted denoising loss is numerically
identical to the unit-variance MSE form.

**`sigma_data` is a property of the corpus, not a constant.** The 0.5 default
is the image-domain value; using it on embeddings would misplace the whole noise
schedule. `estimate_sigma_data` measures it from the training split at startup
and the value is recorded in the checkpoint and the report.

## Normalisation, and why it isn't optional

mpnet embeddings are unit-norm with per-dimension standard deviations that vary
substantially. EDM assumes a *single scalar* `σ_data`, so without whitening the
noise level that is mid-schedule for a high-variance coordinate has already
destroyed a low-variance one, and no single schedule is right for both. This is
the workplan's fourth failure mode ("samples don't converge to data marginals")
and `EmbeddingNormalizer(mode="whiten")` is the fix; `mode="global"` is retained
as the ablation baseline that should show the failure.

## Guidance in the EDM parameterisation

`grad_x log p(x;σ) = (D(x;σ) − x)/σ²`, so tilting by `∏ V_k^{w_k}` shifts the
denoiser output:

```
D̃(x;σ) = D(x;σ) + σ² · Σ_k w_k ∇_x log V_k(x)
```

the EDM analogue of `ε̃ = ε_θ − σ_t Σ w_k ∇log V_k` from the screening task. The
prefactor **grows** with σ here, so the noise-aware weight shaping that the toy
project needed is needed again; `weight_from_sigma` adapts a σ-parameterised
weight into the step-indexed `WeightFn` that `GuidanceSpec` fixed in W1, so the
Phase 0 API survives unchanged. `target="x0_hat"` evaluates verifiers at
`D(x;σ)` — in EDM the Tweedie posterior mean *is* the denoiser output — which is
what we'll want in embedding space, where a policy verifier applied to a noisy
point is meaningless.

## Open questions for review

1. **`num_tokens = 12` is a guess.** Ablate before the 50-GPU-hour run.
2. **adaLN parameter share.** Over half the model is modulation; is depth or
   conditioning capacity the binding constraint?
3. **Whitening vs. re-normalising to the sphere.** Whitening leaves samples off
   the unit sphere that mpnet embeddings live on; the decoder retrieves by
   cosine so norm is discarded anyway, but a spherical parameterisation is the
   more principled alternative and I have not implemented it.
4. **`sigma_max = 80` is the image default.** After whitening `σ_data ≈ 1`, so
   80 is ~80σ of noise; that is probably more than needed and wastes schedule.
   Worth measuring the σ at which the denoiser stops beating the mean predictor.
