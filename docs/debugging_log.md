# Debugging log

Running record of what failed and why. Entries are append-only.

## W1 — porting

**Toy sampler divergence at moderate guidance weights (carried over from screening).**
Constant-weight guidance diverges from `w = 2` upward: mean ‖x‖ 19.5 at `w = 2`,
132 at `w = 3`, 997 at `w = 8`, against a data radius of 2. Cause is not
numerical — it is that a constant weight applies the full tilt at every noise
level (see `docs/porting_notes.md`, working notes). Fixed by
`gaussian_tilt_weight`. Recorded here because the same failure will reappear in
768-dim with `s²` recalibrated per dimension, and the symptom (sample norms
exploding while compliance reads 1.00) is easy to misread as a verifier problem.

**`clip_x0` is available but unused in reported runs.** Clamping the predicted
clean sample does stop the divergence, but it makes compliance non-monotonic in
`w` (0.96 at `w = 3`, 0.82 at `w = 10`) because clipping fights the guidance
rather than shaping it. Kept as a parameter, off by default. In embedding space
the equivalent knob is a norm clip to the embedding sphere, which is a different
and probably better-behaved operation — revisit at W8.

**Verifier gradient magnitudes are not comparable across verifiers.**
`TargetPointVerifier` has gradient `|x − c|/σ²`, so at `σ = 0.5` it is up to 16×
the half-plane verifier's unit gradient and wins every cell of a `(w1, w2)` grid.
The first grid showed no trade-off at all and looked like a null result. Fixed by
rescaling to `σ = 1` and choosing `w2 ∈ [0, 1]` against `w1 ∈ [0, 4]`. Direct
consequence for Phase 2: V_P, V_τ and V_E will have wildly different natural
gradient scales, so the `normalized` composition strategy and per-spec
`grad_clip` exist for this reason and their `λ_k` need tuning, not defaults.

**Untrained denoisers make trajectory-based assertions flaky.** A test asserting
that the mean pairwise gradient cosine is negative for two opposed verifiers
failed: with a randomly initialised denoiser the samples wander past the target
point, after which the two gradients agree. Replaced with an assertion that the
conflict is detected at some step (`min(cosine) < −0.5`) plus a direct check on
the data manifold itself. Lesson for Phase 2 tests: assert on gradients at known
points, not on where an untrained sampler happens to go.

## W2 — architecture and preconditioning

**Self-attention over a length-1 sequence.** The proposed architecture applies
transformer blocks to a single 768-dim vector, which makes self-attention an
expensive identity. Resolved by splitting the sample into 12 tokens of 64 dims
(`docs/diffusion_architecture.md`). Flagged rather than silently fixed because
the slicing is semantically arbitrary and deserves an ablation.

**Zero-init output head makes wiring tests vacuous.** `token_out` is
zero-initialised so training starts from the identity denoiser — correct, but it
means every output is exactly zero at init, so "conditioning changes the output"
passed trivially while proving nothing. The test fixture now perturbs the head
to stand in for a partially trained network, and a separate test asserts the
zero-init property itself. Any future test that checks "X changes the output"
must use the perturbed fixture.

**Retrieval similarity is nearly uninformative in high dimension.** In the
1,500-step smoke run, mean nearest-neighbour cosine to the advice bank was 0.736
for model samples and 0.737 for matched Gaussian noise. With a 2k-entry bank in
128 dimensions, the *maximum* cosine over the bank is high for almost any query,
so "samples retrieve to plausible advice" cannot be established by this number
alone. The baseline is now reported alongside it so the comparison is forced.
Rely on conditional recall@k against the unconditional control, and on sliced
Wasserstein, for the actual W3/W4 decisions — and on reading decoded text, which
no metric replaces.

**Per-dimension std correlation was measured in the wrong space.** After
whitening, every reference dimension has unit variance by construction, so
correlating per-dimension standard deviations compares against a constant and
returns noise (0.08–0.12 in early runs, meaninglessly). The distribution report
now decodes both sets to raw embedding space first. Generalises: any statistic
about the *shape* of the data distribution must be computed after inverting the
normaliser.

**`sigma_max = 80` is inherited from images and probably wrong here.** After
whitening `sigma_data ≈ 1`, so the schedule spends its top end at ~80σ of noise
where the denoiser can only predict the mean. Not yet changed — it needs the
real corpus to measure where the denoiser stops beating a mean predictor — but
it is a live suspect if the high-σ loss bucket flatlines.
