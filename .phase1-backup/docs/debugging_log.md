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
