# Stage 6B-3 v2 implementation lock

This file records implementation details for the original v1 pre-registration
plus `PROTOCOL_AMENDMENT.md`. It must be committed with the v2 validator and
launchers before any v2 GPU smoke or formal run.

- The v1 protocol and implementation remain unchanged in
  `experiments/stage6b_3_one_step_validation_v1/`.
- The v2 amendment is locked at commit
  `7a504251b7b1616017f46d74d530ec830cb5d7f8` with SHA256
  `0a285cc3383c35d77d8a8e0c8d212fecc6ce1747617ff295ab2344bcc72a4c49`.
- The standalone v2 validator is `validate_one_step.py`; existing Stage 6A,
  Stage 6B-1 and Stage 6B-2 code and frozen evidence are not modified.
- Candidate and reference gradients use the exact Stage 6B-1 completion-only
  helper, 1024-token truncation, BF16 model precision, evaluation mode and
  batch size one.
- Before any eta cell, both smoke and formal modes recompute the mean gradient
  of all 100 references and one no-update gradient for every one of the 100
  candidates. The resulting dot vector must have Spearman correlation at
  least `0.999` with the frozen Stage 6B-1 vector and maximum norm-product-
  normalised discrepancy no greater than `0.002`.
- The preflight writes all 100 candidate audit rows and a summary before any
  positive-eta update. It also verifies exact trainable/frozen parameter
  hashes. Pearson correlation, sign agreement, Top50 overlap and the old v1
  tolerance are reported only as audit descriptors.
- Every candidate gradient recomputed for an eta cell must independently meet
  the same `0.002` normalised-discrepancy limit before the manual update is
  applied. The old v1 element-wise tolerance is never used for v2 pass/fail.
- Every reference loss is evaluated separately with model batch size one. The
  reported reference loss is the arithmetic mean of exactly 100 per-example
  losses, never a token-count-weighted batch loss.
- The initial trainable tensors are saved exactly in each tensor's loaded
  dtype. For each candidate and eta, the manual update is computed as
  `float32(theta_0) - eta * float32(g_i)` and cast back to the tensor's original
  dtype. No optimizer object or optimizer state exists.
- The v2 smoke uses candidate index 0 and all five frozen eta values only after
  its all-100 no-update preflight passes. The formal run uses candidate indices
  0 through 99 and all five eta values.
- Every candidate/eta cell records all 100 individual updated reference losses.
- The frozen Stage 6B-1 dot product remains the predictor in every inferential
  and descriptive predicted-change field. Recomputed dots are audit values and
  cannot replace the frozen predictor.
- Candidate-loss and candidate-gradient-norm covariates use the frozen Stage
  6B-1 score-table values; recomputed values remain in the audit output.
- Spearman ranks use average ranks for exact ties. The permutation and bootstrap
  generators are independently initialised NumPy PCG64 generators with seed 42.
  The bootstrap interval uses NumPy's linear percentile method.
- Sign diagnostics use exact numerical sign with no post-hoc tolerance.
- The partial-rank diagnostic rank-transforms all variables, removes an
  intercept plus the three ranked covariates by ordinary least squares and
  correlates the two residual vectors.
- Linear-scaling diagnostics compare each secondary eta's actual changes with
  the primary-eta actual changes multiplied by `eta / 5e-4`.
- Each PBS launcher runs a validator copy extracted from the locked Git commit.
  It records the amendment and implementation in provenance. Untracked Hugging
  Face Arrow caches are permitted, but tracked or staged differences are not.
- v2 smoke and formal output/log paths are distinct from all v1 and diagnostic
  paths. No previous evidence is overwritten.

The scientific endpoint, sample, eta grid, primary eta, test direction, seeds,
decision rule and interpretation boundaries remain those locked by v1. Stage
6C stays deferred until complete Stage 6B evidence is reviewed and archived.
