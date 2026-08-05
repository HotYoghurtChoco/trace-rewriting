# Stage 6B-3 implementation lock

This file records implementation details that do not change the frozen
`PRE_REGISTRATION.md` protocol.  It must be committed before any Stage 6B-3
GPU smoke or formal run.

- The standalone validator is `validate_one_step.py`; existing Stage 6A and
  Stage 6B scoring code is not modified.
- Candidate gradients and reference gradients use the exact Stage 6B-1
  completion-only helper, 1024-token truncation, BF16 model precision,
  evaluation mode, and batch size one.
- Every reference loss is evaluated separately with model batch size one.
  The reported reference loss is the arithmetic mean of exactly 100
  per-example losses, never a token-count-weighted batch loss.
- The initial trainable tensors are saved exactly in each tensor's loaded
  dtype.  For each candidate and eta, the candidate gradient is recomputed
  from the restored initial state in FP32.  The manual update is computed as
  `float32(theta_0) - eta * float32(g_i)` and cast back to each tensor's
  original dtype.  No optimizer object or optimizer state exists.
- The smoke uses candidate index 0 and all five frozen eta values.  The formal
  run uses all candidate indices 0 through 99 and all five eta values.
- Every candidate/eta cell records all 100 individual updated reference losses.
- Inferential and descriptive predicted-change fields use the frozen Stage 6B-1
  dot product.  The independently recomputed dot product is retained only as
  the pre-registered execution-validity check.
- Candidate-loss and candidate-gradient-norm covariates use the frozen Stage
  6B-1 score-table values; recomputed values remain in the audit output.
- Spearman ranks use average ranks for exact ties.  The permutation and
  bootstrap generators are independently initialised NumPy PCG64 generators
  with seed 42.  The bootstrap interval uses NumPy's linear percentile method.
- Sign diagnostics use exact numerical sign with no post-hoc tolerance.
- The partial-rank diagnostic rank-transforms all variables, removes an
  intercept plus the three ranked covariates by ordinary least squares, and
  correlates the two residual vectors.
- Linear-scaling diagnostics compare each secondary eta's actual changes with
  the primary-eta actual changes multiplied by `eta / 5e-4`.
- The PBS launcher executes a validator copy extracted from the locked Git
  commit.  Untracked Hugging Face Arrow caches are permitted, but tracked or
  staged differences are prohibited.

The implementation may be repaired after a failed smoke, but the pre-registered
scientific endpoint, sample, eta grid, test direction, seeds, and decision rule
must remain unchanged.
