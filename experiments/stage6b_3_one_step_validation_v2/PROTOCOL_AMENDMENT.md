# Stage 6B-3 Diagnostic-Informed Protocol Amendment v2

## Registration record

- Amendment date: 2026-08-06
- Project: UNSW COMP4952 / Honours Project
- Student: Daniel / Hongyan Chen
- Supervisor: Bao Gia Doan / Bao Doan
- Repository branch: `research/gradient-feedback`
- Original v1 pre-registration commit:
  `7e2f8221d3e51e28959899b28aa37b3de896868e`
- Original v1 pre-registration SHA256:
  `edff3069693faef03e219078f24a1a5694071fbd8b8fb70cb3756256642a0c53`
- Locked v1 execution commit:
  `a5f7ac3f17fc59854d57daa65e0ad669ae53b9b7`
- Diagnostic commit:
  `38f27182c8f2bced05dff4275a230c64a795af42`
- Status: diagnostic-informed protocol amendment drafted after a technical
  smoke failure and numerical diagnostic, but before any positive-learning-rate
  update or Stage 6B-3 scientific endpoint was evaluated.

## 1. Why this amendment exists

The original v1 protocol remains preserved and must not be edited or
relabelled. Its validity check 10 required every recomputed candidate/reference
gradient dot product to agree with the frozen Stage 6B-1 value within:

`max(1e-6, 1e-4 * abs(frozen_dot))`

Smoke job `8924042.kman.restech.unsw.edu.au` stopped at that check for candidate
0 and `eta = 0`, before any parameter update or scientific one-step outcome was
computed. It remains an invalid technical run and must not be treated as a
Stage 6B-3 result.

Diagnostic job `8924283.kman.restech.unsw.edu.au` then tested numerical
reproducibility only. It performed no optimizer step, no manual positive-eta
update, and no scientific endpoint evaluation. It completed with
`Exit_status = 0`; `Stageout_status = 1` is retained in the audit evidence.

The diagnostic found:

- reference-gradient repeat cosine: `0.99999947178795123`;
- minimum candidate-gradient repeat cosine: `0.99993223072421444`;
- minimum frozen-versus-recomputed Spearman correlation: `0.99998799879988021`;
- minimum frozen-versus-recomputed Pearson correlation: `0.99999962329218206`;
- sign agreement: `100/100` in all four repeat combinations;
- Top50 overlap: `50/50` in all four repeat combinations;
- maximum norm-product-normalised dot discrepancy:
  `0.00085073219193706779`;
- failures under the original element-wise tolerance: `44` to `81` of 100;
- identical trainable and frozen parameter SHA256 values before and after the
  diagnostic.

This shows that the original element-wise absolute/relative tolerance is not a
numerically reproducible validity rule for the locked BF16/CUDA procedure even
though the gradient signal's ordering, sign and magnitude are highly stable.
The amendment changes only that technical validity rule and the no-update audit
needed to evaluate it.

## 2. Frozen diagnostic disclosure

The following diagnostic evidence was inspected before this amendment:

- failed smoke job: `8924042.kman.restech.unsw.edu.au`;
- diagnostic job: `8924283.kman.restech.unsw.edu.au`;
- diagnostic method version:
  `stage6b_3_gradient_reproducibility_diagnostic_v1`;
- diagnostic script SHA256:
  `dcd5bb5fc29a3ead3dbed5af016c72387a6b09b26eeac4c5a52a3caf45c958d1`;
- diagnostic PBS SHA256:
  `98bb9d497bfc3410b1319688394f63a7eb012411ff527f68eef03032e77fdefc`;
- immutable HOME backup:
  `/home/z5463756/honour/results/trace-rewriting/stage6b-gradient-feedback-qwen3-1.7b-v1/one_step_validation/diagnostics/stage6b_3diag_8924283`;
- backed-up run-manifest SHA256:
  `f4545b802907a1fb7f942ffdee4f5d7447d84e14991bbc6bf9a79a0f43d3cee1`;
- backed-up post-run-audit manifest SHA256:
  `54caab29b552ee4137b138530062aa1a62e501c8cd43bf8e2e22565f32719a20`.

No actual one-step reference-loss changes at a positive learning rate, primary
correlation, permutation p-value, bootstrap interval, or secondary scientific
outcome had been generated or inspected when this amendment was drafted.

## 3. Scope of the amendment

The original v1 pre-registration governs Stage 6B-3 except for the explicit
replacement of validity check 10 below and the additional no-update vector
reproducibility preflight needed to evaluate it.

The following remain unchanged:

- the same 100 candidates and their ascending original-index order;
- the same 100 frozen references;
- the same initial Candidate adapter state;
- the same completion-only gradient and loss procedure;
- evaluation mode, BF16 model precision and batch size one;
- the eta grid `0`, `1e-4`, `5e-4`, `1e-3`, `5e-3`;
- primary eta `5e-4`;
- the frozen Stage 6B-1 dot product as the inferential predictor;
- the primary Spearman endpoint;
- the positive-direction one-sided 100,000-permutation test;
- statistical seed 42, significance level 0.05 and decision rule;
- all secondary analyses and interpretation boundaries;
- all v1 validity checks other than check 10;
- the zero-eta absolute reference-loss-change limit of `1e-6`.

The amendment cannot be used to exclude candidates, select an eta, change the
test direction, or reinterpret the locked Stage 6B-2 null/reverse result.

## 4. Replacement validity check 10

Before any positive-learning-rate update in the formal run, the implementation
must perform a no-update reproducibility preflight from the exact initial state:

1. Recompute the arithmetic-mean gradient of all 100 frozen references using
   the locked Stage 6B-1 procedure.
2. Recompute one candidate gradient for each of the 100 candidates, without
   changing any parameter.
3. Compute the 100 recomputed candidate/reference dot products in frozen
   candidate-index order.
4. Compare that vector with the 100 frozen Stage 6B-1 dot products.

For candidate `i`, define the norm-product-normalised discrepancy:

`e_i = abs(recomputed_dot_i - frozen_dot_i) /`
`      (frozen_candidate_gradient_norm_i * frozen_reference_gradient_norm)`

The denominator must be finite and strictly positive. The formal execution is
valid under replacement check 10 only if:

- every recomputed dot and every `e_i` is finite;
- Spearman correlation between the frozen and recomputed 100-dot vectors is at
  least `0.999`;
- `max(e_i)` is no greater than `0.002`.

The threshold `0.002` and vector-rank threshold `0.999` are explicitly
diagnostic-informed. They were fixed after observing the no-update diagnostic
above and before observing any positive-eta or scientific endpoint result.

The preflight must also report Pearson correlation, exact sign agreement,
Top50 overlap, absolute-error summaries and normalised-error summaries. These
are audit descriptors, not additional pass/fail criteria.

For every candidate gradient actually used in an eta cell, the implementation
must record its recomputed dot and normalised discrepancy. Each such discrepancy
must also be no greater than `0.002`; otherwise the run is invalid. The old v1
tolerance `max(1e-6, 1e-4 * abs(frozen_dot))` must still be reported for audit
comparison but must not be used as a v2 pass/fail rule.

The frozen Stage 6B-1 dot products remain the predictor in every primary and
secondary inferential calculation. Recomputed dots are execution-audit values
only and cannot replace the frozen predictor after outcomes are inspected.

## 5. Smoke and formal boundaries

The v2 smoke must first execute the same all-100-candidate no-update vector
preflight required by the formal run. Only after that preflight passes may the
smoke apply updates for candidate index 0 at all five frozen eta values. It
must apply the per-cell `0.002` normalised-discrepancy limit, all unchanged v1
validity checks, and the zero-eta control. It does not apply positive-eta
updates to the other 99 candidates, evaluate the 100-candidate primary
endpoint, or produce a scientific result.

The v2 formal run must begin with the all-100-candidate no-update preflight. No
positive-eta update may occur unless that preflight passes. A formal run is
invalid if it begins before both this amendment and the corresponding v2
execution code are committed and locked.

The failed v1 smoke, completed diagnostic, v1 protocol and v1 execution code
must remain preserved. They must not be overwritten, deleted, retagged as
successful, or silently replaced.

## 6. Interpretation and reporting boundary

This amendment is not an independent blind pre-registration. It is a
transparent, diagnostic-informed repair of a technical numerical-validity
criterion before the scientific experiment was run.

Any positive, null or reverse v2 scientific result must be retained and
reported under the unchanged v1 endpoint and decision rule. Secondary analyses
cannot rescue a failed primary endpoint. Stage 6C remains deferred until the
complete Stage 6B evidence is reviewed and archived.
