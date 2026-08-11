# Stage 6C scorer-equivalence protocol amendment v4

Date: 2026-08-11 (Australia/Sydney)

Status: Pre-rewrite amendment. No Stage 6C rewrite or student-outcome
result existed when this amendment was made.

## Scope

This amendment changes only the hard-gate role of the per-candidate
gradient norm in the Stage 6C scorer-equivalence verifier.

It does not modify:

- `optimize/gradient_feedback.py`;
- candidate or reference inputs;
- tokenization or token metadata;
- candidate loss, normalized-dot, cosine, or Spearman thresholds;
- reference-gradient equivalence;
- parameter-integrity and no-update checks;
- candidate count, identity, ordering, or frozen Stage 6B-1c artifacts.

## Preserved v3 evidence

The two v3 formal runs remain failures and are not overwritten or
retroactively relabelled:

- job 8963589 stopped after candidate 0 failed the numerical
  candidate-gradient-norm comparison;
- job 8965953 completed all 100 candidates and returned failure because
  14 of 100 candidates were outside that same comparison tolerance.

For job 8965953:

- candidate loss, normalized dot, cosine, token metadata, both Spearman
  vector gates, reference equivalence, input identity, and parameter
  integrity passed;
- all 14 numerical norm discrepancies were in GradLow50;
- the maximum candidate-gradient-norm relative difference was
  approximately 0.124636%;
- the v3 scalar relative tolerance was 0.01%.

These observations are consistent with numerical reproducibility being
the source of the discrepancy, but do not by themselves prove its cause.

## v4 rule

The candidate-gradient-norm hard gate now requires both the frozen and
recomputed norms to be finite and strictly positive.

The original v3 numerical comparison remains recorded for every
candidate using the unchanged scalar tolerance:

- relative tolerance: `1e-4`;
- absolute floor: `1e-6`;
- absolute and relative errors remain recorded;
- within/outside-tolerance results and candidate indices are summarised.

That numerical comparison is diagnostic only and is not included in the
v4 overall PASS/FAIL decision. No tolerance value is loosened.

The unchanged cosine absolute-error threshold remains an independent v4
hard gate. Its original derivation using both v3 norm comparisons is
historical context, not a claim that both norm comparisons remain v4 hard
gates.

## Interpretation boundary

A v4 PASS establishes scorer implementation equivalence under the
amended, pre-declared protocol only. It is not evidence that CosRewrite
improves rewrite quality or distilled-student accuracy.

A v4 FAIL preserves diagnostic evidence and must be investigated before
rewrite generation.

The required sequence remains:

1. commit and provenance-check the v4 amendment;
2. run and audit v4 smoke;
3. run and audit v4 formal;
4. only after the formal gate passes, begin rewrite generation.
