# Stage 6C v4 Scorer-Equivalence Formal Audit

## Scope

This checkpoint validates that the reusable Stage 6C gradient scorer reproduces the frozen scorer closely enough before Stage 6C rewrite generation.

This run did not generate rewrites, train a student, perform an optimizer step, or update model parameters.

## Execution identity

- Status: `PASS`
- Method: `stage6c_gradient_scorer_equivalence_v4`
- Mode: `formal`
- PBS job: `8967237.kman.restech.unsw.edu.au`
- Job name: `tr-s6c-eq-v4`
- Exit status: `0`
- Wall time: `00:02:00`
- Execution branch: `research/stage6c-cosrewrite`
- Execution commit: `c5449b34a525993981d7653baadedb13e2a4d909`

The later Git commit containing this audit record is evidence-only and must not be confused with the execution commit above.

## Formal results

- Candidate count: `100`
- Candidate indices: complete and ordered from `0` through `99`
- Candidates failing v4 hard gates: `0`
- Frozen-versus-recomputed dot Spearman: `0.9999519952`
- Frozen-versus-recomputed cosine Spearman: `0.9999639964`
- Formal Spearman minimum: `0.999`
- Maximum normalized dot discrepancy: `0.00060164`
- Normalized dot discrepancy limit: `0.002`
- Maximum gradient-cosine absolute error: `0.00058473`
- Gradient-cosine absolute-error limit: `0.00225`
- Candidate gradient-norm validity gate: `100/100 PASS`
- Input and reference identity checks: `PASS`
- Parameter integrity: `PASS_NO_PARAMETER_UPDATE`
- Optimizer created: `false`
- Optimizer step performed: `false`

The legacy v3 gradient-norm tolerance diagnostic reported `84 True` and `16 False`. This diagnostic is non-binding under the amended v4 protocol and does not represent 16 v4 hard-gate failures.

## Archived evidence

HOME archive root:

`/home/z5463756/honour/results/trace-rewriting/stage6c-cosrewrite-qwen3-1.7b-v1/scorer_equivalence`

Full archive manifest:

- File: `STAGE6C_SCORER_EQUIVALENCE_SHA256SUMS`
- SHA256: `486bb3628781e2e5ade59176a2f7f27b951d9b014bf86cf266ca33fdc5609e05`

Critical v4 formal artifacts:

| Archived file | SHA256 |
|---|---|
| `formal_8967237/COMPLETE` | `f2eba1a990c7a5bf95bacd1adc091b7db87295b29459e3500c2447ae87abae7f` |
| `formal_8967237/run.log` | `4fa712800c6fe3dda5a452d85cbe632d8d935355af7938495e9e50b9787382a8` |
| `formal_8967237/stage6c_scorer_equivalence_candidates.jsonl` | `f5f8683333695366b3b2b4ecf7d57b8ab19a418660461f22b0d5e47a57c9ce83` |
| `formal_8967237/stage6c_scorer_equivalence_summary.json` | `790353eda15b7a815c635ee561e02a64231c617ac4c98a43cf4fd3f288596899` |
| `job_records/qstat-xf-formal-8967237.txt` | `de4fe16d615f2afc751674027dd2fbd7afad279c39bd26b90495b25bd216a715` |
| `job_records/tr-s6c-eq-v4-formal-8967237.pbs.log` | `b12fee57b1b6fcd0b6e3fc217bb8252f575ceb015850ce7227e36bcef8242438` |

The complete manifest also covers the retained earlier v3 attempts and all v4 smoke evidence.

## Checkpoint boundary

The scorer-equivalence checkpoint is complete and passed. This establishes implementation equivalence only; it does not establish that the Stage 6C rewriting method improves resistance to student distillation.

At this checkpoint:

- Stage 6C rewrites have not been generated.
- A Stage 6C student has not been trained.
- Stage 6C method effectiveness remains unknown.
- The next activity is review and preparation of rewrite-generation implementation.
