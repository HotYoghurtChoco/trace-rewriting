# Stage 6B-3 One-Step Gradient Prediction Validation v1

## Pre-registration

- Registration date: 2026-08-05
- Project: UNSW COMP4952 / Honours Project
- Student: Daniel / Hongyan Chen
- Supervisor: Bao Gia Doan / Bao Doan
- Repository branch: `research/gradient-feedback`
- Parent commit before pre-registration: `6a9bb3b6403fe72b05193b4e06906167af061750`
- Status: Protocol locked before validator implementation or GPU execution

## 1. Purpose and stage boundary

Stage 6B-3 tests whether the first-order gradient prediction computed in
Stage 6B-1 predicts the actual change in frozen reference loss after one
manual SGD update on a candidate trace.

Stage 6B-2 compared cosine-ranked High50 and Low50 training sets and produced
a locked null/reverse result:

- High Rewrite answer-forced accuracy: 0.77
- Low Rewrite answer-forced accuracy: 0.80
- High minus Low: -0.03

Stage 6B-3 does not replace or reinterpret that result. It directly tests the
local one-step dot-product prediction and does not claim to measure final
student accuracy or full OPRO performance.

## 2. Frozen provenance

### 2.1 Score artifacts

Candidate score table:

`/srv/scratch/z5463756/honour/results/stage6b-gradient-feedback-qwen3-1.7b-v1/scoring/stage6b_1c_8918577/stage6b_1c_candidate_scores.csv`

SHA256:

`e324e84644fde5974f9f7399338b62c629fa7ec374389f297129fb2811026efa`

Scoring summary:

`/srv/scratch/z5463756/honour/results/stage6b-gradient-feedback-qwen3-1.7b-v1/scoring/stage6b_1c_8918577/stage6b_1c_summary.json`

SHA256:

`8d00ae1e5997005e60426e3ef97133531f2ebbc2fbf77deadd4db588b6dfbee0`

Scoring method:

- Method version: `stage6b_1c_gradient_scoring_v1`
- Method-lock SHA256:
  `c366851cb049882a5b642cc08bff22e0fef087efa2cee934b6f02bbc5185a337`
- Preflight SHA256:
  `0dbc7296aff3ac48f200e40902e634cb3413cb26b05f48fedbb1a191fb568cdf`
- Gradient mode: `eval_no_dropout`
- Candidate count: 100
- Reference count: 100
- Candidate truncated count: 0
- Reference truncated count: 0
- Scoring seed: 888
- Trainable tensor count: 392
- Trainable parameter count: 17,432,576
- Optimizer step during scoring: false

### 2.2 Initial adapter state

The initial state theta_0 is the Stage 6A Candidate final adapter:

`/srv/scratch/z5463756/honour/results/stage6a-formal-pilot-qwen3-1.7b-v1/stage6a_formal_pilot_candidate_0/finetuned_model/qwen3-1.7b-base-ea980cb0a6c2/adapter`

Adapter configuration SHA256:

`fa981a6e6d4a88c8672e6816aed76f30ca3bced2ade2a147e3f81ad32a92a95d`

Adapter model SHA256:

`bf31d1b724b94aeaaed8ff90c1142884e89496264ac6225527321a9d46545a2e`

The clean-comparator adapter is not the initial state for Stage 6B-3.

### 2.3 Frozen reference data

Reference data:

`/srv/scratch/z5463756/honour/results/stage6b-gradient-feedback-qwen3-1.7b-v1/frozen_data/gradient_reference.jsonl`

SHA256:

`517161b29ce6eba59dd7bbd7519670fde22e3d7f7258b00700dfd7939eae166d`

Reference manifest SHA256:

`7b32ac84b9a0f97a6135e5e197c33bac534ab67fa2aaa0528182be05d87124f1`

Frozen-data SHA256SUMS file SHA256:

`3ce595a067f8f4d895bdb5b20a1e0b4cc58415cc542b481f786a7f665eb2b5ea`

Candidate identities and their ascending original-index order are frozen by
the candidate score table.

## 3. Research question

For each candidate trace i, does the first-order predicted change in frozen
reference loss agree with the actual change after one SGD update from the
same initial adapter state?

Let:

- g_i be the completion-only gradient for candidate i;
- g_ref be the arithmetic mean of the 100 per-example reference gradients;
- eta be the fixed learning rate;
- L_ref be the arithmetic mean of the 100 per-example completion-only
  reference losses.

The predicted change is:

`x_i(eta) = -eta * <g_ref, g_i>`

The actual change is:

`y_i(eta) = L_ref(theta_0 - eta * g_i) - L_ref(theta_0)`

The frozen score field
`predicted_reference_loss_change_per_unit_step` is exactly
`-gradient_dot`.

## 4. Hypothesis and sign interpretation

Primary hypothesis:

Candidates with a larger predicted reference-loss change will also produce a
larger actual reference-loss change.

Therefore, the primary expected Spearman correlation between predicted and
actual change is positive.

Sign interpretation:

- `gradient_dot > 0` predicts a negative loss change and therefore lower
  reference loss;
- `gradient_dot < 0` predicts a positive loss change and therefore higher
  reference loss;
- more positive `-eta * gradient_dot` represents greater predicted damage.

Cosine similarity is a secondary directional metric. It is not identical to
the dot-product prediction because it removes gradient magnitude.

## 5. Fixed experimental design

For every candidate and every learning rate:

1. Restore exactly the same initial Candidate adapter state theta_0.
2. Put the model in evaluation mode with dropout disabled.
3. Use the same tokenizer, chat template, completion-only labels, truncation
   rules, model precision and gradient procedure as Stage 6B-1.
4. Recompute the single-candidate gradient g_i from theta_0.
5. Update only the same 392 trainable LoRA tensors.
6. Apply the manual update:

   `theta_i = theta_0 - eta * g_i`

7. Evaluate all 100 frozen references.
8. Calculate the arithmetic mean of their per-example completion-only losses.
9. Record actual loss change relative to the theta_0 baseline.
10. Restore theta_0 before processing another learning rate or candidate.

Candidate-gradient batch size is one candidate trace. Reference losses must be
calculated per example before their arithmetic mean is taken; they must not be
replaced by a token-count-weighted global mean.

The following are prohibited:

- Adam or AdamW;
- momentum;
- weight decay;
- gradient clipping;
- gradient normalisation;
- optimizer-state reuse;
- carrying an updated adapter into another candidate or learning rate;
- updating frozen base-model parameters.

## 6. Learning-rate grid

The fixed learning rates are:

- Control: `0`
- Secondary: `1e-4`
- Primary: `5e-4`
- Secondary: `1e-3`
- Secondary: `5e-3`

The zero learning rate is an implementation and reset control. It is not a
scientific endpoint.

The primary learning rate cannot be replaced by a secondary learning rate
after inspecting results.

## 7. Primary endpoint

The primary endpoint uses all 100 candidates at `eta = 5e-4`:

`Spearman(predicted_reference_loss_change, actual_reference_loss_change)`

where:

`predicted_reference_loss_change = -5e-4 * gradient_dot`

No candidate may be excluded on the basis of its score, sign, loss, length or
observed outcome.

### Primary statistical test

- Positive-direction, one-sided permutation test
- 100,000 permutations
- Statistical seed: 42
- P-value calculation:
  `(1 + count(permuted_rho >= observed_rho)) / 100001`
- Significance level: 0.05

A 10,000-resample percentile bootstrap 95% confidence interval, using seed 42,
must also be reported. The confidence interval is descriptive and is not an
additional pass/fail condition.

### Decision rule

- `observed rho > 0` and positive-tail `p < 0.05`:
  supports local first-order ranking prediction;
- negative-direction `p < 0.05`:
  reverse/contradicted result;
- otherwise:
  null/inconclusive result.

The negative-direction classification uses the corresponding lower-tail
permutation probability and does not replace the primary positive-direction
test.

## 8. Secondary analyses

The following analyses must be labelled secondary:

- Spearman correlation at `1e-4`, `1e-3` and `5e-3`;
- Pearson correlation at every positive learning rate;
- OLS calibration of actual change on predicted change, including intercept,
  slope and R-squared;
- theoretical calibration target: intercept zero and slope one;
- predicted-versus-actual sign agreement;
- sign confusion matrix and balanced sign accuracy;
- MAE and RMSE;
- cosine similarity as a secondary predictor;
- candidate loss, candidate gradient norm and completion length diagnostics;
- partial-rank diagnostic controlling for candidate loss, gradient norm and
  completion length;
- deviation from linear scaling as the learning rate increases.

Secondary analyses are descriptive and cannot rescue a failed primary
endpoint.

## 9. Fixed statistical and output rules

- Use all 100 candidates in ascending original candidate-index order.
- Preserve original candidate identifiers in every output row.
- Statistical random seed is 42.
- Scoring/model seed remains 888 where applicable.
- Report full-precision machine-readable values.
- Rounded values may only be used in human-readable summaries.
- Preserve individual reference losses or sufficient per-example audit data
  to reconstruct every reported arithmetic mean.
- Do not select a subset, threshold or learning rate after viewing results.

## 10. Required validity checks

A formal run is invalid if any of the following occurs:

1. Any frozen SHA256 value does not match this document.
2. Candidate or reference count is not exactly 100.
3. Candidate order or identity differs from the frozen score table.
4. The initial adapter is not the Candidate final adapter.
5. The number of trainable tensors is not exactly 392.
6. A non-LoRA or otherwise frozen parameter changes.
7. Parameters are not reset exactly to theta_0 before every candidate and
   learning rate.
8. The zero-learning-rate control changes any parameter.
9. The implemented parameter update differs from
   `theta_0 - eta * g_i`.
10. A recomputed gradient dot product differs from the frozen value by more
    than `max(1e-6, 1e-4 * abs(frozen_dot))`.
11. A NaN or infinity appears in a gradient, parameter, loss or statistic.
12. Any required positive-learning-rate result row is missing.
13. The primary endpoint, learning rate, sample, direction or decision rule is
    changed after results are inspected.
14. A formal GPU run begins before the pre-registration commit is created.

The zero-learning-rate reference-loss change must have absolute magnitude no
greater than `1e-6`. A larger control deviation invalidates the run pending
diagnosis.

If an implementation defect invalidates a run, the defect and failed job must
be documented. The implementation may be repaired, but this protocol remains
unchanged and the complete formal experiment must be rerun.

## 11. Interpretation boundary

A positive result supports only the local claim that the frozen gradient
dot-product signal predicts one-step reference-loss change in this setting.

It does not by itself demonstrate:

- lower final GSM8K student accuracy;
- successful multi-epoch degradation;
- improved full OPRO search;
- generalisation to another model, dataset or seed;
- a deployable defence against model extraction.

A null or reverse result must be retained and reported. Stage 6C design must
be based on the complete Stage 6B evidence rather than only favourable
secondary results.

## 12. Planned output and archival requirements

Formal result root:

`/srv/scratch/z5463756/honour/results/stage6b-gradient-feedback-qwen3-1.7b-v1/validation/stage6b_3_one_step_validation_v1`

The formal archive must contain at least:

- a copy of this pre-registration;
- execution-code commit;
- PBS job ID and launcher;
- environment and GPU/backend information;
- run manifest;
- per-candidate machine-readable results;
- primary and secondary statistics;
- validity-check report;
- limitations and interpretation;
- complete SHA256SUMS;
- stdout and stderr logs.

The execution-code commit must be recorded separately from the later
evidence-archive commit. After Stage 6B-3 is completed, the formal results must
also be copied from Scratch to HOME, verified by SHA256, committed as compact
Git evidence where appropriate, pushed to GitHub and assigned a versioned Git
tag.
