# Stage 6B Gradient-Feedback Final Closure

## Closure status

Stage 6B is complete. Its planned scoring, downstream pilot, and same-start
one-step mechanism experiments have all produced valid completed results. The
stage is closed with mixed-to-negative scientific evidence rather than a
positive validation of the original gradient-learnability hypothesis.

This final closure covers:

* Stage 6B-1: frozen gradient alignment scoring and ranking;
* Stage 6B-2: GradHigh50 versus GradLow50 downstream pilot;
* Stage 6B-3: same-start one-step gradient prediction validation;
* all invalid attempts, diagnostics, protocol amendments, and scheduling
  changes needed to interpret those formal results correctly.

The final versioned tag for this complete stage is:

```text
stage6b-gradient-feedback-qwen3-1.7b-v1
```

The earlier annotated tag remains an immutable intermediate checkpoint:

```text
stage6b-1-2-gradient-feedback-v1
-> 3d69dd91524542e0d4be53c6182c4113c6c083e2
```

It is not moved or deleted by this closure.

## Research question and stage decomposition

Stage 6A had shown that, in a controlled Qwen3-1.7B pilot, students trained on
the available optimized rewritten traces learned less effectively than a
matched clean-trace student. Stage 6B asked whether candidate-reference
gradient alignment could explain or predict that learnability difference well
enough to support a gradient-feedback rewriting method.

The stage was separated into three questions:

| Substage | Question | Why it was needed |
| --- | --- | --- |
| 6B-1 | Can candidate-reference gradient alignment be computed and frozen reproducibly? | A stable trace-level signal was required before testing learnability. |
| 6B-2 | Do high-alignment traces train a more accurate held-out student than low-alignment traces? | This was the first downstream directional test of the ranking. |
| 6B-3 | From the same student state, does frozen gradient dot product predict the actual one-step reference-loss change? | This removed the main different-problem-set confound in 6B-2 and tested the local first-order mechanism directly. |

Stage 6C is separate. It converts a score into rewriting feedback and must be
evaluated with matched student training and held-out accuracy. Stage 6B neither
pre-validates nor cancels Stage 6C.

## Frozen inputs shared across Stage 6B

Stage 6B used the final Stage 6A Candidate LoRA adapter and the 100 official
candidate rewrite traces. The frozen reference set contained 100 examples
drawn from seed-137 holdout positions 100--199, disjoint from the Stage 6A
evaluation positions 0--99.

The locked gradient definition used:

```text
392 trainable LoRA tensors
17,432,576 trainable parameters
completion-only loss
evaluation mode with dropout disabled
BF16 model computation
no optimizer update during Stage 6B-1 scoring
100 per-example reference gradients averaged in FP32
```

Important frozen input hashes include:

| Input | SHA256 |
| --- | --- |
| Stage 6B-1 candidate score CSV | `e324e84644fde5974f9f7399338b62c629fa7ec374389f297129fb2811026efa` |
| Stage 6B-1 scoring summary | `8d00ae1e5997005e60426e3ef97133531f2ebbc2fbf77deadd4db588b6dfbee0` |
| Frozen reference JSONL | `517161b29ce6eba59dd7bbd7519670fde22e3d7f7258b00700dfd7939eae166d` |
| Frozen reference manifest | `7b32ac84b9a0f97a6135e5e197c33bac534ab67fa2aaa0528182be05d87124f1` |
| Stage 6A adapter config | `fa981a6e6d4a88c8672e6816aed76f30ca3bced2ade2a147e3f81ad32a92a95d` |
| Stage 6A adapter weights | `bf31d1b724b94aeaaed8ff90c1142884e89496264ac6225527321a9d46545a2e` |

## Stage 6B-1: gradient scoring and ranking

### Purpose and locked method

Stage 6B-1 computed each candidate trace's completion-only LoRA gradient at
the fixed Stage 6A Candidate adapter state. It compared that gradient with the
mean gradient of the 100 frozen reference examples.

The primary trace-level score was cosine similarity. Dot product and the
corresponding first-order `-dot` prediction were retained as diagnostics. The
100 candidates were sorted by descending cosine, with original candidate index
as the tie-breaker:

```text
ranks 1--50:   GradHigh50
ranks 51--100: GradLow50
```

### Stage 6B-1b invalid attempt

```text
job:    8916264
status: invalid; no scientific code executed
```

The shell heredoc was attached to `tee` instead of Python. The Python source
was printed rather than executed, GPU use was zero, and exit code 0 was not
accepted as evidence of a run. The logs were preserved to document the
failure.

### Stage 6B-1b valid reproducibility smoke

```text
job:    8918490
status: PASS
```

The repeated candidate loss was identical and the repeat-gradient cosine was
0.9999559343. This established highly reproducible direction across two
passes, not bitwise-identical gradient tensors.

### Stage 6B-1c full scoring

```text
job:       8918577
status:    PASS
walltime:  00:01:53
```

Formal results were:

| Metric | Result |
| --- | ---: |
| Candidate/reference count | 100 / 100 |
| Overall cosine mean | 0.0810139413 |
| Overall cosine median | 0.0823431383 |
| Cosine minimum / maximum | -0.0728829008 / 0.1656464144 |
| GradHigh50 cosine mean | 0.1126098799 |
| GradLow50 cosine mean | 0.0494180028 |
| Positive / negative cosine | 96 / 4 |
| Dot-product mean | 0.2564384594 |
| Cosine--dot Spearman | 0.9332613261 |
| Mean reference-gradient norm | 1.2550210246 |
| Candidate-gradient norm mean | 2.3717365660 |
| Candidate loss mean | 3.069626 |
| Reference loss mean | 1.1347439781 |

Stage 6B-1 proved that the score could be computed, reproduced closely,
frozen, and used to form separated High/Low groups. It did not prove that high
cosine means greater final student learnability.

## Stage 6B-2: GradHigh50 versus GradLow50 pilot

### Purpose and controls

Stage 6B-2 trained independent students from the same Qwen3-1.7B base model:

```text
GradHigh50 job: 8920030
GradLow50 job:  8920031
```

Each group used 50 traces, the same hyperparameters, seed 888, and the same
100-question held-out evaluation protocol. Each rewrite-trained student had a
clean-trace comparator trained on the same 50 underlying problems as that
group.

An early launcher-generation command was pasted into the shell without being
executed by Python, so expected launchers and frozen inputs were absent. The
missing files were detected before submission; the launchers were regenerated
with a correct Python invocation and their paths and hashes were verified.
This prelaunch error did not contaminate jobs 8920030 or 8920031.

### Locked hypothesis and formal result

The preregistered directional metric was:

```text
GradHigh50 rewrite AF - GradLow50 rewrite AF
```

The locked direction expected a value greater than zero, with at least +0.05
treated as a meaningful signal. A value at or below zero was null/reverse.

| Group | Mean cosine | Clean Raw | Rewrite Raw | Clean AF | Rewrite AF | AF drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GradHigh50 | 0.1126098799 | 0.85 | 0.72 | 0.88 | 0.77 | 0.11 |
| GradLow50 | 0.0494180028 | 0.82 | 0.77 | 0.87 | 0.80 | 0.07 |

The primary result was:

```text
0.77 - 0.80 = -0.03
```

Stage 6B-2 therefore did not support its directional hypothesis and was locked
as **null/reverse**.

The High group's comparator-normalised AF drop exceeded the Low group's by
0.04, and its Raw drop exceeded the Low group's by 0.08. These were secondary,
post-hoc observations. They cannot override the failed primary result or be
reported as validation of the method.

### Why Stage 6B-2 was not the final mechanism test

GradHigh50 and GradLow50 contained different underlying problems. The result
could therefore be influenced by difficulty, trace length, candidate loss,
gradient norm, one training seed, only 50 training examples per group, and a
small unequal number of token-capped generations. The clean comparators only
partially controlled this confounding. This limitation motivated Stage 6B-3.

## Stage 6B-1/6B-2 intermediate archive

The completed intermediate snapshot is stored under:

```text
experiments/stage6b_1_2_gradient_feedback_v1/
```

It contains compact PBS files, method locks, preflight records, invalid and
valid smoke evidence, full scoring outputs, frozen-data manifests, pilot
metrics, and per-example evaluations. It excludes model weights, caches, and
environments.

```text
archive commit: 3d69dd91524542e0d4be53c6182c4113c6c083e2
annotated tag:  stage6b-1-2-gradient-feedback-v1
```

The archive commit is retrospective; Stage 6B-1 and 6B-2 were executed from
`6e22ffdaf0e270f4f8c1ab1ca6268d274a99165a`.

## Stage 6B-3: same-start one-step validation

### Why a same-start design was used

For every candidate, Stage 6B-3 reset the same Stage 6A Candidate adapter
state, manually applied one candidate-gradient step, and measured the same
100-example reference loss. It tested whether the frozen Stage 6B-1 dot
product ranked the observed one-step loss changes.

The grid used 100 candidates and five step sizes (`0`, `0.0001`, `0.0005`,
`0.001`, and `0.005`) for 500 cells. The primary step size was locked to
`0.0005`, and every cell began from identical `theta_0`.

### v1 invalid smoke and diagnostic

The scientific design was preregistered in commit
`7e2f8221d3e51e28959899b28aa37b3de896868e`. The first implementation used
commit `a5f7ac3f17fc59854d57daa65e0ad669ae53b9b7`.

Job 8924042 stopped at candidate 0 and `eta = 0` because its elementwise dot-
reproduction tolerance was too strict for BF16/CUDA recomputation. No
positive-step update or endpoint was evaluated, so this was an invalid
technical run, not an adverse scientific result.

Diagnostic job 8924283, from commit
`38f27182c8f2bced05dff4275a230c64a795af42`, showed extremely stable rank,
sign, and relative magnitude: minimum candidate repeat cosine 0.9999322307,
minimum frozen-versus-recomputed Spearman 0.9999879988, sign agreement 100/100,
and Top-50 overlap 50/50.

### v2 amendment and engineering smoke

Before any formal positive-step outcome was observed, commit
`7a504251b7b1616017f46d74d530ec830cb5d7f8` transparently replaced only the
unsuitable technical tolerance with a vector-level validity gate. The
candidate/reference identities, initial state, step sizes, primary endpoint,
statistics, and interpretation rule were unchanged.

The locked v2 execution commit was:

```text
35442654a8b388f2584d01fd2f7a5b6afb9ff9ed
```

Job 8924540 passed the full no-update preflight and a one-candidate/five-step
engineering smoke. It explicitly did not evaluate the scientific endpoint.

### Formal queue replacement

The original 8-hour job 8927619 waited approximately 318 hours in `csegpu12`.
A `qalter` walltime reduction was rejected by the cluster hook. Based on the
measured smoke runtime, a held two-hour replacement was resource-checked; the
old job was then deleted and the replacement released. This changed queueing
only, not the experiment.

```text
formal job:       9043125.kman.restech.unsw.edu.au
execution commit: 35442654a8b388f2584d01fd2f7a5b6afb9ff9ed
requested GPU:    1 x A100
requested CPUs:   8
requested memory: 46 GB
walltime limit:   02:00:00
observed walltime: 00:51:12
Exit_status:       0
```

### Formal validity and primary result

The formal job completed 500/500 cells, wrote 50,000 per-reference audit rows,
verified its original `SHA256SUMS`, and passed all locked identity, reset,
finite-value, discrepancy, manual-update, and zero-step controls.

The all-100 no-update preflight reproduced the frozen Stage 6B-1 dot ranking
with Spearman 0.9999759976 and maximum normalised discrepancy 0.0008265327,
inside the locked 0.002 limit. The maximum per-cell normalised discrepancy was
0.0010560287, also inside the same locked limit.

At the preregistered primary `eta = 0.0005`:

```text
Spearman rho:                -0.0838763876
positive one-sided p:         0.7990220098
negative one-sided p:         0.2010079899
bootstrap 95% CI:            [-0.2755179884, 0.1105261313]
decision:                     null_or_inconclusive_result
```

The formal run was valid, but the primary gradient-dot ranking hypothesis was
not supported. The result did not establish a statistically reliable reverse
relationship either.

The secondary `eta = 0.005` Spearman was 0.6405400540. Because this was a
secondary step size ten times larger than the primary and cross-step linear
scaling was poor, it is retained only as a step-size-sensitive future lead. It
cannot replace the primary endpoint or retrospectively make Stage 6B-3
positive.

The complete Stage 6B-3 result record and copied formal evidence are indexed
from:

```text
experiments/stage6b_3_one_step_validation_v2/RESULTS.md
```

## Consolidated jobs and decisions

| Stage | Job | Role | Engineering status | Scientific status |
| --- | ---: | --- | --- | --- |
| 6B-1b | 8916264 | first smoke | invalid; Python not executed | none |
| 6B-1b | 8918490 | valid gradient smoke | PASS | reproducibility only |
| 6B-1c | 8918577 | full scoring | PASS | score/ranking frozen |
| 6B-2 | 8920030 | GradHigh50 pilot | COMPLETE | contributes to primary -3 pp |
| 6B-2 | 8920031 | GradLow50 pilot | COMPLETE | contributes to primary -3 pp |
| 6B-3 v1 | 8924042 | first one-step smoke | invalid technical stop | none |
| 6B-3 diagnostic | 8924283 | no-update reproducibility | PASS | no endpoint |
| 6B-3 v2 | 8924540 | engineering smoke | PASS | no endpoint |
| 6B-3 old formal | 8927619 | long-queued formal | deliberately replaced before execution | none |
| 6B-3 formal | 9043125 | complete one-step validation | PASS | primary null/inconclusive |

## Overall Stage 6B conclusion

Stage 6B established that candidate-reference gradient alignment can be
computed and reproduced closely, but it did not validate the original claim
that this signal reliably predicts downstream or local learning quality under
the locked primary tests:

* Stage 6B-1: engineering/scoring success, no learnability claim;
* Stage 6B-2: primary High-minus-Low Rewrite AF = -3 percentage points,
  null/reverse;
* Stage 6B-3: primary one-step ranking Spearman = -0.0838763876 with
  positive-tail `p = 0.7990220098`, null/inconclusive.

The combined evidence therefore does not support presenting cosine or dot
product as a validated defence signal. It also does not prove that gradient
information is universally useless or that CosRewrite must fail. Any later
method claim must come from a separately locked, matched end-to-end student
experiment and held-out accuracy, not from a favourable secondary score alone.

Stage 6B is closed because all planned experiments and their valid outcomes
are complete, not because the original hypothesis was confirmed.

## Durable whole-stage HOME archive

The complete Stage 6B Scratch result root, including Stage 6B-1, Stage 6B-2,
and Stage 6B-3, was copied to the durable Katana HOME snapshot:

```text
/home/z5463756/honour/results/trace-rewriting/
stage6b-gradient-feedback-qwen3-1.7b-v1/full_scratch_snapshot_20260820/
```

The corresponding archive metadata is stored at:

```text
/home/z5463756/honour/results/trace-rewriting/
stage6b-gradient-feedback-qwen3-1.7b-v1/
full_scratch_snapshot_20260820_archive_metadata/
```

The archive was published at `2026-08-20T00:57:41Z` from execution HEAD
`35442654a8b388f2584d01fd2f7a5b6afb9ff9ed`. The verified identity is:

| Audit | Result |
| --- | ---: |
| Scope | complete Stage 6B-1/2/3 Scratch root |
| Source file count | 224 |
| HOME file count | 224 |
| Source file bytes | 367,616,133 |
| HOME file bytes | 367,616,133 |
| Apparent source/HOME size | 351 MB / 351 MB |
| Tree-manifest SHA256 | `0f4dad6d434bd9daa2fbc50ccadc99a427219a3e09af4800494fcdb20d4fb6dc` |
| Archive status | `PASS_EXACT_COPY` |

The source tree was hashed before and after the copy; both manifests matched.
The staging and final HOME manifests and inventories also matched, and a
checksum-based `rsync` dry run reported no remaining difference. Different
allocated-block counts on GPFS and NFS do not indicate a content difference.

The earlier 19 MB Stage 6B-3 HOME copy was an immediate safety checkpoint made
before this whole-stage archive. It is not a second versioned Stage 6B archive
and does not receive a separate commit or tag. The whole-stage snapshot is the
unified raw-results archive; Git retains only compact evidence and provenance.

## Evidence map

| Evidence | Repository path |
| --- | --- |
| Stage 6B-1/6B-2 compact snapshot | `experiments/stage6b_1_2_gradient_feedback_v1/` |
| Stage 6B-3 v1 preregistration, code, and diagnostic | `experiments/stage6b_3_one_step_validation_v1/` |
| Stage 6B-3 v2 amendment and locked code | `experiments/stage6b_3_one_step_validation_v2/` |
| Stage 6B-3 result interpretation | `experiments/stage6b_3_one_step_validation_v2/RESULTS.md` |
| Stage 6B-3 formal compact evidence | `experiments/stage6b_3_one_step_validation_v2/evidence/formal/stage6b_3b_v2_9043125/` |
| Whole-stage HOME archive metadata | `experiments/stage6b_gradient_feedback_closure_v1/evidence/home_archive/full_scratch_snapshot_20260820_archive_metadata/` |
| Whole-stage closure | `experiments/stage6b_gradient_feedback_closure_v1/README.md` |

Full adapters, checkpoints, Hugging Face caches, Python environments, and
model weights remain outside Git. The repository stores compact scientific
evidence, provenance, checksums, and human-readable interpretation.

The old 17 KB `stage6b_3_v2_review_bundle.tar.gz` is also excluded. Its only
contents are the v2 review-patch generator and the already-applied transform
patch; canonical code is preserved by commits `7a504251...` and `35442654...`,
and the bundle is not a formal result artifact.

## Version identity and verification

The final annotated tag `stage6b-gradient-feedback-qwen3-1.7b-v1` resolves the
whole-stage closure commit. The exact execution commits remain recorded
separately because an archive commit cannot be treated as the code that ran.

After checkout, verify the existing Stage 6B-1/6B-2 copied evidence with:

```bash
cd experiments/stage6b_1_2_gradient_feedback_v1
sha256sum -c COPIED_FILES_SHA256SUMS
```

Verify the compact Git subset of Stage 6B-3 with its copy manifest:

```bash
cd experiments/stage6b_3_one_step_validation_v2/evidence/formal/\
stage6b_3b_v2_9043125
sha256sum -c COPIED_FILES_SHA256SUMS
```

The retained `ORIGINAL_RESULT_SHA256SUMS` records hashes for all 19 formal
artifacts, including files deliberately excluded from Git. Run that manifest
against the complete formal result directory, not against the compact Git
subset.

Verify the Git copy of the whole-stage HOME archive metadata with:

```bash
cd experiments/stage6b_gradient_feedback_closure_v1
sha256sum -c COPIED_FILES_SHA256SUMS
```
