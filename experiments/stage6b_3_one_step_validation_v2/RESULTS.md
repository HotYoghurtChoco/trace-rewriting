# Stage 6B-3 One-Step Gradient Prediction Validation Results

## Archive status

Stage 6B-3 is scientifically complete. The formal v2 run passed all locked
execution-validity checks, but its preregistered primary hypothesis was not
supported. The primary result is therefore retained as **null/inconclusive**.

The experiment was executed from:

```text
branch:             research/gradient-feedback
execution commit:   35442654a8b388f2584d01fd2f7a5b6afb9ff9ed
formal job:         9043125.kman.restech.unsw.edu.au
result directory:   stage6b_3b_v2_9043125
method version:     stage6b_3_one_step_validation_v2
```

The later commit containing this result record and copied evidence is an
archive commit. It must not be confused with the execution commit above.

## Why Stage 6B-3 was required

Stage 6B-2 trained separate students on GradHigh50 and GradLow50. Its primary
comparison was downstream-facing, but the two groups contained different
underlying problems. Problem difficulty, trace length, candidate loss, and
gradient norm could therefore be confounded with gradient-cosine rank.

Stage 6B-3 removed that main between-group confound. For every candidate trace,
it reset the same Stage 6A Candidate LoRA adapter state, applied exactly one
manual update, and measured the resulting change in the same frozen 100-example
reference loss.

For candidate `i` and step size `eta`, the locked quantities were:

```text
predicted change x_i(eta) = -eta * <g_ref, g_i>

observed change y_i(eta) =
    L_ref(theta_0 - eta * g_i) - L_ref(theta_0)
```

This is a local one-step mechanism test. It is not a final student-training or
held-out answer-forced-accuracy test.

## Locked design

The scientific design was preregistered before implementation and GPU
execution in:

```text
experiments/stage6b_3_one_step_validation_v1/PRE_REGISTRATION.md
preregistration commit: 7e2f8221d3e51e28959899b28aa37b3de896868e
preregistration SHA256:  edff3069693faef03e219078f24a1a5694071fbd8b8fb70cb3756256642a0c53
```

The locked formal grid contained:

| Role | Step size `eta` |
| --- | ---: |
| Zero-update control | 0 |
| Secondary | 0.0001 |
| **Primary** | **0.0005** |
| Secondary | 0.001 |
| Secondary | 0.005 |

The run evaluated 100 candidates at all five step sizes, for 500 cells. Each
cell reset all 392 trainable LoRA tensors (17,432,576 FP32 parameters) to the
same `theta_0`. The implementation prohibited optimizer state, Adam/AdamW,
momentum, weight decay, clipping, gradient normalisation, base-parameter
updates, and state reuse between cells.

The preregistered primary endpoint was Spearman correlation between predicted
and observed reference-loss changes at `eta = 0.0005`, with:

* positive-direction one-sided permutation test;
* 100,000 permutations;
* statistical seed 42;
* alpha 0.05;
* 10,000-resample percentile bootstrap 95% confidence interval.

The locked decision rule was:

* positive Spearman and positive-tail `p < 0.05`: support;
* negative-tail `p < 0.05`: reverse/contradicted;
* otherwise: null/inconclusive.

## v1 technical stop and why it was not a scientific result

The first v1 smoke used:

```text
v1 implementation commit: a5f7ac3f17fc59854d57daa65e0ad669ae53b9b7
job:                      8924042.kman.restech.unsw.edu.au
```

It stopped at candidate 0 and `eta = 0` because the original validity check
required each recomputed dot product to satisfy a very tight elementwise
absolute/relative tolerance against the frozen Stage 6B-1 dot product.

No positive-step-size update and no scientific endpoint had been evaluated.
Job 8924042 is therefore an invalid technical smoke, not a null, reverse, or
positive gradient-mechanism result.

## Numerical reproducibility diagnostic

The failure motivated a no-update diagnostic rather than a change to the
scientific endpoint:

```text
diagnostic commit: 38f27182c8f2bced05dff4275a230c64a795af42
job:               8924283.kman.restech.unsw.edu.au
```

Key diagnostic results were:

| Audit | Result |
| --- | ---: |
| Reference-gradient repeat cosine | 0.9999994718 |
| Minimum candidate-gradient repeat cosine | 0.9999322307 |
| Minimum frozen-versus-recomputed Spearman | 0.9999879988 |
| Minimum frozen-versus-recomputed Pearson | 0.9999996233 |
| Sign agreement | 100/100 |
| Top-50 overlap | 50/50 |
| Maximum norm-product-normalised discrepancy | 0.0008507322 |

The diagnostic showed that rank, sign, and relative magnitude were extremely
stable even though BF16/CUDA recomputation was not elementwise identical. It
therefore identified the v1 tolerance as an unsuitable engineering validity
criterion. It did not evaluate a positive-`eta` outcome.

## Transparent v2 protocol amendment

Before any formal positive-`eta` outcome was observed, the technical validity
criterion was amended in:

```text
experiments/stage6b_3_one_step_validation_v2/PROTOCOL_AMENDMENT.md
amendment commit: 7a504251b7b1616017f46d74d530ec830cb5d7f8
amendment SHA256:  0a285cc3383c35d77d8a8e0c8d212fecc6ce1747617ff295ab2344bcc72a4c49
```

The v2 gate required all 100 no-update recomputations to be finite, frozen-
versus-recomputed dot-vector Spearman to be at least 0.999, and the maximum
norm-product-normalised discrepancy to be at most 0.002. The same discrepancy
limit also applied before every positive-`eta` cell update.

The amendment did **not** change candidate or reference identities, `theta_0`,
the five step sizes, the primary step size, the frozen Stage 6B-1 predictor,
the primary statistic, the permutation/bootstrap rules, or the interpretation
boundary. The old elementwise tolerance remained an audit-only diagnostic.

## v2 engineering smoke

The locked v2 implementation was tested by:

```text
job:       8924540.kman.restech.unsw.edu.au
exit:      0
walltime:  00:02:59
```

The full 100-candidate no-update preflight passed, after which only candidate 0
was evaluated at the five step sizes. The smoke explicitly recorded
`scientific_endpoint_evaluated=false`; it proved that reset, update, loss
measurement, and evidence writing worked, but it did not produce the Stage
6B-3 conclusion.

## Formal scheduling history

The original formal PBS request was conservatively written for 12 hours. Job
8927619 was submitted with an 8-hour override and remained queued in
`csegpu12` for approximately 318 hours. A later `qalter` request to reduce it
to two hours was rejected by Katana's `no_qalter_place` hook, so that job was
not silently modified.

After the v2 smoke supplied a runtime benchmark, a replacement was submitted
on hold, its requested A100/CPU/memory/walltime resources were checked, the old
job was deleted, and the replacement was released:

```text
replacement formal job: 9043125.kman.restech.unsw.edu.au
requested GPU:          1 x A100
requested CPUs:         8
requested memory:       46 GB
requested walltime:     02:00:00
```

This queue replacement changed scheduling only. It did not change code,
inputs, model state, numerical precision, or the scientific protocol.

The completed scheduler record reported:

```text
job state:               F
execution node:          k110
start time:              2026-08-19 23:03:27 +1000
obit time:               2026-08-19 23:54:46 +1000
observed walltime:       00:51:12
observed memory:         6,263,368 KB
observed CPU percent:    95
Exit_status:             0
```

## Formal execution validity

The formal run recorded `status=PASS` and produced all 100 candidates by five
step sizes:

| Artifact | Row count |
| --- | ---: |
| `stage6b_3_v2_preflight_candidates.jsonl` | 100 |
| `stage6b_3_results.jsonl` | 500 |
| `stage6b_3_results.csv` | 501 including header |
| `stage6b_3_reference_baseline.jsonl` | 100 |
| `stage6b_3_reference_losses.jsonl` | 50,000 |

The result-root `SHA256SUMS` verified every listed artifact. The formal
validity checks all passed, including:

* adapter, candidate, and reference identity;
* frozen input hashes;
* finite values;
* trainable-LoRA-only updates;
* all-100 no-update vector reproducibility preflight;
* per-cell normalised dot discrepancy;
* manual-update formula;
* reset to `theta_0` for every cell;
* zero-`eta` loss and parameter controls;
* frozen and trainable parameter-integrity hashes.

Key formal audit values were:

| Audit | Result | Locked threshold |
| --- | ---: | ---: |
| Frozen-versus-recomputed dot Spearman | 0.9999759976 | at least 0.999 |
| Frozen-versus-recomputed dot Pearson | 0.9999994433 | descriptive |
| Maximum preflight normalised discrepancy | 0.0008265327 | at most 0.002 |
| Maximum per-cell normalised discrepancy | 0.0010560287 | at most 0.002 |
| Sign agreement | 100/100 | descriptive |
| Top-50 overlap | 50/50 | descriptive |
| Maximum zero-control absolute loss change | 0 | at most 0.000001 |
| Maximum reset mismatch count | 0 | 0 |

This establishes that the formal execution was valid. It does not turn a
null scientific endpoint into a positive result.

## Preregistered primary result

At the primary step size `eta = 0.0005`:

```text
Spearman rho:                     -0.0838763876
positive-tail permutation p:       0.7990220098
negative-tail permutation p:       0.2010079899
bootstrap 95% CI:                 [-0.2755179884, 0.1105261313]
sample size:                        100
locked decision:                    null_or_inconclusive_result
```

The predicted ranking did not reliably match the observed one-step
reference-loss-change ranking. The positive-tail test did not support the
preregistered positive hypothesis, while the negative-tail test did not
establish a statistically reliable reverse relationship. The confidence
interval included zero.

The correct primary conclusion is:

> At the preregistered step size, the frozen gradient-dot predictor did not
> reliably rank observed one-step reference-loss changes. The result is
> null/inconclusive, not positive and not statistically established as
> reverse.

## Secondary results

The step-size Spearman correlations were:

| Step size `eta` | Role | Spearman |
| ---: | --- | ---: |
| 0 | zero-update control | not applicable |
| 0.0001 | secondary | -0.2174617462 |
| **0.0005** | **primary** | **-0.0838763876** |
| 0.001 | secondary | -0.1311491149 |
| 0.005 | secondary | 0.6405400540 |

At the primary step size, additional descriptive analyses were also weak:

```text
Pearson:                        -0.1262089156
cosine-predictor Spearman:       0.0667866787
partial-rank correlation:       -0.0410849861
```

The stronger positive correlation at `eta = 0.005` is an exploratory,
step-size-sensitive observation. It cannot replace the preregistered primary
result because it was a secondary step size, ten times larger than the primary
step, and the cross-step linear-scaling diagnostic was poor. It is suitable as
a hypothesis for a future separately locked experiment, not as a post-hoc
claim that Stage 6B-3 succeeded.

## Interpretation boundary

Stage 6B-3 supports the following claims:

* the frozen Stage 6B-1 gradient-dot vector was reproduced extremely closely;
* the one-step formal run was complete and passed its locked validity gates;
* the primary local ranking hypothesis was not supported;
* prediction quality varied strongly with step size in secondary analyses.

It does not establish that:

* gradient information is always useless;
* cosine is harmful;
* the primary relationship is reliably reversed;
* `eta = 0.005` is a validated replacement primary condition;
* CosRewrite succeeds or fails;
* final multi-epoch student accuracy improves or degrades;
* the result generalises to another model, dataset, seed, GPU, or precision.

The formal result is specific to the locked A100/BF16 runtime and the fixed
Qwen3-1.7B LoRA state. GPU or numerical precision may affect very small loss
changes, but this experiment does not identify the A100 model as the cause of
the null result.

## Evidence and verification

The copied formal evidence is archived under:

```text
experiments/stage6b_3_one_step_validation_v2/evidence/formal/
    stage6b_3b_v2_9043125/
```

The complete formal output was also copied byte-for-byte from Scratch to the
durable HOME archive:

```text
/home/z5463756/honour/results/trace-rewriting/
stage6b-gradient-feedback-qwen3-1.7b-v1/one_step_validation/formal_v2/
stage6b_3b_v2_9043125/
```

This 19 MB copy was an immediate, verified safety checkpoint made when the
formal result became available. It is not a separate scientific stage,
closure commit, or version tag. The later whole-Stage 6B HOME snapshot is the
unified durable archive for Stage 6B-1, Stage 6B-2, and Stage 6B-3.

The archive was created at `2026-08-20T00:40:19Z`. Source and HOME contained
the same 20 files and 19,316,449 file bytes. Independent whole-tree manifests,
the original result-root manifest, and a direct byte comparison of the
17,987,485-byte reference-loss audit file all passed. Scratch reported GPFS
storage and HOME reported NFS storage; their allocated-block counts differed,
but both reported the same 19 MB apparent size and identical content hashes.

Archive metadata, including the complete scheduler record, source inventory,
Git status, and source/copy manifests, is stored beside the HOME copy in:

```text
stage6b_3b_v2_9043125_archive_metadata/
```

The complete Stage 6B Scratch result root was subsequently copied to:

```text
/home/z5463756/honour/results/trace-rewriting/
stage6b-gradient-feedback-qwen3-1.7b-v1/full_scratch_snapshot_20260820/
```

That whole-stage snapshot was published at `2026-08-20T00:57:41Z` only after
the source-before, source-after, staging, and final HOME manifests agreed. It
contains 224 files and 367,616,133 file bytes. Its tree-manifest SHA256 is
`0f4dad6d434bd9daa2fbc50ccadc99a427219a3e09af4800494fcdb20d4fb6dc`.
The snapshot therefore subsumes the Stage 6B-3 result checkpoint while also
preserving the complete Stage 6B-1 and Stage 6B-2 raw result tree.

The original result-root `SHA256SUMS` verifies the formal output. A separate
copy manifest at the Stage 6B-3 archive root verifies the exact files copied
into Git. Large model weights, caches, and environments are intentionally not
included in Git.

The 17 KB file `/home/z5463756/stage6b_3_v2_review_bundle.tar.gz` is
explicitly excluded. It contains only the pre-execution review helper
`create_stage6b_3_v2_for_review.sh` and the already-applied
`stage6b_3_v2_transform.patch`; it is not formal Job 9043125 evidence.
