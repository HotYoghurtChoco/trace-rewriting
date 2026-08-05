# Stage 6B-1 and Stage 6B-2 Gradient-Feedback Pilot

## Snapshot status

This directory is a retrospective archival snapshot of the completed
Stage 6B-1 gradient-scoring experiment and Stage 6B-2 GradHigh50 versus
GradLow50 pilot.

The runs were executed from source commit:

```text
6e22ffdaf0e270f4f8c1ab1ca6268d274a99165a
```

The later Git commit containing this directory is an archival commit.
It must not be interpreted as the commit from which the experiments
were originally executed.

This snapshot contains compact scripts, provenance, manifests,
checksums, metrics, and per-example evaluation evidence. Full model
adapters, checkpoints, caches, environments, and large run artifacts
remain outside Git.

## Experiment scope

### Stage 6B-1

Stage 6B-1 measured candidate-reference gradient alignment using:

* the final Stage 6A candidate LoRA adapter;
* trainable LoRA parameters only;
* evaluation mode with dropout disabled;
* completion-only loss;
* no optimizer step;
* 100 candidate rewrite traces;
* 100 disjoint reference examples;
* the arithmetic mean of per-example reference gradients.

The primary candidate score was cosine similarity between each
candidate gradient and the mean reference gradient.

The 100 candidates were ranked in descending cosine order:

* GradHigh50: ranks 1–50;
* GradLow50: ranks 51–100.

### Stage 6B-2

GradHigh50 and GradLow50 were trained in isolated processes from the
same Qwen3-1.7B base model. They used the same controlled
hyperparameters, seed, evaluation set, and evaluation protocol.

Each rewrite-trained student was compared with a clean-trace student
trained on the corresponding 50 underlying problems.

The formal rewrite score stored in `instructions_and_scores.yaml` is:

```text
Clean answer-forced accuracy - Rewrite answer-forced accuracy
```

It is not `1 - Rewrite answer-forced accuracy`.

## Jobs

| Purpose                          |  Job ID | Status                           |
| -------------------------------- | ------: | -------------------------------- |
| First Stage 6B-1 smoke attempt   | 8916264 | Invalid; Python was not executed |
| Valid Stage 6B-1 gradient smoke  | 8918490 | PASS                             |
| Stage 6B-1 full gradient scoring | 8918577 | PASS                             |
| Stage 6B-2 GradHigh50 pilot      | 8920030 | COMPLETE                         |
| Stage 6B-2 GradLow50 pilot       | 8920031 | COMPLETE                         |

The invalid Job 8916264 produced no scientific result. Its heredoc was
attached to `tee` instead of `python`, causing the Python source to be
printed rather than executed. It is retained only as failure evidence.

## Stage 6B-1 results

* Candidate count: 100
* Reference count: 100
* Candidate traces truncated at 1,024 total tokens: 0
* Reference traces truncated at 1,024 total tokens: 0
* Overall gradient-cosine mean: 0.0810139413
* Overall gradient-cosine median: 0.0823431383
* GradHigh50 cosine mean: 0.1126098799
* GradLow50 cosine mean: 0.0494180028
* Positive cosine count: 96
* Negative cosine count: 4

The valid smoke test showed highly reproducible candidate gradients
across two passes, with repeat-gradient cosine 0.9999559343.

## Stage 6B-2 results

| Group      | Mean gradient cosine | Clean Raw | Rewrite Raw | Raw drop | Clean AF | Rewrite AF | AF drop | Clean/Rewrite token caps |
| ---------- | -------------------: | --------: | ----------: | -------: | -------: | ---------: | ------: | -----------------------: |
| GradHigh50 |         0.1126098799 |      0.85 |        0.72 |     0.13 |     0.88 |       0.77 |    0.11 |                    0 / 3 |
| GradLow50  |         0.0494180028 |      0.82 |        0.77 |     0.05 |     0.87 |       0.80 |    0.07 |                    1 / 2 |

The pre-locked primary decision metric was:

```text
GradHigh50 rewrite AF - GradLow50 rewrite AF
= 0.77 - 0.80
= -0.03
```

The locked rule required:

* directional pass: greater than 0;
* meaningful signal: at least 0.05;
* null or reverse: less than or equal to 0.

Therefore, Stage 6B-2 did not pass its primary directional hypothesis.
Under the locked rule, the observed result is null/reverse.

GradHigh50 showed a larger comparator-normalised AF drop than
GradLow50:

```text
0.11 - 0.07 = 0.04
```

It also showed a larger raw-accuracy drop:

```text
0.13 - 0.05 = 0.08
```

These are secondary exploratory observations. They do not override the
failed primary decision rule and must not be reported as validation of
the gradient-feedback method.

## Limitations

Stage 6B-2 is a pilot using:

* one proxy model;
* one training seed;
* 50 training examples per group;
* 100 evaluation examples;
* non-identical problem groups;
* a small number of token-capped generations.

No robustness, statistical-generalisation, or causal claim is
permitted from this pilot alone. Stage 6B-1 and Stage 6B-2 do not
constitute Stage 6B-3 or a completed validation of the proposed method.

## Directory structure

```text
pbs/
    Original Stage 6B launch scripts

evidence/provenance/
    Locked methods and preflight records

evidence/smoke/
    Invalid-attempt evidence and valid smoke evidence

evidence/scoring/
    Full Stage 6B-1 scoring summaries and candidate scores

evidence/frozen_data/
    Frozen-reference manifests and checksums

evidence/pilot/
    Stage 6B-2 manifests, configurations, metrics, and evaluations

COPIED_FILES_SHA256SUMS
    SHA-256 manifest covering all files under pbs/ and evidence/
```

From this directory, verify the copied experiment files with:

```bash
sha256sum -c COPIED_FILES_SHA256SUMS
```
