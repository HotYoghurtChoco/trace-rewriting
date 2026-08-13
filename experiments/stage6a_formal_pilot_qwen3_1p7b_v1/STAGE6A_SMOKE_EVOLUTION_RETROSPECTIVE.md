# Stage 6A Smoke Evolution Retrospective

**Record date:** 2026-08-13 (Australia/Sydney)
**Record type:** Retrospective historical evidence; not a preregistration
**Project:** UNSW COMP4952 / Honours — Trace Rewriting / CosRewrite
**Scope:** Stage 6A v1, v2, v3, and the controlled formal pilot

## Why this separate retrospective exists

The existing 165-line Stage 6A Formal [`README.md`](README.md) is a compact evidence record for the completed controlled formal pilot. A read-only audit on 2026-08-13 found that it does not contain the v1/v2/v3 smoke history, the scientific-validity boundaries of those runs, or the BF16 merge/save diagnosis that motivated the Formal dynamic-LoRA design.

The complete v1/v2/v3 artifacts are also planned to move out of Katana Scratch. A single, tracked retrospective is therefore needed to preserve the reasoning chain before those full artifacts cease to be immediately available on Katana. This file supplements the existing Formal README; it does not replace it, change the historical execution evidence, or claim to have been written before the experiments.

This documentation is committed on `research/stage6c-cosrewrite` because:

- that branch descends from and contains the Stage 6A Formal and Stage 6B evidence history;
- the archive decision is being made to release space for the active Stage 6C GPT-OSS/CosRewrite work;
- the separate `research/gradient-feedback` worktree remains the frozen Stage 6B execution line while its formal job boundary is unresolved;
- adding a later documentation commit does not move the immutable Stage 6A Formal execution tag or alter any earlier execution commit.

Its location beside the Stage 6A Formal evidence reflects the subject of the record. Its branch records when and why the retrospective and cleanup boundary were added.

## Why the old Scratch copies are planned for removal

At the 2026-08-13 checkpoint, the user Scratch allocation was approximately:

```text
128G total
112G used
17G available
```

The three old Stage 6A smoke/diagnostic roots occupied approximately `37.5G`, while the active Stage 6A Formal, Stage 6B, and Stage 6C result roots together occupied less than `1G`. Stage 6C additionally requires space for an isolated vLLM container and later controlled experiments.

The planned removal is therefore storage lifecycle management, not deletion of the scientific record:

1. Git retains the code, immutable execution identities, compact evidence, and this historical interpretation;
2. the complete v1/v2/v3 artifacts are first transferred to a durable Mac archive;
3. archive SHA256 and structure checks must pass;
4. only the redundant Katana Scratch copies then become eligible for exact removal;
5. the Stage 6A Formal, all Stage 6B/6C dependencies, environments, and model caches remain on Katana.

No Scratch copy had been deleted when this retrospective was written.

## Purpose and evidence boundary

Stage 6A established a low-cost, repeatable Qwen3-1.7B downstream-student baseline for the later gradient and CosRewrite experiments. It did not implement or complete the paper authors' unpublished gradient method.

The development sequence was:

```text
Stage 6A v1
→ identify masking, tokenizer, resume, and provenance risks

Stage 6A v2
→ verify that the repaired pipeline executes end to end
→ identify degenerate generation and stopping behaviour

Stage 6A v3
→ diagnose the BF16 dense merge/save failure mode
→ select BF16 base + dynamic LoRA for formal evaluation

Stage 6A Formal
→ independently rerun the controlled 100/100/100 protocol
```

Only the controlled formal pilot is a Stage 6A scientific result. The v1 and v2 metrics are scientifically invalid, and v3 is diagnostic evidence rather than the formal baseline.

## Version summary

| Version | Scale | Main purpose | Long-term status |
|---|---:|---|---|
| v1 | 8 train / 4 held-out | First small Clean-versus-Candidate Qwen pipeline | Scientifically invalid; retain only engineering lessons |
| v2 | Small smoke / 4 held-out | Rerun after the main v1 engineering repairs | Engineering PASS; scientifically invalid |
| v3 | 32 train / 16 held-out | Diagnose unexpectedly weak merged-model evaluation | Diagnostic PASS; identified BF16 merge/save loss |
| Formal | 100 Clean train / 100 Candidate train / 100 held-out | Controlled downstream-student baseline | Completed formal pilot |

## Stage 6A v1

Scratch root:

```text
/srv/scratch/z5463756/honour/results/stage6a-smoke-qwen3-1.7b
```

Known execution attempts included:

```text
8793450  vLLM/ZMQ Unix socket path exceeded 107 characters
8809040  resumed from a partial Clean artifact and completed
```

The first smoke run exposed several interacting engineering risks:

- completion-only label masking;
- PAD/EOS handling;
- tokenizer vocabulary and chat-tail EOS handling;
- partial resume, cached state, and provenance contamination;
- PBS, socket, and `PYTHONPATH` execution issues.

Historical surface metrics were approximately:

```text
Clean Raw/AF      ≈ 75% / 100%
Candidate Raw/AF  ≈ 0% / 0%
```

These values must not be cited as scientific results. The run was too small and was affected by unresolved pipeline-validity risks.

Long-term conclusion:

```text
scientifically invalid
engineering failure knowledge retained
```

## Stage 6A v2

Scratch root:

```text
/srv/scratch/z5463756/honour/results/stage6a-smoke-qwen3-1.7b-v2
```

Known PBS job:

```text
8897599
```

The earlier A100-only Job `8811644` never ran because of its resource constraint. Moving to an ordinary eligible GPU for Job `8897599` was a scheduler adjustment, not a scientific protocol change. Job `8897599` exited successfully after approximately `00:03:21`.

After the principal v1 repairs, v2 completed the pipeline end to end. However, evaluation still used only four held-out problems, and the generated outputs exhibited long loops, abnormal stopping, and clearly degenerate Candidate behaviour, including Arabic-language fragments.

Historical surface metrics were approximately:

```text
Clean Raw/AF      ≈ 25% / 50%
Candidate Raw/AF  ≈ 0% / 0%
```

These values also must not be cited as scientific results.

Long-term conclusion:

```text
engineering PASS
scientifically invalid
```

## Stage 6A v3

Scratch root:

```text
/srv/scratch/z5463756/honour/results/stage6a-smoke-qwen3-1.7b-v3
```

Known PBS job and configuration:

```text
job:               8907681
training rows:     32 per condition
held-out rows:     16
split seed:        137
LoRA rank/alpha:   16 / 16
LoRA dropout:      0.05
epochs:            2
learning rate:     1e-4
```

The same trained LoRA adapters produced sharply different results depending on how they were loaded for inference:

| Model form | Clean Raw/AF | Candidate Raw/AF |
|---|---:|---:|
| BF16 dense merged model | 25.00% / 31.25% | 18.75% / 25.00% |
| BF16 base with the same adapter loaded dynamically | 81.25% / 87.50% | 68.75% / 75.00% |

The diagnostic chain compared adapter changes, base and merged tensors, tokenizer semantics, BF16 versus FP32 merging, dynamic versus merged inference, and attention backends.

The supported diagnosis was:

> The LoRA deltas were small enough that merging them into the BF16 base and saving a dense model rounded away many updates below BF16 resolution. FP32 merging recovered most or all of the lost behaviour, while dynamic loading of the adapter avoided the destructive BF16 merge/save step.

Additional runtime findings bounded the implemented solution:

- tokenizer byte-level hash differences did not imply semantic tokenizer differences;
- vLLM 0.11.0 did not support FP32 dynamic LoRA;
- FP32 FlexAttention produced a CUDA illegal-memory-access failure during diagnosis;
- forcing `TRITON_ATTN` allowed the FP32 merged diagnostic to complete.

This diagnosis determined the formal evaluation design:

```text
BF16 base model
+ dynamically loaded LoRA adapter
```

v3 is therefore valuable diagnostic evidence, but its metrics are not the Stage 6A formal baseline.

## Stage 6A controlled formal pilot

The tracked Formal evidence and full protocol description remain in [`README.md`](README.md). This retrospective records why that formal design followed from the earlier smoke and diagnostic runs.

Formal Scratch root:

```text
/srv/scratch/z5463756/honour/results/stage6a-formal-pilot-qwen3-1.7b-v1
```

Formal identity:

```text
PBS job:          8908886.kman.restech.unsw.edu.au
execution commit: 6e22ffdaf0e270f4f8c1ab1ca6268d274a99165a
execution tag:    stage6a-formal-pilot-qwen3-1.7b-v1
evidence commit:  6a9bb3b6403fe72b05193b4e06906167af061750
```

The Formal run did not continue from the v3 smoke students. It independently trained and evaluated new students using the runtime design established by v3:

```text
100 Clean training traces
100 Candidate training traces
100 held-out GSM8K problems
LoRA rank/alpha/dropout = 16/16/0.05
2 epochs
learning rate = 1e-4
BF16 base + dynamic LoRA inference
```

Formal results:

| Condition | Raw accuracy | Answer-forced accuracy |
|---|---:|---:|
| Clean traces | 82% | 83% |
| Existing optimized rewrite traces | 59% | 62% |
| Clean minus Candidate | 23 pp | 21 pp |

The pre-existing optimized Candidate traces were not generated by a Stage 6A gradient procedure. Under this controlled pilot, they reduced answer-forced proxy accuracy by 21 percentage points relative to Clean traces.

Raw generation used a 1024-token limit and answer-forced continuation used 32 tokens. The Raw token-cap counts were 1/100 for Clean and 13/100 for Candidate. The existing Formal evidence records the associated sensitivity analysis; these cap diagnostics do not turn the v1/v2 surface metrics into valid scientific results.

## Key Git development chain

| Commit | Historical role |
|---|---|
| `0647420fb203b9fe52c41638ffbcd11c76963890` | Paper-aligned accuracy baseline; objective defined as Clean AF minus Candidate AF |
| `20f0c825…` | Completion-only, PAD/EOS, vocabulary, chat-tail, cache, and provenance repairs; the surviving evidence does not contain the complete hash, so it must not be reconstructed or invented |
| `6222bb719cf458b172d96b9b1455d587069b2f06` | Stable v3 training and diagnostic foundation |
| `6e22ffdaf0e270f4f8c1ab1ca6268d274a99165a` | Dynamic-LoRA controlled formal-pilot execution and tag target |

## Dependency boundary for Stage 6B and Stage 6C

The current gradient/CosRewrite chain depends on:

- the Stage 6A Formal final Candidate LoRA adapter;
- Stage 6B frozen candidate/reference identities and reference examples;
- the frozen Stage 6B scorer definition;
- the Stage 6C.1 verified reusable scorer.

It does not depend at runtime on:

- v1 dense smoke artifacts;
- v2 dense smoke artifacts;
- v3 smoke models or its full diagnostic directory.

Removing the three old smoke roots from Katana Scratch therefore does not break the current scorer chain, provided the Formal, Stage 6B, Stage 6C, environment, and model/cache roots remain unchanged.

## Artifact retention and archive status

Sizes measured on 2026-08-13:

| Scratch root | Approximate size | Planned treatment |
|---|---:|---|
| `stage6a-smoke-qwen3-1.7b` | 9.9G | Full off-Katana archive, then eligible for exact Scratch removal |
| `stage6a-smoke-qwen3-1.7b-v2` | 7.6G | Full off-Katana archive, then eligible for exact Scratch removal |
| `stage6a-smoke-qwen3-1.7b-v3` | 20G | Full off-Katana archive, then eligible for exact Scratch removal |

Status at the time of this retrospective:

```text
archive planned/approved
Mac transfer not yet completed
archive SHA256 not yet verified
archive structure not yet verified
Scratch copies not yet deleted
```

The intended Mac archive root is:

```text
/Users/hongyanchen/Documents/Programming/School/UNSW/2026/Honour/results/
```

The final archive subdirectory, archive filename, SHA256, file count, and recovery command must be recorded only after the archive has actually been created and verified.

Deletion is permitted only after all of the following gates pass:

1. the complete v1/v2/v3 artifacts exist in the Mac archive;
2. SHA256 verification passes;
3. archive listing and structure verification pass;
4. all three expected top-level roots and the v3 diagnostics are present;
5. the recovery location is recorded;
6. the user explicitly confirms deletion.

## Assets that must remain on Katana

This archive/cleanup task must not remove or modify:

```text
/srv/scratch/z5463756/honour/results/stage6a-formal-pilot-qwen3-1.7b-v1
/srv/scratch/z5463756/honour/results/stage6b-gradient-feedback-qwen3-1.7b-v1
/srv/scratch/z5463756/honour/results/stage6c-cosrewrite-qwen3-1.7b-v1
/srv/scratch/z5463756/honour/envs/trace-rewriting-official
the Qwen3-1.7B base/tokenizer cache
the fixed GPT-OSS-120B snapshot
```

Large model artifacts, environments, diagnostics, and raw results are not GitHub artifacts. Git retains code, launchers, compact evidence, immutable execution identities, and this retrospective boundary record; the complete old artifacts are retained off Katana in the verified Mac archive.

## Git integrity

This document was written after completion of the Stage 6A experiments. It must not be interpreted as a contemporaneous preregistration.

The Stage 6A Formal execution tag remains attached to the historical execution commit. Adding or later updating this retrospective is a documentation/evidence action and must not rewrite the execution commit or move the execution tag.
