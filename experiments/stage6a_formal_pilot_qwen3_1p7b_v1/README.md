# Stage 6A Controlled Formal Pilot

## Status

This directory contains the compact Git evidence snapshot for the completed Stage 6A controlled formal pilot.

| Field                 | Value                                               |
| --------------------- | --------------------------------------------------- |
| Run                   | `stage6a-formal-pilot-qwen3-1.7b-v1`                |
| Status                | Completed                                           |
| Completion time       | `2026-08-02T00:17:48+10:00`                         |
| PBS job               | `8908886.kman.restech.unsw.edu.au`                  |
| Git commit            | `6e22ffdaf0e270f4f8c1ab1ca6268d274a99165a`          |
| Git tag               | `stage6a-formal-pilot-qwen3-1.7b-v1`                |
| Protocol              | `paper_a2_accuracy_v3_dynamic_lora_formal_pilot_v1` |
| Run kind              | `controlled_formal_pilot_v1`                        |
| Proxy model           | Qwen3-1.7B                                          |
| Evaluation mode       | vLLM dynamic LoRA                                   |
| Stored model artifact | LoRA adapter only                                   |

## Objective

The pilot tested whether the existing optimized GSM8K rewrite traces reduced the performance gained by a proxy student model relative to clean semantic reasoning traces.

This was a controlled one-candidate formal pilot. It was not a complete OPRO search and did not generate new candidate rewrite traces during this run. The candidate condition used the optimized GSM8K traces that already existed before the pilot.

## Data and protocol

Both conditions used 100 training traces selected from the first 100 rows of their respective frozen data sources:

* Clean condition: semantic/original reasoning traces.
* Candidate condition: existing optimized rewrite traces.

Evaluation used 100 held-out GSM8K training examples. These examples were selected by shuffling with seed 137 and taking the first 100 examples.

The run used seed 888 and the following proxy-training configuration:

| Parameter             |    Value |
| --------------------- | -------: |
| LoRA rank             |       16 |
| LoRA alpha            |       16 |
| LoRA dropout          |     0.05 |
| Per-device batch size |        4 |
| Effective batch size  |       16 |
| Epochs                |        2 |
| Learning rate         |   0.0001 |
| Weight decay          |      0.1 |
| Warmup ratio          |      0.1 |
| Training dtype        | bfloat16 |
| Inference dtype       | bfloat16 |

## Results

| Condition                | Raw accuracy | Answer-forced accuracy | Raw token-cap count |
| ------------------------ | -----------: | ---------------------: | ------------------: |
| Clean traces             |         0.82 |                   0.83 |               1/100 |
| Optimized rewrite traces |         0.59 |                   0.62 |              13/100 |
| Clean minus candidate    |         0.23 |                   0.21 |                   — |

The stored Stage 6A objective score was calculated as:

```text
clean answer-forced accuracy - candidate answer-forced accuracy
= 0.83 - 0.62
= 0.21
```

Therefore, under this 100-example pilot protocol, the optimized rewrite condition reduced answer-forced proxy accuracy by 21 percentage points and raw proxy accuracy by 23 percentage points relative to the clean condition.

## Interpretation

The result supports the hypothesis that the existing optimized rewrite traces were less useful for proxy-model distillation than the clean traces under this pilot protocol.

However, this result must be interpreted cautiously. The candidate condition reached the 1,024-token generation cap on 13 examples, compared with only one example in the clean condition. Some of the observed accuracy difference may therefore be associated with truncated generations rather than the rewriting method alone.

Token-cap analysis and example-level error analysis are required before attributing the complete accuracy drop to the optimized rewrite traces.

## Limitations

The principal limitations of this pilot are:

* Only one proxy model was evaluated.
* Only one training seed was used.
* Only one candidate condition was tested.
* Each condition used only 100 training traces.
* Evaluation used only 100 held-out examples.
* The candidate traces already existed before this run.
* The run did not execute a complete instruction-optimization or OPRO loop.
* GPT-OSS-120B did not generate new candidates during this run.
* The candidate condition had substantially more token-cap events than the clean condition.
* This was a controlled formal pilot, not a paper-scale or final thesis experiment.

The pilot establishes that the Stage 6A clean and candidate pipelines can be executed and compared under a controlled protocol. It does not yet establish that the entire observed accuracy reduction was caused by trace rewriting.

## Evidence contents

The compact evidence snapshot is designed to preserve the files required to verify the configuration, inputs, execution provenance, evaluation results, and archive identity of the Stage 6A run.

It includes:

* the run completion record;
* the frozen configuration;
* the input data manifest;
* the candidate instruction and recorded scores;
* clean and candidate evaluation metrics;
* per-example clean and candidate evaluation records;
* model and evaluation metadata;
* the PBS launcher;
* runtime-attempt provenance;
* the Python package inventory;
* Git commit and working-tree provenance;
* hashes of the original input and source files; and
* the checksum manifest for the canonical full archive.

## Excluded artifacts

The compact evidence snapshot deliberately excludes large or regenerable artifacts, including:

* LoRA adapter weights;
* tokenizer files;
* Arrow datasets;
* Hugging Face cache files;
* binary training-argument files; and
* other regenerable model artifacts.

These exclusions keep the Git evidence snapshot small while preserving the information needed to understand and audit the experiment.

## Canonical archive

The authoritative full Stage 6A archive remains on Katana at:

```text
/home/z5463756/honour/results/trace-rewriting/stage6a-formal-pilot-qwen3-1.7b-v1
```

The canonical archive contains 59 files listed in `SHA256SUMS`, all of which passed checksum verification.

The SHA-256 hash of the canonical archive’s `SHA256SUMS` file is:

```text
0b876a4d23d9779be6851537a05d178208a8bfa672e33eeb62bc412e03f49166
```

The Stage 6A annotated Git tag points to the following commit:

```text
stage6a-formal-pilot-qwen3-1.7b-v1
6e22ffdaf0e270f4f8c1ab1ca6268d274a99165a
```

Two additional Arrow cache files were later observed in the scratch working copy. Their creation times were after the formal pilot had completed, and they were not included in the canonical archive manifest. They are therefore treated as later-derived cache files and are intentionally excluded from both the canonical HOME archive and this compact Git evidence snapshot.

## Reproducibility status

The Stage 6A run is preserved through:

1. the tagged source-code commit;
2. the frozen run configuration;
3. the recorded input hashes and manifest;
4. the execution and environment provenance;
5. the clean and candidate evaluation outputs;
6. the canonical archive checksum manifest; and
7. the full verified archive retained on Katana.

This evidence is sufficient to identify the exact Stage 6A pilot, inspect its recorded results, audit its inputs and configuration, and distinguish the compact Git snapshot from the complete archived run.
