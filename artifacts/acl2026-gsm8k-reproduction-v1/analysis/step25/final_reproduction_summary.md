# Trace Rewriting Core Experiment Reproduction Summary

## 1. Reproduction objective

This reproduction evaluates the core GSM8K anti-distillation experiment
from *Protecting Language Models Against Unauthorized Distillation
through Trace Rewriting*.

The reproduction target was initially selected from the first arXiv
version of the paper. The final quantitative comparison is made against
the ACL 2026 publication because it is the definitive published version.
The Clean and Optimized/Rewrite results for this core experiment remained
unchanged across these versions.

The reproduced experiment compares:

- a Clean student trained on `original_trace`;
- a Rewrite student trained on `rewrite_trace`;
- an undistilled Base model included as an additional reference.

The student model is `meta-llama/Llama-3.2-3B`.

## 2. Reproduction scope

The reproduction covers the student-distillation and evaluation stages
of the official GSM8K optimized trace-rewriting experiment.

It uses the official pre-generated original and optimized traces.
Consequently, it does not independently reproduce teacher generation,
rewrite generation, or prompt optimization.

## 3. Experimental setup

| Component | Reproduction setting |
|---|---|
| Training dataset | GSM8K |
| Training examples | 5,231 |
| Evaluation dataset | GSM8K-Platinum test |
| Evaluation examples | 1,209 |
| Student model | Llama-3.2-3B |
| Tokenizer | Llama-3.2-3B-Instruct |
| LoRA rank | 128 |
| LoRA alpha | 128 |
| LoRA dropout | 0 |
| Epochs | 4 |
| Learning rate | 5e-4 |
| Per-device batch size | 2 |
| Effective global batch size | 32 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Scheduler | Cosine |
| Warm-up ratio | 0.1 |
| Random seed | 888 |
| GSM8K generation limit | 1,024 tokens |
| Primary metric | Answer-forced zero-shot accuracy |

Clean and Rewrite used the same model, tokenizer, dataset, seed,
evaluation set, and training hyperparameters. The source trace column
was the only core experimental variable:

- Clean: `original_trace`
- Rewrite: `rewrite_trace`

## 4. Main evaluation results

### 4.1 Reproduction results

| Model | Raw correct | Raw accuracy | AF correct | AF accuracy |
|---|---:|---:|---:|---:|
| Base | 18 / 1,209 | 1.4888% | 47 / 1,209 | 3.8875% |
| Clean | 710 / 1,209 | 58.7262% | 717 / 1,209 | 59.3052% |
| Rewrite | 59 / 1,209 | 4.8801% | 107 / 1,209 | 8.8503% |

Base is an additional reproduction reference and is not the primary
Clean-versus-Rewrite comparison reported by the paper.

### 4.2 Comparison with the ACL 2026 publication

| Metric | ACL publication | Reproduction | Difference |
|---|---:|---:|---:|
| Clean AF accuracy | 57.7000% | 59.3052% | +1.6052 pp |
| Rewrite AF accuracy | 8.6000% | 8.8503% | +0.2503 pp |
| Absolute accuracy drop | 49.1000 pp | 50.4549 pp | +1.3549 pp |
| Relative performance reduction | 85.0953% | 85.0767% | -0.0186 pp |

The reproduction produced:

- Clean AF accuracy: 59.3052%;
- Rewrite AF accuracy: 8.8503%;
- absolute reduction: 50.4549 percentage points;
- relative reduction: 85.0767%.

The relative reduction differs from the official value by only
0.0186
percentage points.

## 5. Correctness-overlap analysis

### 5.1 Raw evaluation

| Clean/Rewrite outcome | Examples |
|---|---:|
| Both correct | 43 |
| Clean correct only | 667 |
| Rewrite correct only | 16 |
| Both incorrect | 483 |

### 5.2 Answer-forced evaluation

| Clean/Rewrite outcome | Examples |
|---|---:|
| Both correct | 80 |
| Clean correct only | 637 |
| Rewrite correct only | 27 |
| Both incorrect | 465 |

Answer forcing rescued:

- 29 Base outputs;
- 8 Clean outputs, while breaking 1 previously correct output;
- 48 Rewrite outputs, without breaking a raw-correct output.

Rewrite therefore benefited more from answer forcing than Clean,
indicating that abnormal formatting and answer extraction contributed
to some of its raw errors. However, its AF accuracy remained only
8.8503%.

## 6. Token-cap analysis

The output traces were decoded and re-tokenized with the saved tokenizer.
These reconstructed lengths are useful for comparison but are not exact
records of the original generation token counts.

| Model/output | Mean length | Median | At least 1,000 tokens |
|---|---:|---:|---:|
| Base Raw | 825.68 | 1,024 | 918 / 1,209 |
| Clean Raw | 952.63 | 1,018 | 1,001 / 1,209 |
| Rewrite Raw | 928.85 | 932 | 1 / 1,209 |
| Rewrite AF | 969.90 | 973 | 140 / 1,209 |

Base and Clean frequently generated near the configured token limit.
In contrast, only one Rewrite Raw output reached at least 1,000
re-tokenized tokens.

The Rewrite accuracy reduction is therefore not primarily explained by
the 1,024-token generation cap. Its outputs generally became unusable
without first reaching the token limit.

Answer-forced outputs include an additional continuation and therefore
must not be interpreted as generation-cap violations when their
re-tokenized lengths exceed 1,024.

## 7. Rewrite-output degeneration analysis

The following markers were measured across all 1,209 Rewrite Raw
outputs:

| Marker | Count | Percentage |
|---|---:|---:|
| `assistant` marker | 1209 | 100.00% |
| Multiple boxed answers | 1209 | 100.00% |
| At least 10 boxed answers | 1209 | 100.00% |
| Unicode replacement character | 1208 | 99.92% |
| Mixed-script fragment | 1208 | 99.92% |
| High line-level repetition | 1174 | 97.11% |
| At least one measured marker | 1209 | 100.00% |

Maximum values observed:

- `assistant` occurrences: 67;
- boxed-answer occurrences: 103;
- line-level repetition score: 0.7279.

These measurements show that the behavior was widespread rather than
limited to a small number of extreme examples.

The symptoms are consistent with highly degenerate or mode-collapse-like
generation behavior. However, output analysis alone does not establish
the internal causal mechanism responsible for that behavior.

## 8. Representative examples

Five representative cases were retained:

| Index | Category | Main observation |
|---:|---|---|
| 32 | Clean correct, Rewrite Raw incorrect, Rewrite AF correct | Rewrite repeatedly produced the correct answer but abnormal formatting interfered with raw extraction |
| 498 | Rewrite Raw correct | Correct answer was repeatedly emitted without a coherent reasoning process |
| 366 | Rewrite rescued by AF | Correct numeric content appeared repeatedly but required answer forcing for successful evaluation |
| 524 | Clean and Rewrite incorrect | Rewrite repeatedly generated an incorrect fixed answer with severe corruption |
| 715 | Least line-repetitive Rewrite case | Generated a sequence of inconsistent candidate answers, showing that line-level repetition does not capture every degeneration type |

The complete examples are stored in:

- `analysis/step23/final_representative_cases.jsonl`
- `analysis/step23/final_representative_cases.txt`

## 9. Katana compatibility changes

Three source files differed from the official repository commit.

### 9.1 Dataset preprocessing parallelism

The fixed `num_proc=96` setting was replaced by an environment-controlled
setting with a default of four processes.

This changes preprocessing resource usage and speed, but not the
preprocessing transformation, data content, training objective, or
evaluation rule.

### 9.2 Tokenizer override

The student weights remained `Llama-3.2-3B`, while
`Llama-3.2-3B-Instruct` was used as the tokenizer to provide a compatible
chat template.

This may contribute to small differences in absolute accuracy.
It was applied identically to Clean and Rewrite, so the controlled
comparison between the two training conditions was preserved.

### 9.3 Hardware

Each formal experiment used one GPU. The effective global batch size
remained 32 through gradient accumulation, preserving the official
training configuration.

## 10. Reproduced and unreproduced components

### Reproduced

- full Clean student training;
- full optimized Rewrite student training;
- full GSM8K-Platinum evaluation;
- Raw and answer-forced accuracy;
- correctness-overlap analysis;
- token-cap analysis;
- Rewrite-output degeneration analysis;
- representative-case analysis;
- comparison with the ACL 2026 core result.

### Not independently reproduced

- original teacher-trace generation;
- optimized rewrite generation with gpt-oss-120b;
- prompt optimization;
- Semantic rewriting;
- gradient-based rewriting;
- ADS and DOGe;
- MATH;
- MMLU and MMLU-Pro;
- other student architectures;
- answer-only distillation;
- adaptive attacks such as paraphrasing and KPOD;
- teacher/rewriter size ablations;
- official rewritten-teacher-trace quality evaluation;
- API watermarking.

## 11. Final conclusion

The core ACL 2026 GSM8K optimized trace-rewriting anti-distillation
result was successfully reproduced.

The Clean student achieved
59.3052% answer-forced accuracy,
compared with the official 57.7%. The Rewrite student
achieved 8.8503%, compared with the
official 8.6%.

The reproduced relative student-performance reduction was
85.0767%, nearly identical to
the official 85.0953%.

The result is therefore both directionally and quantitatively consistent
with the published core experiment. The principal limitation is that
official pre-generated traces were used, meaning the reproduction
validates the student-distillation and evaluation stages rather than the
entire teacher-generation and rewriting pipeline.

## 12. Primary artifact locations

- Base results: `gsm8k/base/`
- Clean results: `gsm8k/clean/`
- Rewrite results: `gsm8k/rewrite/`
- Step 23 analysis: `analysis/step23/`
- Step 24 comparison: `analysis/step24/`
- Final manifest: `analysis/step25/final_artifact_manifest.txt`
