# Thesis Integration: Trace Rewriting Reproduction

> Citation placeholder: [CITE: Protecting Language Models Against
> Unauthorized Distillation through Trace Rewriting]

## Experimental Methodology

The core GSM8K experiment from the trace-rewriting study was reproduced
to evaluate whether rewritten reasoning traces reduce the usefulness of
teacher outputs for downstream student-model distillation.

Two student models were trained using the same architecture and
optimisation configuration. The Clean student was trained using the
`original_trace` field, while the Rewrite student was trained using the
`rewrite_trace` field. Therefore, the source reasoning trace was the
principal controlled experimental variable.

Both conditions used `meta-llama/Llama-3.2-3B` as the student model and
the official pre-generated GSM8K trace dataset containing 5,231 training
examples. Because the official pre-generated traces were used, this
experiment reproduced the student-distillation and evaluation stages,
rather than independently regenerating the teacher and rewritten traces.

## Experimental Setup

The student models were trained using LoRA with rank 128, alpha 128, and
dropout 0. Training was conducted for four epochs with a learning rate
of 5e-4, an effective global batch size of 32, weight decay of 0.1,
gradient clipping at 1.0, a cosine learning-rate scheduler, and a warm-up
ratio of 0.1. Both formal training runs used random seed 888.

Evaluation was performed on all 1,209 examples in the GSM8K-Platinum
test set. The maximum generation length was 1,024 new tokens.

Two evaluation measurements were retained:

1. Raw accuracy, calculated directly from the original generated output.
2. Answer-forced accuracy, calculated after prompting the model to
   provide a final answer in a more extractable form.

Answer-forced zero-shot accuracy was treated as the primary comparison
metric because it corresponds to the metric reported in the published
core experiment.

An undistilled Base model was also evaluated as an additional reference.
It was not part of the primary Clean-versus-Rewrite comparison.

## Implementation Environment

The experiment was conducted on the UNSW Katana high-performance
computing system. Each formal training experiment used one GPU, while
gradient accumulation preserved the effective global batch size of 32.

Two compatibility changes were made to the official repository.

First, the fixed Hugging Face Dataset preprocessing value of
`num_proc=96` was replaced with an environment-controlled setting that
defaulted to four processes. This modification changed preprocessing
resource usage and runtime, but not the preprocessing transformation,
training data, optimisation objective, or evaluation rule.

Second, the `meta-llama/Llama-3.2-3B-Instruct` tokenizer was used with
the `meta-llama/Llama-3.2-3B` model weights to provide the chat template
required by the pipeline. This modification may contribute to small
differences in absolute accuracy. However, the same tokenizer was used
for both Clean and Rewrite, preserving the controlled comparison.

## Main Results

| Condition | Raw correct | Raw accuracy | AF correct | AF accuracy |
|---|---:|---:|---:|---:|
| Base | 18 / 1,209 | 1.4888% | 47 / 1,209 | 3.8875% |
| Clean | 710 / 1,209 | 58.7262% | 717 / 1,209 | 59.3052% |
| Rewrite | 59 / 1,209 | 4.8801% | 107 / 1,209 | 8.8503% |

The Clean student achieved 59.3052%
answer-forced accuracy, while the Rewrite student achieved
8.8503%.

This represents an absolute reduction of
50.4549 percentage points
and a relative performance reduction of
85.0767%.

These results show that the optimised rewritten traces retained much
less value for downstream student training than the original teacher
reasoning traces.

## Comparison with the Published Result

| Metric | ACL 2026 | Reproduction | Difference |
|---|---:|---:|---:|
| Clean AF accuracy | 57.7000% | 59.3052% | +1.6052 pp |
| Rewrite AF accuracy | 8.6000% | 8.8503% | +0.2503 pp |
| Absolute accuracy drop | 49.1000 pp | 50.4549 pp | +1.3549 pp |
| Relative reduction | 85.0953% | 85.0767% | -0.0186 pp |

The reproduced Clean accuracy was
+1.6052 percentage
points relative to the published value, while the reproduced Rewrite
accuracy differed by only
+0.2503 percentage
points.

Most importantly, the reproduced relative performance reduction differed
from the published value by only
0.0186 percentage
points. The reproduced core result is therefore quantitatively
consistent with the ACL publication.

## Correctness-Overlap Analysis

Under raw evaluation, both Clean and Rewrite were correct on 43 test
examples. Clean alone was correct on 667 examples, Rewrite alone was
correct on 16 examples, and both were incorrect on 483 examples.

Under answer-forced evaluation, both models were correct on 80 examples.
Clean alone was correct on 637 examples, Rewrite alone was correct on 27
examples, and both were incorrect on 465 examples.

Answer forcing rescued 48 Rewrite outputs, compared with eight Clean
outputs. One previously raw-correct Clean output became incorrect after
answer forcing.

This indicates that malformed answer structure and extraction failure
contributed to some Rewrite errors. However, Rewrite answer-forced
accuracy remained only 8.8503%.
Therefore, answer-format failure alone does not explain the large
performance gap.

## Token-Cap Analysis

Decoded outputs were re-tokenised using the saved tokenizer to estimate
response length. These reconstructed lengths are useful for comparison,
although they are not exact records of the original generation token
counts.

A total of 918 Base Raw outputs and 1,001 Clean Raw outputs reached at
least 1,000 reconstructed tokens. In contrast, only one Rewrite Raw
output reached at least 1,000 tokens.

The Rewrite Raw outputs had a mean reconstructed length of approximately
928.85 tokens and a median of 932 tokens.

These findings indicate that the Rewrite accuracy reduction was not
primarily caused by the 1,024-token generation limit. Rewrite outputs
usually became unusable before reaching the maximum generation length.

## Rewrite-Output Degeneration

The Rewrite outputs contained widespread abnormal generation patterns.

| Marker | Count | Percentage |
|---|---:|---:|
| `assistant` marker | 1209 | 100.00% |
| Multiple boxed answers | 1209 | 100.00% |
| At least ten boxed answers | 1209 | 100.00% |
| Unicode replacement character | 1208 | 99.92% |
| Mixed-script fragments | 1208 | 99.92% |
| High line-level repetition | 1174 | 97.11% |
| At least one measured marker | 1209 | 100.00% |

All 1,209 Rewrite outputs contained repeated boxed answers and
`assistant` markers. Unicode replacement characters and mixed-script
fragments appeared in 1,208 outputs, while 1,174 outputs met the
high-repetition threshold.

Representative cases showed several different failure modes. Some
outputs repeatedly emitted a correct numerical answer but used abnormal
formatting that interfered with extraction. Some technically correct
outputs repeated the answer without producing coherent reasoning. Other
outputs repeatedly generated an incorrect fixed value, corrupted text,
or inconsistent candidate answers.

These observations provide qualitative evidence consistent with the
rewritten traces substantially degrading downstream student usefulness.
However, output analysis alone cannot establish the internal causal
mechanism responsible for the degeneration.

## Discussion

The reproduced experiment supports the paper's central anti-distillation
claim. Training on the original traces produced a capable student with
59.3052% answer-forced accuracy.
Replacing those traces with the optimised rewritten traces reduced
accuracy to 8.8503%.

The model, tokenizer, dataset size, random seed, training configuration,
and evaluation procedure were held constant between the two conditions.
The trace source was therefore the principal experimental difference.

The close numerical agreement with the ACL result provides evidence that
the official rewritten traces substantially reduce the value of
reasoning traces as student-training supervision.

The additional analysis also suggests that the Rewrite student did not
merely produce more ordinary mathematical mistakes. Its responses
frequently contained repeated answers, leaked role markers, corrupted
characters, mixed-script fragments, and inconsistent candidate answers.

Answer forcing partially reduced errors caused by malformed output
structure, but the large remaining Clean-Rewrite gap shows that the
anti-distillation effect cannot be attributed only to the answer
extraction method.

## Limitations

The reproduction used official pre-generated original and rewritten
traces. It therefore did not independently reproduce:

- teacher trace generation;
- trace generation using the 120B rewriting model;
- prompt optimisation;
- Semantic rewriting;
- gradient-based rewriting;
- ADS and DOGe baselines;
- MATH, MMLU, and MMLU-Pro experiments;
- alternative student models;
- answer-only distillation;
- adaptive paraphrasing and KPOD attacks;
- teacher and rewriter size ablations;
- official trace-quality evaluation;
- API watermarking experiments.

The reproduction covers only the core GSM8K experiment using
Llama-3.2-3B as the student.

The tokenizer override is also a documented implementation difference
and may explain part of the small absolute accuracy difference between
the reproduction and the publication.

Finally, the degeneration analysis identifies observable output
patterns. It does not prove that a specific internal training failure,
such as mode collapse, occurred.

## Reproducibility Statement

The student-distillation and evaluation stages of the core ACL 2026
GSM8K optimised trace-rewriting experiment were successfully reproduced.

The reproduced Clean and Rewrite answer-forced accuracies were
59.3052% and
8.8503%, respectively. This
produced a relative performance reduction of
85.0767%, compared with the
published value of 85.0953%.

The numerical agreement, controlled Clean-versus-Rewrite configuration,
saved evaluation outputs, configuration records, and integrity checks
support the reproducibility of the paper's core student-degradation
result under the tested configuration.
