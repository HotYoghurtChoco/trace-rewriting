# Stage 6C.3 matched CosRewrite experiment

## Objective

Stage 6C.3 tests whether scalar gradient feedback improves a two-round trace
rewriter beyond the benefit of generating and selecting an additional candidate.
The primary downstream outcome is held-out answer-forced (AF) accuracy of the
Qwen3-1.7B student.

## Matched generation

The experiment uses the same frozen 100 training rows used by the Stage 6A
formal pilot. No row is removed based on BaselineRewrite validity or generated
outcomes.

For every row:

1. Generate one shared first-round candidate, C1, from BaselineRewrite.
2. Use C1 as the second-round parent when it is valid; otherwise use
   BaselineRewrite.
3. Generate C2 with the parent's scalar gradient cosine and the direction
   "lower is better".
4. Generate S2 from the same parent and the same sampling seed, without any
   gradient, cosine, ranking, or score feedback.
5. Select the lowest-cosine valid trace from `{C1, C2}` for CosRewrite and from
   `{C1, S2}` for SelectionOnly. Use BaselineRewrite only when neither generated
   candidate in that arm is valid.

The fallback retains the frozen input row so that the matched 100-row cohort is
not silently reduced. It is recorded as a fallback and is not labelled as a
valid generated winner.

The two arms therefore share their problems, BaselineRewrite inputs, C1 traces,
second-round parents, generator settings, second-round seeds, validation rules,
scorer, and selection rule. The presence of scalar feedback in the C2 prompt is
the intended difference.

## Student evaluation

Three student seeds are used: 888, 889, and 890. Each seed trains four
Qwen3-1.7B LoRA students using the Stage 6A recipe: Clean, BaselineRewrite,
SelectionOnly, and CosRewrite. Each condition uses 100 training examples, and
evaluation uses the fixed 100-example GSM8K holdout selected with shuffle seed
137.

The LoRA settings are rank 16, alpha 16, dropout 0.05, two epochs, effective
batch size 16, learning rate 1e-4, weight decay 0.1, and warmup ratio 0.1.
Evaluation uses BF16 base weights with dynamic LoRA loading.

The student launcher passes each seed explicitly on the command line because
the shared configuration loader's CLI default otherwise overrides the YAML
seed with 888. The Stage 6C.3 student entry point resets the random seed
immediately before each Clean, BaselineRewrite, SelectionOnly, or CosRewrite
LoRA adapter is initialized; the trainer also receives that same seed. This
applies to newly created adapters when resuming a partly completed run too.
The Stage 6A training/scoring source and Stage 6C.2 frozen-scorer checks remain
unchanged. Seed control does not imply bitwise-identical GPU execution.

The generated dataset is fixed once; the three student seeds measure student
training variation rather than generator sampling variation.

The primary paired contrast is:

```text
SelectionOnly held-out AF - CosRewrite held-out AF
```

A positive value means CosRewrite produced less useful distillation traces and
therefore improved the defense. BaselineRewrite and Clean are contextual
comparators. The report includes the three seed-level differences, their mean,
and a paired 95% t interval.
