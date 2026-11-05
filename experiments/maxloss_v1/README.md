# MaxLoss Trace Rewriting

This experiment edits the existing 100 Clean reasoning traces to increase
completion loss under a fixed Qwen3-1.7B-Base model, then measures the downstream
effect with fresh Clean and Optimized-Clean students.

## Objective and proposal rule

For each fixed question `q`, optimize the completion `r` under frozen weights
`theta0`: maximize `loss(q, r; theta0)` within the local wording policy.

The loss uses the existing `_tokenize_completion_only_batch` helper from
`optimize/score_candidates.py`. Question and system tokens are masked. The
completion includes the supplied reasoning, final answer, and the template's
assistant suffix/EOS. The model's causal shift is applied once by its own loss.
Inputs longer than 1,024 total tokens remain unchanged; no shortened text is
silently scored as a full completion.

Each round performs:

1. Compute the current loss and its gradient with respect to detached input
   embeddings; model parameters have `requires_grad=False` throughout.
2. Generate the explicit local wording alternatives in `text_edits.py`.
3. Retain edits whose retokenized inputs have the same length and completion
   mask. Rank them by `sum(g_E * (E_candidate - E_current))`.
4. Rebuild the complete candidate tokens AND labels, and measure true loss
   for the higher-ranked half (rounded up), capped at 16 candidates, one at a time.
5. Accept the best candidate if its loss increases by at least 0.0001.
   Stop without an improvement or after five rounds.

The directional score is an input-side approximation with current labels held
fixed. Exact rescoring includes the changed target tokens. Each accepted edit
replaces one local wording span, which can contain more than one token.

The proposal vocabulary covers equivalent discourse connectives and wording
such as `calculate` / `compute`. It excludes quantities, operators, arithmetic
relations, negation, and entities. Final-answer paragraphs, math spans,
equation lines, markup, and verbatim copies of the question are protected.
The source and selected result are also checked with MathVerify. Read the
five before/after examples to assess contextual meaning before the formal run.

## Inputs and software

`config.yaml` identifies the existing Stage6A 100-row cohort, base checkpoint,
and the separate frozen Qwen tokenizer. The scorer loads no adapter and uses
no reference gradients or GPT-OSS rewriter. It reuses the existing Katana
environment; no environment update is required.

Defaults: seed 888, BF16, SDPA, 1,024 total tokens, five rounds, at most
16 scored candidates per round. The base model and tokenizer revision prefixes
are recorded alongside the execution commit, PBS job, runtime configuration,
data identity, timing, and result paths.

## Five-example run

The MaxLoss code uses the dedicated linked worktree at
`/home/z5463756/honour/papers/trace-rewriting-maxloss` on branch
`research/maxloss`. The existing Stage6C worktree and branch
remain in place. Models, environments and prior results retain their existing
paths outside the code worktrees.

From the repository root, after committing the new files:

```bash
qsub experiments/maxloss_v1/pbs/maxloss_smoke.pbs
```

The PBS request is one H200, four CPUs, 32 GB host RAM, and 15 minutes. It
selects the first five source rows that pass the original-answer check, fit
the token budget, and have wording proposals. Selection precedes gradient or
loss results. Rows with no token-aligned improvement are reported unchanged.

Results are written to a new directory for each job:

```text
/srv/scratch/z5463756/honour/results/maxloss-qwen3-1.7b-v1/smoke_JOBID/
```

- `run_config.json`: actual inputs, checkpoint, software and execution metadata.
- `records.jsonl`: row identities, original/selected traces, losses, every
  scored edit, accepted edits, stop reasons and per-row timing.
- `examples.md`: readable before/after examples and accepted replacements.
- `summary.json`: counts, loss means, stop reasons, timing and GPU memory.
- `OPTIMIZATION_COMPLETE`: the execution finished and wrote its summary.

The PBS stdout file also prints the exact result directory and per-row progress.
Search outputs are flushed after each row; a failed run retains those records.

## Formal generation and downstream comparison

After reviewing the five examples, keep or revise the proposal configuration
and size the formal job from the measured runtime. Within a GPU allocation:

```bash
python experiments/maxloss_v1/run_optimization.py \
  --mode formal --output-dir /path/to/new/formal_output
```

Formal mode preserves all 100 row identities and order. An uneditable,
over-budget, or unsuccessful row keeps its original Clean trace. The resulting
Hugging Face dataset is saved at `formal_output/optimized_dataset`.

Prepare a new matched student run:

```bash
python experiments/maxloss_v1/prepare_students.py \
  --generation-dir /path/to/new/formal_output \
  --working-dir /path/to/new/students_seed_888 --seed 888
```

Use the existing seeded driver under the same Katana training environment:

```bash
python experiments/stage6c_cosrewrite_v1/run_stage6c3_students.py \
  --config /path/to/new/students_seed_888/config.yaml --seed 888
```

It trains two fresh students with the established LoRA recipe: r/alpha 16/16,
dropout 0.05, two epochs, learning rate 1e-4, batch size 4 and effective batch
size 16. Evaluation uses the existing 100 held-out questions, Raw and AF, and
dynamic LoRA inference. `prepare_students.py` applies seed-137 ordering to
the optimized arm to match the Clean shuffle performed by the existing driver.
Both arms therefore use the same problem order and seeded initialization.

The primary comparison is `Clean AF - Optimized-Clean AF`; positive values
indicate lower student accuracy after learning from the optimized traces.
The existing BaselineRewrite results are historical method references.
The original experiment code and old result directories are unchanged.

## Local checks

```bash
python -m unittest experiments.maxloss_v1.test_optimizer -v
```

The CPU tests cover the completion mask (including PAD=EOS), fresh candidate
labels, numerical input-gradient agreement, frozen weights, actual-loss ascent,
edit budgets, unchanged fallback, token limits and protected text. The real
checkpoint's loss changes and runtime are obtained from the Katana sample run.
