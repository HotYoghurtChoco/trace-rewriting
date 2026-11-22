# Gradient-guided mathematical-step compression

This experiment trains one new arm on the 5,231 original GSM8K problems. Each of the three students uses the same rewritten corpus, its archived seed-matched Clean comparison, the original Clean-based training order, and the same 1,209 GSM8K-Platinum evaluation questions.

The pinned local gpt-oss-120b first reads the full teacher trace and writes a standalone sequence of mathematical steps. Needed facts from the draft can survive even if the formal solution omitted them. The code appends the source's verified final answer and requires the resulting base to be shorter than the original. On this base, the frozen Qwen3-1.7B-Base computes the completion loss and ranks mathematical steps by mean token embedding gradient norm. The highest-ranked step is the anchor. Its highest-ranked immediate mathematical neighbor, when present, can be merged with it.

One proposal call returns two replacements for that passage. Each replacement goes back into the complete base. A separate call checks the full solutions against the question, teacher trace and reference derivation, including quantity relations, units and recomputability. Only the checker sees the gold derivation. The final answer is also checked by Math-Verify. Each eligible alternative must compress or reformulate a mathematical step, remain shorter than the original teacher trace and fit the scorer's 1,280-token chat limit. It can use extra words relative to the compact base to explain an equivalent expression or a unit. The highest valid mean completion loss wins if it exceeds the base by at least 0.0001. Otherwise the valid base is kept. An invalid base falls back to the original. This trains one combined compression-and-selection method. Per-row records and the assembly summary report how often a loss-selected alternative actually enters training.

If the judge reaches its 2,048-token output limit, it can retry once with up to 4,096 tokens, using the prompt length to stay within the 8,192-token server context. The summary counts truncated judge responses, alternatives with verdicts, alternatives scored, and alternatives selected. A corpus with valid compact bases and zero winning alternatives is reported as `compact_base_only`. If every row retains the original trace, assembly stops before student training.

## Katana

Commit the patch in the project repository first. From the repository root, submit the first 100 **formal** rows.

~~~bash
bash experiments/maxloss_step/submit.sh first
~~~

When shard zero completes, submit the remaining 52 shards and three seed-888 students.

~~~bash
bash experiments/maxloss_step/submit.sh full 888
~~~

The default run directory is `$HONOUR_SCRATCH/results/maxloss-step-v1/full`. The first shard writes `parts/000.timing.json` with startup, elapsed time, selection counts and fallback reasons. The previous full run requested 30 minutes per rewriting shard and the longest took 27:56. The new first shard requests 30 minutes. Subsequent shards request the measured full-shard time plus 25%, rounded up to five minutes. After the first 100 rows, PBS can run up to four of the remaining 52 shards at once, with one H200 per shard. With four GPUs continuously available and similar shard runtimes, the remaining work takes about 13 full-shard durations plus queue delays. The last shard contains 31 rows. Submitting `full 888` directly schedules all 53 shards with a 30-minute initial request. Set `REWRITE_WALLTIME` or `SHARD_CONCURRENCY` to override these defaults.

The prior seed-888 training jobs took 36:01, 38:07 and 49:43 for all three arms, with 2:30:00 requested per job. The single-arm H200 jobs request 20 minutes for Qwen3-1.7B, 25 for Llama-3.2-3B and 30 for Qwen3-4B. They can run in parallel after assembly. The CPU assembly and report request five and two minutes. A timeout preserves completed rewrite rows. Rerun the same command for unfinished work. `STUDENT_WALLTIME` overrides a student request.

`af_seed888.csv` lists Clean and new AF, correct counts, percentage-point and relative declines, and the per-model 10% target. `datasets/rewrite_summary.json` lists A/B/base/original selections, corpus compression, and original/base/selected loss on the same valid bases. For a later seed explicitly submit `full 889` or `full 890` using the same assembled dataset. `PAIRED_CLEAN` and `MAXLOSS_STEP_DIR` can point to alternate locations for the archived full comparison and the new run respectively.
