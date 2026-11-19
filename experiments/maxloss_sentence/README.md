# Sentence-level MaxLoss on GSM8K

From the project root on Katana, submit the prior test with the committed code.

~~~bash
bash experiments/maxloss_sentence/submit.sh prior-test
~~~

The login node prints the PBS job IDs and result directory. Sentence rewriting runs once for the cohort. Dataset assembly, student training and AF scoring follow through PBS dependencies.

## Data and comparisons

| Cohort | Training | Evaluation |
| --- | --- | --- |
| Prior test | First 100 published GSM8K Clean rows, in the original Stage 6A order | 100 archived held-out questions and gold solutions |
| Full | All 5,231 published Clean rows | All 1,209 GSM8K-Platinum test questions |

Each model–seed pair trains three fresh LoRA adapters on the same problems in the same order. The arms use Clean traces, word-level MaxLoss edits and sentence-level MaxLoss edits. The students are Qwen3-1.7B-Base, Llama-3.2-3B and Qwen3-4B-Base. Each uses seeds 888, 889 and 890. All nine jobs share the same rewritten training data.

Both searches use the frozen Qwen3-1.7B-Base scorer, the same completion mask and a 1,280-token ceiling for the entire chat input. Over-limit candidates are skipped. Student training uses a 1,280-token limit with each student's tokenizer. The AF evaluator retains its existing 1,024-token answer limit.

The word-level control uses the existing equal-token-length edit policy. When available, matching word edits from formal_9679942 are reused. Previously over-limit rows are searched again. Without matching archived records, the word search runs on the allocated GPU.

## Sentence editing

1. Rank eligible reasoning sentences by the mean L2 norm of their token embedding gradients. Keep equations, markup, embedded question text and the final answer section protected.
2. Send the question, current solution and selected sentence to gpt-oss-120b on the allocated H200. Ask for three complete paraphrases. The Stage 6C vLLM 0.11.0 Apptainer image serves the cached model with MXFP4 weights, BF16 computation and an 8,192-token context. Proposals use temperature 0.6, medium reasoning effort and at most 1,024 output tokens per request. A malformed reply gets one retry. The client reads the final response text.
3. Check each paraphrase's numeric content and structure. In a separate temperature-zero call, the same model rates semantic equivalence in context from 1 to 5. Keep candidates rated 5 and check the full answer with Math-Verify.
4. Rebuild the complete chat input for each surviving candidate and select the largest mean completion loss gain of at least 0.0001. After acceptance, recompute the gradient. If a sentence has no acceptable replacement, move to the next eligible sentence. Try up to five distinct sentences and retain every training row.

The edit percentage equals 100 times the minimum number of token insertions, substitutions and deletions divided by the original solution's token count. It excludes the question and chat template. The mean includes rows with zero edits. By default the experiment measures the percentage. Set MAXLOSS_EDIT_PERCENT to impose a ceiling in a separate MAXLOSS_RESULTS directory.

## PBS requests and output

| Job | Prior test | Full run |
| --- | --- | --- |
| Rewrite, per shard of up to 100 rows | 1 H200, 8 CPUs, 128 GB, 1 hour | Same resources per shard, up to 2 shards at once |
| Student training and AF, per model–seed pair | 1 H200, 8 CPUs, 56 GB, 30 minutes | 1 H200, 8 CPUs, 56 GB, 4 hours |
| Dataset assembly and final summary | 2 separate CPU jobs, each with 2 CPUs, 6 GB, 10 minutes | Same |

PBS allocates resources separately to each of the nine student array members, with up to three running at once. REWRITE_WALLTIME and STUDENT_WALLTIME override their time limits. SHARD_CONCURRENCY and STUDENT_CONCURRENCY set the array concurrency limits.

Results are under $HONOUR_SCRATCH/results/maxloss-sentence-v1/ in a directory named for the cohort. The logs/ directory contains job logs and parts/ contains per-row decisions. The final af.csv has one row per model and seed. AF scores are fractions. The clean_minus_*_pp columns give differences in percentage points.

## Full run

After the prior test produces af.csv, submit the full cohort separately.

~~~bash
bash experiments/maxloss_sentence/submit.sh full
~~~

With the same commit and edit budget, the full run reuses the first 100 rewritten rows and processes the remaining 5,131. A change to either setting reruns the first shard. The full student jobs train on all 5,231 rows. For a local Hugging Face save_to_disk copy of the Platinum test split, export PLATINUM_PATH before submission. Otherwise the run uses the local dataset cache in offline mode.

Rewriting saves each completed row. After a timeout or failed job, resubmit the same cohort and settings to resume. Completed shards and completed model–seed comparisons are reused.
