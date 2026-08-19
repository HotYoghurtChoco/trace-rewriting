# Stage 6C.2 Workflow Test Record

Record date: 2026-08-19 (Australia/Sydney)

Branch: `research/stage6c-cosrewrite`

## 1. Outcome and claim boundary

The Stage 6C.2 pre-formal workflow test is complete. The final batch run was
PBS job `9039134.kman.restech.unsw.edu.au` at execution commit
`2222a20db48bddec825865709f1c763eaabdd847`. All five fixed problems passed and
the PBS exit status was zero.

This outcome establishes that the following workflow ran on an H200:

1. use fixed GPT-OSS-120B weights and a fixed container runtime;
2. generate C1;
3. compute cosine with the frozen Qwen scorer;
4. provide the black-box cosine feedback to the rewriter and generate C2;
5. validate answers, traces, scorer tokens, parameter integrity, and lineage;
6. preserve checkpoints, results, and provenance across a five-problem batch.

Stage 6C.2 is **not** a CosRewrite method-effect experiment. It did not lock a
formal SelectionOnly control, create the formal rewrite dataset, train a
student, or measure held-out Raw/AF. This record therefore cannot support a
claim that CosRewrite is effective or ineffective. Those tests belong to Stage
6C.3.

This file was added to Git after every job listed below had finished. The later
Git commit containing this record is an evidence-only record commit and must
not be reported as the execution commit of any run.

## 2. Inputs and external runtime objects

- The 100 frozen BaselineRewrite traces come from the Stage 6A formal candidate
  dataset. They are the existing optimized rewrite traces, not 100 traces newly
  generated in Stage 6C.2 or a newly selected paper subset.
- Stage 6C.1 reusable scorer equivalence passed separately in job `8967237` at
  execution commit `c5449b34a525993981d7653baadedb13e2a4d909` and was archived
  separately. It is a dependency of Stage 6C.2, not part of this Stage 6C.2
  record.
- Rewriter model: `openai/gpt-oss-120b`.
- Model revision: `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
- OCI manifest digest:
  `sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a`.
- Apptainer SIF SHA256:
  `e10fa3b1e526c5f375e2221d0afd1461963f729e0a7867c0d88bee10eb065ec3`.

The model revision and SIF SHA256 identify external objects that are not stored
in Git. They do not create a separate Stage 6C.2 file-hash archive.

Unless stated otherwise, result directory names below are relative to:

```text
/srv/scratch/z5463756/honour/results/
stage6c-cosrewrite-qwen3-1.7b-v1/backend_preflight/
```

`backend_preflight` is retained here because it is part of the historical path
used by the completed runs.

## 3. Stage 6C.2a: H200 environment

| Job | Execution commit | PBS outcome | Conclusion |
|---|---|---|---|
| `8983448` | `318b7a183aa4bcc7140e42e52916f4515eb6f694` | Exit 0, `00:01:30` | H200, BF16, torch, CUDA, vLLM import/serve options, and the parser option were available |

The launcher requested one H200, 4 CPUs, 16 GB of host memory, and 15 minutes.
The detected H200 had 140.065 GiB and compute capability 9.0. The runtime used
Python 3.11.3, torch 2.8.0+cu128, and vLLM 0.11.0.

Result directory: `h200_environment_8983448`.

The `COMPLETE` marker explicitly records `model_loaded=false` and
`model_download_attempted=false`. Stage 6C.2a therefore established environment
readiness only; it did not establish that the GPT-OSS weights or API worked.

## 4. Stage 6C.2b: fixed GPT-OSS snapshot

| Job | Execution commit | PBS outcome | Conclusion |
|---|---|---|---|
| `8990062` | `ddebaeea7418ed4b5957706657197314971bbe8c` | Exit 0 | Download and validation of the fixed revision's top-level HF safetensors passed |

Result directory: `gptoss_snapshot_download_8990062`.

The selected weights totalled 65,276,859,410 bytes and used 15 top-level shard
symlinks. vLLM required only the root safetensors, so `metal/` and `original/`
were excluded instead of downloading the approximately 182 GiB full
repository. Stage 6C.2b did not load the model or call the API.

## 5. Stage 6C.2c: host failure to container API pass

| Version or purpose | Job | Execution commit | Outcome | Cause or correction |
|---|---:|---|---|---|
| Host API v1 | `8990140` | `a503dcbb6139a9a941a1a3c72fa320a7cc78ae94` | Exit 1 | Approximately 65.9651 GiB of weights loaded successfully, but host GLIBC 2.28 was older than the GLIBC 2.29 required by the vLLM `_moe_C` extension, so `topk_softmax` was not registered. This was not weight corruption or OOM |
| Container acquisition v1 | `8993678` | `cfc0da1c7e5528135331cbe3de612f751cfb80ae` | FAIL | `mksquashfs` exceeded the 16 GB cgroup limit |
| Container acquisition v2 | `8993853` | `ee3705ebe989de4901140f0683846ebde833bb63` | FAIL gate | A temporary SIF was built with 32 GB, but applying a GPU-dependent gate on the CPU node was invalid |
| Container acquisition v3 | `8994052` | `e4eb1b0b37ef4ec317496724d4ef9f68e8ce6fa5` | PASS | Acquisition retained CPU-safe checks and moved GPU checks to the H200 job |
| Container API v2 | `8995484` | `f519187c9e9bba867cd15d12d935f380dca00caf` | Exit 0, `00:04:14` | CUDA extensions, real model loading, health, model listing, chat completion, and parser output passed inside the container |

The corresponding result directories are:

```text
gptoss_h200_api_8990140
vllm_container_acquire_8993678
vllm_container_acquire_8993853
vllm_container_acquire_8994052
gptoss_h200_container_api_8995484
```

The final container API used vLLM 0.11.0, torch 2.8.0+cu128, transformers
4.57.0, Python 3.12.11, MXFP4/BF16, maximum model length 8192, GPU memory
utilization 0.75, and the `openai_gptoss` parser. The real response contained
`content="19"` and `reasoning_content="Just answer 19."`. GPU memory was
approximately 110,217 MiB when the server was ready and returned to 0 MiB after
shutdown.

## 6. Stage 6C.2d: single-problem loop v1-v6

The fixed problem was row 0. Workflow-test settings were temperature 0.6,
maximum tokens 1024, medium reasoning effort, at most two generation attempts
per round, scoring seed 888, C1 seed 888001, and C2 seed 888101. These settings
are not the Stage 6C.3 formal protocol lock.

| Version | Job | Execution commit | PBS outcome | Reached point, failure cause, or correction |
|---|---:|---|---|---|
| v1 | `9001324` | `78e1927bdfb7e2e68e4cfc1cd27ff69635429462` | Exit 1, `00:00:10` | Snapshot entries were symlinks; `find -type f` returned 0 instead of 15 and stopped before vLLM startup |
| v2 | `9003273` | `2327ff0d80c0e92801a88c7b2aa3f31441fca185` | Exit 1, `00:04:07` | Changed to `find -L`; API startup, C1, Qwen loading, and the 100-reference recomputation were reached; the strict A100 reference gate failed, but the actual H200 values were not persisted |
| v3 | `9005658` | `a4247b9264cadf2e5fefd381da51eed59634e373` | Exit 1, `00:04:08` | Added persistent diagnostics; the A100-to-H200 reference-loss error was 9.45 times tolerance and the norm error was 30.99 times tolerance; the run stopped safely before candidate scoring |
| v4 | `9010570` | `581c0391eafcc8e335176f39ccb143b1034e2256` | Exit 0, `00:03:44` | Completed two independent H200 scorer repeats; A100-to-H200 score equivalence failed in both repeats, while H200-to-H200 repeat stability passed; this supported the later H200-native runtime policy |
| v5 | `9032662` | `e4b1c9da19d505075a55e873eb4d2ba5fd8f800e` | Exit 1, `00:04:19` | Complete Baseline/C1/C2 artifacts were produced; host memory reached the 32 GB cgroup limit, and the old repetition detector also treated LaTeX punctuation tokens as repeated degeneration, falsely failing the correct baseline |
| v6 | `9033138` | `a70c9e65fcd5165a4c0b0e4f10d7e59fc90cd742` | Exit 0, `00:04:06` | Replaced the detector with Unicode alphanumeric eight-token windows, added regression tests, and raised host memory to 64 GB; every single-problem loop gate passed |

The key v3 values were:

| Metric | Frozen/A100 | H200 v3 | Absolute error | Tolerance |
|---|---:|---:|---:|---:|
| Reference loss | 1.1347439781 | 1.1358162263 | 0.0010722482 | 0.0001134744 |
| Mean gradient norm | 1.2550210246 | 1.2511321186 | 0.0038889060 | 0.0001255021 |

These measurements establish cross-runtime numerical drift. A single run does
not prove that the cause was hardware alone.

Across the two v4 H200 repeats, cosine Spearman was 0.9999879988, dot Spearman
was 0.9999759976, and both the lowest-ten and highest-ten overlaps were 10/10.
Every H200 repeat-stability gate passed. At the same time, both repeats failed
equivalence against the frozen A100 scores. Later runs therefore required
consistency within the same fixed H200 runtime and did not claim A100/H200
numerical equivalence.

Stage 6C.2d v5 had two independent problems: an insufficient host-memory
request and a LaTeX false positive in trace validation. The earlier
`future feature annotations is not defined` message observed with Katana's
default `python3` was not a v5 job failure; `py_compile` passed with the official
Python 3.11 environment.

The v6 result directory is
`gptoss_qwen_single_problem_loop_v6_9033138`. All closed-loop gates passed. The
BaselineRewrite, C1, and C2 cosine values were 0.04941048, 0.08723686, and
0.08894748 respectively, so the temporary lowest-valid-cosine rule selected
BaselineRewrite. This single-problem ranking is not a method-effect claim.

## 7. Stage 6C.2e: five-problem batch v1-v2

| Version | Job | Execution commit | PBS outcome | Reached point, failure cause, or correction |
|---|---:|---|---|---|
| v1 | `9037482` | `6a3290f6cbb517b078ebbee29efaede3954c2562` | Exit 1, `00:04:13` | Initially selected rows 0-4 in dataset order; rows 0, 1, and 2 passed, while the frozen baselines for rows 3 and 4 were boxed answers only and did not meet the locked full-trace validity rules. This was not a memory, API, C1, or C2 failure |
| v2 | `9039134` | `2222a20db48bddec825865709f1c763eaabdd847` | Exit 0, `00:04:16` | Audited eligibility across all 100 frozen baselines before generation, then fixed the first five valid rows 0, 1, 2, 6, and 9; all five passed |

The v1 result directory is
`gptoss_qwen_five_problem_batch_2e_v1_9037482`.

The v2 result directory is
`gptoss_qwen_five_valid_problem_batch_2e_v2_9039134`.

Of the 100 baselines, 19 met full-trace validity before the scorer token gate
and 81 did not. Stage 6C.2e v2 selected the first five valid rows
`[0, 1, 2, 6, 9]`. Selection did not inspect generation outcomes or cosine
values and therefore did not choose problems based on method performance.

Stage 6C.2e v2 requested one H200, 8 CPUs, 128 GB of host memory, and 30
minutes. Actual wall time was `00:04:16`, with approximately 65.94 GB of host
memory used. The reference gradient was recomputed once for the full batch, all
five problem checkpoint roundtrips passed, scorer parameters remained
unchanged, `COMPLETE` was present, and `FAILED` was absent.

The five descriptive cosine results were:

| Row | Baseline | C1 | C2 | Temporary lowest-valid-cosine winner |
|---:|---:|---:|---:|---|
| 0 | 0.048583 | 0.094155 | 0.075667 | BaselineRewrite |
| 1 | -0.003752 | 0.022702 | 0.071236 | BaselineRewrite |
| 2 | 0.034554 | 0.161294 | 0.065742 | BaselineRewrite |
| 6 | 0.017628 | 0.058749 | 0.032982 | BaselineRewrite |
| 9 | -0.000810 | 0.035883 | 0.025449 | BaselineRewrite |

C2 was lower than C1 in four of five cases and higher in one. C2 was higher
than BaselineRewrite in all five cases, and the temporary rule selected
BaselineRewrite in all five. These values are retained only as workflow
diagnostics and input to Stage 6C.3 design. Without SelectionOnly, student
training, and AF measurement, they cannot isolate the effect of cosine feedback
or determine the final effectiveness of CosRewrite.

## 8. Key evidence locations

The main existing result directories are:

```text
h200_environment_8983448/
gptoss_snapshot_download_8990062/
vllm_container_acquire_8994052/
gptoss_h200_container_api_8995484/
h200_scorer_equivalence_v4_9010570/
gptoss_qwen_single_problem_loop_v6_9033138/
gptoss_qwen_five_problem_batch_2e_v1_9037482/
gptoss_qwen_five_valid_problem_batch_2e_v2_9039134/
```

The scheduler logs for the final two passing runs are:

```text
/srv/scratch/z5463756/honour/logs/stage6c/tr-s6c-loop-v6.pbs.log
/srv/scratch/z5463756/honour/logs/stage6c/tr-s6c-batch-2e-v2.pbs.log
```

The current tracked launchers and drivers are:

```text
experiments/stage6c_cosrewrite_v1/run_single_problem_closed_loop.py
experiments/stage6c_cosrewrite_v1/run_multi_problem_batch.py
experiments/stage6c_cosrewrite_v1/config/stage6c_2e_batch_preflight_lock.json
experiments/stage6c_cosrewrite_v1/pbs/stage6c_2a_h200_environment_preflight.pbs
experiments/stage6c_cosrewrite_v1/pbs/stage6c_2b_gptoss_snapshot_download.pbs
experiments/stage6c_cosrewrite_v1/pbs/stage6c_2c1_vllm_container_acquire.pbs
experiments/stage6c_cosrewrite_v1/pbs/stage6c_2c_gptoss_h200_api_preflight.pbs
experiments/stage6c_cosrewrite_v1/pbs/stage6c_2d_h200_scorer_equivalence.pbs
experiments/stage6c_cosrewrite_v1/pbs/stage6c_2d_single_problem_closed_loop.pbs
experiments/stage6c_cosrewrite_v1/pbs/stage6c_2e_multi_problem_batch.pbs
```

## 9. Record policy and next stage

Stage 6C.2 was a workflow test supporting Stage 6C.3, so this checkpoint adds
one compact Git record only:

- no tag is created for 2a, 2b, 2c, 2d, 2e, or any small version;
- no Stage 6C.2 `SHA256SUMS` file is created;
- full Scratch results, model weights, SIF files, environments, and caches are
  not copied into Git;
- failed jobs and their correction chain remain recorded and are not
  overwritten as passes;
- Git commits identify this record and the code, while each job ID, execution
  commit, and result path identify an actual run; the model revision and SIF
  SHA identify external dependencies.

The next stage is the Stage 6C.3 formal method experiment. Before formal
generation, it still requires an executable protocol lock and a matched
SelectionOnly implementation. Formal rewrite generation, matched student
training, fixed held-out Raw/AF evaluation, and pre-declared statistical
analysis follow. After Stage 6C.3 produces the formal student/AF results, that
stage should receive the complete manifest, durable result copy, and any
necessary immutable tag.
