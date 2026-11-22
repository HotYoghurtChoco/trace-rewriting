"""Assemble the shared rewrite corpus and train seed-matched students against existing Clean."""

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
import shutil
import tempfile

import yaml

from experiments.maxloss_sentence import data
from experiments.maxloss_sentence.rewriter import MODEL, REVISION


ARM = "step"
N_TRAIN, N_EVAL = data.SIZES["full"]
N_SHARDS = math.ceil(N_TRAIN / data.SHARD)
CELLS = [(m, s) for m in data.STUDENTS for s in data.SEEDS]


def read(path):
    return json.loads(path.read_text())


def init(run, prior, commit):
    prior = prior.resolve()
    reference = read(prior / "run.json")
    n, m = reference["train"], reference["eval"]
    if (n, m, reference["scorer_and_student_input_cap"]) != (N_TRAIN, N_EVAL, data.CAP):
        raise ValueError("Expected the completed 5231/1209 Clean comparison at 1280 tokens")
    if not (prior / "af.csv").is_file() or not (prior / "datasets/evaluation").is_dir():
        raise ValueError(f"Missing the completed full comparison at {prior}")
    meta = {"method": "gradient_guided_math_step_compression", "source_commit": commit,
            "profile": "full", "train": n, "eval": m, "train_source": reference["train_source"],
            "paired_clean": str(prior), "paired_clean_commit": reference["source_commit"],
            "rewrite_seed": 888, "default_student_seed": 888, "student_checkpoints": reference["student_checkpoints"],
            "input_cap": data.CAP, "rewriter": MODEL, "rewriter_revision": REVISION,
            "rounds": 1, "replacements_per_passage": 2, "minimum_gain_over_compact_base": data.MIN_GAIN}
    record = run / "run.json"
    if record.is_file() and read(record) != meta:
        raise ValueError("This run already contains a different source commit or Clean comparison")
    (run / "parts").mkdir(parents=True, exist_ok=True)
    if not record.exists():
        data.atomic_json(record, meta)


def collect(run):
    from datasets import load_from_disk
    dest = run / "datasets"
    if (dest / "rewrite_summary.json").exists():
        return
    prior = Path(read(run / "run.json")["paired_clean"]) / "datasets"
    clean = load_from_disk(str(prior / "original_traces"))
    evaluation = load_from_disk(str(prior / "evaluation"))
    if (len(clean), len(evaluation)) != (N_TRAIN, N_EVAL):
        raise ValueError("Archived dataset size differs from the paired cohort")
    originals = iter(clean)
    texts, reasons, selected = [], Counter(), Counter()
    original_tokens = final_tokens = paired = 0
    alternatives_judged = alternatives_scored = judge_truncations = 0
    loss_sums = dict.fromkeys(("original_loss", "base_loss", "final_loss"), 0.0)
    for i in range(N_SHARDS):
        with (run / "parts" / f"{i:03}.jsonl").open() as part:
            for row in map(json.loads, part):
                source = next(originals, None)
                if (source is None or row["index"] != len(texts)
                        or any(row[k] != source[k] for k in ("problem", "solution", "original_trace"))):
                    raise ValueError(f"Row {len(texts)} differs from the original Clean cohort")
                step = row["step"]
                texts.append(row["rewrite_trace"])
                selected[step["selected"]] += 1
                reasons[step["reason"]] += 1
                original_tokens += step["original_tokens"]
                final_tokens += step["final_tokens"]
                verdicts = step.get("verdicts", {})
                proposals = step["proposals"]
                alternatives_judged += sum(label in verdicts for label in proposals)
                alternatives_scored += sum("loss" in proposal for proposal in proposals.values())
                judge_truncations += step.get("judge_stats", {}).get("truncated_responses", 0)
                if step["selected"] != "original":
                    paired += 1
                    for key in loss_sums:
                        loss_sums[key] += step[key]
    if len(texts) != N_TRAIN:
        raise ValueError(f"Expected {N_TRAIN} rows, found {len(texts)}")
    if not paired:
        raise ValueError(f"All {N_TRAIN} rows retained the original trace. No new training corpus was assembled. "
                         f"Reasons={dict(reasons)}, judge_truncated_responses={judge_truncations}")
    wins = selected["A"] + selected["B"]
    report = {"train": N_TRAIN, "eval": N_EVAL, "reason_counts": dict(reasons),
              "selected": {label: selected[label] for label in ("A", "B", "base", "original")},
              "rewrite_outcome": "loss_selected_alternatives" if wins else "compact_base_only",
              "validated_compact_bases_used": paired, "maxloss_alternatives_used": wins,
              "alternatives_with_judge_verdict": alternatives_judged, "alternatives_scored": alternatives_scored,
              "judge_truncated_responses": judge_truncations,
              "original_body_tokens": original_tokens, "training_body_tokens": final_tokens,
              "corpus_token_reduction_percent": 100 * (original_tokens - final_tokens) / original_tokens,
              "mean_completion_loss_on_valid_bases": {
                  key: total / paired for key, total in loss_sums.items()}}
    with tempfile.TemporaryDirectory(dir=run, prefix="assemble-") as tmp:
        stage = Path(tmp) / "datasets"
        corpus = clean.add_column("rewrite_trace", texts).shuffle(seed=137, keep_in_memory=True)
        corpus.save_to_disk(str(stage / ARM))
        data.atomic_json(stage / "rewrite_summary.json", report)
        stage.rename(dest)
    print(json.dumps(report), flush=True)


def relative_result(clean, candidate, n):
    clean_correct, step_correct = round(clean * n), round(candidate * n)
    if not clean_correct:
        raise ValueError("Relative AF decline requires a nonzero Clean baseline")
    return {"clean_correct": clean_correct, "step_correct": step_correct, "net_fewer_correct": clean_correct - step_correct,
            "drop_pp": 100 * (clean - candidate), "relative_drop_percent": 100 * (clean_correct - step_correct) / clean_correct,
            "target_max_correct": (9 * clean_correct) // 10, "met_10pct_relative": 10 * step_correct <= 9 * clean_correct}


def train_student(run, model, seed):
    from experiments.maxloss_sentence.first_accepted import train_one
    cell = run / "students" / f"{model}-{seed}"
    if (cell / "af.json").is_file():
        return
    prior = Path(read(run / "run.json")["paired_clean"])
    old_cell = prior / "students" / f"{model}-{seed}"
    ref = read(old_cell / "af.json")
    old_work = old_cell / Path(ref["scores_path"]).parents[1].name
    cfg = yaml.safe_load((old_work / "config.yaml").read_text())
    proxy, = cfg["proxy_models"]
    names = (Path(proxy["name"]).name, Path(proxy["tokenizer"]).name)
    if (names != data.STUDENTS[model] or (ref["model"], ref["seed"]) != (model, seed)
            or (cfg["seed"], cfg["proxy_max_length"], cfg["dataset_size_used_for_optimize"],
                cfg["eval_dataset_size_for_optimize"]) != (seed, data.CAP, N_TRAIN, N_EVAL)):
        raise ValueError(f"Archived configuration does not match {model} / {seed}")
    cell.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="attempt-", dir=cell))
    # Local directory entries keep the trainer's Arrow caches away from the archived datasets.
    shutil.copytree(prior / "datasets/original_traces", work / "original_traces")
    shutil.copytree(run / "datasets" / ARM, work / ARM)
    cfg.update(working_dir=str(work), original_traces_path=str(work / "original_traces"),
               evaluation_dataset_path=str(prior / "datasets/evaluation"),
               stage6a_run_kind="math_step_compression", stage6a_candidate_generation="gradient_selected_math_steps",
               trace_edit_policy="math_step_compression_maxloss")
    (work / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    iteration = work / cfg["experiment_folder"]
    iteration.mkdir()
    candidate = {"name": ARM, "instruction": "Gradient-guided mathematical-step compression MaxLoss", "score": None}
    (iteration / "candidate_instructions.yaml").write_text(yaml.safe_dump({"candidate_instructions": [candidate]}))
    # This paired trainer reuses the completed Clean adapter and its saved evaluation.
    train_one(cfg, work, old_work, names[0])
    scores_path = iteration / "instructions_and_scores.yaml"
    score, = yaml.safe_load(scores_path.read_text())["candidate_instructions"]
    metrics, = score["proxy_metrics"]
    clean, candidate = metrics["clean_af_accuracy"], metrics["candidate_af_accuracy"]
    if clean != ref["clean"] or any(metrics[k]["num_examples"] != N_EVAL for k in ("clean_evaluation", "candidate_evaluation")):
        raise ValueError(f"The reported Clean AF or evaluation size differs from {old_cell}")
    result = {"model": model, "seed": seed, "train": N_TRAIN, "eval": N_EVAL, "clean_af": clean,
              "step_af": candidate, **relative_result(clean, candidate, N_EVAL), "old_word_af": ref["word"],
              "old_sentence_af": ref["sentence"], "scores_path": str(scores_path), "job": os.getenv("PBS_JOBID")}
    data.atomic_json(cell / "af.json", result)
    print(json.dumps(result), flush=True)


def summarize(run, seed):
    rows = [read(run / "students" / f"{model}-{seed}" / "af.json") for model in data.STUDENTS]
    dest = run / f"af_seed{seed}.csv"
    temp = dest.with_suffix(".tmp")
    with temp.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(dest)
    for row in rows:
        print(f"{row['model']} {row['clean_correct']} -> {row['step_correct']}/{N_EVAL} "
              f"relative drop={row['relative_drop_percent']:.2f}% target<={row['target_max_correct']}")
    print(f"all_three_met_10pct_relative={all(row['met_10pct_relative'] for row in rows)} table={dest}")


def walltime(run):
    first = run / "parts/000.timing.json"
    if not (first.is_file() and (run / "parts/000.jsonl").is_file()):
        return "00:30:00"
    # Allow 25% variation, then round up to a five-minute PBS slot.
    minutes = 5 * math.ceil(read(first)["full_shard_seconds"] * 1.25 / 300)
    return f"{minutes // 60:02}:{minutes % 60:02}:00"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "pending-shards", "walltime", "collect", "student", "summary"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.stage == "init":
        init(run, Path(os.environ["PAIRED_CLEAN"]), os.environ["SOURCE_COMMIT"])
    elif args.stage == "pending-shards":
        print(" ".join(str(i) for i in range(N_SHARDS) if not (run / "parts" / f"{i:03}.jsonl").exists()))
    elif args.stage == "walltime":
        print(walltime(run))
    elif args.stage == "collect":
        collect(run)
    elif args.stage == "student":
        train_student(run, *CELLS[args.index])
    else:
        summarize(run, int(os.environ["STUDENT_SEED"]))


if __name__ == "__main__":
    main()
