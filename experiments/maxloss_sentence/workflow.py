"""Prepare fixed cohorts, run a paired comparison for each seed and collect AF scores."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

import yaml

from experiments.maxloss_sentence import data
from experiments.maxloss_sentence.rewriter import MODEL, REVISION


def init(root, profile, commit, budget, platinum):
    if budget is not None and not 0 < budget <= 100:
        raise ValueError("Use an edit budget in (0, 100], or omit it to measure edits without a target")
    run = root / profile
    run.mkdir(parents=True, exist_ok=True)
    (run / "parts").mkdir(exist_ok=True)
    n, m = data.SIZES[profile]
    if platinum and profile == "full":
        platinum = str(Path(platinum).expanduser().resolve())
    meta = {"profile": profile, "train": n, "eval": m, "source_commit": commit,
            "train_source": "data/gsm8k/semantic:original_trace",
            "eval_source": str(data.EVAL_100.relative_to(data.ROOT)) if profile == "prior-test" else
                           (platinum or "madrylab/gsm8k-platinum:test"),
            "platinum_path": platinum if profile == "full" else None,
            "max_edit_percent": budget, "scorer_and_student_input_cap": data.CAP,
            "rewriter": MODEL, "rewriter_revision": REVISION, "rewrite_seed": 888,
            "student_checkpoints": data.STUDENTS, "training_seeds": data.SEEDS,
            "candidates_per_sentence": 3, "sentence_attempts": data.ROUNDS, "minimum_loss_gain": data.MIN_GAIN}
    if profile == "full":
        prior = root / "prior-test"
        if not (prior / "af.csv").is_file():
            raise ValueError("Complete the prior test before submitting the full cohort")
        old = json.loads((prior / "run.json").read_text())
        meta["reuse_prior_shard"] = old["source_commit"] == commit and old["max_edit_percent"] == budget
    record = run / "run.json"
    if record.exists():
        if json.loads(record.read_text()) != json.loads(json.dumps(meta)):
            raise ValueError("This result directory belongs to different settings; use another MAXLOSS_RESULTS")
    else:
        data.atomic_json(record, meta)


def collect(run):
    from datasets import Dataset
    settings = json.loads((run / "run.json").read_text())
    dest = run / "datasets"
    if (dest / "rewrite_summary.json").is_file():
        return
    rows = []
    summary = {arm: {"changed": 0, "sum_percent": 0., "paired": 0, "before": 0., "after": 0.}
               for arm in data.ARMS}
    ratings = Counter()
    format_errors = interpreted = 0
    for shard in range((settings["train"] + data.SHARD - 1) // data.SHARD):
        parent = run.parent / "prior-test" if settings.get("reuse_prior_shard") and shard == 0 else run
        with (parent / "parts" / f"{shard:03}.jsonl").open() as handle:
            for row in map(json.loads, handle):
                if row["index"] != len(rows):
                    raise ValueError(f"Expected source row {len(rows)}, found {row['index']}")
                compact = {k: row[k] for k in ("problem", "solution", "original_trace")}
                for arm in data.ARMS:
                    result = row[arm]
                    compact[arm] = result["text"]
                    stat = summary[arm]
                    stat["changed"] += result["text"] != row["original_trace"]
                    stat["sum_percent"] += result["changed_percent"]
                    if result["loss_before"] is not None:
                        stat["paired"] += 1
                        stat["before"] += result["loss_before"]
                        stat["after"] += result["loss_after"]
                for turn in row["sentence"].get("rounds", []):
                    if turn.get("format_error"):
                        format_errors += 1
                    else:
                        interpreted += 1
                    ratings.update(t["semantic_rating"] for t in turn["proposals"] if "semantic_rating" in t)
                rows.append(compact)
    if len(rows) != settings["train"]:
        raise ValueError(f"Expected {settings['train']} rows, found {len(rows)}")
    if format_errors and not interpreted:
        raise ValueError("All rewriter rounds failed structured parsing; inspect the rewrite log")
    train = Dataset.from_list(rows)
    eval_data = data.load_eval(settings["profile"], settings["platinum_path"])
    for stat in summary.values():
        stat["mean_changed_percent_all_rows"] = stat.pop("sum_percent") / len(rows)
        for metric in ("before", "after"):
            total = stat.pop(metric)
            stat["mean_loss_" + metric] = total / stat["paired"] if stat["paired"] else None
    with tempfile.TemporaryDirectory(prefix="assemble-", dir=run) as temp:
        stage = Path(temp) / "datasets"
        stage.mkdir()
        common = ["problem", "solution", "original_trace"]
        train.select_columns(common).save_to_disk(str(stage / "original_traces"))
        shuffled = train.shuffle(seed=137, keep_in_memory=True)
        for arm in data.ARMS:
            arm_data = shuffled.select_columns(common + [arm]).rename_column(arm, "rewrite_trace")
            arm_data.save_to_disk(str(stage / arm))
        eval_data.save_to_disk(str(stage / "evaluation"))
        data.atomic_json(stage / "rewrite_summary.json", {
            "train": len(rows), "eval": len(eval_data), "arms": summary, "semantic_ratings": dict(ratings),
            "malformed_rounds": format_errors, "eval_questions_also_in_train":
                len(set(train["problem"]) & set(eval_data["problem"])),
        })
        stage.rename(dest)


def train_student(run, model_key, seed):
    from experiments.maxloss_v1.prepare_students import matched_config
    from experiments.maxloss_v1.train_transfer_students import run as train_matched

    settings = json.loads((run / "run.json").read_text())
    source = run / "datasets"
    cell = run / "students" / f"{model_key}-{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    if (cell / "af.json").is_file():
        return
    # Keep any interrupted attempt and publish the cell's metrics only after all three arms finish.
    work = Path(tempfile.mkdtemp(prefix="attempt-", dir=cell))
    models = Path(os.environ["HONOUR_SCRATCH"]) / "models"
    checkpoint, tok = (str(models / name) for name in data.STUDENTS[model_key])
    for name in ("original_traces", *data.ARMS):
        # The trainer writes below each arm directory. Give each job its own directory entries.
        shutil.copytree(source / name, work / name, copy_function=os.link)
    template = yaml.safe_load(data.TRAINING_CONFIG.read_text())
    cfg = matched_config(template, work, checkpoint, tok, seed)
    # The inherited candidate metadata describes the earlier word-only search.
    cfg = {key: value for key, value in cfg.items() if not key.startswith("stage6a_candidate_")}
    cfg.update(dataset_size_used_for_optimize=settings["train"], eval_dataset_size_for_optimize=settings["eval"],
               proxy_max_length=data.CAP, evaluation_dataset_path=str(source / "evaluation"),
               stage6a_protocol="sentence_maxloss_" + settings["profile"], stage6a_run_kind="sentence_comparison",
               stage6a_source_selection=settings["train_source"], stage6a_validation_selection=settings["eval_source"],
               trace_edit_policy="word_and_sentence_maxloss", rewriter_model_name=MODEL)
    (work / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    iteration = work / cfg["experiment_folder"]
    iteration.mkdir()
    labels = {"word": "Local word MaxLoss", "sentence": "Sentence-level MaxLoss"}
    candidates = [{"name": arm, "instruction": labels[arm], "score": None} for arm in data.ARMS]
    (iteration / "candidate_instructions.yaml").write_text(yaml.safe_dump({"candidate_instructions": candidates}))
    train_matched(argparse.Namespace(**cfg))
    scores = yaml.safe_load((iteration / "instructions_and_scores.yaml").read_text())["candidate_instructions"]
    result = {"model": model_key, "seed": seed, "train": settings["train"], "eval": settings["eval"]}
    for score in scores:
        metrics, = score["proxy_metrics"]
        if any(metrics[k]["num_examples"] != settings["eval"] for k in ("clean_evaluation", "candidate_evaluation")):
            raise ValueError("AF score used the wrong evaluation size")
        result["clean"] = metrics["clean_af_accuracy"]
        result[score["name"]] = metrics["candidate_af_accuracy"]
    if not all(arm in result for arm in data.ARMS):
        raise ValueError("Missing an experimental arm")
    for arm in data.ARMS:
        result["clean_minus_" + arm + "_pp"] = 100 * (result["clean"] - result[arm])
    result["scores_path"] = str(iteration / "instructions_and_scores.yaml")
    result["job"] = os.getenv("PBS_JOBID")
    data.atomic_json(cell / "af.json", result)


def summarize(run):
    rows = [json.loads((run / "students" / f"{model}-{seed}" / "af.json").read_text())
            for model in data.STUDENTS for seed in data.SEEDS]
    temp = run / "af.csv.tmp"
    with temp.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(run / "af.csv")
    print(run / "af.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "collect", "student", "summary", "pending-shards", "pending-students"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.stage == "init":
        pct = os.getenv("MAXLOSS_EDIT_PERCENT")
        init(run_dir.parent, run_dir.name, os.environ["SOURCE_COMMIT"], float(pct) if pct else None,
             os.getenv("PLATINUM_PATH") or None)
    elif args.stage == "pending-shards":
        n = data.SIZES[run_dir.name][0]
        first = int(json.loads((run_dir / "run.json").read_text()).get("reuse_prior_shard", False))
        print(" ".join(str(i) for i in range(first, (n + data.SHARD - 1) // data.SHARD)
                       if not (run_dir / "parts" / f"{i:03}.jsonl").exists()))
    elif args.stage == "pending-students":
        cells = ((m, s) for m in data.STUDENTS for s in data.SEEDS)
        print(" ".join(str(i) for i, (model, seed) in enumerate(cells)
                       if not (run_dir / "students" / f"{model}-{seed}" / "af.json").exists()))
    elif args.stage == "collect":
        collect(run_dir)
    elif args.stage == "student":
        model_index, seed_index = divmod(args.index, len(data.SEEDS))
        train_student(run_dir, list(data.STUDENTS)[model_index], data.SEEDS[seed_index])
    else:
        summarize(run_dir)
