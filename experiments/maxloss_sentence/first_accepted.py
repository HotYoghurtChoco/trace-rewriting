"""Train one Qwen3-4B seed-889 student on the first accepted sentence per trace."""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import yaml


ARM = "first_sentence"
EVAL_DIR = "search_validation_evaluation"
EXAMPLES = "evaluated_examples.jsonl"


def first_accepted(row):
    original = row["original_trace"]
    for turn in row["sentence"].get("rounds", []):
        replacement = turn["accepted"]
        if replacement is None:
            continue
        start, source = turn["start"], turn["source"]
        end = start + len(source)
        if start < 0 or original[start:end] != source:
            raise ValueError(f"First accepted span differs from original on row {row['index']}")
        return original[:start] + replacement + original[end:]
    return original


def training_texts(records_path, prior_sentence_dataset):
    with records_path.open() as handle:
        records = {row["problem"]: row for row in map(json.loads, handle)}
    if len(records) != 100 or len(prior_sentence_dataset) != 100:
        raise ValueError("Expected 100 training questions")

    texts = []
    changed = different = 0
    for prior in prior_sentence_dataset:
        row = records.pop(prior["problem"])
        final = row["sentence"]["text"]
        if (row["original_trace"], final) != (prior["original_trace"], prior["rewrite_trace"]):
            raise ValueError(f"Rewrite history does not match the paired dataset on row {row['index']}")
        text = first_accepted(row)
        texts.append(text)
        changed += text != row["original_trace"]
        different += text != final
    return texts, {"one_sentence_changed": changed, "original_retained": len(texts) - changed,
                   "different_from_multi_sentence": different}


def whole_output_loop(trace):
    """Match a whole response repeating a 1-128 character period at least eight times."""
    text = "".join(trace.split())
    return any(text[p:] == text[:-p] for p in range(1, min(128, len(text) // 8) + 1))


def outcomes(path):
    with path.open() as handle:
        rows = {r["problem"]: (r["solution"], r["is_af_correct"], whole_output_loop(r["trace"]))
                for r in map(json.loads, handle)}
    if len(rows) != 100:
        raise ValueError(f"Expected 100 evaluation questions in {path}")
    return rows


def compare(previous, current):
    if previous.keys() != current.keys():
        raise ValueError("Evaluation questions differ between arms")
    gained = lost = recovered = 0
    for problem, (gold, was_correct, was_looping) in previous.items():
        new_gold, correct, looping = current[problem]
        if new_gold != gold:
            raise ValueError(f"Evaluation solution changed for {problem}")
        gained += correct and not was_correct
        lost += was_correct and not correct
        recovered += was_looping and not was_correct and not looping and correct
    result = {"evaluation_questions": len(current)}
    for name, rows in (("multi_sentence", previous), (ARM, current)):
        result[name] = {"af_correct": sum(correct for _, correct, _ in rows.values()),
                        "whole_output_loops": sum(loop for _, _, loop in rows.values())}
    result["paired"] = {"wrong_to_correct": gained, "correct_to_wrong": lost,
                        "looping_wrong_to_nonlooping_correct": recovered}
    return result


def train_one(cfg, work, prior_work, model):
    from experiments.maxloss_v1.train_transfer_students import run as train_matched
    from optimize import score_candidates as students

    # The normal trainer recognizes the completed Clean adapter and skips its training.
    clean_parent = prior_work / "clean_comparators"
    clean_metrics = yaml.safe_load((clean_parent / model / EVAL_DIR / "metrics.yaml").read_text())
    (work / "clean_comparators").symlink_to(clean_parent, target_is_directory=True)
    clean_model = work / "clean_comparators" / model
    evaluate = students.evaluate_candidate_af_accuracy

    def paired_evaluation(artifact_dir, *args, **kwargs):
        if Path(artifact_dir) == clean_model:
            return clean_metrics
        return evaluate(artifact_dir, *args, **kwargs)

    students.evaluate_candidate_af_accuracy = paired_evaluation
    try:
        train_matched(argparse.Namespace(**cfg))
    finally:
        students.evaluate_candidate_af_accuracy = evaluate


def run(prior, output):
    from datasets import load_from_disk
    from experiments.maxloss_sentence.data import atomic_json

    result_file = output / "comparison.json"
    if result_file.is_file():
        print(result_file.read_text())
        return
    reference = json.loads((prior / "students/qwen4b-889/af.json").read_text())
    prior_work = Path(reference["scores_path"]).parents[1]
    cfg = yaml.safe_load((prior_work / "config.yaml").read_text())
    if (cfg["seed"], cfg["proxy_max_length"]) != (889, 1280):
        raise ValueError("Expected the seed-889, 1280-token reference")
    proxy, = cfg["proxy_models"]
    model = Path(proxy["name"]).name
    prior_examples = prior_work / "sentence/finetuned_model" / model / EVAL_DIR / EXAMPLES
    previous = outcomes(prior_examples)

    dataset = load_from_disk(str(prior / "datasets/sentence"))
    texts, counts = training_texts(prior / "parts/000.jsonl", dataset)
    work = Path(tempfile.mkdtemp(prefix="attempt-", dir=output))
    print(f"Training inputs: {counts}\nWork directory: {work}", flush=True)
    # Keep the exact row order of the prior sentence arm, including its Clean-based batch grouping.
    dataset.remove_columns("rewrite_trace").add_column("rewrite_trace", texts).save_to_disk(str(work / ARM))
    shutil.copytree(prior / "datasets/original_traces", work / "original_traces")
    cfg.update(working_dir=str(work), original_traces_path=str(work / "original_traces"),
               trace_edit_policy="first_accepted_sentence", stage6a_run_kind="first_accepted_sentence")
    (work / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    iteration = work / cfg["experiment_folder"]
    iteration.mkdir()
    candidate = {"name": ARM, "instruction": "First accepted sentence-level MaxLoss edit", "score": None}
    (iteration / "candidate_instructions.yaml").write_text(yaml.safe_dump({"candidate_instructions": [candidate]}))
    train_one(cfg, work, prior_work, model)

    examples = work / ARM / "finetuned_model" / model / EVAL_DIR / EXAMPLES
    report = compare(previous, outcomes(examples))
    report.update(model=model, seed=cfg["seed"], training_inputs=counts,
                  multi_sentence_examples=str(prior_examples), first_sentence_examples=str(examples),
                  source_commit=os.environ.get("SOURCE_COMMIT"), job=os.environ.get("PBS_JOBID"))
    atomic_json(result_file, report)
    print(result_file.read_text(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.prior.resolve(), args.output.resolve())
