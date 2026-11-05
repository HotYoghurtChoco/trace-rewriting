"""Prepare a seed-matched Clean / Optimized-Clean downstream comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]


def matched_config(template, workdir, base_model, tokenizer, seed):
    config = dict(template)
    config.update(
        seed=seed, working_dir=str(workdir), original_traces_path=str(workdir / "original_traces"),
        proxy_models=[{"name": base_model, "tokenizer": tokenizer}],
        stage6a_protocol=f"maxloss_v1_seed_{seed}",
        stage6a_run_kind="maxloss_v1",
        stage6a_source_selection="same_100_rows_with_matched_training_order",
        stage6a_candidate_generation="frozen_base_input_gradient_wording_edits",
        rewriter_model_name="local_gradient_wording_search", optimization_prompt="MaxLoss fixed-student completion loss.",
        dataset_size_used_for_optimize=100, eval_dataset_size_for_optimize=100,
        rescore=False, experiment_folder="iter_0", score_type="acc",
    )
    # Remove historical descriptions that refer to the old candidate source.
    for key in ("stage6a_reference_run", "stage6a_clean_source_path", "stage6a_candidate_source_path"):
        config.pop(key, None)
    return config


def run(args):
    from datasets import load_from_disk

    generation = args.generation_dir.resolve()
    summary = json.loads((generation / "summary.json").read_text())
    if summary["status"] != "complete" or summary["mode"] != "formal" or summary["row_indices"] != list(range(100)):
        raise ValueError("Student training requires the complete, ordered 100-row formal output")
    metadata = json.loads((generation / "run_config.json").read_text())
    config = metadata["config"]
    data = load_from_disk(str(generation / "optimized_dataset"))
    if len(data) != 100:
        raise ValueError("Expected 100 optimized rows")
    workdir = args.working_dir.resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    data.select_columns(["problem", "solution", "original_trace"]).save_to_disk(str(workdir / "original_traces"))
    # warmup_models shuffles Clean with seed 137. Apply the same permutation
    # to the candidate arm so both trainers see matching problem order.
    data.shuffle(seed=137).save_to_disk(str(workdir / "optimized_clean"))
    template = yaml.safe_load((REPO / config["training_config"]).read_text())
    training = matched_config(template, workdir, config["base_model"], config["tokenizer"], args.seed)
    (workdir / "config.yaml").write_text(yaml.safe_dump(training, sort_keys=False))
    (workdir / "iter_0").mkdir()
    candidates = {"candidate_instructions": [{"name": "optimized_clean", "instruction": "Input-gradient optimized Clean traces.", "score": None}]}
    (workdir / "iter_0/candidate_instructions.yaml").write_text(yaml.safe_dump(candidates, sort_keys=False))
    (workdir / "input_record.json").write_text(json.dumps({
        "generation_dir": str(generation), "seed": args.seed, "rows": 100,
        "training_order": "both arms use the same seed137 permutation before Trainer shuffling",
        "primary_contrast": "clean_af_minus_optimized_clean_af",
    }, indent=2) + "\n")
    print(f"Prepared two fresh students at {workdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=888)
    run(parser.parse_args())
