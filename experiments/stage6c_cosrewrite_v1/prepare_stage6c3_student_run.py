"""Prepare one matched Stage 6C.3 student-training run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

from datasets import Dataset, load_from_disk
import yaml


ARM_NAMES = ("baseline_rewrite", "selection_only", "cosrewrite")
INSTRUCTION = (
    "You are a math teacher. You will be given a math problem and you will "
    "solve it step by step. You will output your final solution like "
    "\\\\boxed{{ANSWER}}. Be sure to include relevant units within the brackets "
    "and fully evaluate arithmetic expressions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer", required=True)
    return parser.parse_args()


def atomic_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_dataset(path: Path, expected: Dataset) -> None:
    expected_rows = expected.to_list()
    if path.exists():
        actual = load_from_disk(str(path))
        if actual.to_list() != expected_rows:
            raise RuntimeError(f"Existing student input differs: {path}")
        return
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    expected.save_to_disk(str(temporary))
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    generation_dir = args.generation_dir.resolve()
    working_dir = args.working_dir.resolve()
    summary_path = generation_dir / "generation_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Generation summary is missing: {summary_path}")
    generation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if generation_summary.get("status") != "PASS":
        raise RuntimeError("Matched generation did not complete successfully")
    if int(generation_summary.get("row_count", -1)) != 100:
        raise RuntimeError("Matched generation does not contain 100 rows")

    source_datasets = {
        arm: load_from_disk(str(generation_dir / "datasets" / arm))
        for arm in ARM_NAMES
    }
    for arm, dataset in source_datasets.items():
        if len(dataset) != 100:
            raise RuntimeError(f"{arm} contains {len(dataset)} rows instead of 100")

    identity_columns = ("problem", "solution", "original_trace")
    baseline_rows = source_datasets["baseline_rewrite"]
    for row_index in range(100):
        expected = {
            column: baseline_rows[row_index][column]
            for column in identity_columns
        }
        for arm in ARM_NAMES[1:]:
            actual = {
                column: source_datasets[arm][row_index][column]
                for column in identity_columns
            }
            if actual != expected:
                raise RuntimeError(f"Matched-arm identity mismatch at row {row_index}")

    working_dir.mkdir(parents=True, exist_ok=True)
    original = baseline_rows.select_columns(list(identity_columns))
    save_dataset(working_dir / "original_traces", original)
    for arm, dataset in source_datasets.items():
        save_dataset(working_dir / arm, dataset)

    protocol = f"stage6c3_matched_students_v1_seed_{args.seed}"
    config = {
        "seed": args.seed,
        "original_traces_path": str(working_dir / "original_traces"),
        "dataset_size_used_for_optimize": 100,
        "eval_dataset_size_for_optimize": 100,
        "score_type": "acc",
        "batch_size_for_scoring": 1,
        "rescore": False,
        "stage6a_protocol": protocol,
        "rewriter_model_name": "openai/gpt-oss-120b",
        "rewrite_temperature": 0.6,
        "rewrite_max_tokens": 1024,
        "proxy_weight_decay": 0.1,
        "proxy_warmup_ratio": 0.1,
        "proxy_models": [
            {
                "name": args.base_model,
                "tokenizer": args.tokenizer,
            }
        ],
        "instruction_generation": INSTRUCTION,
        "rewrite_prompt_template": (
            "{instruction}\n\nOriginal trace: {original_trace}"
            "\n\nYour solution:\n\n"
        ),
        "optimization_prompt": "Stage 6C.3 matched downstream-student comparison.",
        "generation_prompt_template": (
            INSTRUCTION + "\n\nProblem: {problem}\n\nYour solution:"
        ),
        "working_dir": str(working_dir),
        "experiment_folder": "iter_0",
        "proxy_lora_r": 16,
        "proxy_lora_alpha": 16,
        "proxy_lora_dropout": 0.05,
        "proxy_per_device_train_batch_size": 4,
        "proxy_effective_batch_size": 16,
        "proxy_train_epochs": 2,
        "proxy_learning_rate": 0.0001,
        "stage6a_run_kind": "stage6c3_matched_students_v1",
        "stage6a_source_selection": "same_frozen_100_rows_as_stage6a",
        "stage6a_validation_selection": (
            "gsm8k_train_holdout_shuffle_seed_137_first_100"
        ),
        "stage6a_score_formula": "clean_af_minus_candidate_af",
        "stage6a_model_artifact": "adapter_only",
        "stage6a_evaluation_mode": "vllm_dynamic_lora",
        "stage6a_training_dtype": "bfloat16",
        "stage6a_inference_dtype": "bfloat16",
        "stage6c3_primary_contrast": "selection_only_af_minus_cosrewrite_af",
    }
    candidates = {
        "candidate_instructions": [
            {
                "name": arm,
                "instruction": f"Stage 6C.3 matched {arm} trace condition.",
                "score": None,
            }
            for arm in ARM_NAMES
        ]
    }
    atomic_yaml(working_dir / "config.yaml", config)
    atomic_yaml(working_dir / "iter_0" / "candidate_instructions.yaml", candidates)
    atomic_json(
        working_dir / "input_manifest.json",
        {
            "protocol": protocol,
            "seed": args.seed,
            "row_count": 100,
            "generation_dir": str(generation_dir),
            "arms": list(ARM_NAMES),
            "base_model": args.base_model,
            "tokenizer": args.tokenizer,
        },
    )
    print(f"Prepared Stage 6C.3 student run: seed={args.seed} root={working_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
