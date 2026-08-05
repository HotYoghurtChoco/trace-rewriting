#!/usr/bin/env python3
"""Diagnose BF16 gradient-dot reproducibility without updating parameters.

This is a technical diagnostic prompted by failed Stage 6B-3 smoke job
8924042.  It computes two reference-gradient passes and two candidate-gradient
passes, but it never applies a positive-eta update and never evaluates the
pre-registered one-step scientific endpoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import datasets
import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from optimize.score_candidates import (
    _load_training_tokenizer,
    _tokenize_completion_only_batch,
)


METHOD_VERSION = "stage6b_3_gradient_reproducibility_diagnostic_v1"
EXPECTED_BRANCH = "research/gradient-feedback"
LOCKED_EXECUTION_COMMIT = "a5f7ac3f17fc59854d57daa65e0ad669ae53b9b7"
LOCKED_VALIDATOR_SHA256 = (
    "a9545cecdd319fa0e418566d5c8881498ae0618803c30ed0240a6fd0c512a3e6"
)
FAILED_SMOKE_JOB_ID = "8924042.kman.restech.unsw.edu.au"
FAILED_SMOKE_OUTPUT = Path(
    "/srv/scratch/z5463756/honour/results/"
    "stage6b-gradient-feedback-qwen3-1.7b-v1/one_step_validation/"
    "smoke/stage6b_3a_8924042"
)
REPEATS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-diagnostic-commit", required=True)
    return parser.parse_args()


def load_locked_module(path: Path):
    spec = importlib.util.spec_from_file_location("stage6b3_locked_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load locked validator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    temporary = path.with_name(path.name + ".incomplete")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    ).stdout.strip()


def quantiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("quantile input must be a non-empty finite vector")
    return {
        "min": float(np.min(array)),
        "q25": float(np.percentile(array, 25, method="linear")),
        "median": float(np.percentile(array, 50, method="linear")),
        "q75": float(np.percentile(array, 75, method="linear")),
        "q95": float(np.percentile(array, 95, method="linear")),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def cosine(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    if left.keys() != right.keys():
        raise RuntimeError("Gradient key mismatch")
    dot = torch.zeros((), device="cuda", dtype=torch.float32)
    left_squared = torch.zeros((), device="cuda", dtype=torch.float32)
    right_squared = torch.zeros((), device="cuda", dtype=torch.float32)
    for name in left:
        dot.add_((left[name] * right[name]).sum())
        left_squared.add_(left[name].square().sum())
        right_squared.add_(right[name].square().sum())
    denominator = math.sqrt(float(left_squared.item())) * math.sqrt(
        float(right_squared.item())
    )
    if denominator == 0.0:
        raise RuntimeError("Gradient cosine has zero denominator")
    value = float(dot.item()) / denominator
    if not math.isfinite(value):
        raise RuntimeError("Gradient cosine is non-finite")
    return max(-1.0, min(1.0, value))


def gradient_l2_distance(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    if left.keys() != right.keys():
        raise RuntimeError("Gradient key mismatch")
    squared = torch.zeros((), device="cuda", dtype=torch.float32)
    for name in left:
        squared.add_((left[name] - right[name]).square().sum())
    return math.sqrt(float(squared.item()))


def gradient_norm(gradient: Mapping[str, torch.Tensor]) -> float:
    squared = torch.zeros((), device="cuda", dtype=torch.float32)
    for value in gradient.values():
        squared.add_(value.square().sum())
    result = math.sqrt(float(squared.item()))
    if result <= 0.0 or not math.isfinite(result):
        raise RuntimeError("Gradient norm is not finite and positive")
    return result


def gradient_dot(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    if left.keys() != right.keys():
        raise RuntimeError("Gradient key mismatch")
    value = torch.zeros((), device="cuda", dtype=torch.float32)
    for name in left:
        value.add_((left[name] * right[name]).sum())
    result = float(value.item())
    if not math.isfinite(result):
        raise RuntimeError("Gradient dot is non-finite")
    return result


def rank_top50(values: Sequence[float]) -> set[int]:
    if len(values) != 100:
        raise ValueError("Top-50 diagnostic requires exactly 100 values")
    ranked = sorted(range(100), key=lambda index: (-values[index], index))
    return set(ranked[:50])


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    # The locked PBS launcher creates output_dir/provenance before execution.
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden_existing = (
        output_dir / "diagnostic_summary.json",
        output_dir / "candidate_reproducibility.jsonl",
        output_dir / "reference_reproducibility.jsonl",
        output_dir / "COMPLETE_DIAGNOSTIC_NOT_FORMAL",
    )
    if any(path.exists() for path in forbidden_existing):
        raise RuntimeError("Diagnostic output directory contains prior results")

    validator_path = (
        repo_root
        / "experiments/stage6b_3_one_step_validation_v1/validate_one_step.py"
    )
    locked = load_locked_module(validator_path)

    actual_branch = git_output(repo_root, "branch", "--show-current")
    actual_commit = git_output(repo_root, "rev-parse", "HEAD")
    if actual_branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Unexpected branch: {actual_branch}")
    if actual_commit != args.expected_diagnostic_commit:
        raise RuntimeError(
            "HEAD does not match the qsub-locked diagnostic commit: "
            f"{actual_commit}"
        )
    ancestor_check = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            LOCKED_EXECUTION_COMMIT,
            actual_commit,
        ],
        check=False,
    )
    if ancestor_check.returncode != 0:
        raise RuntimeError("Locked Stage 6B-3 execution commit is not an ancestor")
    if git_output(repo_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked or staged repository changes exist")
    if locked.file_sha256(validator_path) != LOCKED_VALIDATOR_SHA256:
        raise RuntimeError("Locked validator SHA256 mismatch")

    formal_root = locked.EXPECTED_PATHS["formal_root"]
    stage6b_root = locked.EXPECTED_PATHS["stage6b_root"]
    prereg_path = (
        repo_root
        / "experiments/stage6b_3_one_step_validation_v1/PRE_REGISTRATION.md"
    )
    score_helper_path = repo_root / "optimize/score_candidates.py"
    score_dir = stage6b_root / "scoring/stage6b_1c_8918577"
    score_csv_path = score_dir / "stage6b_1c_candidate_scores.csv"
    score_summary_path = score_dir / "stage6b_1c_summary.json"
    method_lock_path = stage6b_root / "provenance/stage6b_1c_method_lock.yaml"
    preflight_path = stage6b_root / "provenance/stage6b_1c_preflight.json"
    frozen_dir = stage6b_root / "frozen_data"
    reference_path = frozen_dir / "gradient_reference.jsonl"
    reference_manifest_path = frozen_dir / "gradient_reference_manifest.json"
    frozen_sums_path = frozen_dir / "SHA256SUMS"
    config_path = formal_root / "provenance/config.yaml"
    candidate_yaml_path = formal_root / "iter_0/candidate_instructions.yaml"

    required_hashes = {
        prereg_path: locked.EXPECTED_SHA256["pre_registration"],
        score_helper_path: locked.EXPECTED_SHA256["score_candidates"],
        score_csv_path: locked.EXPECTED_SHA256["score_csv"],
        score_summary_path: locked.EXPECTED_SHA256["score_summary"],
        method_lock_path: locked.EXPECTED_SHA256["method_lock"],
        preflight_path: locked.EXPECTED_SHA256["preflight"],
        reference_path: locked.EXPECTED_SHA256["reference"],
        reference_manifest_path: locked.EXPECTED_SHA256["reference_manifest"],
        frozen_sums_path: locked.EXPECTED_SHA256["frozen_sums"],
    }
    for path, expected in required_hashes.items():
        locked.require_sha(path, expected, path.name)

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with candidate_yaml_path.open(encoding="utf-8") as handle:
        candidate_yaml = yaml.safe_load(handle)
    with method_lock_path.open(encoding="utf-8") as handle:
        method_lock = yaml.safe_load(handle)
    with preflight_path.open(encoding="utf-8") as handle:
        preflight = json.load(handle)
    with score_summary_path.open(encoding="utf-8") as handle:
        frozen_summary = json.load(handle)
    reference_rows = locked.load_jsonl(reference_path)
    with score_csv_path.open(encoding="utf-8", newline="") as handle:
        score_rows = list(csv.DictReader(handle))

    if method_lock.get("status") != "LOCKED_BEFORE_FULL_SCORE_DISTRIBUTION":
        raise RuntimeError("Stage 6B-1 method lock is not valid")
    if preflight.get("status") != "PASS" or preflight.get("scores_computed") is not False:
        raise RuntimeError("Stage 6B-1 preflight is not valid")
    if int(config["seed"]) != locked.SCORING_SEED:
        raise RuntimeError("Unexpected scoring seed")

    model_name = config["proxy_models"][0]["name"]
    tokenizer_name = config["proxy_models"][0]["tokenizer"]
    instruction = config["instruction_generation"]
    candidate_name = candidate_yaml["candidate_instructions"][0]["name"]
    candidate_path = formal_root / candidate_name
    adapter_path = (
        candidate_path / "finetuned_model" / Path(model_name).name / "adapter"
    )
    locked.require_sha(
        adapter_path / "adapter_config.json",
        locked.EXPECTED_SHA256["adapter_config"],
        "adapter_config",
    )
    locked.require_sha(
        adapter_path / "adapter_model.safetensors",
        locked.EXPECTED_SHA256["adapter_model"],
        "adapter_model",
    )

    candidate_dataset = datasets.load_from_disk(str(candidate_path))
    candidate_rows = [candidate_dataset[index] for index in range(len(candidate_dataset))]
    if len(candidate_rows) != 100 or len(reference_rows) != 100 or len(score_rows) != 100:
        raise RuntimeError("Candidate, reference, and score counts must all be 100")
    if [int(row["candidate_index"]) for row in score_rows] != list(range(100)):
        raise RuntimeError("Frozen candidate score order is not 0..99")
    numeric_score_fields = (
        "gradient_cosine",
        "gradient_dot",
        "candidate_loss",
        "candidate_gradient_norm",
    )
    for index, row in enumerate(score_rows):
        for field in numeric_score_fields:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(
                    f"Non-finite frozen score: candidate={index}, field={field}"
                )
    if sorted(int(row["gradient_rank"]) for row in score_rows) != list(range(1, 101)):
        raise RuntimeError("Frozen candidate ranks are not exactly 1..100")
    for index, (candidate_row, score_row) in enumerate(
        zip(candidate_rows, score_rows, strict=True)
    ):
        if locked.text_sha256(candidate_row["problem"]) != score_row["problem_sha256"]:
            raise RuntimeError(f"Candidate identity mismatch at index {index}")
    for index, row in enumerate(reference_rows):
        if int(row["gradient_reference_index"]) != index:
            raise RuntimeError(f"Reference order mismatch at index {index}")
        if locked.text_sha256(row["problem"]) != row["problem_sha256"]:
            raise RuntimeError(f"Reference identity mismatch at index {index}")
    if locked.file_sha256(config_path) != preflight["config_sha256"]:
        raise RuntimeError("Config/preflight mismatch")
    if locked.file_sha256(candidate_yaml_path) != preflight["candidate_yaml_sha256"]:
        raise RuntimeError("Candidate-YAML/preflight mismatch")
    if locked.ordered_rows_sha256(candidate_rows, "rewrite_trace") != preflight[
        "candidate_ordered_rows_sha256"
    ]:
        raise RuntimeError("Candidate ordered-row mismatch")
    if locked.ordered_rows_sha256(reference_rows, "solution") != preflight[
        "reference_ordered_rows_sha256"
    ]:
        raise RuntimeError("Reference ordered-row mismatch")
    if locked.adapter_file_records(adapter_path) != preflight["adapter_files"]:
        raise RuntimeError("Adapter inventory mismatch")

    random.seed(locked.SCORING_SEED)
    np.random.seed(locked.SCORING_SEED)
    torch.manual_seed(locked.SCORING_SEED)
    torch.cuda.manual_seed_all(locked.SCORING_SEED)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")

    device = torch.device("cuda")
    compute_dtype = torch.bfloat16
    tokenizer = _load_training_tokenizer(tokenizer_name)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer special tokens are incomplete")
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    if len(trainable) != locked.EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError("Unexpected trainable tensor count")
    if sum(parameter.numel() for _, parameter in trainable) != (
        locked.EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("Unexpected trainable parameter count")
    if not all("lora_" in name for name, _ in trainable):
        raise RuntimeError("A non-LoRA parameter is trainable")

    trainable_sha_before = locked.named_parameter_sha256(trainable)
    frozen_sha_before = locked.named_parameter_sha256(frozen)
    trainable_versions_before = {name: parameter._version for name, parameter in trainable}
    frozen_versions_before = {name: parameter._version for name, parameter in frozen}

    def build_feature(problem: str, response: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        encoded = _tokenize_completion_only_batch(
            tokenizer=tokenizer,
            instruction=instruction,
            problems=[problem],
            responses=[response],
            model_name=model_name,
        )
        full_ids = encoded["input_ids"][0]
        full_mask = encoded["completion_mask"][0]
        kept_ids = full_ids[: locked.MAX_LENGTH]
        kept_mask = full_mask[: locked.MAX_LENGTH]
        if len(full_ids) != len(full_mask) or sum(kept_mask) <= 0:
            raise RuntimeError("Invalid token/mask feature")
        metadata = {
            "total_tokens_before_truncation": len(full_ids),
            "completion_tokens_before_truncation": sum(full_mask),
            "completion_tokens_kept": sum(kept_mask),
            "completion_tokens_removed": sum(full_mask[locked.MAX_LENGTH :]),
            "eos_in_kept_completion": any(
                token_id == tokenizer.eos_token_id and mask_value == 1
                for token_id, mask_value in zip(kept_ids, kept_mask, strict=True)
            ),
        }
        return {"input_ids": kept_ids, "completion_mask": kept_mask}, metadata

    def model_batch(feature: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        collated = collator([dict(feature)])
        labels = collated["labels"][0]
        if int((labels != -100).sum().item()) != sum(feature["completion_mask"]):
            raise RuntimeError("Completion-only label mask mismatch")
        return {
            key: value.to(device)
            for key, value in collated.items()
            if key in {"input_ids", "attention_mask", "labels", "position_ids"}
        }

    candidate_features = []
    candidate_metadata = []
    for row in candidate_rows:
        feature, metadata = build_feature(row["problem"], row["rewrite_trace"])
        candidate_features.append(feature)
        candidate_metadata.append(metadata)
    reference_features = []
    reference_metadata = []
    for row in reference_rows:
        feature, metadata = build_feature(row["problem"], row["solution"])
        reference_features.append(feature)
        reference_metadata.append(metadata)
    if any(item["completion_tokens_removed"] for item in candidate_metadata):
        raise RuntimeError("A candidate completion was truncated")
    if any(item["completion_tokens_removed"] for item in reference_metadata):
        raise RuntimeError("A reference completion was truncated")

    def gradient_snapshot(feature: Mapping[str, Any]) -> Tuple[float, Dict[str, torch.Tensor]]:
        model.zero_grad(set_to_none=True)
        batch = model_batch(feature)
        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            output = model(**batch)
            loss = output.loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss")
        loss.backward()
        torch.cuda.synchronize()
        gradients = {}
        for name, parameter in trainable:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError(f"Invalid gradient for {name}")
            gradients[name] = parameter.grad.detach().float().clone()
        result = float(loss.detach().cpu())
        del batch, output, loss
        return result, gradients

    def reference_pass(pass_index: int) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        mean_gradient = {
            name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
            for name, parameter in trainable
        }
        rows = []
        for index, (source, feature, metadata) in enumerate(
            zip(reference_rows, reference_features, reference_metadata, strict=True)
        ):
            loss, gradient = gradient_snapshot(feature)
            norm = gradient_norm(gradient)
            for name in mean_gradient:
                mean_gradient[name].add_(gradient[name])
            rows.append(
                {
                    "pass_index": pass_index,
                    "reference_index": index,
                    "problem_sha256": source["problem_sha256"],
                    "loss": loss,
                    "gradient_norm": norm,
                    "tokens": metadata,
                }
            )
            del gradient
            if (index + 1) % 10 == 0:
                print(f"reference_pass_{pass_index}_progress: {index + 1}/100")
        for value in mean_gradient.values():
            value.div_(100)
        return rows, mean_gradient

    print("===== REFERENCE REPRODUCIBILITY =====")
    reference_rows_pass1, reference_gradient_pass1 = reference_pass(1)
    reference_rows_pass2, reference_gradient_pass2 = reference_pass(2)
    reference_norm1 = gradient_norm(reference_gradient_pass1)
    reference_norm2 = gradient_norm(reference_gradient_pass2)
    reference_cosine = cosine(reference_gradient_pass1, reference_gradient_pass2)
    reference_distance = gradient_l2_distance(
        reference_gradient_pass1, reference_gradient_pass2
    )
    print(f"reference_norm_pass1: {reference_norm1:.17g}")
    print(f"reference_norm_pass2: {reference_norm2:.17g}")
    print(f"reference_repeat_cosine: {reference_cosine:.17g}")
    print(f"reference_l2_distance: {reference_distance:.17g}")

    print("===== CANDIDATE REPRODUCIBILITY =====")
    candidate_diagnostics = []
    for index, (source, feature, metadata, score_row) in enumerate(
        zip(
            candidate_rows,
            candidate_features,
            candidate_metadata,
            score_rows,
            strict=True,
        )
    ):
        loss1, gradient1 = gradient_snapshot(feature)
        loss2, gradient2 = gradient_snapshot(feature)
        norm1 = gradient_norm(gradient1)
        norm2 = gradient_norm(gradient2)
        frozen_dot = float(score_row["gradient_dot"])
        frozen_norm = float(score_row["candidate_gradient_norm"])
        prereg_tolerance = max(1e-6, 1e-4 * abs(frozen_dot))
        dots = {
            "candidate_pass1_reference_pass1": gradient_dot(
                gradient1, reference_gradient_pass1
            ),
            "candidate_pass1_reference_pass2": gradient_dot(
                gradient1, reference_gradient_pass2
            ),
            "candidate_pass2_reference_pass1": gradient_dot(
                gradient2, reference_gradient_pass1
            ),
            "candidate_pass2_reference_pass2": gradient_dot(
                gradient2, reference_gradient_pass2
            ),
        }
        errors = {name: abs(value - frozen_dot) for name, value in dots.items()}
        normalized_errors = {
            name: error / (frozen_norm * float(frozen_summary["mean_reference_gradient_norm"]))
            for name, error in errors.items()
        }
        candidate_diagnostics.append(
            {
                "candidate_index": index,
                "problem_sha256": score_row["problem_sha256"],
                "gradient_rank": int(score_row["gradient_rank"]),
                "gradient_group": score_row["gradient_group"],
                "tokens": metadata,
                "frozen_candidate_loss": float(score_row["candidate_loss"]),
                "candidate_loss_pass1": loss1,
                "candidate_loss_pass2": loss2,
                "candidate_loss_repeat_absolute_difference": abs(loss1 - loss2),
                "frozen_candidate_gradient_norm": frozen_norm,
                "candidate_gradient_norm_pass1": norm1,
                "candidate_gradient_norm_pass2": norm2,
                "candidate_gradient_repeat_cosine": cosine(gradient1, gradient2),
                "candidate_gradient_repeat_l2_distance": gradient_l2_distance(
                    gradient1, gradient2
                ),
                "frozen_gradient_dot": frozen_dot,
                "frozen_gradient_cosine": float(score_row["gradient_cosine"]),
                "preregistered_gradient_dot_tolerance": prereg_tolerance,
                "recomputed_gradient_dots": dots,
                "absolute_errors_from_frozen_dot": errors,
                "normalized_errors_by_frozen_norm_product": normalized_errors,
                "within_preregistered_tolerance": {
                    name: error <= prereg_tolerance for name, error in errors.items()
                },
            }
        )
        del gradient1, gradient2
        torch.cuda.empty_cache()
        if (index + 1) % 10 == 0:
            print(f"candidate_progress: {index + 1}/100")

    combinations = (
        "candidate_pass1_reference_pass1",
        "candidate_pass1_reference_pass2",
        "candidate_pass2_reference_pass1",
        "candidate_pass2_reference_pass2",
    )
    frozen_dots = [float(row["gradient_dot"]) for row in score_rows]
    frozen_top50 = rank_top50(frozen_dots)
    combination_summary = {}
    for name in combinations:
        recomputed = [
            row["recomputed_gradient_dots"][name] for row in candidate_diagnostics
        ]
        errors = [
            row["absolute_errors_from_frozen_dot"][name]
            for row in candidate_diagnostics
        ]
        normalized_errors = [
            row["normalized_errors_by_frozen_norm_product"][name]
            for row in candidate_diagnostics
        ]
        top50 = rank_top50(recomputed)
        combination_summary[name] = {
            "pearson_frozen_vs_recomputed": locked.pearson_correlation(
                frozen_dots, recomputed
            ),
            "spearman_frozen_vs_recomputed": locked.spearman_correlation(
                frozen_dots, recomputed
            ),
            "sign_agreement_count": sum(
                int(np.sign(frozen) == np.sign(current))
                for frozen, current in zip(frozen_dots, recomputed, strict=True)
            ),
            "preregistered_tolerance_pass_count": sum(
                int(error <= max(1e-6, 1e-4 * abs(frozen)))
                for frozen, error in zip(frozen_dots, errors, strict=True)
            ),
            "preregistered_tolerance_failure_count": sum(
                int(error > max(1e-6, 1e-4 * abs(frozen)))
                for frozen, error in zip(frozen_dots, errors, strict=True)
            ),
            "absolute_error": quantiles(errors),
            "normalized_error_by_frozen_norm_product": quantiles(normalized_errors),
            "top50_overlap_count": len(frozen_top50 & top50),
            "top50_changed_membership_count": len(frozen_top50 ^ top50) // 2,
        }

    trainable_sha_after = locked.named_parameter_sha256(trainable)
    frozen_sha_after = locked.named_parameter_sha256(frozen)
    if trainable_sha_after != trainable_sha_before:
        raise RuntimeError("Trainable parameters changed during diagnostic")
    if frozen_sha_after != frozen_sha_before:
        raise RuntimeError("Frozen parameters changed during diagnostic")
    if any(
        parameter._version != trainable_versions_before[name]
        for name, parameter in trainable
    ):
        raise RuntimeError("Trainable parameter version changed during diagnostic")
    if any(
        parameter._version != frozen_versions_before[name]
        for name, parameter in frozen
    ):
        raise RuntimeError("Frozen parameter version changed during diagnostic")

    reference_loss_pass1 = [row["loss"] for row in reference_rows_pass1]
    reference_loss_pass2 = [row["loss"] for row in reference_rows_pass2]
    summary = {
        "status": "DIAGNOSTIC_COMPLETE_NOT_FORMAL",
        "method_version": METHOD_VERSION,
        "failed_smoke_job_id": FAILED_SMOKE_JOB_ID,
        "failed_smoke_output": str(FAILED_SMOKE_OUTPUT),
        "scientific_endpoint_evaluated": False,
        "positive_eta_update_performed": False,
        "optimizer_created": False,
        "candidate_count": len(candidate_diagnostics),
        "reference_count": len(reference_rows_pass1),
        "reference_pass_count": REPEATS,
        "candidate_pass_count": REPEATS,
        "frozen_reference_loss_mean": float(frozen_summary["reference_loss_mean"]),
        "frozen_mean_reference_gradient_norm": float(
            frozen_summary["mean_reference_gradient_norm"]
        ),
        "reference_reproducibility": {
            "loss_mean_pass1": statistics.fmean(reference_loss_pass1),
            "loss_mean_pass2": statistics.fmean(reference_loss_pass2),
            "loss_max_absolute_repeat_difference": max(
                abs(left - right)
                for left, right in zip(
                    reference_loss_pass1, reference_loss_pass2, strict=True
                )
            ),
            "mean_gradient_norm_pass1": reference_norm1,
            "mean_gradient_norm_pass2": reference_norm2,
            "mean_gradient_repeat_cosine": reference_cosine,
            "mean_gradient_l2_distance": reference_distance,
            "mean_gradient_l2_distance_normalized_by_pass1_norm": (
                reference_distance / reference_norm1
            ),
        },
        "candidate_reproducibility": {
            "loss_repeat_absolute_difference": quantiles(
                [
                    row["candidate_loss_repeat_absolute_difference"]
                    for row in candidate_diagnostics
                ]
            ),
            "gradient_repeat_cosine": quantiles(
                [
                    row["candidate_gradient_repeat_cosine"]
                    for row in candidate_diagnostics
                ]
            ),
        },
        "frozen_dot_comparisons": combination_summary,
        "parameter_integrity": {
            "trainable_sha256_before": trainable_sha_before,
            "trainable_sha256_after": trainable_sha_after,
            "frozen_sha256_before": frozen_sha_before,
            "frozen_sha256_after": frozen_sha_after,
            "status": "PASS_NO_PARAMETER_UPDATE",
        },
    }

    run_config = {
        "method_version": METHOD_VERSION,
        "branch": actual_branch,
        "diagnostic_commit": actual_commit,
        "locked_execution_commit": LOCKED_EXECUTION_COMMIT,
        "locked_validator_sha256": LOCKED_VALIDATOR_SHA256,
        "pre_registration_commit": locked.EXPECTED_PREREG_COMMIT,
        "scoring_seed": locked.SCORING_SEED,
        "compute_dtype": str(compute_dtype),
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "adapter_path": str(adapter_path),
        "candidate_count": 100,
        "reference_count": 100,
        "reference_pass_count": REPEATS,
        "candidate_pass_count": REPEATS,
        "eta_values": [],
        "manual_update_implemented": False,
        "optimizer_implemented": False,
        "purpose": (
            "Quantify numerical reproducibility after failed smoke 8924042; "
            "do not evaluate one-step scientific outcomes"
        ),
    }

    reference_output_rows = reference_rows_pass1 + reference_rows_pass2
    reference_count = atomic_jsonl(
        output_dir / "reference_reproducibility.jsonl", reference_output_rows
    )
    candidate_count = atomic_jsonl(
        output_dir / "candidate_reproducibility.jsonl", candidate_diagnostics
    )
    if reference_count != 200 or candidate_count != 100:
        raise RuntimeError("Diagnostic output row count mismatch")
    atomic_json(output_dir / "diagnostic_run_config.json", run_config)
    atomic_json(output_dir / "diagnostic_summary.json", summary)

    print("===== DIAGNOSTIC SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("DIAGNOSTIC_COMPLETE_NOT_FORMAL")


if __name__ == "__main__":
    main()
