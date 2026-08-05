#!/usr/bin/env python3
"""Validate Stage 6B-1 gradient predictions with one manual SGD step.

This executable implements the protocol frozen in PRE_REGISTRATION.md.  It
does not train a student, run OPRO, or reinterpret the Stage 6B-2 result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
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


METHOD_VERSION = "stage6b_3_one_step_validation_v1"
EXPECTED_BRANCH = "research/gradient-feedback"
EXPECTED_PREREG_COMMIT = "7e2f8221d3e51e28959899b28aa37b3de896868e"

EXPECTED_PATHS = {
    "formal_root": Path(
        "/srv/scratch/z5463756/honour/results/"
        "stage6a-formal-pilot-qwen3-1.7b-v1"
    ),
    "stage6b_root": Path(
        "/srv/scratch/z5463756/honour/results/"
        "stage6b-gradient-feedback-qwen3-1.7b-v1"
    ),
}

EXPECTED_SHA256 = {
    "pre_registration": (
        "edff3069693faef03e219078f24a1a5694071fbd8b8fb70cb3756256642a0c53"
    ),
    "score_csv": (
        "e324e84644fde5974f9f7399338b62c629fa7ec374389f297129fb2811026efa"
    ),
    "score_summary": (
        "8d00ae1e5997005e60426e3ef97133531f2ebbc2fbf77deadd4db588b6dfbee0"
    ),
    "method_lock": (
        "c366851cb049882a5b642cc08bff22e0fef087efa2cee934b6f02bbc5185a337"
    ),
    "preflight": (
        "0dbc7296aff3ac48f200e40902e634cb3413cb26b05f48fedbb1a191fb568cdf"
    ),
    "reference": (
        "517161b29ce6eba59dd7bbd7519670fde22e3d7f7258b00700dfd7939eae166d"
    ),
    "reference_manifest": (
        "7b32ac84b9a0f97a6135e5e197c33bac534ab67fa2aaa0528182be05d87124f1"
    ),
    "frozen_sums": (
        "3ce595a067f8f4d895bdb5b20a1e0b4cc58415cc542b481f786a7f665eb2b5ea"
    ),
    "adapter_config": (
        "fa981a6e6d4a88c8672e6816aed76f30ca3bced2ade2a147e3f81ad32a92a95d"
    ),
    "adapter_model": (
        "bf31d1b724b94aeaaed8ff90c1142884e89496264ac6225527321a9d46545a2e"
    ),
    "score_candidates": (
        "2035787b8b7d4e64ff166893cebabcf506facef3c449fcc7a7f927c514596c67"
    ),
}

ETA_GRID = (0.0, 1e-4, 5e-4, 1e-3, 5e-3)
PRIMARY_ETA = 5e-4
SCORING_SEED = 888
STATISTICAL_SEED = 42
PERMUTATIONS = 100_000
BOOTSTRAP_RESAMPLES = 10_000
MAX_LENGTH = 1024
EXPECTED_CANDIDATE_COUNT = 100
EXPECTED_REFERENCE_COUNT = 100
EXPECTED_TRAINABLE_TENSORS = 392
EXPECTED_TRAINABLE_PARAMETERS = 17_432_576


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("smoke", "formal"))
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-execution-commit", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, found {actual}"
        )
    print(f"{label}_sha256: {actual}")
    return actual


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    ).stdout.strip()


def ordered_rows_sha256(
    rows: Sequence[Mapping[str, Any]],
    response_key: str,
) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        record = {
            "index": index,
            "problem": row["problem"],
            response_key: row[response_key],
        }
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def adapter_file_records(adapter_path: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(adapter_path.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "relative_path": str(path.relative_to(adapter_path)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return records


def named_parameter_sha256(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
) -> str:
    """Hash parameter metadata and exact bytes without retaining CPU copies."""
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        metadata = {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        digest.update(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
        cpu_tensor = parameter.detach().contiguous().view(torch.uint8).cpu()
        digest.update(cpu_tensor.numpy().tobytes(order="C"))
        del cpu_tensor
    return digest.hexdigest()


def rankdata_average(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("rankdata input must be a finite one-dimensional array")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def pearson_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("correlation inputs must be equal one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("correlation inputs must be finite")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(
        np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    )
    if denominator == 0.0:
        raise ValueError("correlation is undefined for a constant input")
    value = float(np.dot(x_centered, y_centered) / denominator)
    if not math.isfinite(value):
        raise ValueError("correlation is not finite")
    return max(-1.0, min(1.0, value))


def spearman_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return pearson_correlation(rankdata_average(left), rankdata_average(right))


def permutation_test(
    predicted: Sequence[float],
    actual: Sequence[float],
) -> Dict[str, Any]:
    x_rank = rankdata_average(predicted)
    y_rank = rankdata_average(actual)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(
        np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    )
    if denominator == 0.0:
        raise ValueError("permutation statistic is undefined")
    observed = max(
        -1.0,
        min(1.0, float(np.dot(x_centered, y_centered) / denominator)),
    )
    rng = np.random.default_rng(STATISTICAL_SEED)
    upper_count = 0
    lower_count = 0
    for _ in range(PERMUTATIONS):
        permuted = y_centered[rng.permutation(len(y_centered))]
        rho = max(
            -1.0,
            min(1.0, float(np.dot(x_centered, permuted) / denominator)),
        )
        upper_count += rho >= observed
        lower_count += rho <= observed
    return {
        "observed_spearman_rho": observed,
        "positive_tail_count": upper_count,
        "positive_tail_p": (1 + upper_count) / (PERMUTATIONS + 1),
        "negative_tail_count": lower_count,
        "negative_tail_p": (1 + lower_count) / (PERMUTATIONS + 1),
        "permutations": PERMUTATIONS,
        "seed": STATISTICAL_SEED,
        "rng": "numpy.random.Generator(PCG64)",
    }


def bootstrap_spearman_ci(
    predicted: Sequence[float],
    actual: Sequence[float],
) -> Dict[str, Any]:
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    rng = np.random.default_rng(STATISTICAL_SEED)
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, len(x), size=len(x))
        values[index] = spearman_correlation(x[sample], y[sample])
    if not np.isfinite(values).all():
        raise ValueError("bootstrap produced a non-finite statistic")
    lower, upper = np.percentile(values, [2.5, 97.5], method="linear")
    return {
        "confidence_level": 0.95,
        "method": "percentile",
        "percentile_method": "linear",
        "lower": float(lower),
        "upper": float(upper),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": STATISTICAL_SEED,
        "rng": "numpy.random.Generator(PCG64)",
    }


def ols_calibration(
    predicted: Sequence[float],
    actual: Sequence[float],
) -> Dict[str, float]:
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual_sum = float(np.square(y - fitted).sum())
    total_sum = float(np.square(y - y.mean()).sum())
    if total_sum == 0.0:
        raise ValueError("OLS R-squared is undefined for constant outcomes")
    return {
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "r_squared": 1.0 - residual_sum / total_sum,
        "theoretical_intercept": 0.0,
        "theoretical_slope": 1.0,
    }


def sign_metrics(
    predicted: Sequence[float],
    actual: Sequence[float],
) -> Dict[str, Any]:
    predicted_sign = np.sign(np.asarray(predicted, dtype=np.float64)).astype(int)
    actual_sign = np.sign(np.asarray(actual, dtype=np.float64)).astype(int)
    labels = (-1, 0, 1)
    matrix = {
        str(actual_label): {
            str(predicted_label): int(
                np.sum(
                    (actual_sign == actual_label)
                    & (predicted_sign == predicted_label)
                )
            )
            for predicted_label in labels
        }
        for actual_label in labels
    }
    recalls = []
    for label in labels:
        support = int(np.sum(actual_sign == label))
        if support:
            recalls.append(float(np.mean(predicted_sign[actual_sign == label] == label)))
    if not recalls:
        raise ValueError("balanced sign accuracy has no supported classes")
    return {
        "sign_definition": "exact numpy.sign; no post-hoc tolerance",
        "agreement": float(np.mean(predicted_sign == actual_sign)),
        "balanced_accuracy": statistics.fmean(recalls),
        "confusion_matrix_actual_rows_predicted_columns": matrix,
    }


def error_metrics(
    predicted: Sequence[float],
    actual: Sequence[float],
) -> Dict[str, float]:
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(
        predicted, dtype=np.float64
    )
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(math.sqrt(float(np.mean(np.square(difference))))),
    }


def partial_rank_correlation(
    predicted: Sequence[float],
    actual: Sequence[float],
    covariates: Sequence[Sequence[float]],
) -> float:
    x_rank = rankdata_average(predicted)
    y_rank = rankdata_average(actual)
    columns = [
        np.ones(len(x_rank), dtype=np.float64),
        *[rankdata_average(values) for values in covariates],
    ]
    design = np.column_stack(columns)
    x_fit, _, _, _ = np.linalg.lstsq(design, x_rank, rcond=None)
    y_fit, _, _, _ = np.linalg.lstsq(design, y_rank, rcond=None)
    x_residual = x_rank - design @ x_fit
    y_residual = y_rank - design @ y_fit
    return pearson_correlation(x_residual, y_residual)


def finite_number(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite value for {label}: {result}")
    return result


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    formal_root = EXPECTED_PATHS["formal_root"]
    stage6b_root = EXPECTED_PATHS["stage6b_root"]

    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_branch = git_output(repo_root, "branch", "--show-current")
    actual_commit = git_output(repo_root, "rev-parse", "HEAD")
    if actual_branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Unexpected branch: {actual_branch}")
    if actual_commit != args.expected_execution_commit:
        raise RuntimeError(
            "Execution commit mismatch: "
            f"expected {args.expected_execution_commit}, found {actual_commit}"
        )
    git_output(
        repo_root,
        "merge-base",
        "--is-ancestor",
        EXPECTED_PREREG_COMMIT,
        actual_commit,
    )

    print("===== REPOSITORY BOUNDARY =====")
    print(f"branch: {actual_branch}")
    print(f"execution_commit: {actual_commit}")
    print(f"pre_registration_commit: {EXPECTED_PREREG_COMMIT}")

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

    print("\n===== FROZEN SHA256 =====")
    require_sha(prereg_path, EXPECTED_SHA256["pre_registration"], "pre_registration")
    require_sha(score_helper_path, EXPECTED_SHA256["score_candidates"], "score_candidates")
    require_sha(score_csv_path, EXPECTED_SHA256["score_csv"], "score_csv")
    require_sha(score_summary_path, EXPECTED_SHA256["score_summary"], "score_summary")
    require_sha(method_lock_path, EXPECTED_SHA256["method_lock"], "method_lock")
    require_sha(preflight_path, EXPECTED_SHA256["preflight"], "preflight")
    require_sha(reference_path, EXPECTED_SHA256["reference"], "reference")
    require_sha(
        reference_manifest_path,
        EXPECTED_SHA256["reference_manifest"],
        "reference_manifest",
    )
    require_sha(frozen_sums_path, EXPECTED_SHA256["frozen_sums"], "frozen_sums")

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

    reference_rows = load_jsonl(reference_path)
    with score_csv_path.open(encoding="utf-8", newline="") as handle:
        score_rows = list(csv.DictReader(handle))

    if method_lock["status"] != "LOCKED_BEFORE_FULL_SCORE_DISTRIBUTION":
        raise RuntimeError("Stage 6B-1 method lock is not valid")
    if preflight["status"] != "PASS" or preflight["scores_computed"] is not False:
        raise RuntimeError("Stage 6B-1 preflight is not valid")
    if int(config["seed"]) != SCORING_SEED:
        raise RuntimeError("Unexpected scoring seed")
    expected_summary_fields = {
        "status": "PASS",
        "method_version": "stage6b_1c_gradient_scoring_v1",
        "checkpoint_state": "stage6a_candidate_final_adapter",
        "gradient_mode": "eval_no_dropout",
        "optimizer_step_performed": False,
        "seed": SCORING_SEED,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "reference_count": EXPECTED_REFERENCE_COUNT,
        "trainable_tensor_count": EXPECTED_TRAINABLE_TENSORS,
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        "candidate_truncated_count": 0,
        "reference_truncated_count": 0,
        "output_order": "ascending_original_candidate_index",
    }
    for key, expected in expected_summary_fields.items():
        if frozen_summary.get(key) != expected:
            raise RuntimeError(
                f"Frozen Stage 6B-1 summary mismatch for {key}: "
                f"expected {expected!r}, found {frozen_summary.get(key)!r}"
            )
    if frozen_summary.get("method_lock_sha256") != EXPECTED_SHA256["method_lock"]:
        raise RuntimeError("Frozen summary method-lock SHA mismatch")
    if frozen_summary.get("preflight_sha256") != EXPECTED_SHA256["preflight"]:
        raise RuntimeError("Frozen summary preflight SHA mismatch")

    model_name = config["proxy_models"][0]["name"]
    tokenizer_name = config["proxy_models"][0]["tokenizer"]
    instruction = config["instruction_generation"]
    candidate_name = candidate_yaml["candidate_instructions"][0]["name"]
    candidate_path = formal_root / candidate_name
    adapter_path = (
        candidate_path / "finetuned_model" / Path(model_name).name / "adapter"
    )
    adapter_config_path = adapter_path / "adapter_config.json"
    adapter_model_path = adapter_path / "adapter_model.safetensors"
    require_sha(
        adapter_config_path,
        EXPECTED_SHA256["adapter_config"],
        "adapter_config",
    )
    require_sha(
        adapter_model_path,
        EXPECTED_SHA256["adapter_model"],
        "adapter_model",
    )

    candidate_dataset = datasets.load_from_disk(str(candidate_path))
    candidate_rows = [candidate_dataset[index] for index in range(len(candidate_dataset))]

    if len(candidate_rows) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("Candidate count is not exactly 100")
    if len(reference_rows) != EXPECTED_REFERENCE_COUNT:
        raise RuntimeError("Reference count is not exactly 100")
    if len(score_rows) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("Frozen score count is not exactly 100")

    indices = [int(row["candidate_index"]) for row in score_rows]
    if indices != list(range(EXPECTED_CANDIDATE_COUNT)):
        raise RuntimeError("Frozen candidate order is not exactly 0..99")
    if len({row["problem_sha256"] for row in score_rows}) != 100:
        raise RuntimeError("Frozen candidate identities are not unique")
    numeric_score_fields = (
        "gradient_cosine",
        "gradient_dot",
        "predicted_reference_loss_change_per_unit_step",
        "candidate_loss",
        "candidate_gradient_norm",
    )
    for index, row in enumerate(score_rows):
        for key in numeric_score_fields:
            if not math.isfinite(float(row[key])):
                raise RuntimeError(
                    f"Frozen candidate score is non-finite: index={index}, field={key}"
                )
    ranks = [int(row["gradient_rank"]) for row in score_rows]
    if sorted(ranks) != list(range(1, 101)):
        raise RuntimeError("Frozen gradient ranks are not exactly 1..100")
    for index, (candidate_row, score_row) in enumerate(zip(candidate_rows, score_rows, strict=True)):
        if text_sha256(candidate_row["problem"]) != score_row["problem_sha256"]:
            raise RuntimeError(f"Candidate identity mismatch at index {index}")
    for index, row in enumerate(reference_rows):
        if int(row["gradient_reference_index"]) != index:
            raise RuntimeError(f"Reference order mismatch at index {index}")
        if text_sha256(row["problem"]) != row["problem_sha256"]:
            raise RuntimeError(f"Reference identity mismatch at index {index}")
    if len({row["problem_sha256"] for row in reference_rows}) != 100:
        raise RuntimeError("Frozen reference identities are not unique")

    if file_sha256(method_lock_path) != preflight["method_lock_sha256"]:
        raise RuntimeError("Method-lock/preflight mismatch")
    if file_sha256(config_path) != preflight["config_sha256"]:
        raise RuntimeError("Config/preflight mismatch")
    if file_sha256(candidate_yaml_path) != preflight["candidate_yaml_sha256"]:
        raise RuntimeError("Candidate YAML/preflight mismatch")
    if file_sha256(reference_path) != preflight["reference_jsonl_sha256"]:
        raise RuntimeError("Reference/preflight mismatch")
    if ordered_rows_sha256(candidate_rows, "rewrite_trace") != preflight["candidate_ordered_rows_sha256"]:
        raise RuntimeError("Candidate ordered-row hash mismatch")
    if ordered_rows_sha256(reference_rows, "solution") != preflight["reference_ordered_rows_sha256"]:
        raise RuntimeError("Reference ordered-row hash mismatch")
    if adapter_file_records(adapter_path) != preflight["adapter_files"]:
        raise RuntimeError("Adapter file inventory differs from Stage 6B-1")

    print("\n===== INPUT IDENTITY =====")
    print(f"candidate_count: {len(candidate_rows)}")
    print(f"reference_count: {len(reference_rows)}")
    print(f"candidate_order: 0..{len(candidate_rows) - 1}")
    print("input_identity: PASS")

    random.seed(SCORING_SEED)
    np.random.seed(SCORING_SEED)
    torch.manual_seed(SCORING_SEED)
    torch.cuda.manual_seed_all(SCORING_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16")

    device = torch.device("cuda")
    compute_dtype = torch.bfloat16
    tokenizer = _load_training_tokenizer(tokenizer_name)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer special tokens are incomplete")
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )

    print("\n===== MODEL LOAD =====")
    print(f"model_name: {model_name}")
    print(f"tokenizer_name: {tokenizer_name}")
    print(f"candidate_adapter: {adapter_path}")
    print("checkpoint_state: stage6a_candidate_final_adapter")
    print("gradient_mode: eval_no_dropout")
    print("optimizer: none")

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
    if len(trainable) != EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError(f"Expected 392 trainable tensors, found {len(trainable)}")
    if not all("lora_" in name for name, _ in trainable):
        raise RuntimeError("A non-LoRA parameter is trainable")
    trainable_parameter_count = sum(parameter.numel() for _, parameter in trainable)
    if trainable_parameter_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            "Unexpected trainable parameter count: "
            f"{trainable_parameter_count}"
        )
    if any(parameter.grad is not None for _, parameter in frozen):
        raise RuntimeError("A frozen parameter has a gradient before execution")

    theta0 = [parameter.detach().clone() for _, parameter in trainable]
    frozen_versions_before = {name: parameter._version for name, parameter in frozen}
    print(f"trainable_tensor_count: {len(trainable)}")
    print(f"trainable_parameter_count: {trainable_parameter_count}")
    print(f"frozen_tensor_count: {len(frozen)}")
    print("hashing_frozen_parameters_before: start")
    frozen_parameter_sha_before = named_parameter_sha256(frozen)
    theta0_sha = named_parameter_sha256(trainable)
    print(f"frozen_parameter_sha256_before: {frozen_parameter_sha_before}")
    print(f"theta0_parameter_sha256: {theta0_sha}")

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
        if len(full_ids) != len(full_mask):
            raise RuntimeError("Token/mask length mismatch")
        kept_ids = full_ids[:MAX_LENGTH]
        kept_mask = full_mask[:MAX_LENGTH]
        if sum(kept_mask) <= 0:
            raise RuntimeError("Completion has no retained tokens")
        feature = {
            "input_ids": kept_ids,
            "completion_mask": kept_mask,
        }
        metadata = {
            "total_tokens_before_truncation": len(full_ids),
            "total_tokens_kept": len(kept_ids),
            "completion_tokens_before_truncation": sum(full_mask),
            "completion_tokens_kept": sum(kept_mask),
            "completion_tokens_removed": sum(full_mask[MAX_LENGTH:]),
            "eos_in_kept_completion": any(
                token_id == tokenizer.eos_token_id and mask_value == 1
                for token_id, mask_value in zip(kept_ids, kept_mask, strict=True)
            ),
        }
        return feature, metadata

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

    def backward_pass(feature: Mapping[str, Any]) -> float:
        model.zero_grad(set_to_none=True)
        batch = model_batch(feature)
        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            output = model(**batch)
            loss = output.loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite backward loss")
        loss.backward()
        torch.cuda.synchronize()
        for name, parameter in trainable:
            if parameter.grad is None:
                raise RuntimeError(f"Missing gradient: {name}")
            if not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError(f"Non-finite gradient: {name}")
        result = float(loss.detach().cpu())
        del batch, output, loss
        return result

    def current_gradient_norm() -> float:
        squared = torch.zeros((), device=device, dtype=torch.float32)
        for _, parameter in trainable:
            squared.add_(parameter.grad.detach().float().square().sum())
        norm = math.sqrt(float(squared.item()))
        return finite_number(norm, "gradient_norm")

    def trainable_mismatch_count() -> int:
        mismatch = torch.zeros((), device=device, dtype=torch.int64)
        for (_, parameter), initial in zip(trainable, theta0, strict=True):
            mismatch.add_(torch.count_nonzero(parameter.detach() != initial))
        return int(mismatch.item())

    def reset_to_theta0() -> None:
        with torch.no_grad():
            for (_, parameter), initial in zip(trainable, theta0, strict=True):
                parameter.copy_(initial)
        model.zero_grad(set_to_none=True)
        mismatch = trainable_mismatch_count()
        if mismatch != 0:
            raise RuntimeError(f"Theta0 reset mismatch count: {mismatch}")

    def gradient_snapshot(feature: Mapping[str, Any]) -> Tuple[float, Dict[str, torch.Tensor], float]:
        loss = backward_pass(feature)
        gradients = {
            name: parameter.grad.detach().float().clone()
            for name, parameter in trainable
        }
        norm = current_gradient_norm()
        return loss, gradients, norm

    def gradient_dot(
        gradients: Mapping[str, torch.Tensor],
        reference_gradient: Mapping[str, torch.Tensor],
    ) -> float:
        if gradients.keys() != reference_gradient.keys():
            raise RuntimeError("Gradient key mismatch")
        value = torch.zeros((), device=device, dtype=torch.float32)
        for name in gradients:
            value.add_((gradients[name] * reference_gradient[name]).sum())
        return finite_number(value.item(), "gradient_dot")

    def apply_manual_update(
        gradients: Mapping[str, torch.Tensor],
        eta: float,
    ) -> Dict[str, Any]:
        changed_elements = torch.zeros((), device=device, dtype=torch.int64)
        changed_tensors = 0
        squared_update = torch.zeros((), device=device, dtype=torch.float32)
        max_abs_update = torch.zeros((), device=device, dtype=torch.float32)
        formula_mismatch = torch.zeros((), device=device, dtype=torch.int64)
        with torch.no_grad():
            for (name, parameter), initial in zip(trainable, theta0, strict=True):
                expected = (initial.float() - eta * gradients[name]).to(parameter.dtype)
                parameter.copy_(expected)
                formula_mismatch.add_(torch.count_nonzero(parameter != expected))
                difference = parameter.detach().float() - initial.float()
                tensor_changed = int(torch.count_nonzero(difference).item())
                changed_elements.add_(tensor_changed)
                changed_tensors += tensor_changed > 0
                squared_update.add_(difference.square().sum())
                max_abs_update = torch.maximum(max_abs_update, difference.abs().max())
                del expected, difference
        stats = {
            "formula_mismatch_count": int(formula_mismatch.item()),
            "changed_tensor_count": int(changed_tensors),
            "changed_element_count": int(changed_elements.item()),
            "update_l2_norm": finite_number(
                math.sqrt(float(squared_update.item())), "update_l2_norm"
            ),
            "update_max_abs": finite_number(max_abs_update.item(), "update_max_abs"),
            "update_rule": (
                "float32(theta0) - eta * float32(gradient), cast to each "
                "original trainable-parameter dtype"
            ),
        }
        if stats["formula_mismatch_count"] != 0:
            raise RuntimeError("Manual update differs from implemented formula")
        if eta == 0.0 and (
            stats["changed_tensor_count"] != 0
            or stats["changed_element_count"] != 0
            or stats["update_l2_norm"] != 0.0
            or stats["update_max_abs"] != 0.0
        ):
            raise RuntimeError("Zero-eta control changed trainable parameters")
        return stats

    def evaluate_reference_losses() -> List[float]:
        model.zero_grad(set_to_none=True)
        losses = []
        with torch.no_grad():
            for feature in reference_features:
                batch = model_batch(feature)
                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    output = model(**batch)
                    loss = output.loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite reference loss")
                losses.append(float(loss.detach().cpu()))
                del batch, output, loss
        if len(losses) != EXPECTED_REFERENCE_COUNT:
            raise RuntimeError("Reference-loss count is not exactly 100")
        return losses

    print("\n===== REFERENCE BASELINE AND MEAN GRADIENT =====")
    reset_to_theta0()
    reference_sum = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for name, parameter in trainable
    }
    baseline_reference_losses = []
    baseline_rows = []
    for index, (row, feature, metadata) in enumerate(
        zip(reference_rows, reference_features, reference_metadata, strict=True)
    ):
        loss = backward_pass(feature)
        gradient_norm = current_gradient_norm()
        for name, parameter in trainable:
            reference_sum[name].add_(parameter.grad.detach().float())
        baseline_reference_losses.append(loss)
        baseline_rows.append(
            {
                "reference_index": index,
                "problem_sha256": row["problem_sha256"],
                "baseline_loss": loss,
                "gradient_norm": gradient_norm,
                "tokens": metadata,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"reference_gradient_progress: {index + 1}/100")
    for value in reference_sum.values():
        value.div_(EXPECTED_REFERENCE_COUNT)
    reference_squared_norm = torch.zeros((), device=device, dtype=torch.float32)
    for value in reference_sum.values():
        reference_squared_norm.add_(value.square().sum())
    mean_reference_gradient_norm = finite_number(
        math.sqrt(float(reference_squared_norm.item())),
        "mean_reference_gradient_norm",
    )
    baseline_reference_loss_mean = finite_number(
        statistics.fmean(baseline_reference_losses),
        "baseline_reference_loss_mean",
    )
    frozen_baseline_mean = float(frozen_summary["reference_loss_mean"])
    baseline_tolerance = max(1e-6, 1e-4 * abs(frozen_baseline_mean))
    if abs(baseline_reference_loss_mean - frozen_baseline_mean) > baseline_tolerance:
        raise RuntimeError(
            "Reference baseline differs from frozen Stage 6B-1 summary: "
            f"recomputed={baseline_reference_loss_mean}, "
            f"frozen={frozen_baseline_mean}, tolerance={baseline_tolerance}"
        )
    frozen_reference_norm = float(frozen_summary["mean_reference_gradient_norm"])
    norm_tolerance = max(1e-6, 1e-4 * abs(frozen_reference_norm))
    if abs(mean_reference_gradient_norm - frozen_reference_norm) > norm_tolerance:
        raise RuntimeError("Mean reference-gradient norm differs from frozen summary")
    if trainable_mismatch_count() != 0:
        raise RuntimeError("Backward-only reference aggregation changed theta0")
    print(f"baseline_reference_loss_mean: {baseline_reference_loss_mean}")
    print(f"mean_reference_gradient_norm: {mean_reference_gradient_norm}")
    print("reference_baseline_check: PASS")

    candidate_indices = [0] if args.mode == "smoke" else list(range(100))
    eta_values = list(ETA_GRID)
    print("\n===== ONE-STEP VALIDATION =====")
    print(f"mode: {args.mode}")
    print(f"candidate_indices: {candidate_indices}")
    print(f"eta_grid: {eta_values}")
    print("reference_loss_batch_size: 1")

    reference_audit_path = output_dir / "stage6b_3_reference_losses.jsonl"
    reference_audit_temp = reference_audit_path.with_name(
        reference_audit_path.name + ".incomplete"
    )
    results = []
    max_dot_error = 0.0
    max_reset_mismatch = 0
    max_zero_control_abs_change = 0.0

    with reference_audit_temp.open("w", encoding="utf-8") as reference_handle:
        for candidate_position, candidate_index in enumerate(candidate_indices, start=1):
            score_row = score_rows[candidate_index]
            frozen_dot = float(score_row["gradient_dot"])
            frozen_predicted_per_unit = float(
                score_row["predicted_reference_loss_change_per_unit_step"]
            )
            if frozen_predicted_per_unit != -frozen_dot:
                raise RuntimeError(
                    f"Frozen dot/prediction identity failed for candidate {candidate_index}"
                )
            metadata = candidate_metadata[candidate_index]
            expected_token_fields = {
                "total_tokens_before_truncation": int(
                    score_row["total_tokens_before_truncation"]
                ),
                "completion_tokens_before_truncation": int(
                    score_row["completion_tokens_before_truncation"]
                ),
                "completion_tokens_kept": int(score_row["completion_tokens_kept"]),
                "completion_tokens_removed": int(
                    score_row["completion_tokens_removed"]
                ),
            }
            for key, expected in expected_token_fields.items():
                if metadata[key] != expected:
                    raise RuntimeError(
                        f"Candidate {candidate_index} token metadata mismatch for {key}"
                    )

            for eta in eta_values:
                reset_to_theta0()
                reset_mismatch_before = trainable_mismatch_count()
                max_reset_mismatch = max(max_reset_mismatch, reset_mismatch_before)
                if reset_mismatch_before != 0:
                    raise RuntimeError("Parameters were not reset to theta0")

                candidate_loss, candidate_gradient, candidate_norm = gradient_snapshot(
                    candidate_features[candidate_index]
                )
                recomputed_dot = gradient_dot(candidate_gradient, reference_sum)
                dot_error = abs(recomputed_dot - frozen_dot)
                dot_tolerance = max(1e-6, 1e-4 * abs(frozen_dot))
                max_dot_error = max(max_dot_error, dot_error)
                if dot_error > dot_tolerance:
                    raise RuntimeError(
                        f"Candidate {candidate_index}, eta {eta}: gradient-dot "
                        f"mismatch {dot_error} exceeds {dot_tolerance}"
                    )
                update_stats = apply_manual_update(candidate_gradient, eta)
                model.zero_grad(set_to_none=True)
                updated_losses = evaluate_reference_losses()
                updated_mean = finite_number(
                    statistics.fmean(updated_losses), "updated_reference_loss_mean"
                )
                actual_change = finite_number(
                    updated_mean - baseline_reference_loss_mean,
                    "actual_reference_loss_change",
                )
                recomputed_predicted_change = finite_number(
                    -eta * recomputed_dot,
                    "recomputed_predicted_reference_loss_change",
                )
                predicted_change = finite_number(
                    eta * frozen_predicted_per_unit,
                    "predicted_reference_loss_change",
                )
                if abs(recomputed_predicted_change - predicted_change) > eta * dot_tolerance + 1e-15:
                    raise RuntimeError("Predicted-change reconstruction mismatch")
                if eta == 0.0:
                    max_zero_control_abs_change = max(
                        max_zero_control_abs_change, abs(actual_change)
                    )
                    if abs(actual_change) > 1e-6:
                        raise RuntimeError(
                            f"Zero-eta loss control failed for candidate {candidate_index}: "
                            f"{actual_change}"
                        )

                for reference_index, (baseline_loss, updated_loss, reference_row) in enumerate(
                    zip(
                        baseline_reference_losses,
                        updated_losses,
                        reference_rows,
                        strict=True,
                    )
                ):
                    audit_row = {
                        "candidate_index": candidate_index,
                        "candidate_problem_sha256": score_row["problem_sha256"],
                        "eta": eta,
                        "reference_index": reference_index,
                        "reference_problem_sha256": reference_row["problem_sha256"],
                        "baseline_loss": baseline_loss,
                        "updated_loss": updated_loss,
                        "loss_change": updated_loss - baseline_loss,
                    }
                    reference_handle.write(
                        json.dumps(
                            audit_row,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        )
                    )
                    reference_handle.write("\n")

                result = {
                    "candidate_index": candidate_index,
                    "candidate_problem_sha256": score_row["problem_sha256"],
                    "gradient_rank": int(score_row["gradient_rank"]),
                    "gradient_group": score_row["gradient_group"],
                    "gradient_cosine": float(score_row["gradient_cosine"]),
                    "eta": eta,
                    "frozen_gradient_dot": frozen_dot,
                    "recomputed_gradient_dot": recomputed_dot,
                    "gradient_dot_absolute_error": dot_error,
                    "gradient_dot_tolerance": dot_tolerance,
                    "candidate_loss": candidate_loss,
                    "frozen_candidate_loss": float(score_row["candidate_loss"]),
                    "candidate_gradient_norm": candidate_norm,
                    "frozen_candidate_gradient_norm": float(
                        score_row["candidate_gradient_norm"]
                    ),
                    "completion_tokens_kept": metadata["completion_tokens_kept"],
                    "baseline_reference_loss_mean": baseline_reference_loss_mean,
                    "updated_reference_loss_mean": updated_mean,
                    "predicted_reference_loss_change": predicted_change,
                    "recomputed_predicted_reference_loss_change": (
                        recomputed_predicted_change
                    ),
                    "actual_reference_loss_change": actual_change,
                    "reset_mismatch_count_before": reset_mismatch_before,
                    **update_stats,
                }
                results.append(result)
                reference_handle.flush()
                os.fsync(reference_handle.fileno())

                reset_to_theta0()
                reset_mismatch_after = trainable_mismatch_count()
                max_reset_mismatch = max(max_reset_mismatch, reset_mismatch_after)
                if reset_mismatch_after != 0:
                    raise RuntimeError("Post-cell theta0 reset failed")
                del candidate_gradient, updated_losses
                torch.cuda.empty_cache()
                print(
                    "cell_complete: "
                    f"candidate={candidate_index} eta={eta:.10g} "
                    f"predicted={predicted_change:.17g} "
                    f"actual={actual_change:.17g}"
                )

            print(
                f"candidate_progress: {candidate_position}/{len(candidate_indices)}"
            )
        reference_handle.flush()
        os.fsync(reference_handle.fileno())

    expected_result_count = len(candidate_indices) * len(eta_values)
    if len(results) != expected_result_count:
        raise RuntimeError("Required candidate/eta result row is missing")
    expected_audit_rows = expected_result_count * EXPECTED_REFERENCE_COUNT
    with reference_audit_temp.open(encoding="utf-8") as handle:
        actual_audit_rows = sum(1 for line in handle if line.strip())
    if actual_audit_rows != expected_audit_rows:
        raise RuntimeError(
            f"Reference audit row count mismatch: {actual_audit_rows}"
        )

    reset_to_theta0()
    theta0_sha_after = named_parameter_sha256(trainable)
    if theta0_sha_after != theta0_sha:
        raise RuntimeError("Trainable parameters were not restored exactly to theta0")
    if any(parameter.grad is not None for _, parameter in frozen):
        raise RuntimeError("A frozen parameter acquired a gradient")
    frozen_versions_after = {name: parameter._version for name, parameter in frozen}
    if frozen_versions_after != frozen_versions_before:
        raise RuntimeError("A frozen parameter version changed")
    print("hashing_frozen_parameters_after: start")
    frozen_parameter_sha_after = named_parameter_sha256(frozen)
    if frozen_parameter_sha_after != frozen_parameter_sha_before:
        raise RuntimeError("A frozen parameter value changed")

    analysis: Dict[str, Any] = {}
    if args.mode == "formal":
        by_eta = {
            eta: sorted(
                [row for row in results if row["eta"] == eta],
                key=lambda row: row["candidate_index"],
            )
            for eta in eta_values
        }
        if any(len(rows) != EXPECTED_CANDIDATE_COUNT for rows in by_eta.values()):
            raise RuntimeError("Formal analysis does not contain all 100 candidates")
        secondary = {}
        for eta in ETA_GRID[1:]:
            rows = by_eta[eta]
            predicted = [row["predicted_reference_loss_change"] for row in rows]
            actual = [row["actual_reference_loss_change"] for row in rows]
            cosine = [row["gradient_cosine"] for row in rows]
            covariate_values = {
                "frozen_candidate_loss": [
                    row["frozen_candidate_loss"] for row in rows
                ],
                "frozen_candidate_gradient_norm": [
                    row["frozen_candidate_gradient_norm"] for row in rows
                ],
                "completion_tokens_kept": [
                    row["completion_tokens_kept"] for row in rows
                ],
            }
            covariate_diagnostics = {
                name: {
                    "spearman_with_actual": spearman_correlation(
                        values, actual
                    ),
                    "pearson_with_actual": pearson_correlation(
                        values, actual
                    ),
                    "spearman_with_predicted": spearman_correlation(
                        values, predicted
                    ),
                }
                for name, values in covariate_values.items()
            }
            secondary[str(eta)] = {
                "label": "secondary" if eta != PRIMARY_ETA else "primary_eta_descriptives",
                "spearman": spearman_correlation(predicted, actual),
                "pearson": pearson_correlation(predicted, actual),
                "ols_calibration": ols_calibration(predicted, actual),
                "sign": sign_metrics(predicted, actual),
                "error": error_metrics(predicted, actual),
                "cosine_predictor": {
                    "spearman": spearman_correlation(cosine, actual),
                    "pearson": pearson_correlation(cosine, actual),
                },
                "candidate_covariate_diagnostics": covariate_diagnostics,
            }

        primary_rows = by_eta[PRIMARY_ETA]
        primary_predicted = [
            row["predicted_reference_loss_change"] for row in primary_rows
        ]
        primary_actual = [
            row["actual_reference_loss_change"] for row in primary_rows
        ]
        permutation = permutation_test(primary_predicted, primary_actual)
        bootstrap = bootstrap_spearman_ci(primary_predicted, primary_actual)
        observed = permutation["observed_spearman_rho"]
        positive_p = permutation["positive_tail_p"]
        negative_p = permutation["negative_tail_p"]
        if observed > 0.0 and positive_p < 0.05:
            decision = "supports_local_first_order_ranking_prediction"
        elif observed < 0.0 and negative_p < 0.05:
            decision = "reverse_or_contradicted_result"
        else:
            decision = "null_or_inconclusive_result"

        covariates = [
            [row["frozen_candidate_loss"] for row in primary_rows],
            [row["frozen_candidate_gradient_norm"] for row in primary_rows],
            [row["completion_tokens_kept"] for row in primary_rows],
        ]
        partial_rank = partial_rank_correlation(
            primary_predicted,
            primary_actual,
            covariates,
        )

        primary_actual_by_index = {
            row["candidate_index"]: row["actual_reference_loss_change"]
            for row in primary_rows
        }
        linear_scaling = {}
        for eta in (1e-4, 1e-3, 5e-3):
            rows = by_eta[eta]
            observed_values = [
                row["actual_reference_loss_change"] for row in rows
            ]
            primary_scaled = [
                primary_actual_by_index[row["candidate_index"]]
                * eta
                / PRIMARY_ETA
                for row in rows
            ]
            linear_scaling[str(eta)] = {
                "comparison": "actual_eta_vs_primary_actual_scaled_by_eta_ratio",
                "pearson": pearson_correlation(primary_scaled, observed_values),
                **error_metrics(primary_scaled, observed_values),
            }

        analysis = {
            "primary_endpoint": {
                "eta": PRIMARY_ETA,
                "sample_size": EXPECTED_CANDIDATE_COUNT,
                "direction": "positive_one_sided",
                "alpha": 0.05,
                "permutation_test": permutation,
                "bootstrap_spearman_95_ci": bootstrap,
                "decision": decision,
            },
            "secondary_analyses": secondary,
            "partial_rank_primary": {
                "label": "secondary",
                "covariates": [
                    "frozen_candidate_loss",
                    "frozen_candidate_gradient_norm",
                    "completion_tokens_kept",
                ],
                "method": "Pearson correlation of OLS residuals after average-rank transform",
                "correlation": partial_rank,
            },
            "linear_scaling_diagnostic": {
                "label": "secondary",
                "comparisons": linear_scaling,
            },
        }

    baseline_path = output_dir / "stage6b_3_reference_baseline.jsonl"
    baseline_temp = baseline_path.with_name(baseline_path.name + ".incomplete")
    with baseline_temp.open("w", encoding="utf-8") as handle:
        for row in baseline_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    results_jsonl_path = output_dir / "stage6b_3_results.jsonl"
    results_jsonl_temp = results_jsonl_path.with_name(
        results_jsonl_path.name + ".incomplete"
    )
    with results_jsonl_temp.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    results_csv_path = output_dir / "stage6b_3_results.csv"
    results_csv_temp = results_csv_path.with_name(results_csv_path.name + ".incomplete")
    fieldnames = [
        "candidate_index",
        "candidate_problem_sha256",
        "gradient_rank",
        "gradient_group",
        "gradient_cosine",
        "eta",
        "frozen_gradient_dot",
        "recomputed_gradient_dot",
        "gradient_dot_absolute_error",
        "gradient_dot_tolerance",
        "candidate_loss",
        "frozen_candidate_loss",
        "candidate_gradient_norm",
        "frozen_candidate_gradient_norm",
        "completion_tokens_kept",
        "baseline_reference_loss_mean",
        "updated_reference_loss_mean",
        "predicted_reference_loss_change",
        "recomputed_predicted_reference_loss_change",
        "actual_reference_loss_change",
        "changed_tensor_count",
        "changed_element_count",
        "update_l2_norm",
        "update_max_abs",
        "formula_mismatch_count",
        "reset_mismatch_count_before",
    ]
    with results_csv_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())

    run_config = {
        "method_version": METHOD_VERSION,
        "mode": args.mode,
        "branch": actual_branch,
        "execution_commit": actual_commit,
        "pre_registration_commit": EXPECTED_PREREG_COMMIT,
        "candidate_indices": candidate_indices,
        "eta_grid": eta_values,
        "primary_eta": PRIMARY_ETA,
        "candidate_gradient_batch_size": 1,
        "reference_loss_batch_size": 1,
        "reference_loss_aggregation": "arithmetic mean of 100 per-example losses",
        "scoring_seed": SCORING_SEED,
        "statistical_seed": STATISTICAL_SEED,
        "compute_dtype": str(compute_dtype),
        "trainable_parameter_dtypes": {
            dtype: sum(
                parameter.numel()
                for _, parameter in trainable
                if str(parameter.dtype) == dtype
            )
            for dtype in sorted({str(parameter.dtype) for _, parameter in trainable})
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "validator_runtime_copy_sha256": file_sha256(Path(__file__)),
        "input_sha256": EXPECTED_SHA256,
    }
    summary = {
        "status": "PASS",
        "method_version": METHOD_VERSION,
        "mode": args.mode,
        "stage_boundary": "local one-step reference-loss validation only",
        "candidate_count": len(candidate_indices),
        "eta_count": len(eta_values),
        "result_row_count": len(results),
        "reference_audit_row_count": actual_audit_rows,
        "baseline_reference_loss_mean": baseline_reference_loss_mean,
        "frozen_baseline_reference_loss_mean": frozen_baseline_mean,
        "mean_reference_gradient_norm": mean_reference_gradient_norm,
        "frozen_mean_reference_gradient_norm": frozen_reference_norm,
        "max_gradient_dot_absolute_error": max_dot_error,
        "max_reset_mismatch_count": max_reset_mismatch,
        "max_zero_control_abs_loss_change": max_zero_control_abs_change,
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": trainable_parameter_count,
        "theta0_parameter_sha256_before": theta0_sha,
        "theta0_parameter_sha256_after": theta0_sha_after,
        "frozen_parameter_sha256_before": frozen_parameter_sha_before,
        "frozen_parameter_sha256_after": frozen_parameter_sha_after,
        "validity_checks": {
            "frozen_input_sha256": "PASS",
            "candidate_and_reference_identity": "PASS",
            "adapter_identity": "PASS",
            "trainable_lora_only": "PASS",
            "theta0_reset": "PASS",
            "zero_eta_parameter_control": "PASS",
            "zero_eta_loss_control": "PASS",
            "manual_update_formula": "PASS",
            "frozen_gradient_dot_recomputation": "PASS",
            "frozen_parameter_unchanged": "PASS",
            "all_required_rows_present": "PASS",
            "finite_values": "PASS",
        },
        **analysis,
    }

    atomic_json(output_dir / "stage6b_3_run_config.json", run_config)
    atomic_json(output_dir / "stage6b_3_summary.json", summary)
    os.replace(baseline_temp, baseline_path)
    os.replace(results_jsonl_temp, results_jsonl_path)
    os.replace(results_csv_temp, results_csv_path)
    os.replace(reference_audit_temp, reference_audit_path)

    print("\n===== FINAL SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"summary_path: {output_dir / 'stage6b_3_summary.json'}")
    print(f"results_path: {results_jsonl_path}")
    print(f"reference_audit_path: {reference_audit_path}")
    print(f"stage6b_3_{args.mode}: PASS")


if __name__ == "__main__":
    main()
