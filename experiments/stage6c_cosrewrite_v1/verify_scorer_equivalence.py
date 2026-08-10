#!/usr/bin/env python3
"""Verify the reusable Stage 6C scorer against frozen Stage 6B-1c.

This is an engineering equivalence test.  It performs no optimizer step,
generates no rewrite, and evaluates no Stage 6C effectiveness endpoint.

Unlike the first draft, PASS requires all of the following:

* the complete frozen candidate traces, reference solutions, score files,
  configuration, tokenizer helper, and adapter match Stage 6B-1c;
* reference loss and mean-gradient norm reproduce within their locked gates;
* every tested candidate reproduces loss, gradient norm, dot, and cosine;
* token metadata is unchanged; and
* every trainable and frozen model parameter is byte-identical before and
  after scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import datasets
import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM

from optimize.gradient_feedback import CompletionOnlyGradientScorer
from optimize.score_candidates import _load_training_tokenizer


METHOD_VERSION = "stage6c_gradient_scorer_equivalence_v2"
SCORING_SEED = 888
EXPECTED_CANDIDATE_COUNT = 100
EXPECTED_REFERENCE_COUNT = 100
EXPECTED_TRAINABLE_TENSORS = 392
EXPECTED_TRAINABLE_PARAMETERS = 17_432_576
REPRODUCIBILITY_SPEARMAN_MIN = 0.999
NORMALIZED_DOT_DISCREPANCY_MAX = 0.002
SCALAR_RELATIVE_TOLERANCE = 1e-4
SCALAR_ABSOLUTE_FLOOR = 1e-6

# Conservative bound implied by the official 0.002 normalised-dot gate and
# two 1e-4 norm gates.  In the worst case:
#   0.002 / (0.9999 ** 2) + (1 / (0.9999 ** 2) - 1)
# is approximately 0.00220043.  The rounded 0.00225 limit is therefore
# compatible with all component gates while still making cosine pass/fail.
COSINE_ABSOLUTE_ERROR_MAX = 0.00225
FROZEN_INTERNAL_CONSISTENCY_TOLERANCE = 1e-12

EXPECTED_SHA256 = {
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

DEFAULT_FORMAL_ROOT = Path(
    "/srv/scratch/z5463756/honour/results/"
    "stage6a-formal-pilot-qwen3-1.7b-v1"
)
DEFAULT_STAGE6B_ROOT = Path(
    "/srv/scratch/z5463756/honour/results/"
    "stage6b-gradient-feedback-qwen3-1.7b-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=DEFAULT_FORMAL_ROOT,
    )
    parser.add_argument(
        "--stage6b-root",
        type=Path,
        default=DEFAULT_STAGE6B_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def atomic_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
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


def finite_number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite value for {label}: {result}")
    return result


def scalar_tolerance(expected: float) -> float:
    return max(
        SCALAR_ABSOLUTE_FLOOR,
        SCALAR_RELATIVE_TOLERANCE * abs(expected),
    )


def parse_csv_bool(value: Any, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"Invalid boolean for {label}: {value!r}")


def rankdata_average(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Ranks require a finite one-dimensional vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def spearman_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    x = rankdata_average(left)
    y = rankdata_average(right)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined")
    return max(-1.0, min(1.0, float(np.dot(x, y) / denominator)))


def relative_error(actual: float, expected: float) -> float:
    denominator = max(abs(expected), 1e-15)
    return abs(actual - expected) / denominator


def main() -> None:
    args = parse_args()
    formal_root = args.formal_root.resolve()
    stage6b_root = args.stage6b_root.resolve()
    output_dir = args.output_dir.resolve()
    runtime_root = Path(__file__).resolve().parents[2]
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = formal_root / "provenance/config.yaml"
    candidate_yaml_path = formal_root / "iter_0/candidate_instructions.yaml"
    method_lock_path = stage6b_root / "provenance/stage6b_1c_method_lock.yaml"
    preflight_path = stage6b_root / "provenance/stage6b_1c_preflight.json"
    reference_path = stage6b_root / "frozen_data/gradient_reference.jsonl"
    reference_manifest_path = (
        stage6b_root / "frozen_data/gradient_reference_manifest.json"
    )
    score_dir = stage6b_root / "scoring/stage6b_1c_8918577"
    score_csv_path = score_dir / "stage6b_1c_candidate_scores.csv"
    score_summary_path = score_dir / "stage6b_1c_summary.json"
    score_helper_path = runtime_root / "optimize/score_candidates.py"

    required_paths = (
        config_path,
        candidate_yaml_path,
        method_lock_path,
        preflight_path,
        reference_path,
        reference_manifest_path,
        score_csv_path,
        score_summary_path,
        score_helper_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    print("===== CRITICAL FROZEN INPUTS =====")
    verified_sha256 = {
        "score_candidates": require_sha(
            score_helper_path,
            EXPECTED_SHA256["score_candidates"],
            "score_candidates",
        ),
        "score_csv": require_sha(
            score_csv_path,
            EXPECTED_SHA256["score_csv"],
            "score_csv",
        ),
        "score_summary": require_sha(
            score_summary_path,
            EXPECTED_SHA256["score_summary"],
            "score_summary",
        ),
        "method_lock": require_sha(
            method_lock_path,
            EXPECTED_SHA256["method_lock"],
            "method_lock",
        ),
        "preflight": require_sha(
            preflight_path,
            EXPECTED_SHA256["preflight"],
            "preflight",
        ),
        "reference": require_sha(
            reference_path,
            EXPECTED_SHA256["reference"],
            "reference",
        ),
        "reference_manifest": require_sha(
            reference_manifest_path,
            EXPECTED_SHA256["reference_manifest"],
            "reference_manifest",
        ),
    }

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with candidate_yaml_path.open(encoding="utf-8") as handle:
        candidate_yaml = yaml.safe_load(handle)
    with method_lock_path.open(encoding="utf-8") as handle:
        method_lock = yaml.safe_load(handle)
    with preflight_path.open(encoding="utf-8") as handle:
        preflight = json.load(handle)
    with reference_manifest_path.open(encoding="utf-8") as handle:
        reference_manifest = json.load(handle)
    with score_summary_path.open(encoding="utf-8") as handle:
        frozen_summary = json.load(handle)
    with score_csv_path.open(encoding="utf-8", newline="") as handle:
        score_rows = list(csv.DictReader(handle))
    reference_rows = load_jsonl(reference_path)

    if method_lock.get("status") != "LOCKED_BEFORE_FULL_SCORE_DISTRIBUTION":
        raise RuntimeError("Stage 6B-1 method lock is invalid")
    if method_lock["candidate_scoring"].get("primary_score") != (
        "cosine_candidate_gradient_with_mean_reference_gradient"
    ):
        raise RuntimeError("Unexpected Stage 6B-1 primary score")
    if method_lock["ranking"].get("direction") != (
        "descending_primary_cosine"
    ):
        raise RuntimeError("Unexpected Stage 6B-1 ranking direction")
    if preflight.get("status") != "PASS":
        raise RuntimeError("Stage 6B-1 preflight did not pass")
    if preflight.get("scores_computed") is not False:
        raise RuntimeError("Stage 6B-1 preflight was not score-blind")
    if int(config["seed"]) != SCORING_SEED:
        raise RuntimeError("Unexpected scoring seed")

    expected_summary_fields = {
        "status": "PASS",
        "stage": "stage6b_1c_full_gradient_scoring",
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

    gradient_reference_manifest = reference_manifest.get("gradient_reference", {})
    if reference_manifest.get("schema") != "stage6b_gradient_reference_manifest_v1":
        raise RuntimeError("Unexpected reference-manifest schema")
    if gradient_reference_manifest.get("rows") != EXPECTED_REFERENCE_COUNT:
        raise RuntimeError("Reference manifest row count mismatch")
    if gradient_reference_manifest.get("jsonl_file_sha256") != (
        EXPECTED_SHA256["reference"]
    ):
        raise RuntimeError("Reference manifest/file identity mismatch")
    if reference_manifest.get("separation_checks", {}).get("passed") is not True:
        raise RuntimeError("Reference-set separation checks did not pass")

    model_name = config["proxy_models"][0]["name"]
    tokenizer_name = config["proxy_models"][0]["tokenizer"]
    instruction = config["instruction_generation"]
    candidate_name = candidate_yaml["candidate_instructions"][0]["name"]
    candidate_path = formal_root / candidate_name
    adapter_path = (
        candidate_path
        / "finetuned_model"
        / Path(model_name).name
        / "adapter"
    )
    if not candidate_path.is_dir():
        raise FileNotFoundError(candidate_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(adapter_path)

    adapter_config_path = adapter_path / "adapter_config.json"
    adapter_model_path = adapter_path / "adapter_model.safetensors"
    verified_sha256["adapter_config"] = require_sha(
        adapter_config_path,
        EXPECTED_SHA256["adapter_config"],
        "adapter_config",
    )
    verified_sha256["adapter_model"] = require_sha(
        adapter_model_path,
        EXPECTED_SHA256["adapter_model"],
        "adapter_model",
    )

    candidate_dataset = datasets.load_from_disk(str(candidate_path))
    candidate_rows = [
        candidate_dataset[index]
        for index in range(len(candidate_dataset))
    ]
    if len(candidate_rows) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("Candidate count is not exactly 100")
    if len(reference_rows) != EXPECTED_REFERENCE_COUNT:
        raise RuntimeError("Reference count is not exactly 100")
    if len(score_rows) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("Frozen score count is not exactly 100")

    indices = [int(row["candidate_index"]) for row in score_rows]
    if indices != list(range(EXPECTED_CANDIDATE_COUNT)):
        raise RuntimeError("Frozen candidate order is not exactly 0..99")
    ranks = [int(row["gradient_rank"]) for row in score_rows]
    if sorted(ranks) != list(range(1, EXPECTED_CANDIDATE_COUNT + 1)):
        raise RuntimeError("Frozen ranks are not exactly 1..100")
    if len({row["problem_sha256"] for row in score_rows}) != 100:
        raise RuntimeError("Frozen candidate identities are not unique")

    frozen_reference_norm = finite_number(
        frozen_summary["mean_reference_gradient_norm"],
        "frozen_mean_reference_gradient_norm",
    )
    if frozen_reference_norm <= 0.0:
        raise RuntimeError("Frozen mean reference-gradient norm is not positive")

    frozen_numeric_fields = (
        "gradient_cosine",
        "gradient_dot",
        "predicted_reference_loss_change_per_unit_step",
        "candidate_loss",
        "candidate_gradient_norm",
    )
    for index, row in enumerate(score_rows):
        for key in frozen_numeric_fields:
            finite_number(row[key], f"frozen_{key}_{index}")
        rank = int(row["gradient_rank"])
        expected_group = "GradHigh50" if rank <= 50 else "GradLow50"
        if row["gradient_group"] != expected_group:
            raise RuntimeError(f"Frozen group/rank mismatch at index {index}")
        frozen_dot = float(row["gradient_dot"])
        frozen_prediction = float(
            row["predicted_reference_loss_change_per_unit_step"]
        )
        if frozen_prediction != -frozen_dot:
            raise RuntimeError(f"Frozen dot/prediction mismatch at index {index}")
        frozen_norm = float(row["candidate_gradient_norm"])
        if frozen_norm <= 0.0:
            raise RuntimeError(f"Frozen candidate norm is not positive: {index}")
        reconstructed_cosine = frozen_dot / (
            frozen_norm * frozen_reference_norm
        )
        if abs(reconstructed_cosine - float(row["gradient_cosine"])) > (
            FROZEN_INTERNAL_CONSISTENCY_TOLERANCE
        ):
            raise RuntimeError(f"Frozen cosine identity mismatch at index {index}")

    ranked_indices = [
        int(row["candidate_index"])
        for row in sorted(score_rows, key=lambda item: int(item["gradient_rank"]))
    ]
    if ranked_indices != frozen_summary.get("ranking_candidate_indices"):
        raise RuntimeError("Frozen score/summary ranking mismatch")

    for index, (candidate, frozen) in enumerate(
        zip(candidate_rows, score_rows, strict=True)
    ):
        if text_sha256(str(candidate["problem"])) != frozen["problem_sha256"]:
            raise RuntimeError(f"Candidate identity mismatch at index {index}")
    if len({str(row["problem"]) for row in candidate_rows}) != 100:
        raise RuntimeError("Candidate problems are not unique")

    for index, row in enumerate(reference_rows):
        if int(row["gradient_reference_index"]) != index:
            raise RuntimeError(f"Reference order mismatch at index {index}")
        if text_sha256(str(row["problem"])) != row["problem_sha256"]:
            raise RuntimeError(f"Reference identity mismatch at index {index}")
    if len({row["problem_sha256"] for row in reference_rows}) != 100:
        raise RuntimeError("Reference identities are not unique")
    if {str(row["problem"]) for row in candidate_rows} & {
        str(row["problem"]) for row in reference_rows
    }:
        raise RuntimeError("Candidate/reference problem sets overlap")

    if file_sha256(method_lock_path) != preflight["method_lock_sha256"]:
        raise RuntimeError("Method-lock/preflight mismatch")
    if file_sha256(config_path) != preflight["config_sha256"]:
        raise RuntimeError("Config/preflight mismatch")
    if file_sha256(candidate_yaml_path) != preflight["candidate_yaml_sha256"]:
        raise RuntimeError("Candidate YAML/preflight mismatch")
    if file_sha256(reference_path) != preflight["reference_jsonl_sha256"]:
        raise RuntimeError("Reference/preflight mismatch")

    candidate_ordered_rows_hash = ordered_rows_sha256(
        candidate_rows,
        "rewrite_trace",
    )
    reference_ordered_rows_hash = ordered_rows_sha256(
        reference_rows,
        "solution",
    )
    if candidate_ordered_rows_hash != preflight["candidate_ordered_rows_sha256"]:
        raise RuntimeError("Candidate problem/rewrite-trace identity mismatch")
    if reference_ordered_rows_hash != preflight["reference_ordered_rows_sha256"]:
        raise RuntimeError("Reference problem/solution identity mismatch")
    if adapter_file_records(adapter_path) != preflight["adapter_files"]:
        raise RuntimeError("Adapter inventory differs from Stage 6B-1")
    if preflight.get("model_name") != model_name:
        raise RuntimeError("Model/config/preflight identity mismatch")
    if preflight.get("tokenizer_name") != tokenizer_name:
        raise RuntimeError("Tokenizer/config/preflight identity mismatch")
    if preflight.get("candidate_name") != candidate_name:
        raise RuntimeError("Candidate-name/preflight identity mismatch")

    verified_sha256["config"] = file_sha256(config_path)
    verified_sha256["candidate_yaml"] = file_sha256(candidate_yaml_path)

    print("\n===== COMPLETE INPUT IDENTITY =====")
    print(f"candidate_count: {len(candidate_rows)}")
    print(f"reference_count: {len(reference_rows)}")
    print(f"candidate_ordered_rows_sha256: {candidate_ordered_rows_hash}")
    print(f"reference_ordered_rows_sha256: {reference_ordered_rows_hash}")
    print("candidate_trace_identity: PASS")
    print("reference_solution_identity: PASS")
    print("adapter_identity: PASS")

    random.seed(SCORING_SEED)
    np.random.seed(SCORING_SEED)
    torch.manual_seed(SCORING_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16")
    torch.cuda.manual_seed_all(SCORING_SEED)

    device = torch.device("cuda")
    compute_dtype = torch.bfloat16
    tokenizer = _load_training_tokenizer(tokenizer_name)
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad token")
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token")

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

    scorer = CompletionOnlyGradientScorer(
        model=model,
        tokenizer=tokenizer,
        instruction=instruction,
        model_name=model_name,
        device=device,
        compute_dtype=compute_dtype,
        maximum_total_tokens=1024,
    )
    if scorer.trainable_tensor_count != EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError(
            "Unexpected trainable tensor count: "
            f"{scorer.trainable_tensor_count}"
        )
    if scorer.trainable_parameter_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            "Unexpected trainable parameter count: "
            f"{scorer.trainable_parameter_count}"
        )

    trainable = list(scorer.trainable)
    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    if any(parameter.grad is not None for _, parameter in frozen):
        raise RuntimeError("A frozen parameter has a gradient before scoring")

    trainable_versions_before = {
        name: parameter._version for name, parameter in trainable
    }
    frozen_versions_before = {
        name: parameter._version for name, parameter in frozen
    }
    print("\n===== PARAMETER INTEGRITY BEFORE =====")
    print("hashing_trainable_parameters_before: start")
    trainable_sha_before = named_parameter_sha256(trainable)
    print("hashing_frozen_parameters_before: start")
    frozen_sha_before = named_parameter_sha256(frozen)
    print(f"trainable_parameter_sha256_before: {trainable_sha_before}")
    print(f"frozen_parameter_sha256_before: {frozen_sha_before}")

    print("\n===== REFERENCE MEAN GRADIENT =====")
    reference = scorer.mean_reference_gradient(
        reference_rows,
        progress_every=10,
        progress_callback=lambda done, total: print(
            f"reference_progress: {done}/{total}"
        ),
    )
    if reference.example_count != EXPECTED_REFERENCE_COUNT:
        raise RuntimeError("Recomputed reference count is not exactly 100")
    if any(
        int(item["tokens"]["completion_tokens_removed"]) != 0
        for item in reference.per_example
    ):
        raise RuntimeError("A recomputed reference completion was truncated")

    frozen_reference_loss = finite_number(
        frozen_summary["reference_loss_mean"],
        "frozen_reference_loss_mean",
    )
    recomputed_reference_loss = finite_number(
        reference.loss_mean,
        "recomputed_reference_loss_mean",
    )
    recomputed_reference_norm = finite_number(
        reference.mean_gradient_norm,
        "recomputed_mean_reference_gradient_norm",
    )
    reference_loss_tolerance = scalar_tolerance(frozen_reference_loss)
    reference_norm_tolerance = scalar_tolerance(frozen_reference_norm)
    reference_loss_error = abs(
        recomputed_reference_loss - frozen_reference_loss
    )
    reference_norm_error = abs(
        recomputed_reference_norm - frozen_reference_norm
    )
    reference_loss_gate = reference_loss_error <= reference_loss_tolerance
    reference_norm_gate = reference_norm_error <= reference_norm_tolerance
    if not reference_loss_gate or not reference_norm_gate:
        raise RuntimeError(
            "Reference equivalence gate failed: "
            f"loss_error={reference_loss_error}, "
            f"loss_tolerance={reference_loss_tolerance}, "
            f"norm_error={reference_norm_error}, "
            f"norm_tolerance={reference_norm_tolerance}"
        )

    rank_to_index = {
        int(row["gradient_rank"]): int(row["candidate_index"])
        for row in score_rows
    }
    candidate_indices = (
        [rank_to_index[1], rank_to_index[50], rank_to_index[100]]
        if args.mode == "smoke"
        else list(range(EXPECTED_CANDIDATE_COUNT))
    )

    print("\n===== CANDIDATE EQUIVALENCE =====")
    print(f"mode: {args.mode}")
    print(f"candidate_indices: {candidate_indices}")
    result_rows = []
    frozen_dots = []
    recomputed_dots = []
    frozen_cosines = []
    recomputed_cosines = []

    for position, candidate_index in enumerate(candidate_indices, start=1):
        candidate = candidate_rows[candidate_index]
        frozen_score = score_rows[candidate_index]
        score = scorer.score_response(
            problem=str(candidate["problem"]),
            response=str(candidate["rewrite_trace"]),
            reference=reference,
        )

        frozen_total_tokens = int(
            frozen_score["total_tokens_before_truncation"]
        )
        expected_tokens = {
            "total_tokens_before_truncation": frozen_total_tokens,
            "total_tokens_kept": min(frozen_total_tokens, 1024),
            "completion_tokens_before_truncation": int(
                frozen_score["completion_tokens_before_truncation"]
            ),
            "completion_tokens_kept": int(
                frozen_score["completion_tokens_kept"]
            ),
            "completion_tokens_removed": int(
                frozen_score["completion_tokens_removed"]
            ),
            "eos_in_kept_completion": parse_csv_bool(
                frozen_score["eos_in_kept_completion"],
                f"eos_in_kept_completion_{candidate_index}",
            ),
        }
        actual_tokens = score.tokens.to_dict()
        token_metadata_gate = actual_tokens == expected_tokens
        if not token_metadata_gate:
            raise RuntimeError(
                f"Token metadata mismatch for candidate {candidate_index}: "
                f"expected={expected_tokens}, actual={actual_tokens}"
            )

        frozen_dot = finite_number(
            frozen_score["gradient_dot"],
            f"frozen_gradient_dot_{candidate_index}",
        )
        frozen_candidate_norm = finite_number(
            frozen_score["candidate_gradient_norm"],
            f"frozen_candidate_gradient_norm_{candidate_index}",
        )
        frozen_cosine = finite_number(
            frozen_score["gradient_cosine"],
            f"frozen_gradient_cosine_{candidate_index}",
        )
        frozen_loss = finite_number(
            frozen_score["candidate_loss"],
            f"frozen_candidate_loss_{candidate_index}",
        )

        frozen_norm_product = frozen_candidate_norm * frozen_reference_norm
        if not math.isfinite(frozen_norm_product) or frozen_norm_product <= 0.0:
            raise RuntimeError("Invalid frozen norm product")

        dot_absolute_error = abs(score.gradient_dot - frozen_dot)
        normalized_dot_discrepancy = (
            dot_absolute_error / frozen_norm_product
        )
        dot_gate = (
            normalized_dot_discrepancy
            <= NORMALIZED_DOT_DISCREPANCY_MAX
        )

        loss_absolute_error = abs(score.loss - frozen_loss)
        loss_tolerance = scalar_tolerance(frozen_loss)
        loss_gate = loss_absolute_error <= loss_tolerance

        norm_absolute_error = abs(
            score.gradient_norm - frozen_candidate_norm
        )
        norm_tolerance = scalar_tolerance(frozen_candidate_norm)
        norm_gate = norm_absolute_error <= norm_tolerance

        cosine_absolute_error = abs(
            score.gradient_cosine - frozen_cosine
        )
        cosine_gate = (
            cosine_absolute_error <= COSINE_ABSOLUTE_ERROR_MAX
        )

        failed_gates = [
            name
            for name, passed in (
                ("dot", dot_gate),
                ("candidate_loss", loss_gate),
                ("candidate_gradient_norm", norm_gate),
                ("gradient_cosine", cosine_gate),
                ("token_metadata", token_metadata_gate),
            )
            if not passed
        ]
        if failed_gates:
            raise RuntimeError(
                f"Candidate {candidate_index} equivalence gate failed: "
                + ", ".join(failed_gates)
            )

        row = {
            "candidate_index": candidate_index,
            "gradient_rank": int(frozen_score["gradient_rank"]),
            "gradient_group": frozen_score["gradient_group"],
            "problem_sha256": frozen_score["problem_sha256"],
            "rewrite_trace_sha256": text_sha256(
                str(candidate["rewrite_trace"])
            ),
            "frozen_candidate_loss": frozen_loss,
            "recomputed_candidate_loss": score.loss,
            "candidate_loss_absolute_error": loss_absolute_error,
            "candidate_loss_relative_error": relative_error(
                score.loss,
                frozen_loss,
            ),
            "candidate_loss_tolerance": loss_tolerance,
            "candidate_loss_gate": "PASS",
            "frozen_candidate_gradient_norm": frozen_candidate_norm,
            "recomputed_candidate_gradient_norm": score.gradient_norm,
            "candidate_gradient_norm_absolute_error": norm_absolute_error,
            "candidate_gradient_norm_relative_error": relative_error(
                score.gradient_norm,
                frozen_candidate_norm,
            ),
            "candidate_gradient_norm_tolerance": norm_tolerance,
            "candidate_gradient_norm_gate": "PASS",
            "frozen_gradient_dot": frozen_dot,
            "recomputed_gradient_dot": score.gradient_dot,
            "gradient_dot_absolute_error": dot_absolute_error,
            "frozen_gradient_norm_product": frozen_norm_product,
            "normalized_dot_discrepancy": normalized_dot_discrepancy,
            "normalized_dot_discrepancy_limit": (
                NORMALIZED_DOT_DISCREPANCY_MAX
            ),
            "gradient_dot_gate": "PASS",
            "frozen_gradient_cosine": frozen_cosine,
            "recomputed_gradient_cosine": score.gradient_cosine,
            "gradient_cosine_absolute_error": cosine_absolute_error,
            "gradient_cosine_absolute_error_limit": (
                COSINE_ABSOLUTE_ERROR_MAX
            ),
            "gradient_cosine_gate": "PASS",
            "recomputed_predicted_reference_loss_change_per_unit_step": (
                score.predicted_reference_loss_change_per_unit_step
            ),
            "token_metadata_gate": "PASS",
            "tokens": actual_tokens,
        }
        result_rows.append(row)
        frozen_dots.append(frozen_dot)
        recomputed_dots.append(score.gradient_dot)
        frozen_cosines.append(frozen_cosine)
        recomputed_cosines.append(score.gradient_cosine)
        print(
            "candidate_complete: "
            f"{position}/{len(candidate_indices)} "
            f"index={candidate_index} "
            f"rank={row['gradient_rank']} "
            f"dot_error={normalized_dot_discrepancy:.9g} "
            f"cosine_error={cosine_absolute_error:.9g} "
            f"loss_error={loss_absolute_error:.9g} "
            f"norm_error={norm_absolute_error:.9g}"
        )

    dot_spearman = spearman_correlation(frozen_dots, recomputed_dots)
    cosine_spearman = spearman_correlation(
        frozen_cosines,
        recomputed_cosines,
    )
    formal_dot_vector_gate = (
        dot_spearman >= REPRODUCIBILITY_SPEARMAN_MIN
        if args.mode == "formal"
        else None
    )
    formal_cosine_vector_gate = (
        cosine_spearman >= REPRODUCIBILITY_SPEARMAN_MIN
        if args.mode == "formal"
        else None
    )
    if args.mode == "formal" and not formal_dot_vector_gate:
        raise RuntimeError("Formal frozen/recomputed dot Spearman gate failed")
    if args.mode == "formal" and not formal_cosine_vector_gate:
        raise RuntimeError("Formal frozen/recomputed cosine Spearman gate failed")

    model.zero_grad(set_to_none=True)
    trainable_versions_after = {
        name: parameter._version for name, parameter in trainable
    }
    frozen_versions_after = {
        name: parameter._version for name, parameter in frozen
    }
    print("\n===== PARAMETER INTEGRITY AFTER =====")
    print("hashing_trainable_parameters_after: start")
    trainable_sha_after = named_parameter_sha256(trainable)
    print("hashing_frozen_parameters_after: start")
    frozen_sha_after = named_parameter_sha256(frozen)
    frozen_gradients_absent = all(
        parameter.grad is None for _, parameter in frozen
    )
    parameter_integrity_gate = (
        trainable_sha_after == trainable_sha_before
        and frozen_sha_after == frozen_sha_before
        and trainable_versions_after == trainable_versions_before
        and frozen_versions_after == frozen_versions_before
        and frozen_gradients_absent
    )
    if not parameter_integrity_gate:
        raise RuntimeError("Model parameters changed during scorer verification")
    print(f"trainable_parameter_sha256_after: {trainable_sha_after}")
    print(f"frozen_parameter_sha256_after: {frozen_sha_after}")
    print("parameter_integrity: PASS_NO_PARAMETER_UPDATE")

    summary = {
        "status": "PASS",
        "method_version": METHOD_VERSION,
        "mode": args.mode,
        "execution_commit": args.execution_commit,
        "scope": "scorer_equivalence_only_no_rewrite_no_optimizer_step",
        "candidate_count": len(result_rows),
        "reference_count": reference.example_count,
        "candidate_indices": candidate_indices,
        "trainable_tensor_count": scorer.trainable_tensor_count,
        "trainable_parameter_count": scorer.trainable_parameter_count,
        "gradient_mode": "eval_no_dropout",
        "completion_only_loss": True,
        "maximum_total_tokens": 1024,
        "reference_gradient_accumulation_dtype": "float32",
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "input_identity": {
            "status": "PASS",
            "critical_file_sha256": verified_sha256,
            "candidate_ordered_rows_sha256": candidate_ordered_rows_hash,
            "reference_ordered_rows_sha256": reference_ordered_rows_hash,
            "candidate_trace_identity": "PASS",
            "reference_solution_identity": "PASS",
            "adapter_identity": "PASS",
        },
        "reference_equivalence": {
            "status": "PASS",
            "frozen_loss_mean": frozen_reference_loss,
            "recomputed_loss_mean": recomputed_reference_loss,
            "loss_absolute_error": reference_loss_error,
            "loss_tolerance": reference_loss_tolerance,
            "frozen_mean_gradient_norm": frozen_reference_norm,
            "recomputed_mean_gradient_norm": recomputed_reference_norm,
            "norm_absolute_error": reference_norm_error,
            "norm_tolerance": reference_norm_tolerance,
        },
        "candidate_equivalence": {
            "status": "PASS",
            "per_candidate_dot_gate": "PASS",
            "per_candidate_loss_gate": "PASS",
            "per_candidate_gradient_norm_gate": "PASS",
            "per_candidate_cosine_gate": "PASS",
            "per_candidate_token_metadata_gate": "PASS",
            "normalized_dot_discrepancy_maximum": max(
                row["normalized_dot_discrepancy"]
                for row in result_rows
            ),
            "normalized_dot_discrepancy_limit": (
                NORMALIZED_DOT_DISCREPANCY_MAX
            ),
            "max_candidate_loss_absolute_error": max(
                row["candidate_loss_absolute_error"]
                for row in result_rows
            ),
            "max_candidate_gradient_norm_absolute_error": max(
                row["candidate_gradient_norm_absolute_error"]
                for row in result_rows
            ),
            "max_gradient_cosine_absolute_error": max(
                row["gradient_cosine_absolute_error"]
                for row in result_rows
            ),
            "gradient_cosine_absolute_error_limit": (
                COSINE_ABSOLUTE_ERROR_MAX
            ),
            "frozen_vs_recomputed_dot_spearman": dot_spearman,
            "frozen_vs_recomputed_cosine_spearman": cosine_spearman,
            "formal_spearman_minimum": REPRODUCIBILITY_SPEARMAN_MIN,
            "formal_dot_vector_gate": (
                "PASS"
                if formal_dot_vector_gate is True
                else "NOT_APPLICABLE_TO_SMOKE"
            ),
            "formal_cosine_vector_gate": (
                "PASS"
                if formal_cosine_vector_gate is True
                else "NOT_APPLICABLE_TO_SMOKE"
            ),
        },
        "parameter_integrity": {
            "status": "PASS_NO_PARAMETER_UPDATE",
            "trainable_sha256_before": trainable_sha_before,
            "trainable_sha256_after": trainable_sha_after,
            "frozen_sha256_before": frozen_sha_before,
            "frozen_sha256_after": frozen_sha_after,
            "trainable_version_counters_unchanged": True,
            "frozen_version_counters_unchanged": True,
            "frozen_gradients_absent": frozen_gradients_absent,
        },
        "pass_fail_criteria": {
            "scalar_relative_tolerance": SCALAR_RELATIVE_TOLERANCE,
            "scalar_absolute_floor": SCALAR_ABSOLUTE_FLOOR,
            "normalized_dot_discrepancy_maximum": (
                NORMALIZED_DOT_DISCREPANCY_MAX
            ),
            "cosine_absolute_error_maximum": (
                COSINE_ABSOLUTE_ERROR_MAX
            ),
            "formal_vector_spearman_minimum": (
                REPRODUCIBILITY_SPEARMAN_MIN
            ),
        },
        "interpretation_boundary": (
            "PASS establishes scorer implementation equivalence only; "
            "it is not evidence that Stage 6C improves rewrite quality "
            "or student accuracy."
        ),
    }

    rows_path = output_dir / "stage6c_scorer_equivalence_candidates.jsonl"
    summary_path = output_dir / "stage6c_scorer_equivalence_summary.json"
    if atomic_jsonl(rows_path, result_rows) != len(result_rows):
        raise RuntimeError("Candidate equivalence row count mismatch")
    atomic_json(summary_path, summary)

    print("\n===== FINAL SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"summary_path: {summary_path}")
    print(f"candidate_rows_path: {rows_path}")
    print(f"stage6c_scorer_equivalence_{args.mode}: PASS")


if __name__ == "__main__":
    main()
