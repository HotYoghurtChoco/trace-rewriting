"""Run the Stage 6C.2e five-problem H200 engineering batch preflight."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM


RUNTIME_REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (RUNTIME_REPO_ROOT / "src", RUNTIME_REPO_ROOT):
    import_root_text = str(import_root)
    if import_root_text not in sys.path:
        sys.path.insert(0, import_root_text)

from experiments.stage6c_cosrewrite_v1 import (  # noqa: E402
    run_single_problem_closed_loop as single,
)
from experiments.stage6c_cosrewrite_v1 import (  # noqa: E402
    verify_scorer_equivalence as equivalence,
)
from optimize.gradient_feedback import CompletionOnlyGradientScorer  # noqa: E402
from optimize.score_candidates import _load_training_tokenizer  # noqa: E402


METHOD = "stage6c_2e_five_problem_batch_preflight_v1_h200_native"
CLAIM_BOUNDARY = (
    "Engineering five-problem batch and resource preflight only: no formal "
    "protocol lock, no SelectionOnly implementation, no formal rewrite "
    "dataset, no method-effect claim, and no student/AF result."
)
SCHEMA = "stage6c_2e_five_problem_batch_result_v1"
ROW_INDICES = (0, 1, 2, 3, 4)
PROBLEM_GATE_NAMES = (
    "fixed_input_identity",
    "locked_row_index",
    "shared_h200_native_reference_reused",
    "round1_prompt_has_no_gradient_derived_feedback",
    "trace_repetition_validator_regression",
    "baseline_answer_and_trace_valid",
    "c1_answer_and_trace_valid",
    "c1_scored_by_frozen_qwen",
    "c2_received_exact_black_box_cosine_feedback",
    "c2_answer_and_trace_valid",
    "c2_scored_by_frozen_qwen",
    "generator_and_scorer_coexisted_on_h200",
    "scorer_parameters_unchanged",
    "cosine_decrease_not_required_for_engineering_pass",
    "engineering_final_selection_completed",
)
LOCK_REL = (
    "experiments/stage6c_cosrewrite_v1/config/"
    "stage6c_2e_batch_preflight_lock.json"
)
SINGLE_DRIVER_REL = (
    "experiments/stage6c_cosrewrite_v1/run_single_problem_closed_loop.py"
)
EXPECTED_LOCK_SHA256 = (
    "5f5a0dd4f46f71e002140d722d1896ac4d130d5e9cd3faa238437dd36e8bef35"
)
EXPECTED_SINGLE_DRIVER_SHA256 = (
    "345bf8977eb0f5be76b0690167c5d5cb9177a958381ce06584c2a9777b4d2aa0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-generation-attempts", type=int, default=2)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def marker_values(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def load_and_validate_lock(repo_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    path = repo_root / LOCK_REL
    single.require(path.is_file(), f"Missing Stage 6C.2e lock: {path}")
    actual_sha256 = single.file_sha256(path)
    single.require(
        actual_sha256 == EXPECTED_LOCK_SHA256,
        f"Unexpected Stage 6C.2e lock SHA256: {actual_sha256}",
    )
    lock = read_json(path)
    single.require(lock.get("method") == METHOD, "Unexpected method in 6C.2e lock")
    single.require(
        tuple(lock.get("cohort", {}).get("row_indices", [])) == ROW_INDICES,
        "Unexpected fixed cohort in 6C.2e lock",
    )
    generator = lock.get("generator", {})
    single.require(
        float(generator.get("temperature")) == args.temperature,
        "Temperature differs from 6C.2e lock",
    )
    single.require(
        int(generator.get("maximum_tokens")) == args.max_tokens,
        "Maximum tokens differ from 6C.2e lock",
    )
    single.require(
        str(generator.get("reasoning_effort")) == args.reasoning_effort,
        "Reasoning effort differs from 6C.2e lock",
    )
    single.require(
        int(generator.get("maximum_generation_attempts"))
        == args.max_generation_attempts,
        "Generation retry budget differs from 6C.2e lock",
    )
    single.require(
        lock.get("claim_boundary", {}).get("formal_protocol_locked") is False,
        "6C.2e was mislabeled as a formal protocol lock",
    )
    single.require(
        lock.get("claim_boundary", {}).get("method_effect_claimed") is False,
        "6C.2e unexpectedly claims a method effect",
    )
    return lock


def generation_seed(scoring_seed: int, row_index: int, offset: int) -> int:
    single.require(row_index in ROW_INDICES, "Seed requested for an unlocked row")
    single.require(offset in (1, 101), "Unexpected generation-seed offset")
    return scoring_seed * 1000 + row_index * 1000 + offset


def batch_regression(lock: Mapping[str, Any], scoring_seed: int) -> dict[str, Any]:
    seeds = {
        f"row_{row_index:04d}_c1": generation_seed(scoring_seed, row_index, 1)
        for row_index in ROW_INDICES
    }
    seeds.update(
        {
            f"row_{row_index:04d}_c2": generation_seed(
                scoring_seed,
                row_index,
                101,
            )
            for row_index in ROW_INDICES
        }
    )
    repetition = single.repetition_validator_regression()
    gates = {
        "five_unique_locked_rows": len(set(ROW_INDICES)) == 5,
        "lock_rows_match_driver_rows": tuple(
            lock.get("cohort", {}).get("row_indices", [])
        )
        == ROW_INDICES,
        "ten_unique_generation_seeds": len(set(seeds.values())) == 10,
        "row_zero_c1_seed_matches_v6": seeds["row_0000_c1"]
        == scoring_seed * 1000 + 1,
        "row_zero_c2_seed_matches_v6": seeds["row_0000_c2"]
        == scoring_seed * 1000 + 101,
        "trace_repetition_regression": repetition.get("status") == "PASS",
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "generation_seeds": seeds,
        "trace_repetition_regression": repetition,
    }


def selected_cohort(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cohort = []
    for row_index in ROW_INDICES:
        row = candidate_rows[row_index]
        hashes = {
            key: single.text_sha256(str(row[key]))
            for key in ("problem", "solution", "original_trace", "rewrite_trace")
        }
        cohort.append(
            {
                "row_index": row_index,
                "field_sha256": hashes,
                "row_identity_sha256": canonical_json_sha256(hashes),
            }
        )
    return cohort


def build_batch_plan(
    *,
    args: argparse.Namespace,
    lock: Mapping[str, Any],
    inputs: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fingerprint_payload = {
        "method": METHOD,
        "execution_commit": args.execution_commit,
        "lock_sha256": EXPECTED_LOCK_SHA256,
        "single_driver_sha256": EXPECTED_SINGLE_DRIVER_SHA256,
        "row_indices": list(ROW_INDICES),
        "cohort": list(cohort),
        "generator": lock["generator"],
        "scorer": lock["scorer"],
        "controller": lock["controller"],
        "candidate_ordered_rows_sha256": inputs["preflight"][
            "candidate_ordered_rows_sha256"
        ],
        "reference_ordered_rows_sha256": inputs["preflight"][
            "reference_ordered_rows_sha256"
        ],
        "verified_fixed_input_sha256": inputs["verified_fixed_input_sha256"],
    }
    return {
        "schema": "stage6c_2e_batch_plan_v1",
        "method": METHOD,
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_commit": args.execution_commit,
        "job_id": args.job_id,
        "row_indices": list(ROW_INDICES),
        "cohort": list(cohort),
        "lock_sha256": EXPECTED_LOCK_SHA256,
        "single_driver_sha256": EXPECTED_SINGLE_DRIVER_SHA256,
        "config_fingerprint": canonical_json_sha256(fingerprint_payload),
        "fingerprint_payload": fingerprint_payload,
    }


def artifact_records(problem_dir: Path) -> list[dict[str, Any]]:
    excluded = {
        "artifact_manifest.json",
        "PROBLEM_COMPLETE",
        "PROBLEM_FAILED",
    }
    records = []
    for path in sorted(item for item in problem_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(problem_dir).as_posix()
        if relative in excluded or relative.endswith(".partial"):
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": single.file_sha256(path),
            }
        )
    return records


def validate_problem_checkpoint(
    problem_dir: Path,
    *,
    row_index: int,
    execution_commit: str,
    config_fingerprint: str,
) -> dict[str, Any]:
    complete_path = problem_dir / "PROBLEM_COMPLETE"
    failed_path = problem_dir / "PROBLEM_FAILED"
    manifest_path = problem_dir / "artifact_manifest.json"
    result_path = problem_dir / "problem_result.json"
    single.require(complete_path.is_file(), f"Missing checkpoint marker: {complete_path}")
    single.require(not failed_path.exists(), f"Conflicting failed marker: {failed_path}")
    single.require(manifest_path.is_file(), f"Missing checkpoint manifest: {manifest_path}")
    single.require(result_path.is_file(), f"Missing checkpoint result: {result_path}")
    marker = marker_values(complete_path)
    single.require(marker.get("status") == "PASS", "Checkpoint marker is not PASS")
    single.require(marker.get("method") == METHOD, "Checkpoint method mismatch")
    single.require(int(marker.get("row_index", "-1")) == row_index, "Checkpoint row mismatch")
    single.require(
        marker.get("execution_commit") == execution_commit,
        "Checkpoint execution commit mismatch",
    )
    single.require(
        marker.get("config_fingerprint") == config_fingerprint,
        "Checkpoint config fingerprint mismatch",
    )
    single.require(
        marker.get("manifest_sha256") == single.file_sha256(manifest_path),
        "Checkpoint manifest SHA256 mismatch",
    )
    manifest = read_json(manifest_path)
    single.require(
        manifest.get("schema") == "stage6c_2e_problem_artifact_manifest_v1",
        "Checkpoint manifest schema mismatch",
    )
    single.require(manifest.get("method") == METHOD, "Checkpoint manifest method mismatch")
    single.require(
        int(manifest.get("row_index", -1)) == row_index,
        "Checkpoint manifest row mismatch",
    )
    single.require(
        manifest.get("execution_commit") == execution_commit,
        "Checkpoint manifest execution commit mismatch",
    )
    single.require(
        manifest.get("config_fingerprint") == config_fingerprint,
        "Checkpoint manifest fingerprint mismatch",
    )
    manifest_records = manifest.get("files")
    single.require(
        isinstance(manifest_records, list) and bool(manifest_records),
        "Checkpoint manifest has no artifact records",
    )
    manifest_paths = [str(record.get("path")) for record in manifest_records]
    single.require(
        len(manifest_paths) == len(set(manifest_paths)),
        "Checkpoint manifest contains duplicate paths",
    )
    for record in manifest_records:
        relative = Path(str(record["path"]))
        single.require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe relative path in checkpoint manifest",
        )
        artifact = problem_dir / relative
        single.require(artifact.is_file(), f"Missing checkpoint artifact: {artifact}")
        single.require(
            artifact.stat().st_size == int(record["size_bytes"]),
            f"Checkpoint size mismatch: {relative}",
        )
        single.require(
            single.file_sha256(artifact) == record["sha256"],
            f"Checkpoint SHA256 mismatch: {relative}",
        )
    single.require(
        manifest_records == artifact_records(problem_dir),
        "Checkpoint manifest does not match the complete artifact inventory",
    )
    result = read_json(result_path)
    single.require(result.get("status") == "PASS", "Checkpoint result is not PASS")
    single.require(result.get("method") == METHOD, "Checkpoint result method mismatch")
    single.require(int(result.get("row_index", -1)) == row_index, "Checkpoint result row mismatch")
    single.require(
        result.get("execution_commit") == execution_commit,
        "Checkpoint result execution commit mismatch",
    )
    single.require(
        result.get("config_fingerprint") == config_fingerprint,
        "Checkpoint result fingerprint mismatch",
    )
    result_gates = result.get("gates")
    single.require(isinstance(result_gates, dict), "Checkpoint result gates are absent")
    single.require(
        set(result_gates) == set(PROBLEM_GATE_NAMES),
        "Checkpoint result gate inventory mismatch",
    )
    single.require(all(result_gates.values()), "Checkpoint result contains a failed gate")
    return result


def copy_resumable_checkpoints(
    *,
    resume_from: Path | None,
    output_dir: Path,
    execution_commit: str,
    plan: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if resume_from is None:
        return {}, {
            "requested": False,
            "source": None,
            "source_plan_valid": None,
            "resumed_row_indices": [],
        }
    source = resume_from.resolve()
    single.require(source.is_dir(), f"Resume source is not a directory: {source}")
    single.require(source != output_dir, "Resume source equals the new output directory")
    source_plan_path = source / "batch_plan.json"
    single.require(source_plan_path.is_file(), "Resume source has no batch plan")
    source_plan = read_json(source_plan_path)
    single.require(source_plan.get("method") == METHOD, "Resume source method mismatch")
    single.require(
        source_plan.get("execution_commit") == execution_commit,
        "Resume source execution commit mismatch",
    )
    single.require(
        source_plan.get("config_fingerprint") == plan["config_fingerprint"],
        "Resume source config fingerprint mismatch",
    )
    resumed = {}
    for row_index in ROW_INDICES:
        source_problem_dir = source / "problems" / f"row_{row_index:04d}"
        if not (source_problem_dir / "PROBLEM_COMPLETE").is_file():
            continue
        result = validate_problem_checkpoint(
            source_problem_dir,
            row_index=row_index,
            execution_commit=execution_commit,
            config_fingerprint=str(plan["config_fingerprint"]),
        )
        destination = output_dir / "problems" / f"row_{row_index:04d}"
        temporary = destination.with_name(destination.name + ".partial")
        single.require(not destination.exists(), f"Resume destination exists: {destination}")
        single.require(not temporary.exists(), f"Resume temporary exists: {temporary}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_problem_dir, temporary)
        os.replace(temporary, destination)
        validate_problem_checkpoint(
            destination,
            row_index=row_index,
            execution_commit=execution_commit,
            config_fingerprint=str(plan["config_fingerprint"]),
        )
        resumed[row_index] = result
    return resumed, {
        "requested": True,
        "source": str(source),
        "source_plan_valid": True,
        "resumed_row_indices": sorted(resumed),
    }


def prepare_scorer_and_reference(
    *,
    inputs: Mapping[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    single.require(torch.cuda.is_available(), "CUDA is unavailable to the Qwen scorer")
    single.require(torch.cuda.is_bf16_supported(), "CUDA device does not support BF16")
    scoring_seed = int(inputs["config"]["seed"])
    random.seed(scoring_seed)
    np.random.seed(scoring_seed)
    torch.manual_seed(scoring_seed)
    torch.cuda.manual_seed_all(scoring_seed)
    device = torch.device("cuda")
    compute_dtype = torch.bfloat16

    single.nvidia_snapshot(output_dir / "gpu_memory_before_scorer_load.csv")
    tokenizer = _load_training_tokenizer(inputs["tokenizer_name"])
    single.require(tokenizer.pad_token_id is not None, "Tokenizer has no pad token")
    single.require(tokenizer.eos_token_id is not None, "Tokenizer has no EOS token")
    base_model = AutoModelForCausalLM.from_pretrained(
        inputs["model_name"],
        torch_dtype=compute_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(inputs["adapter_path"]),
        is_trainable=True,
        local_files_only=True,
    )
    model.to(device)
    scorer = CompletionOnlyGradientScorer(
        model=model,
        tokenizer=tokenizer,
        instruction=inputs["config"]["instruction_generation"],
        model_name=inputs["model_name"],
        device=device,
        compute_dtype=compute_dtype,
        maximum_total_tokens=1024,
    )
    single.require(
        scorer.trainable_tensor_count == equivalence.EXPECTED_TRAINABLE_TENSORS,
        "Unexpected trainable tensor count",
    )
    single.require(
        scorer.trainable_parameter_count
        == equivalence.EXPECTED_TRAINABLE_PARAMETERS,
        "Unexpected trainable parameter count",
    )
    trainable = list(scorer.trainable)
    parameter_versions_before = {
        name: parameter._version for name, parameter in model.named_parameters()
    }
    trainable_sha_before = equivalence.named_parameter_sha256(trainable)
    reference = scorer.mean_reference_gradient(
        inputs["reference_rows"],
        progress_every=10,
        progress_callback=lambda done, total: print(
            f"reference_progress={done}/{total}",
            flush=True,
        ),
    )
    reference_count_gate = int(reference.example_count) == 100
    reference_truncated_count = sum(
        int(item["tokens"]["completion_tokens_removed"] > 0)
        for item in reference.per_example
    )
    reference_token_gate = reference_truncated_count == 0
    h200_reference_loss = float(reference.loss_mean)
    h200_reference_norm = float(reference.mean_gradient_norm)
    reference_loss_gate = math.isfinite(h200_reference_loss)
    reference_norm_gate = (
        math.isfinite(h200_reference_norm) and h200_reference_norm > 0.0
    )
    loss_error = abs(
        h200_reference_loss - single.H200_REFERENCE_SENTINEL["loss_expected"]
    )
    norm_error = abs(
        h200_reference_norm
        - single.H200_REFERENCE_SENTINEL["mean_gradient_norm_expected"]
    )
    loss_sentinel_gate = (
        loss_error <= single.H200_REFERENCE_SENTINEL["loss_tolerance"]
    )
    norm_sentinel_gate = (
        norm_error
        <= single.H200_REFERENCE_SENTINEL["mean_gradient_norm_tolerance"]
    )
    h200_device_gate = "H200" in torch.cuda.get_device_name(device)
    parameter_versions_after_reference = {
        name: parameter._version for name, parameter in model.named_parameters()
    }
    trainable_sha_after_reference = equivalence.named_parameter_sha256(trainable)
    versions_unchanged = parameter_versions_after_reference == parameter_versions_before
    sha_unchanged = trainable_sha_after_reference == trainable_sha_before
    gates = {
        "reference_count_100": reference_count_gate,
        "no_reference_completion_truncated": reference_token_gate,
        "reference_loss_finite": reference_loss_gate,
        "reference_norm_finite_positive": reference_norm_gate,
        "reference_loss_matches_h200_v4_sentinel": loss_sentinel_gate,
        "reference_norm_matches_h200_v4_sentinel": norm_sentinel_gate,
        "device_is_h200": h200_device_gate,
        "parameter_versions_unchanged": versions_unchanged,
        "trainable_sha256_unchanged": sha_unchanged,
    }
    h200_native_reference_gate = all(gates.values())
    per_example_path = output_dir / "reference_recompute_per_example.jsonl"
    single.atomic_jsonl(per_example_path, reference.per_example)
    diagnostics = {
        "schema": "stage6c_2e_h200_native_reference_diagnostics_v1",
        "status": "PASS" if h200_native_reference_gate else "FAIL",
        "method": METHOD,
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_commit": args.execution_commit,
        "job_id": args.job_id,
        "scorer_runtime_policy": single.SCORER_RUNTIME_POLICY,
        "policy_evidence": single.H200_POLICY_EVIDENCE,
        "a100_scalar_or_score_equivalence_gate_applied": False,
        "mean_reference_gradient_recomputations": 1,
        "reference_count": int(reference.example_count),
        "reference_per_example_file_sha256": single.file_sha256(per_example_path),
        "reference_loss": {
            "h200_native": h200_reference_loss,
            "sentinel_expected": single.H200_REFERENCE_SENTINEL["loss_expected"],
            "sentinel_absolute_error": loss_error,
            "sentinel_tolerance": single.H200_REFERENCE_SENTINEL["loss_tolerance"],
        },
        "reference_mean_gradient_norm": {
            "h200_native": h200_reference_norm,
            "sentinel_expected": single.H200_REFERENCE_SENTINEL[
                "mean_gradient_norm_expected"
            ],
            "sentinel_absolute_error": norm_error,
            "sentinel_tolerance": single.H200_REFERENCE_SENTINEL[
                "mean_gradient_norm_tolerance"
            ],
        },
        "gates": gates,
        "model": {
            "base_model_path": str(inputs["model_name"]),
            "tokenizer_path": str(inputs["tokenizer_name"]),
            "adapter_path": str(inputs["adapter_path"]),
            "compute_dtype": str(compute_dtype),
            "trainable_tensor_count": int(scorer.trainable_tensor_count),
            "trainable_parameter_count": int(scorer.trainable_parameter_count),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "torch_compiled_cuda": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_memory": single.torch_memory_snapshot(),
        },
        "parameter_integrity_after_reference": {
            "versions_unchanged": versions_unchanged,
            "trainable_sha256_before": trainable_sha_before,
            "trainable_sha256_after": trainable_sha_after_reference,
            "trainable_sha256_unchanged": sha_unchanged,
        },
        "verified_source_sha256": inputs["verified_source_sha256"],
        "verified_fixed_input_sha256": inputs["verified_fixed_input_sha256"],
    }
    single.atomic_json(
        output_dir / "reference_recompute_diagnostics.json",
        diagnostics,
    )
    print(
        "h200_native_reference_loss_gate={} value={:.17g} error={:.17g} "
        "tolerance={:.17g}".format(
            "PASS" if reference_loss_gate and loss_sentinel_gate else "FAIL",
            h200_reference_loss,
            loss_error,
            single.H200_REFERENCE_SENTINEL["loss_tolerance"],
        ),
        flush=True,
    )
    print(
        "h200_native_reference_norm_gate={} value={:.17g} error={:.17g} "
        "tolerance={:.17g}".format(
            "PASS" if reference_norm_gate and norm_sentinel_gate else "FAIL",
            h200_reference_norm,
            norm_error,
            single.H200_REFERENCE_SENTINEL["mean_gradient_norm_tolerance"],
        ),
        flush=True,
    )
    single.require(
        h200_native_reference_gate,
        "H200-native reference validity or scorer-integrity gate failed",
    )
    single.nvidia_snapshot(output_dir / "gpu_memory_server_and_scorer_ready.csv")
    single.atomic_json(
        output_dir / "torch_memory_server_and_scorer_ready.json",
        single.torch_memory_snapshot(),
    )
    return {
        "base_model": base_model,
        "compute_dtype": compute_dtype,
        "device": device,
        "diagnostics": diagnostics,
        "h200_native_reference_gate": h200_native_reference_gate,
        "model": model,
        "parameter_versions_before": parameter_versions_before,
        "reference": reference,
        "scorer": scorer,
        "tokenizer": tokenizer,
        "trainable": trainable,
        "trainable_sha_before": trainable_sha_before,
    }


def process_problem(
    *,
    row_index: int,
    row: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    problem_dir: Path,
    config_fingerprint: str,
    inputs: Mapping[str, Any],
    state: Mapping[str, Any],
    args: argparse.Namespace,
    regression: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.time()
    single.require(not problem_dir.exists(), f"Problem directory exists: {problem_dir}")
    api_dir = problem_dir / "api"
    api_dir.mkdir(parents=True)
    problem = str(row["problem"])
    solution = str(row["solution"])
    baseline = str(row["rewrite_trace"])
    field_sha256 = {
        key: single.text_sha256(str(row[key]))
        for key in ("problem", "solution", "original_trace", "rewrite_trace")
    }
    fixed_input_identity = (
        int(expected_identity.get("row_index", -1)) == row_index
        and expected_identity.get("field_sha256") == field_sha256
        and expected_identity.get("row_identity_sha256")
        == canonical_json_sha256(field_sha256)
    )
    input_identity = {
        "row_index": row_index,
        "field_sha256": field_sha256,
        "row_identity_sha256": canonical_json_sha256(field_sha256),
        "matches_locked_cohort": fixed_input_identity,
    }
    single.atomic_json(problem_dir / "input_identity.json", input_identity)
    baseline_validity = single.validate_trace(
        solution=solution,
        trace=baseline,
        parents={},
        finish_reason=None,
    )
    c1_user_prompt = (
        f"Problem:\n{problem}\n\n"
        f"Current trace:\n{baseline}\n\n"
        "Create one materially different but concise complete solution trace for "
        "the same problem. Preserve the same correct final answer."
    )
    c1_prompt_normalized = single.normalize_text(single.SYSTEM_PROMPT + "\n" + c1_user_prompt)
    round1_leakage = [
        term for term in single.ROUND1_FORBIDDEN_TERMS if term in c1_prompt_normalized
    ]
    single.require(
        not round1_leakage,
        f"Round-1 prompt contains forbidden feedback language: {round1_leakage}",
    )
    scoring_seed = int(inputs["config"]["seed"])
    c1, c1_attempts = single.generate_candidate(
        label="c1",
        user_prompt=c1_user_prompt,
        solution=solution,
        parents={"BaselineRewrite": baseline},
        seed_start=generation_seed(scoring_seed, row_index, 1),
        args=args,
        api_dir=api_dir,
    )
    single.atomic_json(problem_dir / "c1_attempts.json", c1_attempts)

    scorer = state["scorer"]
    reference = state["reference"]
    baseline_score = single.score_trace(scorer, reference, problem, baseline)
    c1_score = single.score_trace(scorer, reference, problem, c1["content"])
    single.add_scorer_token_gate(baseline_validity, baseline_score)
    single.add_scorer_token_gate(c1["validity"], c1_score)
    if baseline_validity["valid"] and c1["validity"]["valid"]:
        c2_parent_label = "C1"
        c2_parent = c1["content"]
        feedback_target_score = c1_score
    else:
        c2_parent_label = "BaselineRewrite"
        c2_parent = baseline
        feedback_target_score = baseline_score
    rendered_feedback_value = f"{float(feedback_target_score['gradient_cosine']):.12f}"
    feedback_text = (
        "Black-box scorer feedback for the current parent:\n"
        "- primary metric: gradient cosine with the fixed mean reference gradient\n"
        f"- measured value: {rendered_feedback_value}\n"
        "- target direction: lower is better\n"
        "Use only this scalar summary; no token-level attribution is available."
    )
    c2_user_prompt = (
        f"Problem:\n{problem}\n\n"
        f"Current parent trace ({c2_parent_label}):\n{c2_parent}\n\n"
        f"{feedback_text}\n\n"
        "Create one materially different but concise complete solution trace. "
        "Preserve the same correct final answer. Do not mention the feedback or "
        "evaluation in the returned trace."
    )
    normalized_c2_prompt = single.normalize_text(c2_user_prompt)
    single.require("gradient cosine" in normalized_c2_prompt, "C2 prompt lacks cosine feedback")
    single.require(
        rendered_feedback_value in c2_user_prompt,
        "C2 prompt lacks exact feedback value",
    )
    single.require("lower is better" in normalized_c2_prompt, "C2 prompt lacks feedback direction")
    c2, c2_attempts = single.generate_candidate(
        label="c2",
        user_prompt=c2_user_prompt,
        solution=solution,
        parents={"BaselineRewrite": baseline, "C1": c1["content"]},
        seed_start=generation_seed(scoring_seed, row_index, 101),
        args=args,
        api_dir=api_dir,
    )
    single.atomic_json(problem_dir / "c2_attempts.json", c2_attempts)
    c2_score = single.score_trace(scorer, reference, problem, c2["content"])
    single.add_scorer_token_gate(c2["validity"], c2_score)
    trainable_sha_after = equivalence.named_parameter_sha256(state["trainable"])
    parameter_versions_after = {
        name: parameter._version
        for name, parameter in state["model"].named_parameters()
    }
    parameter_integrity = (
        state["trainable_sha_before"] == trainable_sha_after
        and state["parameter_versions_before"] == parameter_versions_after
    )
    prompt_audit = {
        "round1_shared_pool_compatible": not round1_leakage,
        "round1_forbidden_terms_found": round1_leakage,
        "c1_prompt_sha256": single.text_sha256(
            single.SYSTEM_PROMPT + "\n" + c1_user_prompt
        ),
        "c2_prompt_sha256": single.text_sha256(
            single.SYSTEM_PROMPT + "\n" + c2_user_prompt
        ),
        "c2_contains_external_black_box_scalar": True,
        "c2_contains_exact_feedback_value": rendered_feedback_value in c2_user_prompt,
        "c2_feedback_has_token_attribution": False,
    }
    gates = {
        "fixed_input_identity": fixed_input_identity,
        "locked_row_index": row_index in ROW_INDICES,
        "shared_h200_native_reference_reused": state[
            "h200_native_reference_gate"
        ],
        "round1_prompt_has_no_gradient_derived_feedback": not round1_leakage,
        "trace_repetition_validator_regression": regression.get("status") == "PASS",
        "baseline_answer_and_trace_valid": bool(baseline_validity["valid"]),
        "c1_answer_and_trace_valid": bool(c1["validity"]["valid"]),
        "c1_scored_by_frozen_qwen": True,
        "c2_received_exact_black_box_cosine_feedback": bool(
            prompt_audit["c2_contains_exact_feedback_value"]
        ),
        "c2_answer_and_trace_valid": bool(c2["validity"]["valid"]),
        "c2_scored_by_frozen_qwen": True,
        "generator_and_scorer_coexisted_on_h200": True,
        "scorer_parameters_unchanged": parameter_integrity,
        "cosine_decrease_not_required_for_engineering_pass": True,
    }
    lineage_rows = [
        {
            "candidate_id": "BaselineRewrite",
            "round": 0,
            "parent_id": None,
            "trace": baseline,
            "trace_sha256": single.text_sha256(baseline),
            "validity": baseline_validity,
            "score": baseline_score,
        },
        {
            "candidate_id": "C1",
            "round": 1,
            "parent_id": "BaselineRewrite",
            "trace": c1["content"],
            "trace_sha256": single.text_sha256(c1["content"]),
            "generation_attempt": c1["attempt"],
            "generation_seed": c1["seed"],
            "validity": c1["validity"],
            "score": c1_score,
        },
        {
            "candidate_id": "C2",
            "round": 2,
            "parent_id": c2_parent_label,
            "parent_trace_sha256": single.text_sha256(c2_parent),
            "trace": c2["content"],
            "trace_sha256": single.text_sha256(c2["content"]),
            "generation_attempt": c2["attempt"],
            "generation_seed": c2["seed"],
            "validity": c2["validity"],
            "score": c2_score,
        },
    ]
    selection = single.select_engineering_winner(lineage_rows)
    single.atomic_json(problem_dir / "engineering_selection.json", selection)
    gates["engineering_final_selection_completed"] = selection["status"] == "PASS"
    single.require(
        set(gates) == set(PROBLEM_GATE_NAMES),
        "Internal per-problem gate inventory mismatch",
    )
    status = "PASS" if all(gates.values()) else "FAIL"
    single.atomic_jsonl(problem_dir / "trace_lineage.jsonl", lineage_rows)
    cosine_diagnostics = {
        "baseline": float(baseline_score["gradient_cosine"]),
        "c1": float(c1_score["gradient_cosine"]),
        "c2": float(c2_score["gradient_cosine"]),
        "c1_minus_baseline": float(c1_score["gradient_cosine"])
        - float(baseline_score["gradient_cosine"]),
        "c2_minus_c1": float(c2_score["gradient_cosine"])
        - float(c1_score["gradient_cosine"]),
        "c2_minus_baseline": float(c2_score["gradient_cosine"])
        - float(baseline_score["gradient_cosine"]),
        "improvement_is_not_a_pass_gate": True,
    }
    result = {
        "schema": "stage6c_2e_problem_result_v1",
        "status": status,
        "method": METHOD,
        "claim_boundary": CLAIM_BOUNDARY,
        "job_id": args.job_id,
        "execution_commit": args.execution_commit,
        "config_fingerprint": config_fingerprint,
        "row_index": row_index,
        "input_identity": input_identity,
        "model_id": single.MODEL_ID,
        "runtime_policy": {
            "name": single.SCORER_RUNTIME_POLICY,
            "device": torch.cuda.get_device_name(state["device"]),
            "fixed_data_model_and_scorer_definition": True,
            "a100_scalar_or_score_equivalence_required": False,
        },
        "generator_settings": {
            "status": "engineering_candidate_settings_not_formal_protocol",
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "maximum_generation_attempts": args.max_generation_attempts,
            "c1_seed_start": generation_seed(scoring_seed, row_index, 1),
            "c2_seed_start": generation_seed(scoring_seed, row_index, 101),
        },
        "feedback": {
            "target_candidate": c2_parent_label,
            "metric": "gradient_cosine",
            "value": float(feedback_target_score["gradient_cosine"]),
            "rendered_value": rendered_feedback_value,
            "direction": "lower_is_better",
            "token_attribution_claimed": False,
            "feedback_text_sha256": single.text_sha256(feedback_text),
        },
        "prompt_audit": prompt_audit,
        "lineage": {
            "c1_parent": "BaselineRewrite",
            "c2_parent": c2_parent_label,
            "fallback_rule": (
                "Use C1 only when BaselineRewrite and C1 pass answer, trace, "
                "and scorer-token gates; otherwise retain BaselineRewrite as "
                "C2 parent."
            ),
            "formal_parent_selection_rule_locked": False,
        },
        "cosine_diagnostics": cosine_diagnostics,
        "engineering_final_selection": selection,
        "gates": gates,
        "elapsed_seconds": time.time() - started,
        "selection_only_implemented": False,
        "formal_rewrite_dataset_created": False,
        "student_trained": False,
        "af_measured": False,
    }
    single.atomic_json(problem_dir / "problem_result.json", result)
    single.nvidia_snapshot(problem_dir / "gpu_memory_after_problem.csv")
    manifest = {
        "schema": "stage6c_2e_problem_artifact_manifest_v1",
        "method": METHOD,
        "row_index": row_index,
        "execution_commit": args.execution_commit,
        "config_fingerprint": config_fingerprint,
        "files": artifact_records(problem_dir),
    }
    single.atomic_json(problem_dir / "artifact_manifest.json", manifest)
    manifest_sha256 = single.file_sha256(problem_dir / "artifact_manifest.json")
    marker_name = "PROBLEM_COMPLETE" if status == "PASS" else "PROBLEM_FAILED"
    atomic_text(
        problem_dir / marker_name,
        "\n".join(
            [
                f"status={status}",
                f"method={METHOD}",
                f"row_index={row_index}",
                f"execution_commit={args.execution_commit}",
                f"config_fingerprint={config_fingerprint}",
                f"manifest_sha256={manifest_sha256}",
                "",
            ]
        ),
    )
    return result


def failure_result(
    *,
    row_index: int,
    problem_dir: Path,
    args: argparse.Namespace,
    config_fingerprint: str,
    error: Exception,
) -> dict[str, Any]:
    problem_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "stage6c_2e_problem_failure_v1",
        "status": "FAIL",
        "method": METHOD,
        "row_index": row_index,
        "execution_commit": args.execution_commit,
        "config_fingerprint": config_fingerprint,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "gates": {},
    }
    single.atomic_json(problem_dir / "problem_failure.json", result)
    atomic_text(
        problem_dir / "PROBLEM_FAILED",
        "\n".join(
            [
                "status=FAIL",
                f"method={METHOD}",
                f"row_index={row_index}",
                f"execution_commit={args.execution_commit}",
                f"config_fingerprint={config_fingerprint}",
                f"error_type={type(error).__name__}",
                "",
            ]
        ),
    )
    return result


def summarize_results(
    *,
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    regression: Mapping[str, Any],
    reference_diagnostics: Mapping[str, Any],
    results: Mapping[int, Mapping[str, Any]],
    checkpoint_audit: Mapping[str, Any],
    resume_audit: Mapping[str, Any],
    parameter_integrity: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    passed = [row for row in results.values() if row.get("status") == "PASS"]
    winner_counts = Counter(
        str(row.get("engineering_final_selection", {}).get("winner_candidate_id"))
        for row in passed
    )
    c2_parent_counts = Counter(
        str(row.get("lineage", {}).get("c2_parent")) for row in passed
    )
    c2_minus_c1 = [
        float(row["cosine_diagnostics"]["c2_minus_c1"]) for row in passed
    ]
    c2_minus_baseline = [
        float(row["cosine_diagnostics"]["c2_minus_baseline"])
        for row in passed
    ]
    gates = {
        "fixed_five_row_cohort": sorted(results) == list(ROW_INDICES),
        "batch_regression": regression.get("status") == "PASS",
        "h200_native_reference_valid": reference_diagnostics.get("status") == "PASS",
        "all_five_problem_results_pass": len(passed) == len(ROW_INDICES),
        "all_checkpoint_roundtrips_valid": checkpoint_audit.get("status") == "PASS",
        "scorer_parameters_unchanged": parameter_integrity,
        "resume_capability_implemented": True,
        "cosine_decrease_not_required_for_engineering_pass": True,
        "formal_claims_disabled": all(
            lock["claim_boundary"][key] is False
            for key in (
                "formal_protocol_locked",
                "formal_rewrite_dataset_created",
                "method_effect_claimed",
                "selection_only_implemented",
                "student_trained",
                "af_measured",
            )
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "schema": SCHEMA,
        "status": status,
        "method": METHOD,
        "claim_boundary": CLAIM_BOUNDARY,
        "job_id": args.job_id,
        "execution_commit": args.execution_commit,
        "config_fingerprint": plan["config_fingerprint"],
        "row_indices": list(ROW_INDICES),
        "problem_count": len(ROW_INDICES),
        "passed_problem_count": len(passed),
        "failed_problem_count": len(ROW_INDICES) - len(passed),
        "gates": gates,
        "batch_regression": regression,
        "reference": {
            "status": reference_diagnostics.get("status"),
            "mean_reference_gradient_recomputations": 1,
            "reference_count": reference_diagnostics.get("reference_count"),
            "scorer_runtime_policy": single.SCORER_RUNTIME_POLICY,
        },
        "checkpoint_audit": checkpoint_audit,
        "resume_audit": resume_audit,
        "descriptive_cosine_only": {
            "inference_performed": False,
            "winner_counts": dict(sorted(winner_counts.items())),
            "c2_parent_counts": dict(sorted(c2_parent_counts.items())),
            "c2_lower_than_c1_count": sum(value < 0.0 for value in c2_minus_c1),
            "c2_equal_to_c1_count": sum(value == 0.0 for value in c2_minus_c1),
            "c2_higher_than_c1_count": sum(value > 0.0 for value in c2_minus_c1),
            "c2_lower_than_baseline_count": sum(
                value < 0.0 for value in c2_minus_baseline
            ),
            "c2_equal_to_baseline_count": sum(
                value == 0.0 for value in c2_minus_baseline
            ),
            "c2_higher_than_baseline_count": sum(
                value > 0.0 for value in c2_minus_baseline
            ),
            "c2_minus_c1_values": c2_minus_c1,
            "c2_minus_baseline_values": c2_minus_baseline,
            "improvement_is_not_a_pass_gate": True,
        },
        "problem_results": [
            {
                "row_index": row_index,
                "status": result.get("status"),
                "winner_candidate_id": result.get(
                    "engineering_final_selection",
                    {},
                ).get("winner_candidate_id"),
                "c2_parent": result.get("lineage", {}).get("c2_parent"),
                "cosine_diagnostics": result.get("cosine_diagnostics"),
                "false_gates": [
                    name
                    for name, value in result.get("gates", {}).items()
                    if not value
                ],
            }
            for row_index, result in sorted(results.items())
        ],
        "elapsed_seconds": elapsed_seconds,
        "resources_locked": lock["resources"],
        "selection_only_implemented": False,
        "formal_protocol_locked": False,
        "formal_rewrite_dataset_created": False,
        "student_trained": False,
        "af_measured": False,
        "scientific_result": False,
    }


def run(args: argparse.Namespace) -> int:
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    single.require(
        not (output_dir / "batch_result.json").exists(),
        "Batch output already exists",
    )
    single.require(args.max_generation_attempts > 0, "Retry budget must be positive")
    single.require(0.0 <= args.temperature <= 2.0, "Temperature outside [0, 2]")
    single.require(args.max_tokens > 0, "Maximum tokens must be positive")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=RUNTIME_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    single.require(
        head == args.execution_commit,
        "Batch execution commit differs from repository HEAD",
    )
    single_driver_path = RUNTIME_REPO_ROOT / SINGLE_DRIVER_REL
    single.require(
        single.file_sha256(single_driver_path) == EXPECTED_SINGLE_DRIVER_SHA256,
        "Imported v6 single-problem driver SHA256 mismatch",
    )
    lock = load_and_validate_lock(RUNTIME_REPO_ROOT, args)
    inputs = single.load_inputs(RUNTIME_REPO_ROOT)
    scoring_seed = int(inputs["config"]["seed"])
    regression = batch_regression(lock, scoring_seed)
    single.atomic_json(output_dir / "batch_regression.json", regression)
    single.require(regression["status"] == "PASS", "6C.2e batch regression failed")
    print("batch_regression_gate=PASS", flush=True)

    candidate_path = inputs["formal_root"] / inputs["candidate_name"]
    dataset = load_from_disk(str(candidate_path))
    candidate_rows = [dataset[index] for index in range(len(dataset))]
    single.require(len(candidate_rows) == 100, "Candidate dataset size changed")
    cohort = selected_cohort(candidate_rows)
    plan = build_batch_plan(
        args=args,
        lock=lock,
        inputs=inputs,
        cohort=cohort,
    )
    single.atomic_json(output_dir / "batch_plan.json", plan)
    resumed_results, resume_audit = copy_resumable_checkpoints(
        resume_from=args.resume_from,
        output_dir=output_dir,
        execution_commit=args.execution_commit,
        plan=plan,
    )
    single.atomic_json(output_dir / "resume_audit.json", resume_audit)
    pending = [row_index for row_index in ROW_INDICES if row_index not in resumed_results]
    single.require(pending, "All five rows were already complete; no new work is required")
    print(f"pending_row_indices={','.join(str(value) for value in pending)}", flush=True)

    model_list = single.get_json(f"{args.api_base_url.rstrip('/')}/models")
    single.atomic_json(output_dir / "api_models_before_batch.json", model_list)
    model_ids = [
        item.get("id")
        for item in model_list.get("data", [])
        if isinstance(item, dict)
    ]
    single.require(
        single.MODEL_ID in model_ids,
        f"Served model list does not contain {single.MODEL_ID}",
    )

    state = prepare_scorer_and_reference(
        inputs=inputs,
        output_dir=output_dir,
        args=args,
    )
    results: dict[int, dict[str, Any]] = dict(resumed_results)
    for position, row_index in enumerate(pending, start=1):
        print(
            f"batch_problem_start={position}/{len(pending)} row_index={row_index}",
            flush=True,
        )
        problem_dir = output_dir / "problems" / f"row_{row_index:04d}"
        try:
            result = process_problem(
                row_index=row_index,
                row=candidate_rows[row_index],
                expected_identity=cohort[row_index],
                problem_dir=problem_dir,
                config_fingerprint=str(plan["config_fingerprint"]),
                inputs=inputs,
                state=state,
                args=args,
                regression=regression,
            )
        except Exception as error:
            result = failure_result(
                row_index=row_index,
                problem_dir=problem_dir,
                args=args,
                config_fingerprint=str(plan["config_fingerprint"]),
                error=error,
            )
        results[row_index] = result
        print(
            f"batch_problem_status={result.get('status')} row_index={row_index}",
            flush=True,
        )

    trainable_sha_after = equivalence.named_parameter_sha256(state["trainable"])
    parameter_versions_after = {
        name: parameter._version
        for name, parameter in state["model"].named_parameters()
    }
    parameter_integrity = (
        state["trainable_sha_before"] == trainable_sha_after
        and state["parameter_versions_before"] == parameter_versions_after
    )
    checkpoint_rows = []
    checkpoint_failures = []
    for row_index in ROW_INDICES:
        problem_dir = output_dir / "problems" / f"row_{row_index:04d}"
        try:
            validate_problem_checkpoint(
                problem_dir,
                row_index=row_index,
                execution_commit=args.execution_commit,
                config_fingerprint=str(plan["config_fingerprint"]),
            )
            checkpoint_rows.append(row_index)
        except Exception as error:
            checkpoint_failures.append(
                {
                    "row_index": row_index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    checkpoint_audit = {
        "status": "PASS" if len(checkpoint_rows) == len(ROW_INDICES) else "FAIL",
        "validated_row_indices": checkpoint_rows,
        "failures": checkpoint_failures,
        "resume_copy_path_implemented": True,
        "actual_interruption_recovery_exercised": bool(resume_audit["requested"]),
    }
    single.atomic_json(output_dir / "checkpoint_audit.json", checkpoint_audit)

    del state["reference"]
    del state["scorer"]
    del state["model"]
    del state["base_model"]
    del state["tokenizer"]
    del state["trainable"]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    single.nvidia_snapshot(output_dir / "gpu_memory_after_scorer_release.csv")
    single.atomic_json(
        output_dir / "torch_memory_after_scorer_release.json",
        single.torch_memory_snapshot(),
    )

    summary = summarize_results(
        args=args,
        plan=plan,
        lock=lock,
        regression=regression,
        reference_diagnostics=state["diagnostics"],
        results=results,
        checkpoint_audit=checkpoint_audit,
        resume_audit=resume_audit,
        parameter_integrity=parameter_integrity,
        elapsed_seconds=time.time() - started,
    )
    single.atomic_json(output_dir / "batch_result.json", summary)
    print(f"multi_problem_batch_status={summary['status']}", flush=True)
    return 0 if summary["status"] == "PASS" else 1


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(run(args))
    except SystemExit:
        raise
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        single.atomic_json(
            args.output_dir / "driver_failure.json",
            {
                "status": "FAIL",
                "method": METHOD,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
