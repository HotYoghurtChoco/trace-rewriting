"""Run the Stage 6C.2d v5 H200-native CosRewrite engineering preflight."""

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
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RUNTIME_REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (RUNTIME_REPO_ROOT / "src", RUNTIME_REPO_ROOT):
    import_root_text = str(import_root)
    if import_root_text not in sys.path:
        sys.path.insert(0, import_root_text)

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from math_verify import parse, verify
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    StringExtractionConfig,
)
from peft import PeftModel
from transformers import AutoModelForCausalLM

from experiments.stage6c_cosrewrite_v1 import verify_scorer_equivalence as equivalence
from optimize.gradient_feedback import CompletionOnlyGradientScorer
from optimize.score_candidates import _load_training_tokenizer


METHOD = "stage6c_2d_single_problem_closed_loop_v5_h200_native"
CLAIM_BOUNDARY = (
    "H200-native engineering preflight only: no executable formal-protocol "
    "lock, no SelectionOnly comparison, no method-effect claim, and no "
    "student/AF result."
)
MODEL_ID = "openai/gpt-oss-120b"
ROW_INDEX = 0
SCORER_RUNTIME_POLICY = "h200_native_v1"
H200_REFERENCE_SENTINEL = {
    "loss_expected": 1.1358162263035774,
    "loss_tolerance": 0.00011358162263035775,
    "mean_gradient_norm_expected": 1.2511129669699297,
    "mean_gradient_norm_tolerance": 0.00012511129669699297,
}
H200_POLICY_EVIDENCE = {
    "method": "stage6c_2d_h200_scorer_equivalence_v4",
    "job_id": "9010570.kman.restech.unsw.edu.au",
    "execution_commit": "581c0391eafcc8e335176f39ccb143b1034e2256",
    "a100_score_equivalence": "FAIL",
    "h200_repeat_stability": "PASS",
    "reference_sentinel_source": "v4_h200_repeat_1_with_locked_repeat_tolerances",
}
ENGINEERING_SELECTION_PRIORITY = ("BaselineRewrite", "C1", "C2")
EXPECTED_SOURCE_SHA256 = {
    "gradient_feedback": "ce1b8f89876b3460ece5aff3ff01e78dc31dc3bc37d8fdc3b2e8edf9770db951",
    "score_candidates": "2035787b8b7d4e64ff166893cebabcf506facef3c449fcc7a7f927c514596c67",
    "evaluate": "02c4cb26f90587c6aeddaf7a4dee96c810f16c4e342c4e02595778e918ab0227",
    "equivalence": "a839fec90420b1c2195bbed906b6a03dba0762a4dccc454ee9eb24270d272c9f",
}
EXPECTED_ROW_SHA256 = {
    "problem": "73429d562982300e6643d83a84428a22e0656773b67dad037d2de2f8bc0b45b9",
    "solution": "3eeba09368f91af378ea6532181536ff0c5a80960ebb7dc122e449c37f5dc489",
    "original_trace": "f16551fe2adebd32d916463b70ec82844f4f13b81c9b5bb245f6331c64a58e75",
    "rewrite_trace": "a1e58c733462ad3db15ad5151e0a253063962effe2c5d48252d396a595a43a99",
}
TRACE_THRESHOLDS = {
    "minimum_characters": 120,
    "minimum_words": 20,
    "minimum_numeric_tokens": 2,
    "maximum_identical_nonempty_line_count": 2,
    "maximum_repeated_eight_token_window_count": 2,
}
LEAKAGE_TERMS = (
    "feedback",
    "cosine",
    "gradient",
    "scorer",
    "score",
    "prompt",
    "instruction",
    "gradient cosine",
    "reference gradient",
    "black-box scorer",
    "lower is better",
    "scorer feedback",
    "cosine feedback",
    "prompt instructions",
)
ROUND1_FORBIDDEN_TERMS = (
    "gradient",
    "cosine",
    "feedback",
    "ranking",
    "lower is better",
)
SYSTEM_PROMPT = (
    "Return only a complete rewritten solution trace for the supplied math "
    "problem. Preserve the correct final answer and relevant units. Include "
    "enough mathematical reasoning that the response is not answer-only. "
    "Do not include meta-commentary. End with the final answer in "
    "\\boxed{ANSWER} form."
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
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, path)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def nvidia_snapshot(path: Path) -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    path.write_text(completed.stdout, encoding="utf-8")
    return completed.stdout.strip()


def torch_memory_snapshot() -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "device_name": torch.cuda.get_device_name(0),
        "free_gib": free_bytes / (1024**3),
        "total_gib": total_bytes / (1024**3),
        "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
        "maximum_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }


def answer_correct(solution: str, trace: str) -> tuple[bool, str | None]:
    extraction = [
        ExprExtractionConfig(),
        LatexExtractionConfig(),
        StringExtractionConfig(),
    ]
    try:
        expected = parse(solution)
        observed = parse(trace, extraction_config=extraction)
        return bool(verify(expected, observed)), None
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def repeated_window_count(text: str, size: int = 8) -> int:
    tokens = re.findall(r"\w+|[^\w\s]", text.casefold())
    if len(tokens) < size:
        return 1
    windows = Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))
    return max(windows.values(), default=1)


def validate_trace(
    *,
    solution: str,
    trace: str,
    parents: Mapping[str, str],
    finish_reason: str | None,
) -> dict[str, Any]:
    stripped = trace.strip()
    words = re.findall(r"\S+", stripped)
    numeric_tokens = re.findall(r"(?<!\w)[-+]?\d+(?:\.\d+)?(?!\w)", stripped)
    nonempty_lines = [line.strip().casefold() for line in stripped.splitlines() if line.strip()]
    line_counts = Counter(line for line in nonempty_lines if len(line) >= 12)
    maximum_line_count = max(line_counts.values(), default=1)
    maximum_window_count = repeated_window_count(stripped)
    normalized = normalize_text(stripped)
    duplicate_of = [label for label, parent in parents.items() if normalized == normalize_text(parent)]
    leakage = [term for term in LEAKAGE_TERMS if term in normalized]
    has_reasoning_marker = bool(
        re.search(
            r"(?:because|therefore|thus|since|so|calculate|equals|hence|[=+\-×÷])",
            stripped,
            flags=re.IGNORECASE,
        )
    )
    correct, verification_error = answer_correct(solution, stripped)

    reasons = []
    if not stripped:
        reasons.append("empty")
    if len(stripped) < TRACE_THRESHOLDS["minimum_characters"]:
        reasons.append("too_short_characters")
    if len(words) < TRACE_THRESHOLDS["minimum_words"]:
        reasons.append("answer_only_or_too_short_words")
    if len(numeric_tokens) < TRACE_THRESHOLDS["minimum_numeric_tokens"]:
        reasons.append("too_few_numeric_tokens")
    if not has_reasoning_marker:
        reasons.append("no_reasoning_marker")
    if maximum_line_count > TRACE_THRESHOLDS["maximum_identical_nonempty_line_count"]:
        reasons.append("repeated_lines")
    if maximum_window_count > TRACE_THRESHOLDS["maximum_repeated_eight_token_window_count"]:
        reasons.append("repeated_token_window")
    if duplicate_of:
        reasons.append("exact_normalized_duplicate")
    if leakage:
        reasons.append("prompt_or_feedback_leakage")
    if not correct:
        reasons.append("gold_answer_mismatch_or_unparseable")
    if finish_reason is not None and finish_reason != "stop":
        reasons.append("generation_did_not_finish_with_stop")

    return {
        "valid_before_scorer_token_gate": not reasons,
        "reasons": reasons,
        "answer_correct": correct,
        "answer_verification_error": verification_error,
        "characters": len(stripped),
        "words": len(words),
        "numeric_tokens": len(numeric_tokens),
        "has_reasoning_marker": has_reasoning_marker,
        "maximum_identical_nonempty_line_count": maximum_line_count,
        "maximum_repeated_eight_token_window_count": maximum_window_count,
        "duplicate_of": duplicate_of,
        "leakage_terms": leakage,
        "finish_reason": finish_reason,
        "thresholds": TRACE_THRESHOLDS,
    }


def add_scorer_token_gate(validity: dict[str, Any], score: Mapping[str, Any]) -> None:
    tokens = score["tokens"]
    token_gate = (
        int(tokens["completion_tokens_removed"]) == 0
        and bool(tokens["eos_in_kept_completion"])
    )
    validity["scorer_token_gate"] = token_gate
    validity["scorer_tokens"] = tokens
    validity["valid"] = bool(validity["valid_before_scorer_token_gate"] and token_gate)
    if not token_gate:
        validity["reasons"] = list(validity["reasons"]) + [
            "scorer_completion_truncated_or_missing_eos"
        ]


def request_json(url: str, payload: Mapping[str, Any], timeout_seconds: int = 900) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"API connection failed: {error}") from error


def get_json(url: str, timeout_seconds: int = 60) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"API GET failed for {url}: {error}") from error


def generate_candidate(
    *,
    label: str,
    user_prompt: str,
    solution: str,
    parents: Mapping[str, str],
    seed_start: int,
    args: argparse.Namespace,
    api_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts = []
    selected = None
    for attempt_index in range(args.max_generation_attempts):
        attempt = attempt_index + 1
        seed = seed_start + attempt_index
        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "seed": seed,
            "stream": False,
        }
        stem = f"{label}_attempt_{attempt:02d}"
        atomic_json(api_dir / f"{stem}_request.json", payload)
        started = time.time()
        try:
            response = request_json(
                f"{args.api_base_url.rstrip('/')}/chat/completions",
                payload,
            )
            atomic_json(api_dir / f"{stem}_response.json", response)
            choices = response.get("choices")
            require(isinstance(choices, list) and len(choices) == 1, "API did not return exactly one choice")
            choice = choices[0]
            message = choice.get("message")
            require(isinstance(message, dict), "API choice has no message object")
            content = message.get("content")
            require(isinstance(content, str), "API message content is not text")
            require("reasoning_content" in message, "GPT-OSS reasoning_content field is absent")
            reasoning_content = message.get("reasoning_content")
            require(
                reasoning_content is None or isinstance(reasoning_content, str),
                "reasoning_content is neither text nor null",
            )
            finish_reason = choice.get("finish_reason")
            validity = validate_trace(
                solution=solution,
                trace=content,
                parents=parents,
                finish_reason=finish_reason,
            )
            record = {
                "label": label,
                "attempt": attempt,
                "seed": seed,
                "elapsed_seconds": time.time() - started,
                "response_id": response.get("id"),
                "finish_reason": finish_reason,
                "content": content,
                "content_sha256": text_sha256(content),
                "reasoning_content": reasoning_content,
                "reasoning_content_sha256": (
                    text_sha256(reasoning_content) if isinstance(reasoning_content, str) else None
                ),
                "usage": response.get("usage"),
                "validity": validity,
            }
            attempts.append(record)
            if validity["valid_before_scorer_token_gate"]:
                selected = record
                break
        except Exception as error:
            error_record = {
                "label": label,
                "attempt": attempt,
                "seed": seed,
                "elapsed_seconds": time.time() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            atomic_json(api_dir / f"{stem}_error.json", error_record)
            attempts.append(error_record)

    if selected is None:
        successful = [item for item in attempts if isinstance(item.get("content"), str)]
        require(successful, f"{label} produced no scoreable text in bounded attempts")
        selected = successful[-1]
    selected = dict(selected)
    selected["selected_as_first_valid_attempt"] = bool(
        selected["validity"]["valid_before_scorer_token_gate"]
    )
    return selected, attempts


def score_trace(
    scorer: CompletionOnlyGradientScorer,
    reference: Any,
    problem: str,
    trace: str,
) -> dict[str, Any]:
    score = scorer.score_response(problem=problem, response=trace, reference=reference).to_dict()
    for key in (
        "loss",
        "gradient_norm",
        "gradient_dot",
        "gradient_cosine",
        "predicted_reference_loss_change_per_unit_step",
    ):
        require(math.isfinite(float(score[key])), f"Non-finite scorer output: {key}")
    return score


def select_engineering_winner(
    lineage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one valid trace for the single-problem engineering preflight.

    This deterministic rule only demonstrates that the controller can finish
    the loop.  It is deliberately not the locked formal CosRewrite selection
    or tie policy.
    """

    by_id = {str(row["candidate_id"]): row for row in lineage_rows}
    require(
        set(by_id) == set(ENGINEERING_SELECTION_PRIORITY),
        "Engineering selection did not receive BaselineRewrite, C1, and C2",
    )
    priority = {
        candidate_id: index
        for index, candidate_id in enumerate(ENGINEERING_SELECTION_PRIORITY)
    }
    eligible = []
    excluded = []
    for candidate_id in ENGINEERING_SELECTION_PRIORITY:
        row = by_id[candidate_id]
        validity = row.get("validity", {})
        if not bool(validity.get("valid")):
            excluded.append(
                {
                    "candidate_id": candidate_id,
                    "reasons": list(validity.get("reasons", [])),
                }
            )
            continue
        cosine = float(row["score"]["gradient_cosine"])
        require(math.isfinite(cosine), f"Non-finite selection cosine for {candidate_id}")
        eligible.append(
            {
                "candidate_id": candidate_id,
                "gradient_cosine": cosine,
                "priority": priority[candidate_id],
                "trace_sha256": row["trace_sha256"],
            }
        )

    require(eligible, "No valid trace is eligible for engineering selection")
    ranked = sorted(
        eligible,
        key=lambda item: (item["gradient_cosine"], item["priority"]),
    )
    winner = ranked[0]
    return {
        "status": "PASS",
        "policy": "engineering_lowest_valid_h200_native_cosine_v1",
        "scope": "engineering_only_not_formal_protocol",
        "tie_break": "BaselineRewrite_then_C1_then_C2",
        "eligible_ranked": ranked,
        "excluded": excluded,
        "winner_candidate_id": winner["candidate_id"],
        "winner_gradient_cosine": winner["gradient_cosine"],
        "winner_trace_sha256": winner["trace_sha256"],
        "formal_selection_rule_locked": False,
    }


def load_inputs(repo_root: Path) -> dict[str, Any]:
    source_paths = {
        "gradient_feedback": repo_root / "optimize/gradient_feedback.py",
        "score_candidates": repo_root / "optimize/score_candidates.py",
        "evaluate": repo_root / "src/evaluate.py",
        "equivalence": repo_root / "experiments/stage6c_cosrewrite_v1/verify_scorer_equivalence.py",
    }
    verified_source_sha256 = {}
    for label, path in source_paths.items():
        require(path.is_file(), f"Missing source file: {path}")
        actual = file_sha256(path)
        require(actual == EXPECTED_SOURCE_SHA256[label], f"Unexpected {label} SHA256: {actual}")
        verified_source_sha256[label] = actual

    formal_root = equivalence.DEFAULT_FORMAL_ROOT.resolve()
    stage6b_root = equivalence.DEFAULT_STAGE6B_ROOT.resolve()
    config_path = formal_root / "provenance/config.yaml"
    candidate_yaml_path = formal_root / "iter_0/candidate_instructions.yaml"
    method_lock_path = stage6b_root / "provenance/stage6b_1c_method_lock.yaml"
    preflight_path = stage6b_root / "provenance/stage6b_1c_preflight.json"
    reference_path = stage6b_root / "frozen_data/gradient_reference.jsonl"
    reference_manifest_path = stage6b_root / "frozen_data/gradient_reference_manifest.json"

    # These hashes retain the fixed data/model/scorer definition.  V5 does not
    # load or gate on any scalar or candidate score computed on the A100.
    fixed_input_paths = {
        "method_lock": method_lock_path,
        "preflight": preflight_path,
        "reference": reference_path,
        "reference_manifest": reference_manifest_path,
        "score_candidates": repo_root / "optimize/score_candidates.py",
    }
    verified_fixed_input_sha256 = {}
    for label, path in fixed_input_paths.items():
        require(path.is_file(), f"Missing fixed input: {path}")
        actual = file_sha256(path)
        expected = equivalence.EXPECTED_SHA256[label]
        require(actual == expected, f"Unexpected fixed {label} SHA256: {actual}")
        verified_fixed_input_sha256[label] = actual

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with candidate_yaml_path.open(encoding="utf-8") as handle:
        candidate_yaml = yaml.safe_load(handle)
    with preflight_path.open(encoding="utf-8") as handle:
        preflight = json.load(handle)

    require(file_sha256(config_path) == preflight["config_sha256"], "Config/preflight identity mismatch")
    require(
        file_sha256(candidate_yaml_path) == preflight["candidate_yaml_sha256"],
        "Candidate YAML/preflight identity mismatch",
    )
    require(preflight.get("status") == "PASS", "Stage 6B preflight status is not PASS")
    require(int(config["seed"]) == int(equivalence.SCORING_SEED), "Scoring seed mismatch")

    candidate_name = candidate_yaml["candidate_instructions"][0]["name"]
    candidate_path = formal_root / candidate_name
    model_name = config["proxy_models"][0]["name"]
    tokenizer_name = config["proxy_models"][0]["tokenizer"]
    adapter_path = candidate_path / "finetuned_model" / Path(model_name).name / "adapter"
    require(candidate_path.is_dir(), f"Missing candidate dataset: {candidate_path}")
    require(adapter_path.is_dir(), f"Missing adapter: {adapter_path}")
    require(
        file_sha256(adapter_path / "adapter_config.json")
        == equivalence.EXPECTED_SHA256["adapter_config"],
        "Adapter config identity mismatch",
    )
    require(
        file_sha256(adapter_path / "adapter_model.safetensors")
        == equivalence.EXPECTED_SHA256["adapter_model"],
        "Adapter model identity mismatch",
    )
    require(
        equivalence.adapter_file_records(adapter_path) == preflight["adapter_files"],
        "Adapter inventory differs from frozen preflight",
    )

    candidate_dataset = load_from_disk(str(candidate_path))
    candidate_rows = [candidate_dataset[index] for index in range(len(candidate_dataset))]
    reference_rows = equivalence.load_jsonl(reference_path)
    require(len(candidate_rows) == 100, "Candidate dataset does not contain 100 rows")
    require(len(reference_rows) == 100, "Reference set does not contain 100 rows")
    require(
        equivalence.ordered_rows_sha256(candidate_rows, "rewrite_trace")
        == preflight["candidate_ordered_rows_sha256"],
        "Candidate ordered-row identity mismatch",
    )
    require(
        equivalence.ordered_rows_sha256(reference_rows, "solution")
        == preflight["reference_ordered_rows_sha256"],
        "Reference ordered-row identity mismatch",
    )
    require(
        not ({str(row["problem"]) for row in candidate_rows} & {str(row["problem"]) for row in reference_rows}),
        "Candidate/reference problem sets overlap",
    )

    row = dict(candidate_rows[ROW_INDEX])
    for key, expected in EXPECTED_ROW_SHA256.items():
        require(text_sha256(str(row[key])) == expected, f"Row-0 {key} identity mismatch")

    return {
        "repo_root": repo_root,
        "formal_root": formal_root,
        "stage6b_root": stage6b_root,
        "config": config,
        "preflight": preflight,
        "reference_rows": reference_rows,
        "row": row,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "adapter_path": adapter_path,
        "verified_source_sha256": verified_source_sha256,
        "verified_fixed_input_sha256": verified_fixed_input_sha256,
        "candidate_name": candidate_name,
    }


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api_dir = output_dir / "api"
    api_dir.mkdir(exist_ok=True)
    require(not (output_dir / "closed_loop_result.json").exists(), "Closed-loop output already exists")
    require(args.max_generation_attempts > 0, "max-generation-attempts must be positive")
    require(0.0 <= args.temperature <= 2.0, "temperature is outside [0, 2]")
    require(args.max_tokens > 0, "max-tokens must be positive")

    repo_root = RUNTIME_REPO_ROOT
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == args.execution_commit, "Driver execution commit differs from repository HEAD")

    inputs = load_inputs(repo_root)
    row = inputs["row"]
    problem = str(row["problem"])
    solution = str(row["solution"])
    baseline = str(row["rewrite_trace"])
    baseline_validity = validate_trace(
        solution=solution,
        trace=baseline,
        parents={},
        finish_reason=None,
    )

    model_list = get_json(f"{args.api_base_url.rstrip('/')}/models")
    atomic_json(api_dir / "model_list.json", model_list)
    model_ids = [item.get("id") for item in model_list.get("data", []) if isinstance(item, dict)]
    require(MODEL_ID in model_ids, f"Served model list does not contain {MODEL_ID}")

    c1_user_prompt = (
        f"Problem:\n{problem}\n\n"
        f"Current trace:\n{baseline}\n\n"
        "Create one materially different but concise complete solution trace for "
        "the same problem. Preserve the same correct final answer."
    )
    c1_prompt_normalized = normalize_text(SYSTEM_PROMPT + "\n" + c1_user_prompt)
    round1_leakage = [term for term in ROUND1_FORBIDDEN_TERMS if term in c1_prompt_normalized]
    require(not round1_leakage, f"Round-1 prompt contains forbidden feedback language: {round1_leakage}")
    c1, c1_attempts = generate_candidate(
        label="c1",
        user_prompt=c1_user_prompt,
        solution=solution,
        parents={"BaselineRewrite": baseline},
        seed_start=int(inputs["config"]["seed"]) * 1000 + 1,
        args=args,
        api_dir=api_dir,
    )
    atomic_json(output_dir / "c1_attempts.json", c1_attempts)

    require(torch.cuda.is_available(), "CUDA is unavailable to the Qwen scorer")
    require(torch.cuda.is_bf16_supported(), "CUDA device does not support BF16")
    scoring_seed = int(inputs["config"]["seed"])
    random.seed(scoring_seed)
    np.random.seed(scoring_seed)
    torch.manual_seed(scoring_seed)
    torch.cuda.manual_seed_all(scoring_seed)
    device = torch.device("cuda")
    compute_dtype = torch.bfloat16

    nvidia_snapshot(output_dir / "gpu_memory_before_scorer_load.csv")
    tokenizer = _load_training_tokenizer(inputs["tokenizer_name"])
    require(tokenizer.pad_token_id is not None, "Tokenizer has no pad token")
    require(tokenizer.eos_token_id is not None, "Tokenizer has no EOS token")
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
    require(
        scorer.trainable_tensor_count == equivalence.EXPECTED_TRAINABLE_TENSORS,
        "Unexpected trainable tensor count",
    )
    require(
        scorer.trainable_parameter_count == equivalence.EXPECTED_TRAINABLE_PARAMETERS,
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
            f"reference_progress={done}/{total}", flush=True
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
    h200_reference_loss_error = abs(
        h200_reference_loss - H200_REFERENCE_SENTINEL["loss_expected"]
    )
    h200_reference_norm_error = abs(
        h200_reference_norm
        - H200_REFERENCE_SENTINEL["mean_gradient_norm_expected"]
    )
    h200_reference_loss_sentinel_gate = (
        h200_reference_loss_error <= H200_REFERENCE_SENTINEL["loss_tolerance"]
    )
    h200_reference_norm_sentinel_gate = (
        h200_reference_norm_error
        <= H200_REFERENCE_SENTINEL["mean_gradient_norm_tolerance"]
    )
    h200_device_gate = "H200" in torch.cuda.get_device_name(device)

    parameter_versions_after_reference = {
        name: parameter._version for name, parameter in model.named_parameters()
    }
    trainable_sha_after_reference = equivalence.named_parameter_sha256(trainable)
    parameter_versions_unchanged = (
        parameter_versions_after_reference == parameter_versions_before
    )
    trainable_sha_unchanged = trainable_sha_after_reference == trainable_sha_before
    h200_native_reference_gate = all(
        (
            reference_count_gate,
            reference_token_gate,
            reference_loss_gate,
            reference_norm_gate,
            h200_reference_loss_sentinel_gate,
            h200_reference_norm_sentinel_gate,
            h200_device_gate,
            parameter_versions_unchanged,
            trainable_sha_unchanged,
        )
    )

    reference_per_example_path = (
        output_dir / "reference_recompute_per_example.jsonl"
    )
    atomic_jsonl(reference_per_example_path, reference.per_example)
    reference_diagnostics = {
        "schema": "stage6c_2d_h200_native_reference_diagnostics_v1",
        "status": "PASS" if h200_native_reference_gate else "FAIL",
        "method": METHOD,
        "claim_boundary": CLAIM_BOUNDARY,
        "scorer_runtime_policy": SCORER_RUNTIME_POLICY,
        "policy_evidence": H200_POLICY_EVIDENCE,
        "a100_scalar_or_score_equivalence_gate_applied": False,
        "execution_commit": args.execution_commit,
        "job_id": args.job_id,
        "scoring_seed": scoring_seed,
        "reference_count": int(reference.example_count),
        "reference_per_example_file_sha256": file_sha256(
            reference_per_example_path
        ),
        "model": {
            "base_model_path": str(inputs["model_name"]),
            "tokenizer_path": str(inputs["tokenizer_name"]),
            "adapter_path": str(inputs["adapter_path"]),
            "compute_dtype": str(compute_dtype),
            "model_training": bool(model.training),
            "use_cache": bool(model.config.use_cache),
            "trainable_tensor_count": int(scorer.trainable_tensor_count),
            "trainable_parameter_count": int(scorer.trainable_parameter_count),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "torch_compiled_cuda": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_memory": torch_memory_snapshot(),
        },
        "reference_loss": {
            "h200_native": h200_reference_loss,
            "finite_gate": "PASS" if reference_loss_gate else "FAIL",
            "sentinel_expected": H200_REFERENCE_SENTINEL["loss_expected"],
            "sentinel_absolute_error": h200_reference_loss_error,
            "sentinel_tolerance": H200_REFERENCE_SENTINEL["loss_tolerance"],
            "sentinel_gate": (
                "PASS" if h200_reference_loss_sentinel_gate else "FAIL"
            ),
        },
        "reference_mean_gradient_norm": {
            "h200_native": h200_reference_norm,
            "finite_positive_gate": "PASS" if reference_norm_gate else "FAIL",
            "sentinel_expected": H200_REFERENCE_SENTINEL[
                "mean_gradient_norm_expected"
            ],
            "sentinel_absolute_error": h200_reference_norm_error,
            "sentinel_tolerance": H200_REFERENCE_SENTINEL[
                "mean_gradient_norm_tolerance"
            ],
            "sentinel_gate": (
                "PASS" if h200_reference_norm_sentinel_gate else "FAIL"
            ),
        },
        "h200_native_reference_gates": {
            "reference_count_100": reference_count_gate,
            "no_reference_completion_truncated": reference_token_gate,
            "reference_loss_finite": reference_loss_gate,
            "reference_norm_finite_positive": reference_norm_gate,
            "reference_loss_matches_h200_v4_sentinel": (
                h200_reference_loss_sentinel_gate
            ),
            "reference_norm_matches_h200_v4_sentinel": (
                h200_reference_norm_sentinel_gate
            ),
            "device_is_h200": h200_device_gate,
            "parameter_versions_unchanged": parameter_versions_unchanged,
            "trainable_sha256_unchanged": trainable_sha_unchanged,
        },
        "parameter_integrity_after_reference": {
            "versions_unchanged": parameter_versions_unchanged,
            "trainable_sha256_before": trainable_sha_before,
            "trainable_sha256_after": trainable_sha_after_reference,
            "trainable_sha256_unchanged": trainable_sha_unchanged,
        },
        "verified_source_sha256": inputs["verified_source_sha256"],
        "verified_fixed_input_sha256": inputs["verified_fixed_input_sha256"],
    }
    atomic_json(
        output_dir / "reference_recompute_diagnostics.json",
        reference_diagnostics,
    )
    print(
        "h200_native_reference_loss_gate={} value={:.17g} error={:.17g} "
        "tolerance={:.17g}".format(
            "PASS"
            if reference_loss_gate and h200_reference_loss_sentinel_gate
            else "FAIL",
            h200_reference_loss,
            h200_reference_loss_error,
            H200_REFERENCE_SENTINEL["loss_tolerance"],
        ),
        flush=True,
    )
    print(
        "h200_native_reference_norm_gate={} value={:.17g} error={:.17g} "
        "tolerance={:.17g}".format(
            "PASS"
            if reference_norm_gate and h200_reference_norm_sentinel_gate
            else "FAIL",
            h200_reference_norm,
            h200_reference_norm_error,
            H200_REFERENCE_SENTINEL["mean_gradient_norm_tolerance"],
        ),
        flush=True,
    )
    require(
        h200_native_reference_gate,
        "H200-native reference validity or scorer-integrity gate failed",
    )

    baseline_score = score_trace(scorer, reference, problem, baseline)
    c1_score = score_trace(scorer, reference, problem, c1["content"])
    add_scorer_token_gate(baseline_validity, baseline_score)
    add_scorer_token_gate(c1["validity"], c1_score)

    nvidia_snapshot(output_dir / "gpu_memory_server_and_scorer_ready.csv")
    atomic_json(output_dir / "torch_memory_server_and_scorer_ready.json", torch_memory_snapshot())

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
    normalized_c2_prompt = normalize_text(c2_user_prompt)
    require("gradient cosine" in normalized_c2_prompt, "C2 prompt lacks cosine feedback")
    require(rendered_feedback_value in c2_user_prompt, "C2 prompt lacks the exact feedback value")
    require("lower is better" in normalized_c2_prompt, "C2 prompt lacks feedback direction")

    c2, c2_attempts = generate_candidate(
        label="c2",
        user_prompt=c2_user_prompt,
        solution=solution,
        parents={"BaselineRewrite": baseline, "C1": c1["content"]},
        seed_start=int(inputs["config"]["seed"]) * 1000 + 101,
        args=args,
        api_dir=api_dir,
    )
    atomic_json(output_dir / "c2_attempts.json", c2_attempts)
    nvidia_snapshot(output_dir / "gpu_memory_during_c2_with_scorer.csv")

    c2_score = score_trace(scorer, reference, problem, c2["content"])
    add_scorer_token_gate(c2["validity"], c2_score)
    trainable_sha_after = equivalence.named_parameter_sha256(trainable)
    parameter_versions_after = {
        name: parameter._version for name, parameter in model.named_parameters()
    }
    parameter_integrity = (
        trainable_sha_before == trainable_sha_after
        and parameter_versions_before == parameter_versions_after
    )

    prompt_audit = {
        "round1_shared_pool_compatible": not round1_leakage,
        "round1_forbidden_terms_found": round1_leakage,
        "c1_prompt_sha256": text_sha256(SYSTEM_PROMPT + "\n" + c1_user_prompt),
        "c2_prompt_sha256": text_sha256(SYSTEM_PROMPT + "\n" + c2_user_prompt),
        "c2_contains_external_black_box_scalar": True,
        "c2_contains_exact_feedback_value": rendered_feedback_value in c2_user_prompt,
        "c2_feedback_has_token_attribution": False,
    }
    gates = {
        "fixed_input_identity": True,
        "h200_native_reference_valid": h200_native_reference_gate,
        "round1_prompt_has_no_gradient_derived_feedback": not round1_leakage,
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
            "trace_sha256": text_sha256(baseline),
            "validity": baseline_validity,
            "score": baseline_score,
        },
        {
            "candidate_id": "C1",
            "round": 1,
            "parent_id": "BaselineRewrite",
            "trace": c1["content"],
            "trace_sha256": text_sha256(c1["content"]),
            "generation_attempt": c1["attempt"],
            "generation_seed": c1["seed"],
            "validity": c1["validity"],
            "score": c1_score,
        },
        {
            "candidate_id": "C2",
            "round": 2,
            "parent_id": c2_parent_label,
            "parent_trace_sha256": text_sha256(c2_parent),
            "trace": c2["content"],
            "trace_sha256": text_sha256(c2["content"]),
            "generation_attempt": c2["attempt"],
            "generation_seed": c2["seed"],
            "validity": c2["validity"],
            "score": c2_score,
        },
    ]
    engineering_selection = select_engineering_winner(lineage_rows)
    atomic_json(output_dir / "engineering_selection.json", engineering_selection)
    gates["engineering_final_selection_completed"] = (
        engineering_selection["status"] == "PASS"
    )
    status = "PASS" if all(gates.values()) else "FAIL"
    atomic_jsonl(output_dir / "trace_lineage.jsonl", lineage_rows)
    atomic_json(
        output_dir / "reference_summary.json",
        {
            "policy": SCORER_RUNTIME_POLICY,
            "a100_scalar_or_score_equivalence_gate_applied": False,
            "reference_count": reference.example_count,
            "h200_native_loss_mean": h200_reference_loss,
            "h200_native_mean_gradient_norm": h200_reference_norm,
            "h200_native_sentinel": H200_REFERENCE_SENTINEL,
            "h200_native_loss_sentinel_error": h200_reference_loss_error,
            "h200_native_norm_sentinel_error": h200_reference_norm_error,
            "completion_truncated_count": reference_truncated_count,
        },
    )
    result = {
        "status": status,
        "method": METHOD,
        "claim_boundary": CLAIM_BOUNDARY,
        "job_id": args.job_id,
        "execution_commit": args.execution_commit,
        "row_index": ROW_INDEX,
        "problem_sha256": text_sha256(problem),
        "model_id": MODEL_ID,
        "runtime_policy": {
            "name": SCORER_RUNTIME_POLICY,
            "device": torch.cuda.get_device_name(device),
            "fixed_data_model_and_scorer_definition": True,
            "a100_scalar_or_score_equivalence_required": False,
            "basis": H200_POLICY_EVIDENCE,
            "h200_reference_sentinel": H200_REFERENCE_SENTINEL,
        },
        "generator_settings": {
            "status": "engineering_candidate_settings_not_formal_protocol",
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "maximum_generation_attempts": args.max_generation_attempts,
        },
        "scorer": {
            "primary_metric": "gradient_cosine",
            "direction": "lower_is_better",
            "other_metrics_role": "diagnostics_only",
            "maximum_total_tokens": 1024,
            "reference_count": reference.example_count,
            "optimizer_created": False,
            "optimizer_step_performed": False,
            "trainable_parameter_sha256_before": trainable_sha_before,
            "trainable_parameter_sha256_after": trainable_sha_after,
            "parameter_integrity": parameter_integrity,
            "reference_policy": SCORER_RUNTIME_POLICY,
            "h200_native_reference_loss_mean": h200_reference_loss,
            "h200_native_reference_gradient_norm": h200_reference_norm,
            "h200_native_reference_sentinel": H200_REFERENCE_SENTINEL,
            "h200_native_reference_loss_sentinel_error": (
                h200_reference_loss_error
            ),
            "h200_native_reference_norm_sentinel_error": (
                h200_reference_norm_error
            ),
            "a100_scalar_or_score_comparison_performed": False,
        },
        "feedback": {
            "target_candidate": c2_parent_label,
            "metric": "gradient_cosine",
            "value": float(feedback_target_score["gradient_cosine"]),
            "rendered_value": rendered_feedback_value,
            "direction": "lower_is_better",
            "token_attribution_claimed": False,
            "feedback_text_sha256": text_sha256(feedback_text),
        },
        "prompt_audit": prompt_audit,
        "lineage": {
            "c1_parent": "BaselineRewrite",
            "c2_parent": c2_parent_label,
            "fallback_rule": (
                "Use C1 only when BaselineRewrite and C1 pass answer, trace, and "
                "scorer-token gates; otherwise retain BaselineRewrite as C2 parent."
            ),
            "formal_parent_selection_rule_locked": False,
        },
        "cosine_diagnostics": {
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
        },
        "engineering_final_selection": engineering_selection,
        "gates": gates,
        "verified_source_sha256": inputs["verified_source_sha256"],
        "verified_fixed_input_sha256": inputs["verified_fixed_input_sha256"],
        "candidate_name": inputs["candidate_name"],
        "baseline_duplicate_rows_known_from_prior_audit": [18, 19],
        "selection_only_implemented": False,
        "formal_rewrite_dataset_created": False,
        "student_trained": False,
        "af_measured": False,
    }
    atomic_json(output_dir / "closed_loop_result.json", result)

    del reference, scorer, model, base_model, tokenizer, trainable
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    nvidia_snapshot(output_dir / "gpu_memory_after_scorer_release.csv")
    atomic_json(output_dir / "torch_memory_after_scorer_release.json", torch_memory_snapshot())
    print(f"closed_loop_status={status}", flush=True)
    return 0 if status == "PASS" else 1


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(run(args))
    except SystemExit:
        raise
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
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
