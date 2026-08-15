#!/usr/bin/env python3
"""Compare two independent H200 scorer diagnostics.

This comparator consumes two complete ``hardware`` runs from
``verify_scorer_equivalence.py``.  It does not score traces, modify a model,
or change any locked tolerance.  Its only purpose is to separate a stable
cross-runtime offset from H200 run-to-run instability before Stage 6C.2d is
allowed to continue.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SUMMARY_NAME = "stage6c_scorer_equivalence_summary.json"
ROWS_NAME = "stage6c_scorer_equivalence_candidates.jsonl"
HARDWARE_DIAGNOSTIC_VERSION = "stage6c_2d_h200_scorer_equivalence_v4"
EXPECTED_CANDIDATE_COUNT = 100
EXPECTED_REFERENCE_COUNT = 100

# These are the already-locked Stage 6C.1 v4 equivalence bounds.  They are
# repeated here so this lightweight comparator does not import the model and
# dataset dependencies used by the scorer process.
REPRODUCIBILITY_SPEARMAN_MIN = 0.999
NORMALIZED_DOT_DISCREPANCY_MAX = 0.002
COSINE_ABSOLUTE_ERROR_MAX = 0.00225
SCALAR_RELATIVE_TOLERANCE = 1e-4
SCALAR_ABSOLUTE_FLOOR = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-one", type=Path, required=True)
    parser.add_argument("--repeat-two", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def scalar_tolerance(expected: float) -> float:
    return max(
        SCALAR_ABSOLUTE_FLOOR,
        SCALAR_RELATIVE_TOLERANCE * abs(expected),
    )


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {label}: {result}")
    return result


def ordered_rows(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> list[Mapping[str, Any]]:
    require(
        len(rows) == EXPECTED_CANDIDATE_COUNT,
        f"{label} does not contain exactly 100 candidates",
    )
    by_index = {int(row["candidate_index"]): row for row in rows}
    require(
        len(by_index) == EXPECTED_CANDIDATE_COUNT,
        f"{label} contains duplicate candidate indices",
    )
    require(
        sorted(by_index) == list(range(EXPECTED_CANDIDATE_COUNT)),
        f"{label} candidate indices are not 0..99",
    )
    return [by_index[index] for index in range(EXPECTED_CANDIDATE_COUNT)]


def top_indices(values: Sequence[float], count: int, *, reverse: bool) -> list[int]:
    return sorted(
        range(len(values)),
        key=lambda index: (values[index], index),
        reverse=reverse,
    )[:count]


def main() -> None:
    args = parse_args()
    repeat_dirs = [args.repeat_one.resolve(), args.repeat_two.resolve()]
    summaries = [load_json(path / SUMMARY_NAME) for path in repeat_dirs]
    rows = [
        ordered_rows(load_jsonl(path / ROWS_NAME), f"repeat_{index + 1}")
        for index, path in enumerate(repeat_dirs)
    ]

    for index, summary in enumerate(summaries, start=1):
        require(summary.get("status") == "COMPLETE", f"repeat {index} incomplete")
        require(summary.get("mode") == "hardware", f"repeat {index} wrong mode")
        require(
            summary.get("method_version") == HARDWARE_DIAGNOSTIC_VERSION,
            f"repeat {index} wrong method version",
        )
        require(
            summary.get("execution_commit") == args.execution_commit,
            f"repeat {index} execution commit mismatch",
        )
        require(
            int(summary.get("candidate_count", -1)) == EXPECTED_CANDIDATE_COUNT,
            f"repeat {index} candidate count mismatch",
        )
        require(
            int(summary.get("reference_count", -1)) == EXPECTED_REFERENCE_COUNT,
            f"repeat {index} reference count mismatch",
        )
        require(
            summary.get("parameter_integrity", {}).get("status")
            == "PASS_NO_PARAMETER_UPDATE",
            f"repeat {index} parameter integrity failed",
        )
        require(
            "H200" in str(summary.get("runtime", {}).get("device_name", "")),
            f"repeat {index} did not run on an H200",
        )

    identity_one = summaries[0]["input_identity"]
    identity_two = summaries[1]["input_identity"]
    input_identity_gate = identity_one == identity_two
    runtime_identity_gate = summaries[0]["runtime"] == summaries[1]["runtime"]

    frozen_identity_gate = all(
        (
            int(left["gradient_rank"]) == int(right["gradient_rank"])
            and left["problem_sha256"] == right["problem_sha256"]
            and left["rewrite_trace_sha256"] == right["rewrite_trace_sha256"]
            and finite_float(left["frozen_gradient_dot"], "frozen dot left")
            == finite_float(right["frozen_gradient_dot"], "frozen dot right")
            and finite_float(left["frozen_gradient_cosine"], "frozen cosine left")
            == finite_float(right["frozen_gradient_cosine"], "frozen cosine right")
        )
        for left, right in zip(rows[0], rows[1], strict=True)
    )
    token_metadata_gate = all(
        left["tokens"] == right["tokens"]
        for left, right in zip(rows[0], rows[1], strict=True)
    )

    dots = [
        [finite_float(row["recomputed_gradient_dot"], "recomputed dot") for row in run]
        for run in rows
    ]
    cosines = [
        [
            finite_float(row["recomputed_gradient_cosine"], "recomputed cosine")
            for row in run
        ]
        for run in rows
    ]
    losses = [
        [
            finite_float(row["recomputed_candidate_loss"], "candidate loss")
            for row in run
        ]
        for run in rows
    ]
    norms = [
        [
            finite_float(row["recomputed_candidate_gradient_norm"], "candidate norm")
            for row in run
        ]
        for run in rows
    ]

    dot_spearman = spearman_correlation(dots[0], dots[1])
    cosine_spearman = spearman_correlation(cosines[0], cosines[1])
    max_cosine_error = max(
        abs(left - right)
        for left, right in zip(*cosines, strict=True)
    )
    max_candidate_loss_error = max(
        abs(left - right) for left, right in zip(*losses, strict=True)
    )
    max_candidate_norm_error = max(
        abs(left - right) for left, right in zip(*norms, strict=True)
    )
    normalized_dot_errors = []
    for left, right, frozen_row in zip(dots[0], dots[1], rows[0], strict=True):
        denominator = finite_float(
            frozen_row["frozen_gradient_norm_product"],
            "frozen gradient norm product",
        )
        require(denominator > 0.0, "Non-positive frozen norm product")
        normalized_dot_errors.append(abs(left - right) / denominator)
    max_normalized_dot_error = max(normalized_dot_errors)

    reference_loss_values = [
        finite_float(
            summary["reference_equivalence"]["recomputed_loss_mean"],
            "reference loss",
        )
        for summary in summaries
    ]
    reference_norm_values = [
        finite_float(
            summary["reference_equivalence"]["recomputed_mean_gradient_norm"],
            "reference norm",
        )
        for summary in summaries
    ]
    reference_loss_error = abs(reference_loss_values[0] - reference_loss_values[1])
    reference_norm_error = abs(reference_norm_values[0] - reference_norm_values[1])
    reference_loss_tolerance = scalar_tolerance(reference_loss_values[0])
    reference_norm_tolerance = scalar_tolerance(reference_norm_values[0])

    gates = {
        "input_identity": input_identity_gate,
        "runtime_identity": runtime_identity_gate,
        "frozen_candidate_identity": frozen_identity_gate,
        "token_metadata": token_metadata_gate,
        "reference_loss_repeat": reference_loss_error <= reference_loss_tolerance,
        "reference_norm_repeat": reference_norm_error <= reference_norm_tolerance,
        "dot_spearman": dot_spearman >= REPRODUCIBILITY_SPEARMAN_MIN,
        "cosine_spearman": cosine_spearman >= REPRODUCIBILITY_SPEARMAN_MIN,
        "normalized_dot_error": (
            max_normalized_dot_error <= NORMALIZED_DOT_DISCREPANCY_MAX
        ),
        "cosine_absolute_error": max_cosine_error <= COSINE_ABSOLUTE_ERROR_MAX,
    }
    repeat_stability_status = "PASS" if all(gates.values()) else "FAIL"
    a100_score_statuses = [
        str(summary["hardware_score_equivalence_status"])
        for summary in summaries
    ]
    hardware_score_equivalence_supported = (
        repeat_stability_status == "PASS"
        and a100_score_statuses == ["PASS", "PASS"]
    )

    low_ten = [top_indices(values, 10, reverse=False) for values in cosines]
    high_ten = [top_indices(values, 10, reverse=True) for values in cosines]
    payload = {
        "status": "COMPLETE",
        "method": "stage6c_2d_h200_scorer_equivalence_v4_repeat_comparison",
        "execution_commit": args.execution_commit,
        "scope": "hardware_score_and_repeat_stability_diagnostic_only",
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "reference_count": EXPECTED_REFERENCE_COUNT,
        "a100_frozen_score_equivalence_statuses": a100_score_statuses,
        "repeat_stability_status": repeat_stability_status,
        "hardware_score_equivalence_supported": (
            hardware_score_equivalence_supported
        ),
        "repeat_stability_gates": {
            name: "PASS" if passed else "FAIL"
            for name, passed in gates.items()
        },
        "repeat_metrics": {
            "dot_spearman": dot_spearman,
            "cosine_spearman": cosine_spearman,
            "spearman_minimum": REPRODUCIBILITY_SPEARMAN_MIN,
            "maximum_normalized_dot_error": max_normalized_dot_error,
            "normalized_dot_error_limit": NORMALIZED_DOT_DISCREPANCY_MAX,
            "maximum_cosine_absolute_error": max_cosine_error,
            "cosine_absolute_error_limit": COSINE_ABSOLUTE_ERROR_MAX,
            "reference_loss_values": reference_loss_values,
            "reference_loss_absolute_error": reference_loss_error,
            "reference_loss_tolerance": reference_loss_tolerance,
            "reference_norm_values": reference_norm_values,
            "reference_norm_absolute_error": reference_norm_error,
            "reference_norm_tolerance": reference_norm_tolerance,
            "maximum_candidate_loss_absolute_error": max_candidate_loss_error,
            "maximum_candidate_norm_absolute_error": max_candidate_norm_error,
            "same_lowest_cosine_candidate": low_ten[0][0] == low_ten[1][0],
            "lowest_ten_overlap": len(set(low_ten[0]).intersection(low_ten[1])),
            "highest_ten_overlap": len(set(high_ten[0]).intersection(high_ten[1])),
        },
        "repeat_runtime": [summary["runtime"] for summary in summaries],
        "interpretation_boundary": (
            "This diagnostic can support H200 score/rank equivalence only. "
            "It does not change the frozen scorer definition, waive the "
            "observed A100/H200 scalar drift, complete the C1-to-C2 loop, "
            "or establish any CosRewrite/student effect."
        ),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
