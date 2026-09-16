"""Summarize the three-seed Stage 6C.3 matched downstream-student result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


ARMS = ("baseline_rewrite", "selection_only", "cosrewrite")
SEEDS = (888, 889, 890)
T_CRITICAL_95_DF2 = 4.3026527297


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_seed(students_root: Path, seed: int) -> dict[str, Any]:
    path = students_root / f"seed_{seed}" / "iter_0" / "instructions_and_scores.yaml"
    if not path.is_file():
        raise RuntimeError(f"Missing student result: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = payload.get("candidate_instructions")
    if not isinstance(entries, list):
        raise RuntimeError(f"Invalid student result: {path}")
    by_name = {entry["name"]: entry for entry in entries}
    if set(by_name) != set(ARMS):
        raise RuntimeError(f"Unexpected arms in {path}: {sorted(by_name)}")

    result = {}
    clean_result = None
    for arm in ARMS:
        metrics = by_name[arm].get("proxy_metrics")
        if not isinstance(metrics, list) or len(metrics) != 1:
            raise RuntimeError(f"Missing proxy metrics for {arm}, seed {seed}")
        metric = metrics[0]
        candidate = metric["candidate_evaluation"]
        clean = metric["clean_evaluation"]
        if int(candidate["num_examples"]) != 100:
            raise RuntimeError(f"Evaluation size changed for {arm}, seed {seed}")
        current_clean = {
            "af_accuracy": float(clean["af_accuracy"]),
            "raw_accuracy": float(clean["raw_accuracy"]),
            "raw_token_cap_count": int(clean["raw_token_cap_count"]),
        }
        if clean_result is not None and current_clean != clean_result:
            raise RuntimeError(f"Clean metrics differ across arms for seed {seed}")
        clean_result = current_clean
        result[arm] = {
            "af_accuracy": float(candidate["af_accuracy"]),
            "raw_accuracy": float(candidate["raw_accuracy"]),
            "raw_token_cap_count": int(candidate["raw_token_cap_count"]),
        }
    result["clean"] = clean_result
    return result


def confidence_interval(values: list[float]) -> tuple[float, float]:
    centre = mean(values)
    half_width = T_CRITICAL_95_DF2 * stdev(values) / math.sqrt(len(values))
    return centre - half_width, centre + half_width


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 6C.3 matched student result",
        "",
        "| Seed | Clean AF | Baseline AF | SelectionOnly AF | CosRewrite AF | SelectionOnly - CosRewrite |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["per_seed"]:
        lines.append(
            "| {seed} | {clean:.3f} | {baseline:.3f} | {selection:.3f} | {cosrewrite:.3f} | {difference:+.3f} |".format(
                seed=item["seed"],
                clean=item["clean"]["af_accuracy"],
                baseline=item["baseline_rewrite"]["af_accuracy"],
                selection=item["selection_only"]["af_accuracy"],
                cosrewrite=item["cosrewrite"]["af_accuracy"],
                difference=item["selection_only_af_minus_cosrewrite_af"],
            )
        )
    primary = summary["primary_contrast"]
    lines.extend(
        [
            "",
            "Primary contrast: `SelectionOnly AF - CosRewrite AF`; positive values favour CosRewrite.",
            "",
            f"Mean difference: **{primary['mean']:+.3f}** ({primary['mean'] * 100:+.1f} percentage points).",
            f"95% paired t interval: [{primary['ci95_low']:+.3f}, {primary['ci95_high']:+.3f}].",
            f"Interpretation: **{primary['interpretation']}**.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    students_root = args.students_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    differences = []
    for seed in SEEDS:
        arms = load_seed(students_root, seed)
        difference = (
            arms["selection_only"]["af_accuracy"]
            - arms["cosrewrite"]["af_accuracy"]
        )
        differences.append(difference)
        per_seed.append(
            {
                "seed": seed,
                **arms,
                "selection_only_af_minus_cosrewrite_af": difference,
            }
        )

    ci_low, ci_high = confidence_interval(differences)
    if ci_low > 0:
        interpretation = "CosRewrite lowered held-out AF consistently"
    elif ci_high < 0:
        interpretation = "CosRewrite increased held-out AF consistently"
    else:
        interpretation = "the three-seed result is inconclusive"

    arm_means = {
        arm: {
            "mean_af_accuracy": mean(item[arm]["af_accuracy"] for item in per_seed),
            "mean_raw_accuracy": mean(item[arm]["raw_accuracy"] for item in per_seed),
            "mean_raw_token_cap_count": mean(
                item[arm]["raw_token_cap_count"] for item in per_seed
            ),
        }
        for arm in ("clean", *ARMS)
    }
    summary = {
        "schema": "stage6c3_matched_student_summary_v1",
        "status": "PASS",
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "arm_means": arm_means,
        "primary_contrast": {
            "definition": "selection_only_af_minus_cosrewrite_af",
            "positive_direction": "CosRewrite lowers held-out AF",
            "values": differences,
            "mean": mean(differences),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "interval": "paired_t_df_2",
            "interpretation": interpretation,
        },
    }
    write_json(output_dir / "stage6c3_student_summary.json", summary)
    write_markdown(output_dir / "stage6c3_student_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
