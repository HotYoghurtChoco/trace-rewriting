"""Generate the matched Stage 6C.3 Baseline, SelectionOnly, and CosRewrite data."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from datasets import Dataset, load_from_disk
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    import_text = str(import_root)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)

from experiments.stage6c_cosrewrite_v1 import (  # noqa: E402
    run_multi_problem_batch as batch,
)
from experiments.stage6c_cosrewrite_v1 import (  # noqa: E402
    run_single_problem_closed_loop as single,
)


METHOD = "stage6c3_matched_cosrewrite_selectiononly_v1"
SCHEMA = "stage6c3_matched_generation_v1"
ROW_COUNT = 100
ARM_NAMES = ("baseline_rewrite", "selection_only", "cosrewrite")
GENERATOR_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def generation_seed(scoring_seed: int, row_index: int, offset: int) -> int:
    return scoring_seed * 1000 + row_index * 1000 + offset


def candidate_record(
    candidate_id: str,
    trace: str,
    validity: Mapping[str, Any],
    score: Mapping[str, Any],
    *,
    parent_id: str | None,
    seed: int | None = None,
    attempt: int | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "trace": trace,
        "validity": dict(validity),
        "score": dict(score),
        "generation_seed": seed,
        "generation_attempt": attempt,
    }


def select_generated_trace(
    candidates: Sequence[Mapping[str, Any]],
    *,
    fallback_trace: str,
) -> dict[str, Any]:
    """Choose the lowest-cosine valid generated candidate, or use Baseline."""

    ranked = []
    excluded = []
    for priority, candidate in enumerate(candidates):
        candidate_id = str(candidate["candidate_id"])
        validity = candidate["validity"]
        if not bool(validity.get("valid")):
            excluded.append(
                {
                    "candidate_id": candidate_id,
                    "reasons": list(validity.get("reasons", [])),
                }
            )
            continue
        cosine = float(candidate["score"]["gradient_cosine"])
        if not math.isfinite(cosine):
            raise RuntimeError(f"Non-finite cosine for {candidate_id}")
        ranked.append(
            {
                "candidate_id": candidate_id,
                "gradient_cosine": cosine,
                "priority": priority,
                "trace": str(candidate["trace"]),
            }
        )

    ranked.sort(key=lambda item: (item["gradient_cosine"], item["priority"]))
    if ranked:
        winner = ranked[0]
        return {
            "selection_status": "selected",
            "winner_candidate_id": winner["candidate_id"],
            "selected_source": winner["candidate_id"],
            "winner_gradient_cosine": winner["gradient_cosine"],
            "rewrite_trace": winner["trace"],
            "used_baseline_fallback": False,
            "eligible_ranked": [
                {key: value for key, value in item.items() if key != "trace"}
                for item in ranked
            ],
            "excluded": excluded,
        }

    return {
        "selection_status": "baseline_fallback",
        "winner_candidate_id": None,
        "selected_source": "BaselineRewriteFallback",
        "winner_gradient_cosine": None,
        "rewrite_trace": fallback_trace,
        "used_baseline_fallback": True,
        "eligible_ranked": [],
        "excluded": excluded,
    }


def score_and_validate(
    *,
    scorer: Any,
    reference: Any,
    problem: str,
    trace: str,
    validity: dict[str, Any],
) -> dict[str, Any]:
    score = single.score_trace(scorer, reference, problem, trace)
    single.add_scorer_token_gate(validity, score)
    return score


def process_row(
    *,
    row_index: int,
    row: Mapping[str, Any],
    output_dir: Path,
    state: Mapping[str, Any],
    scoring_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row_dir = output_dir / "rows" / f"row_{row_index:04d}"
    result_path = row_dir / "row_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("schema") != SCHEMA
            or result.get("method") != METHOD
            or result.get("execution_commit") != args.execution_commit
            or result.get("row_index") != row_index
        ):
            raise RuntimeError(f"Incompatible completed row: {result_path}")
        return result

    row_dir.mkdir(parents=True, exist_ok=True)
    api_dir = row_dir / "api"
    api_dir.mkdir(exist_ok=True)

    problem = str(row["problem"])
    solution = str(row["solution"])
    baseline = str(row["rewrite_trace"])
    scorer = state["scorer"]
    reference = state["reference"]

    baseline_validity = single.validate_trace(
        solution=solution,
        trace=baseline,
        parents={},
        finish_reason=None,
    )
    baseline_score = score_and_validate(
        scorer=scorer,
        reference=reference,
        problem=problem,
        trace=baseline,
        validity=baseline_validity,
    )

    c1_prompt = (
        f"Problem:\n{problem}\n\n"
        f"Current trace:\n{baseline}\n\n"
        "Create one materially different but concise complete solution trace for "
        "the same problem. Preserve the same correct final answer."
    )
    c1, c1_attempts = single.generate_candidate(
        label="c1",
        user_prompt=c1_prompt,
        solution=solution,
        parents={"BaselineRewrite": baseline},
        seed_start=generation_seed(scoring_seed, row_index, 1),
        args=args,
        api_dir=api_dir,
    )
    c1_score = score_and_validate(
        scorer=scorer,
        reference=reference,
        problem=problem,
        trace=c1["content"],
        validity=c1["validity"],
    )

    if c1["validity"]["valid"]:
        second_parent_id = "C1"
        second_parent = c1["content"]
        second_parent_score = c1_score
    else:
        second_parent_id = "BaselineRewrite"
        second_parent = baseline
        second_parent_score = baseline_score

    rendered_cosine = f"{float(second_parent_score['gradient_cosine']):.12f}"
    revision_instruction = (
        "Create one materially different but concise complete solution trace. "
        "Preserve the same correct final answer. Return only the solution trace."
    )
    c2_prompt = (
        f"Problem:\n{problem}\n\n"
        f"Current parent trace ({second_parent_id}):\n{second_parent}\n\n"
        "Black-box scorer feedback for the current parent:\n"
        "- primary metric: gradient cosine with the fixed mean reference gradient\n"
        f"- measured value: {rendered_cosine}\n"
        "- target direction: lower is better\n"
        "Use only this scalar summary; no token-level attribution is available.\n\n"
        f"{revision_instruction}"
    )
    s2_prompt = (
        f"Problem:\n{problem}\n\n"
        f"Current parent trace ({second_parent_id}):\n{second_parent}\n\n"
        f"{revision_instruction}"
    )

    paired_second_seed = generation_seed(scoring_seed, row_index, 101)
    c2, c2_attempts = single.generate_candidate(
        label="c2",
        user_prompt=c2_prompt,
        solution=solution,
        parents={"BaselineRewrite": baseline, "C1": c1["content"]},
        seed_start=paired_second_seed,
        args=args,
        api_dir=api_dir,
    )
    c2_score = score_and_validate(
        scorer=scorer,
        reference=reference,
        problem=problem,
        trace=c2["content"],
        validity=c2["validity"],
    )
    s2, s2_attempts = single.generate_candidate(
        label="s2",
        user_prompt=s2_prompt,
        solution=solution,
        parents={"BaselineRewrite": baseline, "C1": c1["content"]},
        seed_start=paired_second_seed,
        args=args,
        api_dir=api_dir,
    )
    s2_score = score_and_validate(
        scorer=scorer,
        reference=reference,
        problem=problem,
        trace=s2["content"],
        validity=s2["validity"],
    )

    baseline_record = candidate_record(
        "BaselineRewrite",
        baseline,
        baseline_validity,
        baseline_score,
        parent_id=None,
    )
    c1_record = candidate_record(
        "C1",
        c1["content"],
        c1["validity"],
        c1_score,
        parent_id="BaselineRewrite",
        seed=int(c1["seed"]),
        attempt=int(c1["attempt"]),
    )
    c2_record = candidate_record(
        "C2",
        c2["content"],
        c2["validity"],
        c2_score,
        parent_id=second_parent_id,
        seed=int(c2["seed"]),
        attempt=int(c2["attempt"]),
    )
    s2_record = candidate_record(
        "S2",
        s2["content"],
        s2["validity"],
        s2_score,
        parent_id=second_parent_id,
        seed=int(s2["seed"]),
        attempt=int(s2["attempt"]),
    )
    cosrewrite = select_generated_trace(
        (c1_record, c2_record),
        fallback_trace=baseline,
    )
    selection_only = select_generated_trace(
        (c1_record, s2_record),
        fallback_trace=baseline,
    )

    result = {
        "schema": SCHEMA,
        "method": METHOD,
        "execution_commit": args.execution_commit,
        "row_index": row_index,
        "problem": problem,
        "solution": solution,
        "original_trace": str(row["original_trace"]),
        "second_round_parent": second_parent_id,
        "paired_second_round_seed_start": paired_second_seed,
        "candidates": {
            "BaselineRewrite": baseline_record,
            "C1": c1_record,
            "C2": c2_record,
            "S2": s2_record,
        },
        "attempt_counts": {
            "C1": len(c1_attempts),
            "C2": len(c2_attempts),
            "S2": len(s2_attempts),
        },
        "selection_only": selection_only,
        "cosrewrite": cosrewrite,
    }
    atomic_json(result_path, result)
    return result


def dataset_rows(results: Sequence[Mapping[str, Any]], arm: str) -> list[dict[str, str]]:
    rows = []
    for result in results:
        if arm == "baseline_rewrite":
            rewrite_trace = result["candidates"]["BaselineRewrite"]["trace"]
        else:
            rewrite_trace = result[arm]["rewrite_trace"]
        rows.append(
            {
                "problem": str(result["problem"]),
                "solution": str(result["solution"]),
                "original_trace": str(result["original_trace"]),
                "rewrite_trace": str(rewrite_trace),
            }
        )
    return rows


def save_datasets(output_dir: Path, results: Sequence[Mapping[str, Any]]) -> None:
    dataset_root = output_dir / "datasets"
    dataset_root.mkdir(exist_ok=True)
    for arm in ARM_NAMES:
        destination = dataset_root / arm
        if destination.exists():
            existing = load_from_disk(str(destination))
            if existing.to_list() != dataset_rows(results, arm):
                raise RuntimeError(f"Existing dataset differs: {destination}")
            continue
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists():
            raise RuntimeError(f"Incomplete dataset exists: {temporary}")
        Dataset.from_list(dataset_rows(results, arm)).save_to_disk(str(temporary))
        temporary.replace(destination)


def summarize(
    *,
    results: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> dict[str, Any]:
    candidate_valid = Counter()
    winner_counts = {"selection_only": Counter(), "cosrewrite": Counter()}
    fallback_counts = Counter()
    shared_winner_count = 0
    for result in results:
        for candidate_id, candidate in result["candidates"].items():
            candidate_valid[candidate_id] += int(candidate["validity"]["valid"])
        for arm in ("selection_only", "cosrewrite"):
            selection = result[arm]
            winner_counts[arm][selection["selected_source"]] += 1
            fallback_counts[arm] += int(selection["used_baseline_fallback"])
        shared_winner_count += int(
            result["selection_only"]["rewrite_trace"]
            == result["cosrewrite"]["rewrite_trace"]
        )

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "method": METHOD,
        "execution_commit": args.execution_commit,
        "job_id": args.job_id,
        "row_count": len(results),
        "row_indices": list(range(len(results))),
        "generator": {
            "model": single.MODEL_ID,
            "revision": GENERATOR_REVISION,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "maximum_generation_attempts": args.max_generation_attempts,
            "shared_c1": True,
            "paired_c2_s2_seed": True,
        },
        "selection": {
            "cosrewrite_pool": ["C1", "C2"],
            "selection_only_pool": ["C1", "S2"],
            "rule": "lowest_gradient_cosine_among_valid_generated_candidates",
            "fallback": "BaselineRewrite_when_no_generated_candidate_is_valid",
        },
        "candidate_valid_counts": dict(sorted(candidate_valid.items())),
        "winner_counts": {
            arm: dict(sorted(counts.items()))
            for arm, counts in winner_counts.items()
        },
        "baseline_fallback_counts": dict(sorted(fallback_counts.items())),
        "identical_selected_trace_count": shared_winner_count,
        "datasets": {arm: f"datasets/{arm}" for arm in ARM_NAMES},
        "elapsed_seconds": elapsed_seconds,
        "student_trained": False,
        "af_measured": False,
    }


def run(args: argparse.Namespace) -> int:
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_generation_attempts < 1:
        raise ValueError("max-generation-attempts must be positive")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != args.execution_commit:
        raise RuntimeError("execution commit differs from repository HEAD")

    inputs = single.load_inputs(REPO_ROOT)
    candidate_path = inputs["formal_root"] / inputs["candidate_name"]
    dataset = load_from_disk(str(candidate_path))
    if len(dataset) != ROW_COUNT:
        raise RuntimeError(f"Expected {ROW_COUNT} frozen rows, found {len(dataset)}")

    model_list = single.get_json(f"{args.api_base_url.rstrip('/')}/models")
    model_ids = [item.get("id") for item in model_list.get("data", [])]
    if single.MODEL_ID not in model_ids:
        raise RuntimeError(f"Generator API does not serve {single.MODEL_ID}")

    state = batch.prepare_scorer_and_reference(
        inputs=inputs,
        output_dir=output_dir,
        args=args,
    )
    scoring_seed = int(inputs["config"]["seed"])
    results = []
    try:
        for row_index in range(ROW_COUNT):
            print(f"stage6c3_row={row_index + 1}/{ROW_COUNT}", flush=True)
            results.append(
                process_row(
                    row_index=row_index,
                    row=dataset[row_index],
                    output_dir=output_dir,
                    state=state,
                    scoring_seed=scoring_seed,
                    args=args,
                )
            )
    finally:
        del state
        gc.collect()
        torch.cuda.empty_cache()

    save_datasets(output_dir, results)
    summary = summarize(
        results=results,
        args=args,
        elapsed_seconds=time.time() - started,
    )
    atomic_json(output_dir / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
