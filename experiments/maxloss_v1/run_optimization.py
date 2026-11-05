"""Optimize existing Clean traces under a frozen Qwen base checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
for root in (REPO / "src", REPO):
    sys.path.insert(0, str(root))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--limit", type=int, default=5, help="Number of editable examples in smoke mode.")
    return parser.parse_args()


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def answer_status(solution, trace):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
    try:
        expected = parse(solution)
        observed = parse(trace, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig(), StringExtractionConfig()])
        return {"correct": bool(verify(expected, observed)), "error": None}
    except Exception as error:
        return {"correct": False, "error": f"{type(error).__name__}: {error}"}


def run(args):
    import torch
    import yaml
    from datasets import Dataset, load_from_disk
    from transformers import AutoModelForCausalLM, set_seed
    from optimize.score_candidates import _load_training_tokenizer, _align_model_special_tokens, _tokenize_completion_only_batch
    from experiments.maxloss_v1.optimizer import FixedStudent, UnscorableTrace, optimize_trace
    from experiments.maxloss_v1.text_edits import POLICY_VERSION, answer_start, propose_edits

    config = yaml.safe_load(args.config.read_text())
    training_path = REPO / config["training_config"]
    training = yaml.safe_load(training_path.read_text())
    instruction = training["instruction_generation"]
    rows = load_from_disk(config["input_dataset"])
    if len(rows) != int(config["cohort_size"]):
        raise ValueError(f"Expected the existing {config['cohort_size']}-row cohort, got {len(rows)}")
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    set_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("Run this experiment in a Katana GPU job")

    tokenizer = _load_training_tokenizer(config["tokenizer"])
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], torch_dtype=getattr(torch, config["dtype"]),
        attn_implementation=config["attention_implementation"],
        local_files_only=True, low_cpu_mem_usage=True,
    ).to("cuda")
    _align_model_special_tokens(model, tokenizer)

    def encode(problem, trace):
        return _tokenize_completion_only_batch(
            tokenizer, instruction, [problem], [trace], config["base_model"]
        )

    student = FixedStudent(model, encode, max_tokens=int(config["maximum_total_tokens"]))
    # Metadata describes the actual runtime. It does not rerun historical audits.
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    metadata = {
        "config": config, "mode": args.mode, "smoke_limit": args.limit,
        "source_row_count": len(rows), "input_dataset": config["input_dataset"],
        "output_dir": str(output),
        "instruction": instruction, "proposal_policy": POLICY_VERSION,
        "shortlist_policy": "top_half_ceil_capped_at_candidates_per_round",
        "execution_commit": commit.stdout.strip() if commit.returncode == 0 else "unavailable",
        "pbs_job_id": os.getenv("PBS_JOBID"), "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "model_class": type(model).__name__,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_checkpoint": "base_only_no_adapter", "trainable_model_parameters": 0,
        "objective": "mean_completion_loss_including_template_suffix_and_eos",
        "truncation": "none; over-budget rows are retained unchanged",
        "selection": "first_eligible_in_dataset_order" if args.mode == "smoke" else "all_rows_in_dataset_order",
    }
    write_json(output / "run_config.json", metadata)
    results, skipped = [], []
    search_started = time.perf_counter()
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            row_started = time.perf_counter()
            problem, trace = str(row["problem"]), str(row["original_trace"])
            status = answer_status(str(row["solution"]), trace)
            reason = None
            feature = None
            if not status["correct"]:
                reason = "original_answer_mismatch_or_unparseable"
            elif answer_start(trace) is None:
                reason = "no_protectable_final_answer"
            else:
                try:
                    feature = student.feature(problem, trace)
                except UnscorableTrace as error:
                    reason = str(error)
                if feature is not None and not propose_edits(trace, problem):
                    reason = "no_wording_proposals"
            if reason and args.mode == "smoke":
                skipped.append({"row_index": index, "reason": reason})
                continue
            if reason:
                loss = student.loss(feature) if feature is not None else None
                result = {
                    "original_trace": trace, "rewrite_trace": trace,
                    "loss_before": loss, "loss_after": loss, "loss_gain": 0.0 if loss is not None else None,
                    "accepted_edits": 0, "stop_reason": reason, "rounds": [],
                }
            else:
                result = optimize_trace(
                    student, problem, trace, max_rounds=int(config["max_rounds"]),
                    candidates_per_round=int(config["candidates_per_round"]),
                    min_gain=float(config["minimum_loss_gain"]),
                )
                # A parser sees the full changed text, not only the protected suffix.
                final_status = answer_status(str(row["solution"]), result["rewrite_trace"])
                if not final_status["correct"]:
                    result.update(rewrite_trace=trace, loss_after=result["loss_before"], loss_gain=0.0,
                                  accepted_edits=0, stop_reason="final_answer_verification_fallback")
                result["final_answer_verification"] = final_status
            result.update(
                row_index=index, problem=problem, solution=str(row["solution"]),
                original_answer_verification=status, elapsed_seconds=time.perf_counter() - row_started,
            )
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"row={index} edits={result['accepted_edits']} loss={result['loss_before']} -> {result['loss_after']} stop={result['stop_reason']}", flush=True)
            if args.mode == "smoke" and len(results) >= args.limit:
                break

    if not results or (args.mode == "smoke" and len(results) < args.limit):
        write_json(output / "skipped.json", skipped)
        raise RuntimeError(f"Only {len(results)} eligible examples; requested {args.limit}")
    if args.mode == "formal":
        Dataset.from_list([
            {key: result[key] for key in ("problem", "solution", "original_trace", "rewrite_trace")}
            for result in results
        ]).save_to_disk(str(output / "optimized_dataset"))
    scored = [r for r in results if r["loss_before"] is not None]
    summary = {
        "status": "complete", "mode": args.mode, "row_count": len(results),
        "row_indices": [r["row_index"] for r in results],
        "rows_modified": sum(r["rewrite_trace"] != r["original_trace"] for r in results),
        "scored_rows": len(scored),
        "mean_loss_before": sum(r["loss_before"] for r in scored) / len(scored) if scored else None,
        "mean_loss_after": sum(r["loss_after"] for r in scored) / len(scored) if scored else None,
        "stop_reasons": dict(Counter(r["stop_reason"] for r in results)),
        "skipped_before_smoke_selection": skipped,
        "search_seconds": time.perf_counter() - search_started,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    write_json(output / "summary.json", summary)
    with (output / "examples.md").open("w", encoding="utf-8") as handle:
        handle.write("# MaxLoss trace edits\n\n")
        for r in results:
            handle.write(f"## Row {r['row_index']}\n\n{r['problem']}\n\n")
            handle.write(f"Loss: {r['loss_before']} -> {r['loss_after']}; accepted edits: {r['accepted_edits']}; stop: {r['stop_reason']}.\n\n")
            for round_result in r["rounds"]:
                if round_result["accepted_edit"]:
                    edit = round_result["accepted_edit"]
                    handle.write(f"- Round {round_result['round']}: `{edit['before']}` -> `{edit['after']}`\n")
            handle.write(f"\n### Before\n\n```text\n{r['original_trace']}\n```\n\n### After\n\n```text\n{r['rewrite_trace']}\n```\n\n")
    (output / "OPTIMIZATION_COMPLETE").write_text("execution_complete\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
