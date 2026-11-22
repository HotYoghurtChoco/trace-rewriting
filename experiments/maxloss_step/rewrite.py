"""Rewrite one formal shard and measure its startup and per-row cost."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from experiments.maxloss_sentence import data
from experiments.maxloss_sentence.rewrite import resume_index
from experiments.maxloss_step.method import rewrite_row
from experiments.maxloss_step.rewriter import StepRewriter
from experiments.maxloss_v1.run_optimization import load_fixed_student


def run(root, shard, endpoint):
    from transformers import set_seed

    launched = float(os.getenv("MAXLOSS_JOB_STARTED", time.time()))
    rows = data.load_train("full")
    start, end = shard * data.SHARD, min((shard + 1) * data.SHARD, len(rows))
    if not 0 <= start < end:
        raise ValueError(f"Invalid shard {shard}")
    dest = root / "parts" / f"{shard:03}.jsonl"
    if dest.is_file():
        return
    set_seed(888)
    cfg = data.scoring_config(Path(os.environ["HONOUR_SCRATCH"]))
    cfg["maximum_total_tokens"] = data.CAP
    instruction = data.scoring_instruction()
    tokenizer, student = load_fixed_student(cfg, instruction)
    rewriter = StepRewriter(endpoint)
    partial = dest.with_suffix(".partial")
    with partial.open("a+b") as out:
        first = resume_index(out, start, end, rows)
        startup = time.time() - launched
        for i in range(first, end):
            before = time.monotonic()
            row = rewrite_row(student, tokenizer, rewriter, instruction, rows[i], i)
            row.update(seconds=round(time.monotonic() - before, 3), job=os.getenv("PBS_JOBID"))
            out.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
            out.flush()
            print(f"row={i} {row['step']['reason']} seconds={row['seconds']}", flush=True)
        out.seek(0)
        records = [json.loads(line) for line in out]
    reasons = Counter(r["step"]["reason"] for r in records)
    timing = {"rows": len(records), "resumed_rows": first - start, "job": os.getenv("PBS_JOBID"),
              "elapsed_seconds": round(time.time() - launched, 1), "startup_seconds": round(startup, 1),
              "full_shard_seconds": round(startup + sum(r["seconds"] for r in records), 1),
              "selected": dict(Counter(r["step"]["selected"] for r in records)), "reason_counts": dict(reasons)}
    data.atomic_json(dest.with_suffix(".timing.json"), timing)
    print(json.dumps(timing), flush=True)
    partial.replace(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("shard", type=int)
    parser.add_argument("endpoint")
    args = parser.parse_args()
    run(args.run, args.shard, args.endpoint)
