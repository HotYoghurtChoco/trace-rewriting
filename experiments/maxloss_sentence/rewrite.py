"""Rewrite one ordered shard of both MaxLoss conditions at 1,280 input tokens."""

import argparse
import json
import os
from pathlib import Path
import time

from experiments.maxloss_sentence import data
from experiments.maxloss_sentence.method import optimize_sentence, token_distance
from experiments.maxloss_sentence.rewriter import Rewriter
from experiments.maxloss_v1.optimizer import UnscorableTrace, optimize_trace
from experiments.maxloss_v1.run_optimization import answer_status, load_fixed_student
from experiments.maxloss_v1.text_edits import answer_start, propose_edits


def unchanged(trace, reason, loss=None):
    return {"text": trace, "loss_before": loss, "loss_after": loss, "accepted_edits": 0, "stop": reason}


def word_result(student, row, saved, config):
    trace, problem = row["original_trace"], row["problem"]
    same_row = saved and all(saved.get(key) == row[key] for key in ("problem", "solution", "original_trace"))
    # Aligned word edits keep the total token length. Only a previously over-cap row
    # needs another search when all other settings and the row match.
    if same_row and not saved["stop_reason"].startswith("over_token_budget"):
        result = saved
        source = "formal_9679942"
    elif not propose_edits(trace, problem):
        return unchanged(trace, "no_wording_proposals", student.loss(student.feature(problem, trace)))
    else:
        result = optimize_trace(student, problem, trace, max_rounds=config["max_rounds"],
                                candidates_per_round=config["candidates_per_round"], min_gain=config["minimum_loss_gain"])
        source = "word_search_1280"
    if result["rewrite_trace"] != trace and not answer_status(row["solution"], result["rewrite_trace"])["correct"]:
        return unchanged(trace, "final_answer_verification_fallback", result["loss_before"])
    return {"text": result["rewrite_trace"], "loss_before": result["loss_before"], "loss_after": result["loss_after"],
            "accepted_edits": result["accepted_edits"], "stop": result["stop_reason"], "source": source}


def process_row(student, tokenizer, rewriter, instruction, row, index, saved, config, budget):
    trace = row["original_trace"]
    original_ids = tokenizer.encode(trace.strip(), add_special_tokens=False)
    reason = None
    if not answer_status(row["solution"], trace)["correct"]:
        reason = "original_answer_mismatch_or_unparseable"
    elif answer_start(trace) is None:
        reason = "no_protectable_final_answer"
    else:
        try:
            student.feature(row["problem"], trace)
        except UnscorableTrace as error:
            reason = str(error)
    if reason:
        word = unchanged(trace, reason)
        sentence = unchanged(trace, reason)
    else:
        word = word_result(student, row, saved, config)
        sentence = optimize_sentence(student, tokenizer, rewriter, instruction, row, index, budget)
    for arm in (word, sentence):
        ids = tokenizer.encode(arm["text"].strip(), add_special_tokens=False)
        distance = token_distance(original_ids, ids)
        arm.update(body_tokens=len(ids), token_edits=distance,
                   changed_percent=100 * distance / max(1, len(original_ids)))
    return {"index": index, **row, "original_tokens": len(original_ids), "word": word, "sentence": sentence}


def resume_index(file, start, end, rows):
    """Recover an interrupted final write without discarding completed row records."""
    file.seek(0)
    index = start
    while True:
        offset = file.tell()
        line = file.readline()
        if not line:
            return index
        if not line.endswith(b"\n"):
            file.seek(offset)
            file.truncate()
            return index
        saved = json.loads(line)
        if index >= end or saved["index"] != index or any(saved[k] != rows[index][k] for k in rows.column_names):
            raise ValueError(f"Checkpoint differs from source row {index}")
        index += 1


def run(run_dir, shard, endpoint):
    import torch
    from transformers import set_seed

    settings = json.loads((run_dir / "run.json").read_text())
    rows = data.load_train(settings["profile"])
    start, end = data.SHARD * shard, min(data.SHARD * (shard + 1), len(rows))
    if start >= end or shard < 0:
        raise ValueError(f"Shard {shard} is outside this cohort")
    dest = run_dir / "parts" / f"{shard:03}.jsonl"
    if dest.is_file():
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Sentence rewriting needs the Katana GPU job")
    set_seed(888)
    honour = Path(os.environ["HONOUR_SCRATCH"])
    cfg = data.scoring_config(honour)
    cfg["maximum_total_tokens"] = data.CAP
    instruction = data.scoring_instruction()
    tokenizer, student = load_fixed_student(cfg, instruction)
    rewriter = Rewriter(endpoint)
    cached = data.old_word_records(honour, cfg, instruction) if shard == 0 else {}
    partial = dest.with_suffix(".partial")
    with partial.open("a+b") as out:
        first = resume_index(out, start, end, rows)
        for i in range(first, end):
            started = time.monotonic()
            row = process_row(student, tokenizer, rewriter, instruction, rows[i], i, cached.get(i), cfg,
                              settings["max_edit_percent"])
            row.update(job=os.getenv("PBS_JOBID"), seconds=round(time.monotonic() - started, 3))
            out.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
            out.flush()
            print(f"row={i} seconds={row['seconds']} word={row['word']['changed_percent']:.2f}% "
                  f"sentence={row['sentence']['changed_percent']:.2f}%", flush=True)
    partial.replace(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("shard", type=int)
    parser.add_argument("endpoint")
    args = parser.parse_args()
    run(args.run_dir, args.shard, args.endpoint)
