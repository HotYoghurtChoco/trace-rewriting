"""Rejudge the saved candidates from one editing round with their original context."""

import argparse
import json
from pathlib import Path

from experiments.maxloss_sentence.rewriter import Rewriter, round_seeds


def round_input(row, number):
    trace = row["original_trace"]
    for turn in row["sentence"]["rounds"]:
        source, start = turn["source"], turn["start"]
        end = start + len(source)
        if trace[start:end] != source:
            raise ValueError(f"Recorded source differs at round {turn['round']}")
        if turn["round"] == number:
            candidates = [p["sentence"] for p in turn["proposals"] if "semantic_rating" in p]
            if not candidates:
                raise ValueError(f"No previously judged candidates in round {number}")
            _, seed = round_seeds(row["index"], number)
            return dict(problem=row["problem"], trace=trace, sentence=source,
                        candidates=candidates, seed=seed)
        if turn["accepted"] is not None:
            trace = trace[:start] + turn["accepted"] + trace[end:]
    raise ValueError(f"Round {number} is absent")


def check(records, row_number, round_number, rewriter):
    with records.open() as handle:
        row = next((row for row in map(json.loads, handle) if row["index"] + 1 == row_number), None)
    if row is None:
        raise ValueError(f"Row {row_number} is absent from {records}")
    inputs = round_input(row, round_number)
    assessment = rewriter.assess(**inputs)
    candidates = [dict(sentence=sentence, claim=item["claim"], rating=item["rating"])
                  for sentence, item in zip(inputs["candidates"], assessment["judgments"], strict=True)]
    return {"row": row_number, "round": round_number, "source": inputs["sentence"],
            "source_claim": assessment["source_claim"], "candidates": candidates}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("row", type=int, help="One-based row number")
    parser.add_argument("round", type=int, help="One-based editing round")
    parser.add_argument("endpoint")
    args = parser.parse_args()
    result = check(args.records, args.row, args.round, Rewriter(args.endpoint))
    print(json.dumps(result, ensure_ascii=False, indent=2))
