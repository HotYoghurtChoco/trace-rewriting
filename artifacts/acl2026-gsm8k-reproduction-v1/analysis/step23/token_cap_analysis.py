import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer


CAP = 1024
NEAR_CAP = 1000


def load_jsonl(path):
    with Path(path).open() as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def tokenize_lengths(tokenizer, texts, batch_size=64):
    lengths = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )

        lengths.extend(
            len(input_ids)
            for input_ids in encoded["input_ids"]
        )

    return lengths


def accuracy_for_mask(correctness, mask):
    selected = correctness[mask]

    if len(selected) == 0:
        return None

    return float(selected.mean())


def summarize(lengths, correctness):
    lengths = np.asarray(lengths, dtype=np.int64)
    correctness = np.asarray(correctness, dtype=np.bool_)

    near_mask = lengths >= NEAR_CAP
    below_near_mask = lengths < NEAR_CAP
    exact_cap_mask = lengths == CAP
    over_cap_mask = lengths > CAP

    return {
        "count": int(len(lengths)),
        "mean_tokens": float(lengths.mean()),
        "std_tokens": float(lengths.std()),
        "min_tokens": int(lengths.min()),
        "p25_tokens": float(np.percentile(lengths, 25)),
        "median_tokens": float(np.percentile(lengths, 50)),
        "p75_tokens": float(np.percentile(lengths, 75)),
        "p90_tokens": float(np.percentile(lengths, 90)),
        "p95_tokens": float(np.percentile(lengths, 95)),
        "max_tokens": int(lengths.max()),
        "near_cap_threshold": NEAR_CAP,
        "near_cap_count": int(near_mask.sum()),
        "near_cap_percent": float(near_mask.mean() * 100),
        "exact_1024_count": int(exact_cap_mask.sum()),
        "exact_1024_percent": float(exact_cap_mask.mean() * 100),
        "over_1024_count": int(over_cap_mask.sum()),
        "over_1024_percent": float(over_cap_mask.mean() * 100),
        "overall_accuracy": float(correctness.mean()),
        "near_cap_accuracy": accuracy_for_mask(
            correctness,
            near_mask,
        ),
        "below_near_cap_accuracy": accuracy_for_mask(
            correctness,
            below_near_mask,
        ),
        "correct_mean_tokens": (
            float(lengths[correctness].mean())
            if correctness.any()
            else None
        ),
        "incorrect_mean_tokens": (
            float(lengths[~correctness].mean())
            if (~correctness).any()
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--rewrite", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    paths = {
        "base": args.base,
        "clean": args.clean,
        "rewrite": args.rewrite,
    }

    datasets = {
        name: load_jsonl(path)
        for name, path in paths.items()
    }

    counts = {
        name: len(rows)
        for name, rows in datasets.items()
    }

    if len(set(counts.values())) != 1:
        raise RuntimeError(
            f"Dataset row counts differ: {counts}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        local_files_only=True,
    )

    summary = {
        "tokenizer": args.tokenizer,
        "generation_cap": CAP,
        "near_cap_threshold": NEAR_CAP,
        "note": (
            "Lengths are obtained by re-tokenizing decoded output "
            "without special tokens. Near-cap counts are a truncation "
            "proxy, not definitive proof that generation stopped only "
            "because of the cap."
        ),
        "models": {},
    }

    csv_rows = []

    for model_name, rows in datasets.items():
        summary["models"][model_name] = {}

        fields = [
            ("raw", "trace", "is_raw_correct"),
            ("answer_forced", "trace_af", "is_af_correct"),
        ]

        for result_name, trace_field, correct_field in fields:
            texts = [
                row[trace_field]
                for row in rows
            ]
            correctness = [
                bool(row[correct_field])
                for row in rows
            ]

            token_lengths = tokenize_lengths(
                tokenizer,
                texts,
            )

            summary["models"][model_name][result_name] = (
                summarize(
                    token_lengths,
                    correctness,
                )
            )

            for index, (
                row,
                text,
                token_count,
                correct,
            ) in enumerate(zip(
                rows,
                texts,
                token_lengths,
                correctness,
            )):
                csv_rows.append({
                    "model": model_name,
                    "result_type": result_name,
                    "index": index,
                    "correct": correct,
                    "token_count": token_count,
                    "character_count": len(text),
                    "whitespace_word_count": len(text.split()),
                    "near_cap": token_count >= NEAR_CAP,
                    "exact_1024": token_count == CAP,
                    "over_1024": token_count > CAP,
                    "problem_preview": (
                        row["problem"]
                        .replace("\n", " ")[:160]
                    ),
                    "trace_preview": (
                        text
                        .replace("\n", " ")[:160]
                    ),
                })

    output_yaml = Path(args.output_yaml)
    output_csv = Path(args.output_csv)

    with output_yaml.open("w") as f:
        yaml.safe_dump(
            summary,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("Tokenizer:", args.tokenizer)
    print("Generation cap:", CAP)
    print("Near-cap threshold:", NEAR_CAP)
    print()

    for model_name in ["base", "clean", "rewrite"]:
        for result_name in ["raw", "answer_forced"]:
            stats = summary["models"][model_name][result_name]

            print(
                f"{model_name:7s} {result_name:13s} | "
                f"mean={stats['mean_tokens']:.2f} | "
                f"median={stats['median_tokens']:.1f} | "
                f"max={stats['max_tokens']} | "
                f">=1000={stats['near_cap_count']} "
                f"({stats['near_cap_percent']:.2f}%) | "
                f"==1024={stats['exact_1024_count']} | "
                f">1024={stats['over_1024_count']} | "
                f"acc_near={stats['near_cap_accuracy']} | "
                f"acc_below={stats['below_near_cap_accuracy']}"
            )

    print()
    print("Saved YAML:", output_yaml)
    print("Saved CSV:", output_csv)


if __name__ == "__main__":
    main()
