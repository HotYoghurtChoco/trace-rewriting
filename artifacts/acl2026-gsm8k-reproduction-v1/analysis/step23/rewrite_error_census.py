import json
import re
import sys
from pathlib import Path

import yaml


base_path = Path(sys.argv[1])
clean_path = Path(sys.argv[2])
rewrite_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])


def load_jsonl(path):
    with path.open() as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


base_rows = load_jsonl(base_path)
clean_rows = load_jsonl(clean_path)
rewrite_rows = load_jsonl(rewrite_path)

if not (
    len(base_rows)
    == len(clean_rows)
    == len(rewrite_rows)
    == 1209
):
    raise RuntimeError("Unexpected row counts")


def repetition_score(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0.0

    return 1.0 - len(set(lines)) / len(lines)


foreign_script_pattern = re.compile(
    r"[\u0400-\u04FF"  # Cyrillic
    r"\u0600-\u06FF"   # Arabic
    r"\u0900-\u097F"   # Devanagari
    r"\u3040-\u30FF"   # Japanese
    r"\u3400-\u9FFF"   # CJK
    r"\uAC00-\uD7AF]"  # Hangul
)


per_example = []

for index, (base, clean, rewrite) in enumerate(
    zip(base_rows, clean_rows, rewrite_rows)
):
    text = rewrite["trace"]

    assistant_count = len(
        re.findall(
            r"\bassistant\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    boxed_count = len(
        re.findall(r"\\boxed", text)
    )
    rep_score = repetition_score(text)

    flags = {
        "assistant_marker": assistant_count > 0,
        "multiple_boxed_answers": boxed_count >= 2,
        "many_boxed_answers": boxed_count >= 10,
        "replacement_character": "\ufffd" in text,
        "foreign_script_fragment": bool(
            foreign_script_pattern.search(text)
        ),
        "high_line_repetition": rep_score >= 0.3,
    }

    flags["any_corruption_marker"] = any(
        flags.values()
    )

    per_example.append({
        "index": index,
        "base_raw_correct": bool(
            base["is_raw_correct"]
        ),
        "clean_raw_correct": bool(
            clean["is_raw_correct"]
        ),
        "rewrite_raw_correct": bool(
            rewrite["is_raw_correct"]
        ),
        "rewrite_af_correct": bool(
            rewrite["is_af_correct"]
        ),
        "assistant_count": assistant_count,
        "boxed_count": boxed_count,
        "repetition_score": rep_score,
        "flags": flags,
    })


def flag_summary(flag):
    selected = [
        row
        for row in per_example
        if row["flags"][flag]
    ]
    not_selected = [
        row
        for row in per_example
        if not row["flags"][flag]
    ]

    def accuracy(rows, field):
        if not rows:
            return None

        return sum(
            row[field]
            for row in rows
        ) / len(rows)

    return {
        "count": len(selected),
        "percent": round(
            len(selected) / len(per_example) * 100,
            4,
        ),
        "rewrite_raw_accuracy_with_flag": accuracy(
            selected,
            "rewrite_raw_correct",
        ),
        "rewrite_raw_accuracy_without_flag": accuracy(
            not_selected,
            "rewrite_raw_correct",
        ),
        "rewrite_af_accuracy_with_flag": accuracy(
            selected,
            "rewrite_af_correct",
        ),
    }


flag_names = [
    "assistant_marker",
    "multiple_boxed_answers",
    "many_boxed_answers",
    "replacement_character",
    "foreign_script_fragment",
    "high_line_repetition",
    "any_corruption_marker",
]

category_counts = {
    "clean_correct_rewrite_wrong": sum(
        row["clean_raw_correct"]
        and not row["rewrite_raw_correct"]
        for row in per_example
    ),
    "clean_wrong_rewrite_wrong": sum(
        not row["clean_raw_correct"]
        and not row["rewrite_raw_correct"]
        for row in per_example
    ),
    "rewrite_raw_correct": sum(
        row["rewrite_raw_correct"]
        for row in per_example
    ),
    "rewrite_rescued_by_af": sum(
        not row["rewrite_raw_correct"]
        and row["rewrite_af_correct"]
        for row in per_example
    ),
}

summary = {
    "sample_count": len(per_example),
    "category_counts": category_counts,
    "error_marker_statistics": {
        flag: flag_summary(flag)
        for flag in flag_names
    },
    "maximum_counts": {
        "assistant_count": max(
            row["assistant_count"]
            for row in per_example
        ),
        "boxed_count": max(
            row["boxed_count"]
            for row in per_example
        ),
        "repetition_score": max(
            row["repetition_score"]
            for row in per_example
        ),
    },
}

with output_path.open("w") as f:
    yaml.safe_dump(
        summary,
        f,
        sort_keys=False,
        allow_unicode=True,
    )

print("Sample count:", len(per_example))
print()
print("Correctness categories:")
for name, count in category_counts.items():
    print(f"  {name:30s} {count:4d}")

print()
print("Rewrite degeneration markers:")
for flag in flag_names:
    stats = summary["error_marker_statistics"][flag]

    print(
        f"  {flag:28s} "
        f"{stats['count']:4d} "
        f"({stats['percent']:6.2f}%) | "
        f"raw acc with flag="
        f"{stats['rewrite_raw_accuracy_with_flag']}"
    )

print()
print("Maximum counts:")
for name, value in summary["maximum_counts"].items():
    print(f"  {name}: {value}")

print()
print("Saved to:", output_path)
