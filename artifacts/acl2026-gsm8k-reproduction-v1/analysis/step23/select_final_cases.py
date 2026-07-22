import json
import re
import statistics
import sys
from pathlib import Path


base_path = Path(sys.argv[1])
clean_path = Path(sys.argv[2])
rewrite_path = Path(sys.argv[3])
jsonl_output = Path(sys.argv[4])
text_output = Path(sys.argv[5])


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


foreign_pattern = re.compile(
    r"[\u0400-\u04FF"
    r"\u0600-\u06FF"
    r"\u0900-\u097F"
    r"\u3040-\u30FF"
    r"\u3400-\u9FFF"
    r"\uAC00-\uD7AF]"
)


def repetition_score(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0.0

    return 1.0 - len(set(lines)) / len(lines)


def text_metrics(text):
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
    repetition = repetition_score(text)
    replacement = "\ufffd" in text
    foreign = bool(foreign_pattern.search(text))

    degeneration_score = (
        assistant_count
        + boxed_count
        + 100 * repetition
        + 10 * int(replacement)
        + 10 * int(foreign)
    )

    return {
        "assistant_count": assistant_count,
        "boxed_count": boxed_count,
        "repetition_score": repetition,
        "replacement_character": replacement,
        "foreign_script_fragment": foreign,
        "degeneration_score": degeneration_score,
    }


records = []

for index, (base, clean, rewrite) in enumerate(
    zip(base_rows, clean_rows, rewrite_rows)
):
    records.append({
        "index": index,
        "base": base,
        "clean": clean,
        "rewrite": rewrite,
        "metrics": text_metrics(rewrite["trace"]),
    })


used_indices = set()
selected = []


def add_selection(label, record):
    used_indices.add(record["index"])
    selected.append({
        "selection_label": label,
        **record,
    })


def typical_pick(predicate):
    candidates = [
        row
        for row in records
        if predicate(row)
        and row["index"] not in used_indices
    ]

    if not candidates:
        raise RuntimeError("No candidate found")

    median_assistant = statistics.median(
        row["metrics"]["assistant_count"]
        for row in candidates
    )
    median_boxed = statistics.median(
        row["metrics"]["boxed_count"]
        for row in candidates
    )
    median_repetition = statistics.median(
        row["metrics"]["repetition_score"]
        for row in candidates
    )

    def distance(row):
        metrics = row["metrics"]

        return (
            abs(
                metrics["assistant_count"]
                - median_assistant
            ) / max(1, median_assistant)
            + abs(
                metrics["boxed_count"]
                - median_boxed
            ) / max(1, median_boxed)
            + abs(
                metrics["repetition_score"]
                - median_repetition
            )
        )

    return min(candidates, key=distance)


add_selection(
    "typical_clean_correct_rewrite_wrong",
    typical_pick(
        lambda row:
        row["clean"]["is_raw_correct"]
        and not row["rewrite"]["is_raw_correct"]
    ),
)

add_selection(
    "typical_rewrite_raw_correct",
    typical_pick(
        lambda row:
        row["rewrite"]["is_raw_correct"]
    ),
)

add_selection(
    "typical_rewrite_rescued_by_af",
    typical_pick(
        lambda row:
        not row["rewrite"]["is_raw_correct"]
        and row["rewrite"]["is_af_correct"]
    ),
)

extreme_candidates = [
    row
    for row in records
    if not row["clean"]["is_raw_correct"]
    and not row["rewrite"]["is_raw_correct"]
    and row["index"] not in used_indices
]

add_selection(
    "extreme_clean_wrong_rewrite_wrong",
    max(
        extreme_candidates,
        key=lambda row:
        row["metrics"]["degeneration_score"],
    ),
)

least_repetitive_candidates = [
    row
    for row in records
    if row["index"] not in used_indices
]

add_selection(
    "least_repetitive_rewrite_output",
    min(
        least_repetitive_candidates,
        key=lambda row: (
            row["metrics"]["repetition_score"],
            row["metrics"]["assistant_count"]
            + row["metrics"]["boxed_count"],
        ),
    ),
)


def reference_answer(solution):
    if "####" in solution:
        return solution.rsplit("####", 1)[1].strip()

    return "not extracted"


def preview(text, limit=600):
    return (
        text
        .replace("\n", " ")
        .replace("\r", " ")[:limit]
    )


with jsonl_output.open("w") as f:
    for row in selected:
        record = {
            "selection_label": row["selection_label"],
            "index": row["index"],
            "problem": row["rewrite"]["problem"],
            "solution": row["rewrite"]["solution"],
            "reference_answer": reference_answer(
                row["rewrite"]["solution"]
            ),
            "base_raw_correct": bool(
                row["base"]["is_raw_correct"]
            ),
            "clean_raw_correct": bool(
                row["clean"]["is_raw_correct"]
            ),
            "rewrite_raw_correct": bool(
                row["rewrite"]["is_raw_correct"]
            ),
            "rewrite_af_correct": bool(
                row["rewrite"]["is_af_correct"]
            ),
            "rewrite_metrics": row["metrics"],
            "base_trace": row["base"]["trace"],
            "clean_trace": row["clean"]["trace"],
            "rewrite_trace": row["rewrite"]["trace"],
            "rewrite_trace_af": row["rewrite"]["trace_af"],
        }

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


report_lines = []

for row in selected:
    report_lines.extend([
        "=" * 90,
        f"Selection: {row['selection_label']}",
        f"Index: {row['index']}",
        (
            "Correctness: "
            f"Base={row['base']['is_raw_correct']}, "
            f"Clean={row['clean']['is_raw_correct']}, "
            f"Rewrite={row['rewrite']['is_raw_correct']}, "
            f"Rewrite-AF={row['rewrite']['is_af_correct']}"
        ),
        (
            "Rewrite metrics: "
            f"assistant={row['metrics']['assistant_count']}, "
            f"boxed={row['metrics']['boxed_count']}, "
            f"repetition="
            f"{row['metrics']['repetition_score']:.4f}"
        ),
        f"Reference answer: "
        f"{reference_answer(row['rewrite']['solution'])}",
        f"Problem: {row['rewrite']['problem']}",
        "",
        f"Base preview: "
        f"{preview(row['base']['trace'])}",
        "",
        f"Clean preview: "
        f"{preview(row['clean']['trace'])}",
        "",
        f"Rewrite preview: "
        f"{preview(row['rewrite']['trace'])}",
        "",
        f"Rewrite-AF preview: "
        f"{preview(row['rewrite']['trace_af'])}",
        "",
    ])


report = "\n".join(report_lines)

text_output.write_text(
    report,
    encoding="utf-8",
)

print(report)
print()
print("Saved JSONL:", jsonl_output)
print("Saved text report:", text_output)
