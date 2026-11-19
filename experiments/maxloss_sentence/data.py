"""Paired GSM8K cohorts and experiment settings for the sentence comparison."""

import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/maxloss_v1/config.yaml"
SIZES = {"prior-test": (100, 100), "full": (5231, 1209)}
SHARD = 100
CAP = 1280
ROUNDS = 5
MIN_GAIN = 0.0001
ARMS = ("word", "sentence")
STUDENTS = {
    "qwen1p7b": ("qwen3-1.7b-base-ea980cb0a6c2", "qwen3-1.7b-tokenizer-70d244cc86cc"),
    "llama3b": ("llama-3.2-3b-base", "llama-3.2-3b-instruct-tokenizer"),
    "qwen4b": ("qwen3-4b-base-906bfd4b4dc7", "qwen3-4b-tokenizer-1cfa9a720891"),
}
SEEDS = (888, 889, 890)
TRAINING_CONFIG = ROOT / yaml.safe_load(BASE.read_text())["training_config"]
EVAL_100 = (TRAINING_CONFIG.parents[1] / "clean_comparators" / STUDENTS["qwen1p7b"][0]
            / "search_validation_evaluation" / "evaluated_examples.jsonl")


def scoring_instruction():
    record = BASE.parent / "evidence/step2_seed888/formal_9679942/run_config.json"
    return json.loads(record.read_text())["instruction"]


def atomic_json(path, data):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def load_train(profile):
    from datasets import load_from_disk
    ds = load_from_disk(str(ROOT / "data/gsm8k/semantic"))
    if len(ds) != SIZES["full"][0]:
        raise ValueError("Unexpected published GSM8K cohort size")
    return ds.select(range(SIZES[profile][0])).select_columns(["problem", "solution", "original_trace"])


def load_eval(profile, platinum_path=None):
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
    if profile == "prior-test":
        with EVAL_100.open() as f:
            rows = [{k: row[k] for k in ("problem", "solution")} for row in map(json.loads, f)]
        ds = Dataset.from_list(rows)
    else:
        ds = (load_from_disk(str(platinum_path)) if platinum_path else
              load_dataset("madrylab/gsm8k-platinum", split="test"))
        if isinstance(ds, DatasetDict):
            ds = ds["test"]
        if "question" in ds.column_names:
            ds = ds.rename_columns({"question": "problem", "answer": "solution"})
        ds = ds.select_columns(["problem", "solution"])
    if len(ds) != SIZES[profile][1]:
        raise ValueError(f"Expected {SIZES[profile][1]} evaluation questions, found {len(ds)}")
    return ds


def scoring_config(honour):
    config = yaml.safe_load(BASE.read_text())
    model_name, tokenizer_name = STUDENTS["qwen1p7b"]
    config["base_model"] = str(honour / "models" / model_name)
    config["tokenizer"] = str(honour / "models" / tokenizer_name)
    return config


def old_word_records(honour, config, instruction):
    relative = Path("maxloss-qwen3-1.7b-v1/formal_9679942")
    for root in (honour / "results", Path.home() / "honour/results/trace-rewriting"):
        output = root / relative
        if not (output / "OPTIMIZATION_COMPLETE").is_file() or not (output / "records.jsonl").is_file():
            continue
        meta = json.loads((output / "run_config.json").read_text())
        keys = ("base_model_revision", "tokenizer_revision", "max_rounds", "candidates_per_round",
                "minimum_loss_gain", "dtype", "attention_implementation")
        if (meta["config"]["maximum_total_tokens"] == 1024 and meta["instruction"] == instruction
                and all(meta["config"][k] == config[k] for k in keys)):
            with (output / "records.jsonl").open() as f:
                return {r["row_index"]: r for r in map(json.loads, f)}
    return {}
