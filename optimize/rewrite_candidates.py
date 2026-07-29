"""Stage 1 of prompt optimization: rewrite traces with each candidate instruction.

Reads `candidate_instructions.yaml` from the experiment directory, and for every
candidate, generates rewritten traces with the teacher model and saves them to disk.
"""

import logging
import os
import warnings

import datasets
import torch
import yaml
from vllm import LLM, SamplingParams

from utils import ConfigManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

ARTIFACT_MARKER = ".stage6a_complete"


class Instruction:
    def __init__(self, name, text, score=0.0):
        self.name = name
        self.text = text
        self.score = score

    def store_traces(self, traces):
        self.traces = traces


def _stage6a_protocol(cfg):
    return str(getattr(cfg, "stage6a_protocol", "paper_a2_accuracy_v1"))


def _verified_artifact_exists(path, cfg):
    if not os.path.exists(path):
        return False

    marker_path = os.path.join(path, ARTIFACT_MARKER)
    if not os.path.isfile(marker_path):
        raise RuntimeError(
            f"Found an incomplete or legacy rewrite artifact at {path}. "
            "Move that directory aside before regenerating it."
        )

    with open(marker_path, "r") as f:
        actual_protocol = f.read().strip()
    expected_protocol = _stage6a_protocol(cfg)
    if actual_protocol != expected_protocol:
        raise RuntimeError(
            f"Rewrite artifact protocol mismatch at {path}: "
            f"expected {expected_protocol}, found {actual_protocol}."
        )

    return True


def _mark_artifact_complete(path, cfg):
    marker_path = os.path.join(path, ARTIFACT_MARKER)
    with open(marker_path, "w") as f:
        f.write(f"{_stage6a_protocol(cfg)}\n")


def generate_and_save_rewrites(cfg):
    os.makedirs(cfg.working_dir, exist_ok=True)
    model = LLM(model=cfg.rewriter_model_name, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(
        max_tokens=int(getattr(cfg, "rewrite_max_tokens", 1024)),
        temperature=float(getattr(cfg, "rewrite_temperature", 0.6)),
    )
    paper_accuracy_protocol = getattr(cfg, "score_type", None) == "acc"

    dataset = datasets.load_from_disk(cfg.original_traces_path)
    dataset = dataset.shuffle(seed=137).select(range(cfg.dataset_size_used_for_optimize))
    dataset = dataset.select_columns(["problem", "solution", "original_trace"])

    instruction_path = os.path.join(cfg.working_dir, cfg.experiment_folder, "candidate_instructions.yaml")
    with open(instruction_path, "r") as f:
        instruction_data = yaml.safe_load(f)
    instructions = [Instruction(item["name"], item["instruction"]) for item in instruction_data["candidate_instructions"]]
    log.info(f"Loaded {len(instructions)} candidate instructions: {[i.name for i in instructions]}")

    for instruction in instructions:
        save_path = os.path.join(cfg.working_dir, instruction.name)
        artifact_exists = (
            _verified_artifact_exists(save_path, cfg)
            if paper_accuracy_protocol
            else os.path.exists(save_path)
        )
        if artifact_exists:
            if paper_accuracy_protocol:
                log.info(
                    f"Verified existing traces for {instruction.name}; "
                    "skipping generation"
                )
            else:
                log.info(f"Skipping existing traces for {instruction.name}")
            continue

        output_path = (
            f"{save_path}.incomplete"
            if paper_accuracy_protocol
            else save_path
        )
        if paper_accuracy_protocol and os.path.exists(output_path):
            raise RuntimeError(
                f"Found an incomplete rewrite artifact at {output_path}. "
                "Inspect and move that directory aside before retrying."
            )

        log.info(f"Generating traces for {instruction.name}")
        prompts = [
            cfg.rewrite_prompt_template.format(instruction=instruction.text, original_trace=sample["original_trace"])
            for sample in dataset
        ]
        conversations = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        outputs = model.chat(
            messages=conversations,
            sampling_params=sampling_params,
        )
        rewrites = [o.outputs[0].text for o in outputs]
        dataset.add_column("rewrite_trace", rewrites).save_to_disk(output_path)
        if paper_accuracy_protocol:
            _mark_artifact_complete(output_path, cfg)
            os.replace(output_path, save_path)
        log.info(f"Saved to {save_path}")


if __name__ == "__main__":
    cfg = ConfigManager().get_config()
    generate_and_save_rewrites(cfg)
