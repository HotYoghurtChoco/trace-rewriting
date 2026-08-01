"""Stage 2 of prompt optimization: score each candidate instruction.

For every candidate, we warm up a proxy student model on the original traces, then
fine-tune it on the rewritten traces and measure either loss or downstream accuracy.
The score drives the next round of instruction evolution.
"""

import gc
import json
import logging
import multiprocessing
import os
import traceback
import warnings

import datasets
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

from utils import ConfigManager, load_gsm8k

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

MODEL_COMPLETE_MARKER = ".stage6a_complete"
MODEL_METADATA_FILE = "stage6a_model_metadata.yaml"
ADAPTER_SUBDIR = "adapter"


class Instruction:
    def __init__(self, name, text, score=0.0):
        self.name = name
        self.text = text
        self.score = score
        self.proxy_metrics = []

    def store_traces(self, traces):
        self.traces = traces


def _stage6a_protocol(cfg):
    return str(getattr(cfg, "stage6a_protocol", "paper_a2_accuracy_v1"))


def _validate_accuracy_protocol_config(cfg):
    if cfg.score_type != "acc":
        return

    required = {
        "stage6a_model_artifact": "adapter_only",
        "stage6a_evaluation_mode": "vllm_dynamic_lora",
        "stage6a_training_dtype": "bfloat16",
        "stage6a_inference_dtype": "bfloat16",
    }
    for key, expected in required.items():
        actual = getattr(cfg, key, None)
        if actual != expected:
            raise ValueError(
                f"{key} must be {expected!r} for the Stage 6A "
                f"accuracy protocol; found {actual!r}."
            )

    if bool(cfg.rescore):
        raise ValueError(
            "rescore=true is not allowed for the resumable Stage 6A "
            "accuracy protocol."
        )


def _verified_model_exists(
    model_dir,
    cfg,
    expected_base_model=None,
    expected_tokenizer=None,
):
    if not os.path.exists(model_dir):
        return False

    marker_path = os.path.join(model_dir, MODEL_COMPLETE_MARKER)
    if not os.path.isfile(marker_path):
        raise RuntimeError(
            f"Found an incomplete or legacy model artifact at {model_dir}. "
            "Move that directory aside before retraining it."
        )

    with open(marker_path, "r") as f:
        actual_protocol = f.read().strip()
    expected_protocol = _stage6a_protocol(cfg)
    if actual_protocol != expected_protocol:
        raise RuntimeError(
            f"Model artifact protocol mismatch at {model_dir}: "
            f"expected {expected_protocol}, found {actual_protocol}."
        )

    metadata_path = os.path.join(model_dir, MODEL_METADATA_FILE)
    if not os.path.isfile(metadata_path):
        raise RuntimeError(
            f"Model artifact metadata is missing: {metadata_path}"
        )

    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    expected_metadata = {
        "protocol": expected_protocol,
        "artifact_type": "lora_adapter_only",
        "training_dtype": "bfloat16",
        "evaluation_mode": "vllm_dynamic_lora",
    }
    if expected_base_model is not None:
        expected_metadata["base_model"] = expected_base_model
    if expected_tokenizer is not None:
        expected_metadata["tokenizer"] = expected_tokenizer

    for key, expected_value in expected_metadata.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise RuntimeError(
                f"Model artifact metadata mismatch for {key} at "
                f"{model_dir}: expected {expected_value!r}, "
                f"found {actual_value!r}."
            )

    adapter_dir = os.path.join(model_dir, ADAPTER_SUBDIR)
    required_files = (
        os.path.join(adapter_dir, "adapter_config.json"),
    )
    for required_file in required_files:
        if not os.path.isfile(required_file):
            raise RuntimeError(
                f"Required adapter file is missing: {required_file}"
            )

    adapter_weights = (
        os.path.join(adapter_dir, "adapter_model.safetensors"),
        os.path.join(adapter_dir, "adapter_model.bin"),
    )
    if not any(os.path.isfile(path) for path in adapter_weights):
        raise RuntimeError(
            f"No adapter weights were found in {adapter_dir}."
        )

    dense_weights = [
        name
        for name in os.listdir(model_dir)
        if name.startswith("model")
        and name.endswith((".safetensors", ".bin"))
    ]
    if dense_weights:
        raise RuntimeError(
            f"Unexpected dense-model weights in adapter-only artifact "
            f"{model_dir}: {dense_weights}"
        )

    return True


def _mark_model_complete(
    model_dir,
    cfg,
    base_model,
    tokenizer_name,
):
    metadata = {
        "protocol": _stage6a_protocol(cfg),
        "artifact_type": "lora_adapter_only",
        "base_model": base_model,
        "tokenizer": tokenizer_name,
        "training_dtype": "bfloat16",
        "evaluation_mode": "vllm_dynamic_lora",
    }
    metadata_path = os.path.join(model_dir, MODEL_METADATA_FILE)
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    marker_path = os.path.join(model_dir, MODEL_COMPLETE_MARKER)
    with open(marker_path, "w") as f:
        f.write(f"{_stage6a_protocol(cfg)}\n")


def _save_adapter_only_artifact(
    trainer,
    final_model_dir,
    cfg,
    base_model,
    tokenizer_name,
):
    incomplete_dir = f"{final_model_dir}.incomplete"
    if os.path.exists(final_model_dir):
        raise RuntimeError(
            f"Refusing to overwrite existing artifact: {final_model_dir}"
        )
    if os.path.exists(incomplete_dir):
        raise RuntimeError(
            f"Found incomplete artifact: {incomplete_dir}. "
            "Inspect and move it aside before retrying."
        )

    os.makedirs(incomplete_dir, exist_ok=False)
    adapter_dir = os.path.join(incomplete_dir, ADAPTER_SUBDIR)
    trainer.save_model(adapter_dir)
    _mark_model_complete(
        incomplete_dir,
        cfg,
        base_model,
        tokenizer_name,
    )
    _verified_model_exists(
        incomplete_dir,
        cfg,
        expected_base_model=base_model,
        expected_tokenizer=tokenizer_name,
    )
    os.replace(incomplete_dir, final_model_dir)


def _load_training_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left",
    )
    if "llama" in tokenizer_name.lower():
        tokenizer.pad_token_id = 128004
        tokenizer.eos_token_id = 128001
        tokenizer.add_eos_token = False
    elif tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Tokenizer {tokenizer_name} has neither a pad token "
                "nor an EOS token that can be reused for padding."
            )
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name} has no EOS token.")
    return tokenizer


def _align_model_special_tokens(model, tokenizer):
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.add_eos_token = False


def _tokenize_completion_only_batch(
    tokenizer,
    instruction,
    problems,
    responses,
    model_name,
):
    prompt_messages = [
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": problem.strip()},
        ]
        for problem in problems
    ]
    full_messages = [
        prompt
        + [
            {
                "role": "assistant",
                "content": response.strip(),
            }
        ]
        for prompt, response in zip(
            prompt_messages,
            responses,
            strict=True,
        )
    ]
    prompt_tokens = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
    )
    full_tokens = tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
    )

    completion_masks = []
    for sample_index, (prompt_ids, full_ids) in enumerate(
        zip(prompt_tokens, full_tokens, strict=True)
    ):
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Chat-template prefix mismatch for {model_name} "
                f"at batch sample {sample_index}; cannot construct "
                "a reliable completion-only loss mask."
            )
        completion_masks.append(
            [0] * len(prompt_ids)
            + [1] * (len(full_ids) - len(prompt_ids))
        )

    return {
        "input_ids": full_tokens,
        "completion_mask": completion_masks,
    }



def _proxy_training_settings(cfg, accelerator, paper_accuracy_protocol):
    """Resolve and validate proxy-training hyperparameters.

    Accuracy-scoring runs may override these values in YAML. The defaults
    intentionally reproduce the pre-v3 implementation. The legacy
    loss-scoring path remains unchanged.
    """
    if paper_accuracy_protocol:
        lora_r = int(getattr(cfg, "proxy_lora_r", 128))
        lora_alpha = int(getattr(cfg, "proxy_lora_alpha", 128))
        lora_dropout = float(getattr(cfg, "proxy_lora_dropout", 0.0))
        per_device_batch_size = int(
            getattr(cfg, "proxy_per_device_train_batch_size", 4)
        )
        effective_batch_size = int(
            getattr(cfg, "proxy_effective_batch_size", 32)
        )
        epochs = float(getattr(cfg, "proxy_train_epochs", 2))
        learning_rate = float(getattr(cfg, "proxy_learning_rate", 5e-4))
    else:
        lora_r = 128
        lora_alpha = 128
        lora_dropout = 0.0
        per_device_batch_size = 4
        effective_batch_size = 32
        epochs = 1.0
        learning_rate = float(5e-6)

    if lora_r <= 0:
        raise ValueError("proxy_lora_r must be positive")
    if lora_alpha <= 0:
        raise ValueError("proxy_lora_alpha must be positive")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError("proxy_lora_dropout must be in [0, 1)")
    if per_device_batch_size <= 0:
        raise ValueError(
            "proxy_per_device_train_batch_size must be positive"
        )
    if effective_batch_size <= 0:
        raise ValueError("proxy_effective_batch_size must be positive")
    if epochs <= 0:
        raise ValueError("proxy_train_epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("proxy_learning_rate must be positive")

    num_gpus = accelerator.num_processes
    global_micro_batch = num_gpus * per_device_batch_size
    if effective_batch_size % global_micro_batch != 0:
        raise ValueError(
            f"Effective batch size {effective_batch_size} is not divisible "
            f"by {num_gpus} processes x batch size "
            f"{per_device_batch_size}."
        )

    return {
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "per_device_batch_size": per_device_batch_size,
        "effective_batch_size": effective_batch_size,
        "gradient_accumulation_steps": (
            effective_batch_size // global_micro_batch
        ),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "num_gpus": num_gpus,
    }


def save_instruction_scores(instructions, save_path):
    payload = {
        "candidate_instructions": [
            {
                "name": instruction.name,
                "instruction": instruction.text,
                "score": instruction.score,
                "proxy_metrics": instruction.proxy_metrics,
            }
            for instruction in instructions
        ]
    }

    temp_path = f"{save_path}.tmp"
    with open(temp_path, "w") as f:
        yaml.safe_dump(
            payload,
            f,
            sort_keys=False,
            allow_unicode=True,
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, save_path)

def load_search_validation_dataset(cfg):
    dataset = load_gsm8k(split="holdout")
    eval_size = int(cfg.eval_dataset_size_for_optimize)

    if eval_size <= 0:
        raise ValueError("eval_dataset_size_for_optimize must be positive")

    if eval_size > len(dataset):
        raise ValueError(
            f"Requested {eval_size} validation examples, "
            f"but holdout only contains {len(dataset)}"
        )

    return dataset.shuffle(seed=137).select(range(eval_size))

def warmup_models(cfg, accelerator):
    trace_dataset = datasets.load_from_disk(cfg.original_traces_path)
    paper_accuracy_protocol = cfg.score_type == "acc"
    if paper_accuracy_protocol:
        train_size = int(cfg.dataset_size_used_for_optimize)
        trace_dataset = (
            trace_dataset.shuffle(seed=137)
            .select(range(train_size))
        )

    for model_config in cfg.proxy_models:
        model_name = model_config['name']
        tokenizer_name = model_config.get('tokenizer', model_name)
        model_group = (
            "clean_comparators"
            if paper_accuracy_protocol
            else "warmed_up_models"
        )
        final_model_dir = os.path.join(
            cfg.working_dir,
            model_group,
            model_name.split("/")[-1],
        )
        model_exists = (
            _verified_model_exists(
                final_model_dir,
                cfg,
                expected_base_model=model_name,
                expected_tokenizer=tokenizer_name,
            )
            if paper_accuracy_protocol
            else os.path.exists(final_model_dir)
        )
        if model_exists:
            if accelerator.is_main_process: log.info(f"Found existing model at {final_model_dir}, skipping warmup...")
            continue
        else:
            if accelerator.is_main_process: log.info(f"Warmup for model {model_name}... Will save to {final_model_dir}")

        tokenizer = _load_training_tokenizer(tokenizer_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            use_cache=True,
        )
        _align_model_special_tokens(model, tokenizer)

        def preprocess_function(examples):
            trace_colname = 'original_trace'
            return _tokenize_completion_only_batch(
                tokenizer=tokenizer,
                instruction=cfg.instruction_generation,
                problems=examples["problem"],
                responses=examples[trace_colname],
                model_name=model_name,
            )

        dataset = trace_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=16384,
            num_proc=int(os.environ.get("TRACE_NUM_PROC", "4")),
            remove_columns=list(trace_dataset.column_names),
            desc="Preprocessing train dataset",
            load_from_cache_file=False
        )

        training_settings = _proxy_training_settings(
            cfg,
            accelerator,
            paper_accuracy_protocol,
        )
        peft_parameters = LoraConfig(
            r=training_settings["lora_r"],
            lora_alpha=training_settings["lora_alpha"],
            lora_dropout=training_settings["lora_dropout"],
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_parameters)
        if accelerator.is_main_process:
            model.print_trainable_parameters()
            log.info(
                "LoRA hyperparams: "
                f"rank={training_settings['lora_r']}, "
                f"alpha={training_settings['lora_alpha']}, "
                f"dropout={training_settings['lora_dropout']}"
            )

        num_gpus = training_settings["num_gpus"]
        per_device_batch_size = training_settings[
            "per_device_batch_size"
        ]
        gradient_accumulation_steps = training_settings[
            "gradient_accumulation_steps"
        ]
        epochs = training_settings["epochs"]
        learning_rate = training_settings["learning_rate"]

        if accelerator.is_main_process:
            log.info(
                f"hyperparams: bs={per_device_batch_size}, grad_accum={gradient_accumulation_steps}, "
                f"effective_bs={per_device_batch_size * gradient_accumulation_steps * num_gpus}, "
                f"epochs={epochs}, lr={learning_rate}"
            )

        training_args = SFTConfig(
            num_train_epochs=float(epochs),
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=float(learning_rate),
            weight_decay=(
                float(getattr(cfg, "proxy_weight_decay", 0.1))
                if paper_accuracy_protocol
                else float(0.01)
            ),
            warmup_ratio=(
                float(getattr(cfg, "proxy_warmup_ratio", 0.1))
                if paper_accuracy_protocol
                else float(0.03)
            ),
            lr_scheduler_type="cosine",
            max_grad_norm=float(1.0),
            remove_unused_columns=False,
            label_names=["labels"],
            completion_only_loss=True,

            bf16=True,
            fp16=False,
            
            seed=cfg.seed,
            log_level="info",
            logging_steps=10,
            logging_strategy="steps",

            save_strategy="no",
            report_to="none",

            gradient_checkpointing=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            group_by_length=True,
        )
        
        # Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        # Training
        trainer.train()

        if accelerator.is_main_process:
            if paper_accuracy_protocol:
                _save_adapter_only_artifact(
                    trainer,
                    final_model_dir,
                    cfg,
                    base_model=model_name,
                    tokenizer_name=tokenizer_name,
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    return_dict=True,
                    device_map="cpu",
                )
                _align_model_special_tokens(base_model, tokenizer)
                adapter_dir = os.path.join(final_model_dir, "adapter")
                trainer.save_model(adapter_dir)
                model_to_merge = PeftModel.from_pretrained(
                    base_model,
                    adapter_dir,
                )
                merged_model = model_to_merge.merge_and_unload()
                merged_model.save_pretrained(final_model_dir)
                tokenizer.save_pretrained(final_model_dir)
                del merged_model, model_to_merge, base_model
            log.info(f"Warmup for model {model_name} done. Saved at {final_model_dir}")

        trainer.accelerator.free_memory()
        model, tokenizer, dataset = accelerator.free_memory(
            model,
            tokenizer,
            dataset,
        )
        del model, tokenizer, dataset, trainer
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

_DISTRIBUTED_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "FORK_LAUNCHED",
    "VLLM_DP_RANK",
    "VLLM_DP_SIZE",
    "VLLM_DP_LOCAL_RANK",
    "VLLM_DP_MASTER_IP",
    "VLLM_DP_MASTER_PORT",
)


def _evaluate_candidate_af_accuracy_worker(
    artifact_dir,
    base_model_dir,
    tokenizer_name,
    validation_data,
    result_writer,
):
    try:
        for variable_name in _DISTRIBUTED_ENV_VARS:
            os.environ.pop(variable_name, None)

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            first_device = visible_devices.split(",", 1)[0].strip()
            if first_device:
                os.environ["CUDA_VISIBLE_DEVICES"] = first_device

        from evaluate import evaluate_dataset_with_vllm
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        validation_dataset = datasets.Dataset.from_dict(
            validation_data
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
        )
        adapter_dir = os.path.join(artifact_dir, ADAPTER_SUBDIR)
        adapter_config_path = os.path.join(
            adapter_dir,
            "adapter_config.json",
        )
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        adapter_rank = int(adapter_config["r"])

        model = LLM(
            model=base_model_dir,
            tokenizer=tokenizer_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=32768,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            dtype="bfloat16",
            enable_lora=True,
            max_loras=1,
            max_lora_rank=adapter_rank,
        )
        lora_request = LoRARequest(
            "stage6a_adapter",
            1,
            adapter_dir,
        )
        evaluated_dataset, metrics = evaluate_dataset_with_vllm(
            model=model,
            tokenizer=tokenizer,
            dataset=validation_dataset,
            dataset_name="gsm8k",
            temperature=0.0,
            max_new_tokens=1024,
            lora_request=lora_request,
        )

        metrics.update({
            "evaluation_mode": "vllm_dynamic_lora",
            "base_model": base_model_dir,
            "adapter": adapter_dir,
            "tokenizer": tokenizer_name,
            "inference_dtype": "bfloat16",
            "adapter_rank": adapter_rank,
        })

        evaluation_dir = os.path.join(
            artifact_dir,
            "search_validation_evaluation",
        )
        os.makedirs(evaluation_dir, exist_ok=True)
        dataset_path = os.path.join(
            evaluation_dir,
            "evaluated_examples.jsonl",
        )
        dataset_temp_path = f"{dataset_path}.tmp"
        evaluated_dataset.to_json(
            dataset_temp_path,
            orient="records",
            lines=True,
        )
        os.replace(dataset_temp_path, dataset_path)

        metrics_path = os.path.join(evaluation_dir, "metrics.yaml")
        metrics_temp_path = f"{metrics_path}.tmp"
        with open(metrics_temp_path, "w") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(metrics_temp_path, metrics_path)

        result_writer.send(
            {
                "status": "ok",
                "metrics": metrics,
            }
        )
    except Exception:
        result_writer.send(
            {
                "status": "error",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        result_writer.close()


def evaluate_candidate_af_accuracy(
    artifact_dir,
    base_model_dir,
    tokenizer_name,
    validation_dataset,
):
    context = multiprocessing.get_context("spawn")
    result_reader, result_writer = context.Pipe(duplex=False)

    process = context.Process(
        target=_evaluate_candidate_af_accuracy_worker,
        args=(
            artifact_dir,
            base_model_dir,
            tokenizer_name,
            validation_dataset.to_dict(),
            result_writer,
        ),
        daemon=False,
    )

    try:
        process.start()
    except Exception:
        result_reader.close()
        result_writer.close()
        raise

    result_writer.close()

    try:
        try:
            result = result_reader.recv()
        except EOFError:
            result = None
    finally:
        result_reader.close()
        process.join()

    exitcode = process.exitcode
    process.close()

    if exitcode != 0:
        raise RuntimeError(
            "Candidate evaluator failed "
            f"with exit code {exitcode}."
        )

    if result is None:
        raise RuntimeError(
            "Candidate evaluator exited without returning a result."
        )

    if result.get("status") != "ok":
        raise RuntimeError(
            "Candidate evaluator failed in the isolated process:\n"
            f"{result.get('traceback', 'No traceback was returned.')}"
        )

    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(
            "Candidate evaluator returned invalid metrics."
        )
    return metrics


def _proxy_comparison_metrics(
    model_name,
    clean_metrics,
    candidate_metrics,
):
    candidate_score = (
        clean_metrics["af_accuracy"]
        - candidate_metrics["af_accuracy"]
    )
    return candidate_score, {
        "model": model_name,
        "evaluation_mode": "vllm_dynamic_lora",
        "clean_raw_accuracy": clean_metrics["raw_accuracy"],
        "clean_af_accuracy": clean_metrics["af_accuracy"],
        "clean_raw_token_cap_count": clean_metrics[
            "raw_token_cap_count"
        ],
        "candidate_raw_accuracy": candidate_metrics[
            "raw_accuracy"
        ],
        "candidate_af_accuracy": candidate_metrics[
            "af_accuracy"
        ],
        "candidate_raw_token_cap_count": candidate_metrics[
            "raw_token_cap_count"
        ],
        "accuracy_drop": candidate_score,
        "clean_evaluation": clean_metrics,
        "candidate_evaluation": candidate_metrics,
    }


def score_instructions_loss(cfg, instructions, accelerator):
    for instruction in instructions:
        if instruction.score is not None: 
            if accelerator.is_main_process: log.info(f"Skipping already scored instruction: {instruction.name}")
            continue
        if accelerator.is_main_process: log.info(f"Scoring instruction: {instruction.name}")
        dataset = instruction.traces
        scores = []
        for model_config in cfg.proxy_models:
            model_name = os.path.join(cfg.working_dir, "warmed_up_models", model_config['name'].split('/')[-1])

            tokenizer = AutoTokenizer.from_pretrained(
                model_config["tokenizer"] if "tokenizer" in model_config else model_name,
                use_fast=True,
                trust_remote_code=True,
                padding_side="left",
            )
            if "llama" in model_name.lower():
                eot_token_id = 128009
                eos_token_id = 128001
                tokenizer.pad_token_id = 128004
                tokenizer.eos_token_id = eos_token_id
                tokenizer.add_eos_token = False
                eos_token = tokenizer.eos_token
            else:
                eos_token = tokenizer.eos_token
                bos_token = tokenizer.bos_token or ""
                special_tokens = {"pad_token": "[PAD]"}
                tokenizer.add_special_tokens(special_tokens)

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16
            )
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.add_eos_token = False
            model.resize_token_embeddings(len(tokenizer))

            def tokenize_rewritten_trace(examples):
                messages = [[
                    {"role": "system", "content": cfg.instruction_generation},
                    {"role": "user", "content": problem.strip()},
                    {"role": "assistant", "content": response.strip()}]
                    for problem, response in zip(examples["problem"], examples['rewrite_trace'])]
                tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
                labels = tokens.copy()
                return {"input_ids": tokens, "labels": labels}

            tokenized_dataset_rewritten = dataset.map(
                tokenize_rewritten_trace, batched=True, remove_columns=dataset.column_names)

            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
            dataloader_rewritten = DataLoader(tokenized_dataset_rewritten, batch_size=cfg.batch_size_for_scoring, collate_fn=data_collator)

            prepared_model, prepared_dataloader_rewritten = accelerator.prepare(model, dataloader_rewritten)

            with torch.no_grad():
                total_loss_per_process = 0.0
                num_samples = 0
                num_samples_per_process = 0
                iterator_rewritten = tqdm(
                    prepared_dataloader_rewritten, 
                    desc=f"Instruction of {instruction.name} ({model_name.split('/')[-1]})",
                    disable=not accelerator.is_main_process
                )

                for batch in iterator_rewritten:
                    outputs = prepared_model(**batch)
                    loss = outputs.loss
                    batch_size = batch["input_ids"].shape[0]

                    total_loss_per_process += loss * batch_size
                    num_samples += batch_size * accelerator.num_processes
                    num_samples_per_process += batch_size
            accelerator.wait_for_everyone()
            gathered_losses = accelerator.gather(total_loss_per_process)
            if accelerator.is_main_process:
                total_loss_for_model = gathered_losses.sum().item()
                final_loss_rewritten = total_loss_for_model / num_samples
                log.info(f"  [{model_name}] Avg Loss (rewritten): {final_loss_rewritten}")
                scores.append(final_loss_rewritten)
            model, dataloader_rewritten = accelerator.free_memory(prepared_model, prepared_dataloader_rewritten)
            del model, dataloader_rewritten, tokenizer, tokenized_dataset_rewritten
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
        avg_score = sum(scores) / len(scores) if scores else 0
        instruction.score = avg_score
        del dataset

    return instructions

def score_instructions_acc(cfg, instructions, accelerator, validation_dataset):
    checkpoint_path = os.path.join(
        cfg.working_dir,
        cfg.experiment_folder,
        "instructions_and_scores.yaml",
    )
    clean_metrics_by_model = {}

    for instruction in instructions:
        if instruction.score is not None: 
            if accelerator.is_main_process: log.info(f"Skipping already scored instruction: {instruction.name}")
            continue
        if accelerator.is_main_process: log.info(f"Scoring instruction: {instruction.name}")
        dataset = instruction.traces
        scores = []
        proxy_metrics = []
        for model_config in cfg.proxy_models:
            model_name = model_config["name"]
            tokenizer_name = model_config.get("tokenizer", model_name)
            model_short_name = model_name.split("/")[-1]
            clean_model_dir = os.path.join(
                cfg.working_dir,
                "clean_comparators",
                model_short_name,
            )
            final_model_dir = os.path.join(
                cfg.working_dir,
                instruction.name,
                "finetuned_model",
                model_short_name,
            )
            if model_name not in clean_metrics_by_model:
                if accelerator.is_main_process:
                    clean_metrics = evaluate_candidate_af_accuracy(
                        clean_model_dir,
                        model_name,
                        tokenizer_name,
                        validation_dataset,
                    )
                else:
                    clean_metrics = None
                clean_payload = [clean_metrics]
                broadcast_object_list(clean_payload, from_process=0)
                clean_metrics_by_model[model_name] = clean_payload[0]
            clean_metrics = clean_metrics_by_model[model_name]

            if _verified_model_exists(
                final_model_dir,
                cfg,
                expected_base_model=model_name,
                expected_tokenizer=tokenizer_name,
            ):
                if accelerator.is_main_process:
                    log.info(
                        f"{final_model_dir} already exists; "
                        "evaluating the saved proxy model."
                    )
                    candidate_metrics = evaluate_candidate_af_accuracy(
                        final_model_dir,
                        model_name,
                        tokenizer_name,
                        validation_dataset,
                    )
                    candidate_score, comparison_metrics = (
                        _proxy_comparison_metrics(
                            model_name,
                            clean_metrics,
                            candidate_metrics,
                        )
                    )
                    scores.append(candidate_score)
                    proxy_metrics.append(comparison_metrics)
                    log.info(
                        f"  [{model_name}] clean AF: "
                        f"{clean_metrics['af_accuracy']:.4f}, "
                        f"candidate AF: "
                        f"{candidate_metrics['af_accuracy']:.4f}, "
                        f"accuracy drop: {candidate_score:.4f}"
                    )

                accelerator.wait_for_everyone()
                continue

            tokenizer = _load_training_tokenizer(tokenizer_name)

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16
            )
            _align_model_special_tokens(model, tokenizer)

            def preprocess_function(examples):
                trace_colname = 'rewrite_trace'
                return _tokenize_completion_only_batch(
                    tokenizer=tokenizer,
                    instruction=cfg.instruction_generation,
                    problems=examples["problem"],
                    responses=examples[trace_colname],
                    model_name=model_name,
                )

            train_dataset = dataset.map(
                preprocess_function,
                batched=True,
                batch_size=16384,
                num_proc=int(os.environ.get("TRACE_NUM_PROC", "4")),
                remove_columns=list(dataset.column_names),
                desc="Preprocessing train dataset",
                load_from_cache_file=False
            )

            training_settings = _proxy_training_settings(
                cfg,
                accelerator,
                True,
            )
            peft_parameters = LoraConfig(
                r=training_settings["lora_r"],
                lora_alpha=training_settings["lora_alpha"],
                lora_dropout=training_settings["lora_dropout"],
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_parameters)
            if accelerator.is_main_process:
                model.print_trainable_parameters()
                log.info(
                    "LoRA hyperparams: "
                    f"rank={training_settings['lora_r']}, "
                    f"alpha={training_settings['lora_alpha']}, "
                    f"dropout={training_settings['lora_dropout']}"
                )

            num_gpus = training_settings["num_gpus"]
            per_device_batch_size = training_settings[
                "per_device_batch_size"
            ]
            gradient_accumulation_steps = training_settings[
                "gradient_accumulation_steps"
            ]
            epochs = training_settings["epochs"]
            learning_rate = training_settings["learning_rate"]

            if accelerator.is_main_process:
                log.info(
                    f"hyperparams: bs={per_device_batch_size}, grad_accum={gradient_accumulation_steps}, "
                    f"effective_bs={per_device_batch_size * gradient_accumulation_steps * num_gpus}, "
                    f"epochs={epochs}, lr={learning_rate}"
                )

            training_args = SFTConfig(
                num_train_epochs=float(epochs),
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=float(learning_rate),
                weight_decay=float(getattr(cfg, "proxy_weight_decay", 0.1)),
                warmup_ratio=float(getattr(cfg, "proxy_warmup_ratio", 0.1)),
                lr_scheduler_type="cosine",
                max_grad_norm=float(1.0),
                remove_unused_columns=False,
                label_names=["labels"],
                completion_only_loss=True,

                bf16=True,
                fp16=False,
                
                seed=cfg.seed,
                log_level="info",
                logging_steps=10,
                logging_strategy="steps",

                save_strategy="no",
                report_to="none",  # disable wandb and others

                # Performance
                gradient_checkpointing=False,  # disabled for speed
                gradient_checkpointing_kwargs={"use_reentrant": False},
                group_by_length=True,
            )
            
            # Trainer
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )
            # Training
            trainer.train()

            if accelerator.is_main_process:
                _save_adapter_only_artifact(
                    trainer,
                    final_model_dir,
                    cfg,
                    base_model=model_name,
                    tokenizer_name=tokenizer_name,
                )
                log.info(f"Finetuned {model_name}. Saved at {final_model_dir}")

            trainer.accelerator.free_memory()
            model, tokenizer = accelerator.free_memory(model, tokenizer)
            del model, tokenizer, train_dataset, trainer
            gc.collect()
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                candidate_metrics = evaluate_candidate_af_accuracy(
                    final_model_dir,
                    model_name,
                    tokenizer_name,
                    validation_dataset,
                )
                candidate_score, comparison_metrics = (
                    _proxy_comparison_metrics(
                        model_name,
                        clean_metrics,
                        candidate_metrics,
                    )
                )
                scores.append(candidate_score)
                proxy_metrics.append(comparison_metrics)
                log.info(
                    f"  [{model_name}] clean AF: "
                    f"{clean_metrics['af_accuracy']:.4f}, "
                    f"candidate AF: "
                    f"{candidate_metrics['af_accuracy']:.4f}, "
                    f"accuracy drop: {candidate_score:.4f}"
                )

            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if not scores:
                raise RuntimeError(
                    f"No proxy scores were produced for instruction: "
                    f"{instruction.name}"
                )

            instruction.score = sum(scores) / len(scores)
            instruction.proxy_metrics = proxy_metrics
            log.info(
                f"Instruction {instruction.name} average accuracy-drop "
                f"score: {instruction.score:.4f}"
            )

        score_payload = [
            instruction.score if accelerator.is_main_process else None
        ]
        broadcast_object_list(score_payload, from_process=0)
        instruction.score = score_payload[0]

        if accelerator.is_main_process:
            save_instruction_scores(instructions, checkpoint_path)
            log.info(
                f"Checkpointed completed instruction "
                f"{instruction.name} to {checkpoint_path}"
            )

        accelerator.wait_for_everyone()
        del dataset

    return instructions

def run_scoring(cfg):
    _validate_accuracy_protocol_config(cfg)
    accelerator = Accelerator()

    warmup_models(cfg, accelerator)

    candidate_path = os.path.join(
        cfg.working_dir,
        cfg.experiment_folder,
        "candidate_instructions.yaml",
    )
    scores_path = os.path.join(
        cfg.working_dir,
        cfg.experiment_folder,
        "instructions_and_scores.yaml",
    )

    if not cfg.rescore and os.path.exists(scores_path):
        instruction_path = scores_path
        if accelerator.is_main_process:
            log.info(
                f"Resuming instruction scores from {instruction_path}"
            )
    else:
        instruction_path = candidate_path
        if accelerator.is_main_process:
            log.info(
                f"Loading candidate instructions from {instruction_path}"
            )

    with open(instruction_path, "r") as f:
        instruction_data = yaml.safe_load(f)
    candidates_list = instruction_data['candidate_instructions']
    instructions = []
    for item in candidates_list:
        if accelerator.is_main_process: log.info(f"Loading traces for instruction: {item['name']}")
        dataset_path = os.path.join(cfg.working_dir, item['name'])

        # Create an Instruction object and load its data
        instruction = Instruction(item['name'], item['instruction'])
        instruction.traces = datasets.load_from_disk(dataset_path)
        if cfg.score_type == "acc":
            expected_train_size = int(
                cfg.dataset_size_used_for_optimize
            )
            if len(instruction.traces) != expected_train_size:
                raise ValueError(
                    f"Candidate dataset {dataset_path} contains "
                    f"{len(instruction.traces)} rows; expected exactly "
                    f"{expected_train_size}."
                )
        instruction.proxy_metrics = item.get("proxy_metrics", [])
        if cfg.rescore:
            instruction.score = None
        else:
            instruction.score = item.get('score')
        instructions.append(instruction)

    if cfg.score_type == "loss":
        instructions_with_scores = score_instructions_loss(
            cfg,
            instructions,
            accelerator,
        )

        if accelerator.is_main_process:
            log.info("Scoring done.")
            for instruction in instructions_with_scores:
                log.info(
                    f"Instruction: {instruction.name}, "
                    f"Score: {instruction.score}"
                )

            save_instruction_scores(
                instructions_with_scores,
                scores_path,
            )
            log.info(
                f"Saved instruction scores to {scores_path}"
            )

    elif cfg.score_type == "acc":
        with accelerator.main_process_first():
            validation_dataset = load_search_validation_dataset(cfg)

        if accelerator.is_main_process:
            log.info(
                f"Loaded {len(validation_dataset)} "
                "search-validation examples"
            )

        instructions_with_scores = score_instructions_acc(
            cfg,
            instructions,
            accelerator,
            validation_dataset,
        )

        if accelerator.is_main_process:
            log.info("Scoring done.")
            for instruction in instructions_with_scores:
                log.info(
                    f"Instruction: {instruction.name}, "
                    f"Score: {instruction.score}"
                )

            save_instruction_scores(
                instructions_with_scores,
                scores_path,
            )
            log.info(
                f"Saved instruction scores to {scores_path}"
            )

    else:
        raise ValueError(
            f"Unknown score type: {cfg.score_type}"
        )


if __name__ == "__main__":
    config_manager = ConfigManager()
    cfg = config_manager.get_config()

    run_scoring(cfg)
