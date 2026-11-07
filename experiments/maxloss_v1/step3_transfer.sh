#!/bin/bash
# On KDM: bash experiments/maxloss_v1/step3_transfer.sh fetch-qwen
# On a Katana login node: use submit-llama or submit-qwen.
set -euo pipefail

HONOUR_SCRATCH=${HONOUR_SCRATCH:-/srv/scratch/$USER/honour}
models="$HONOUR_SCRATCH/models"
qwen_revision=906bfd4b4dc7f14ee4320094d8b41684abff8539
tokenizer_revision=1cfa9a7208912126459214e8b04321603b3df60c
qwen_model="$models/qwen3-4b-base-${qwen_revision:0:12}"
qwen_tokenizer="$models/qwen3-4b-tokenizer-${tokenizer_revision:0:12}"

case "${1:-}" in
    fetch-qwen)
        unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
        hf="$HONOUR_SCRATCH/envs/trace-rewriting-official/bin/hf"
        # Match the model/tokenizer pairing used by the 1.7B experiment.
        "$hf" download Qwen/Qwen3-4B config.json tokenizer.json tokenizer_config.json merges.txt vocab.json \
            --revision "$tokenizer_revision" --local-dir "$qwen_tokenizer"
        "$hf" download Qwen/Qwen3-4B-Base config.json generation_config.json model.safetensors.index.json \
            model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors \
            --revision "$qwen_revision" --local-dir "$qwen_model"
        exit 0
        ;;
    submit-llama)
        job_name=ml32-llama
        seed_range=889-890
        walltime=00:25:00
        host_memory=46gb
        STUDENT_MODEL="$models/llama-3.2-3b-base"
        STUDENT_TOKENIZER="$models/llama-3.2-3b-instruct-tokenizer"
        RESULT_ROOT="$HONOUR_SCRATCH/results/maxloss-llama3.2-3b-v1"
        ;;
    submit-qwen)
        job_name=ml33-qwen4b
        seed_range=888-890
        walltime=00:35:00
        host_memory=56gb
        STUDENT_MODEL="$qwen_model"
        STUDENT_TOKENIZER="$qwen_tokenizer"
        RESULT_ROOT="$HONOUR_SCRATCH/results/maxloss-qwen3-4b-v1"
        ;;
    *)
        echo "Usage: bash $0 {fetch-qwen|submit-llama|submit-qwen}" >&2
        exit 2
        ;;
esac

formal=maxloss-qwen3-1.7b-v1/formal_9679942
GENERATION_DIR="$HONOUR_SCRATCH/results/$formal"
if [[ ! -f "$GENERATION_DIR/OPTIMIZATION_COMPLETE" ]]; then
    GENERATION_DIR="$HOME/honour/results/trace-rewriting/$formal"
fi

# Check the frozen input and installed shards before requesting a GPU.
python3 - "$STUDENT_MODEL" "$STUDENT_TOKENIZER" "$GENERATION_DIR" <<'PY'
import json
import sys
from pathlib import Path

model, tokenizer, generation = map(Path, sys.argv[1:])
index = model / "model.safetensors.index.json"
if not index.is_file():
    sys.exit(f"Missing model index: {index}")
shards = set(json.loads(index.read_text())["weight_map"].values())
if not shards:
    sys.exit(f"No weight shards listed in {index}")
required = [model / name for name in sorted(shards)]
required += [model / "config.json", tokenizer / "tokenizer.json", tokenizer / "tokenizer_config.json",
             generation / "summary.json", generation / "run_config.json", generation / "optimized_dataset/state.json"]
for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        sys.exit(f"Missing or empty input: {path}")
if not (generation / "OPTIMIZATION_COMPLETE").is_file():
    sys.exit(f"Missing completed formal run: {generation}")
PY

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo"
SOURCE_COMMIT=$(git rev-parse HEAD)
export HONOUR_SCRATCH SOURCE_COMMIT STUDENT_MODEL STUDENT_TOKENIZER GENERATION_DIR RESULT_ROOT
echo "Submitting $job_name, seeds $seed_range, commit $SOURCE_COMMIT" >&2
exec qsub -N "$job_name" -J "$seed_range" \
    -l "select=1:ncpus=8:ngpus=1:mem=$host_memory:gpu_model=H200" -l "walltime=$walltime" \
    -v HONOUR_SCRATCH,SOURCE_COMMIT,STUDENT_MODEL,STUDENT_TOKENIZER,GENERATION_DIR,RESULT_ROOT \
    experiments/maxloss_v1/pbs/maxloss_transfer_seeds.pbs
