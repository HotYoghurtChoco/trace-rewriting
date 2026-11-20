#!/bin/bash
# Submit one cohort. The full run is requested separately after the prior test.
set -euo pipefail
profile=${1:-prior-test}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export HONOUR_SCRATCH=${HONOUR_SCRATCH:-/srv/scratch/$USER/honour}
root=${MAXLOSS_RESULTS:-$HONOUR_SCRATCH/results/maxloss-sentence-v1}
if [[ "$profile" == check-judge ]]; then
    export JUDGE_ROW=${2:?Supply the one-based row number} JUDGE_ROUND=${3:?Supply the round number}
    export JUDGE_RECORDS="$root/prior-test/parts/000.jsonl"
    export RUN_DIR
    RUN_DIR=$(mktemp -d "$root/judge-check.XXXXXXXX")
    mkdir "$RUN_DIR/logs"
    ln -s "$repo" "$RUN_DIR/code"
    qsub -N ml-sent-judge -j oe -o "$RUN_DIR/logs/" \
        -l select=1:ncpus=8:ngpus=1:mem=128gb:gpu_model=H200 -l walltime=00:12:00 \
        -v HONOUR_SCRATCH,RUN_DIR,STAGE=check-judge,TASK_IDS=0,JUDGE_RECORDS,JUDGE_ROW,JUDGE_ROUND \
        "$repo/experiments/maxloss_sentence/job.pbs"
    echo "Result: $RUN_DIR/judgments.json"
    exit 0
fi
case "$profile" in prior-test|full) ;; *) echo "Usage: bash $0 {prior-test|full|check-judge ROW ROUND}" >&2; exit 2 ;; esac
mkdir -p "$root/$profile/logs"
export RUN_DIR
RUN_DIR=$(cd "$root/$profile" && pwd)
exec 9>"$RUN_DIR/.submit.lock"
flock -x 9
export SOURCE_COMMIT
SOURCE_COMMIT=$(git -C "$repo" rev-parse HEAD)
if ! git -C "$repo" cat-file -e "$SOURCE_COMMIT:experiments/maxloss_sentence/job.pbs" 2>/dev/null; then
    echo "Commit the applied patch before submitting so queued jobs use that version." >&2
    exit 1
fi
module load gcc/12.2.0 python/3.11.3
python="$HONOUR_SCRATCH/envs/trace-rewriting-official/bin/python"
cd "$repo"
export PYTHONPATH="$repo:$repo/src"
"$python" -m experiments.maxloss_sentence.workflow init "$RUN_DIR"
if [[ -s "$RUN_DIR/af.csv" ]]; then
    echo "Complete: $RUN_DIR/af.csv"
    exit 0
fi
# Pin the small source tree and the published Clean data once for all queued jobs.
if [[ ! -d "$RUN_DIR/code" ]]; then
    tmp=$(mktemp -d "$RUN_DIR/.code.XXXXXXXX")
    git archive "$SOURCE_COMMIT" -- src optimize experiments data/gsm8k/semantic | tar -x -C "$tmp"
    mv "$tmp" "$RUN_DIR/code"
fi
job="$RUN_DIR/code/experiments/maxloss_sentence/job.pbs"
base=(-j oe -o "$RUN_DIR/logs/")
cpu=(-l select=1:ncpus=2:mem=6gb -l walltime=00:10:00)
# Array indices address only unfinished tasks when resubmitting a failed run.
workflow=("$python" -m experiments.maxloss_sentence.workflow)
if [[ -s "$RUN_DIR/last-summary-job" ]]; then
    previous=$(cat "$RUN_DIR/last-summary-job")
    status=$(qstat -f "$previous" 2>/dev/null || true)
    if [[ "$status" =~ job_state[[:space:]]*=[[:space:]]*[BHQRTWSE] ]]; then
        echo "Already submitted. Summary job $previous, results $RUN_DIR"
        exit 0
    fi
fi
submit_array() {
    local stage=$1 name=$2 limit=$3 memory=$4 duration=$5 list=$6
    shift 6
    local -a items flags=()
    read -ra items <<< "$list"
    if (( ${#items[@]} > 1 )); then
        flags=(-J "0-$(( ${#items[@]} - 1 ))%$limit")
    fi
    local joined
    joined=$(IFS=:; echo "${items[*]}")
    qsub "${base[@]}" -N "$name" "${flags[@]}" "$@" \
        -l "select=1:ncpus=8:ngpus=1:mem=$memory:gpu_model=H200" -l "walltime=$duration" \
        -v "HONOUR_SCRATCH,RUN_DIR,STAGE=$stage,TASK_IDS=$joined" "$job"
}
deps=()
if [[ ! -f "$RUN_DIR/datasets/rewrite_summary.json" ]]; then
    pending=$("${workflow[@]}" pending-shards "$RUN_DIR")
    if [[ -n "$pending" ]]; then
        rewrite=$(submit_array rewrite ml-sent-rewrite "${SHARD_CONCURRENCY:-2}" 128gb "${REWRITE_WALLTIME:-01:00:00}" "$pending")
        deps=(-W "depend=afterok:$rewrite")
        echo "Rewriting    $rewrite"
    fi
    collect=$(qsub "${base[@]}" -N ml-sent-data "${cpu[@]}" "${deps[@]}" \
        -v HONOUR_SCRATCH,RUN_DIR,STAGE=collect "$job")
    deps=(-W "depend=afterok:$collect")
    echo "Assembly     $collect"
fi
pending=$("${workflow[@]}" pending-students "$RUN_DIR")
if [[ -n "$pending" ]]; then
    default_time=00:30:00
    if [[ "$profile" == full ]]; then default_time=04:00:00; fi
    train=$(submit_array student ml-sent-train "${STUDENT_CONCURRENCY:-3}" 56gb \
        "${STUDENT_WALLTIME:-$default_time}" "$pending" "${deps[@]}")
    deps=(-W "depend=afterok:$train")
    echo "Students     $train"
fi
report=$(qsub "${base[@]}" -N ml-sent-af "${cpu[@]}" "${deps[@]}" \
    -v HONOUR_SCRATCH,RUN_DIR,STAGE=summary "$job")
echo "$report" > "$RUN_DIR/last-summary-job"
echo "AF table     $report   $RUN_DIR/af.csv"
