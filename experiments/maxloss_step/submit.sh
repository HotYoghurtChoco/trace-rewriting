#!/bin/bash
# One shared rewrite corpus. Each full invocation schedules one explicit student seed.
set -euo pipefail
mode=${1:-first}
export STUDENT_SEED=${2:-888}
case "$mode:$STUDENT_SEED" in first:888|full:888|full:889|full:890) ;;
    *) echo "Usage: bash $0 first | full [888|889|890]" >&2; exit 2 ;; esac
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export HONOUR_SCRATCH=${HONOUR_SCRATCH:-/srv/scratch/$USER/honour}
export PAIRED_CLEAN=${PAIRED_CLEAN:-$HONOUR_SCRATCH/results/maxloss-sentence-v1/full}
export RUN_DIR=${MAXLOSS_STEP_DIR:-$HONOUR_SCRATCH/results/maxloss-step-v1/full}
export MAXLOSS_PACKAGE=experiments.maxloss_step
mkdir -p "$RUN_DIR"/{logs,jobs}
RUN_DIR=$(cd "$RUN_DIR" && pwd)
exec 9>"$RUN_DIR/.submit.lock"
flock -x 9
module load gcc/12.2.0 python/3.11.3
python="$HONOUR_SCRATCH/envs/trace-rewriting-official/bin/python"
if [[ ! -d "$RUN_DIR/code" ]]; then
    source_commit=$(git -C "$repo" rev-parse HEAD)
    if ! git -C "$repo" cat-file -e "$source_commit:experiments/maxloss_step/submit.sh" 2>/dev/null; then
        echo "Commit the step patch before submitting. The job runs the committed source." >&2
        exit 1
    fi
    tmp=$(mktemp -d "$RUN_DIR/.code.XXXXXXXX")
    git -C "$repo" archive "$source_commit" -- src optimize experiments data/gsm8k/semantic | tar -x -C "$tmp"
    echo "$source_commit" > "$tmp/source-commit"
    mv "$tmp" "$RUN_DIR/code"
fi
cd "$RUN_DIR/code"
export SOURCE_COMMIT
SOURCE_COMMIT=$(cat source-commit)
export PYTHONPATH="$PWD:$PWD/src"
workflow=("$python" -m "$MAXLOSS_PACKAGE.workflow")
"${workflow[@]}" init "$RUN_DIR"
job="$PWD/experiments/maxloss_sentence/job.pbs"
common=(-j oe -o "$RUN_DIR/logs/")
exports=HONOUR_SCRATCH,RUN_DIR,MAXLOSS_PACKAGE,STUDENT_SEED
gpu=(-l select=1:ncpus=8:ngpus=1:mem=128gb:gpu_model=H200)
cpu=(-l select=1:ncpus=2:mem=6gb)
active() {
    local file="$RUN_DIR/jobs/$1" id status
    [[ -s "$file" ]] || return 1
    id=$(cat "$file")
    status=$(qstat -f "$id" 2>/dev/null) || return 1
    [[ "$status" =~ job_state[[:space:]]*=[[:space:]]*[BHQRTWSE] ]] || return 1
    echo "$id"
}
submit() {
    local key=$1 id
    shift
    if ! id=$(active "$key"); then
        id=$(qsub "${common[@]}" "$@" "$job") || return
        if [[ -z "$id" || "$id" == *[[:space:]]* ]]; then
            echo "qsub returned an invalid job ID for $key" >&2
            return 1
        fi
        printf '%s\n' "$id" > "$RUN_DIR/jobs/$key" || return
    fi
    echo "$id"
}
first_part="$RUN_DIR/parts/000.jsonl"
if [[ "$mode" == first ]]; then
    if [[ -f "$first_part" ]]; then
        cat "$RUN_DIR/parts/000.timing.json"
        echo "Suggested rewrite walltime: $("${workflow[@]}" walltime "$RUN_DIR")"
    elif existing=$(active rewrite); then
        echo "The full rewrite array already includes the first shard: $existing"
    else
        first=$(submit first -N ml-step-first "${gpu[@]}" -l "walltime=${REWRITE_WALLTIME:-00:30:00}" \
            -v "$exports,STAGE=rewrite,TASK_IDS=0")
        echo "First 100 formal rows: $first   $RUN_DIR/parts/000.timing.json"
    fi
    exit 0
fi
table="$RUN_DIR/af_seed$STUDENT_SEED.csv"
if [[ -s "$table" ]]; then echo "Complete: $table"; exit 0; fi
if existing=$(active "summary-$STUDENT_SEED"); then echo "Already submitted: $existing   $table"; exit 0; fi
assembled="$RUN_DIR/datasets/rewrite_summary.json"
if [[ "$STUDENT_SEED" != 888 && ! -f "$assembled" ]]; then
    echo "Additional seeds use the assembled seed-888 rewrite corpus. Finish its assembly first." >&2
    exit 1
fi
dependency=()
if [[ ! -f "$assembled" ]]; then
    first_job=
    if [[ ! -f "$first_part" ]] && first_job=$(active first); then
        dependency=(-W "depend=afterok:$first_job")
    fi
    pending=$("${workflow[@]}" pending-shards "$RUN_DIR")
    ids=()
    for i in $pending; do
        if [[ "$i" != 0 || -z "$first_job" ]]; then ids+=("$i"); fi
    done
    if (( ${#ids[@]} )); then
        array=()
        if (( ${#ids[@]} > 1 )); then array=(-J "0-$((${#ids[@]} - 1))%${SHARD_CONCURRENCY:-4}"); fi
        joined=$(IFS=:; echo "${ids[*]}")
        duration=${REWRITE_WALLTIME:-$("${workflow[@]}" walltime "$RUN_DIR")}
        rewrite=$(submit rewrite -N ml-step-rewrite "${gpu[@]}" "${array[@]}" "${dependency[@]}" \
            -l "walltime=$duration" -v "$exports,STAGE=rewrite,TASK_IDS=$joined")
        echo "Rewriting: $rewrite   pending=${#ids[@]}   walltime=$duration"
        dependency=(-W "depend=afterok:$rewrite")
    fi
    collect=$(submit collect -N ml-step-data "${cpu[@]}" "${dependency[@]}" -l walltime=00:05:00 \
        -v "$exports,STAGE=collect")
    echo "Assembly: $collect"
    dependency=(-W "depend=afterok:$collect")
fi
students=()
models=(qwen1p7b llama3b qwen4b)
times=(00:20:00 00:25:00 00:30:00)
for m in "${!models[@]}"; do
    model=${models[m]}
    [[ ! -f "$RUN_DIR/students/$model-$STUDENT_SEED/af.json" ]] || continue
    index=$((3 * m + STUDENT_SEED - 888))
    duration=${STUDENT_WALLTIME:-${times[m]}}
    id=$(submit "$model-$STUDENT_SEED" -N "ml-step-$model" \
        -l select=1:ncpus=8:ngpus=1:mem=56gb:gpu_model=H200 -l "walltime=$duration" "${dependency[@]}" \
        -v "$exports,STAGE=student,TASK_IDS=$index")
    echo "Student $model / $STUDENT_SEED: $id   walltime=$duration"
    students+=("$id")
done
if (( ${#students[@]} )); then
    joined=$(IFS=:; echo "${students[*]}")
    dependency=(-W "depend=afterok:$joined")
fi
report=$(submit "summary-$STUDENT_SEED" -N ml-step-af "${cpu[@]}" "${dependency[@]}" -l walltime=00:02:00 \
    -v "$exports,STAGE=summary")
echo "AF table: $report   $table"
