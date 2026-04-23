#!/bin/bash
#SBATCH --job-name=selfrag-eval
#SBATCH --account=xxxx
#SBATCH --partition=gpu
#SBATCH --output=logs/selfrag-eval-%j.out
#SBATCH --error=logs/selfrag-eval-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 MODEL_PATH EVAL_FILE METRIC [OUTPUT_FILE] [WORLD_SIZE]"
  echo "  MODEL_PATH  : merged model directory"
  echo "  EVAL_FILE   : short-form eval jsonl"
  echo "  METRIC      : match | accuracy"
  echo "  OUTPUT_FILE : optional output json path"
  echo "  WORLD_SIZE  : optional tensor parallel size (default: 1)"
  exit 1
fi

MODEL_PATH="$1"
EVAL_FILE="$2"
METRIC="$3"
OUTPUT_FILE="${4:-}"
WORLD_SIZE="${5:-1}"

if [[ "$METRIC" != "match" && "$METRIC" != "accuracy" ]]; then
  echo "Unsupported metric: $METRIC"
  echo "run_short_form.py supports only: match, accuracy"
  exit 1
fi

mkdir -p logs outputs

REPO_ROOT="/scratch/user/varshagoalla/self-rag"
SCRIPT_PATH="$REPO_ROOT/retrieval_lm/run_short_form.py"

if [[ ! -f "$EVAL_FILE" ]]; then
  echo "Eval file not found: $EVAL_FILE"
  exit 1
fi

if [[ ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Model path not found or not a model directory: $MODEL_PATH"
  exit 1
fi

BASENAME="$(basename "$EVAL_FILE")"
TASK=""
NDOCS=10
MAX_NEW_TOKENS=100

case "$BASENAME" in
  arc_challenge_processed*.jsonl)
    TASK="arc_c"
    NDOCS=5
    MAX_NEW_TOKENS=50
    ;;
  health_claims_processed*.jsonl)
    TASK="fever"
    NDOCS=5
    MAX_NEW_TOKENS=50
    ;;
  popqa_longtail_w_gs*.jsonl)
    TASK=""
    NDOCS=10
    MAX_NEW_TOKENS=100
    ;;
  triviaqa_test_w_gs*.jsonl)
    TASK=""
    NDOCS=10
    MAX_NEW_TOKENS=100
    ;;
  *)
    echo "Warning: unknown eval file pattern: $BASENAME"
    echo "Using defaults: task='', ndocs=10, max_new_tokens=100"
    ;;
esac

if [[ -z "$OUTPUT_FILE" ]]; then
  MODEL_NAME="$(basename "$MODEL_PATH")"
  DATA_NAME="${BASENAME%.jsonl}"
  OUTPUT_FILE="$REPO_ROOT/retrieval_lm/eval_outputs/${MODEL_NAME}_${DATA_NAME}_${METRIC}.json"
fi

echo "MODEL_PATH=$MODEL_PATH"
echo "EVAL_FILE=$EVAL_FILE"
echo "METRIC=$METRIC"
echo "TASK=${TASK:-<none>}"
echo "NDOCS=$NDOCS"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "OUTPUT_FILE=$OUTPUT_FILE"
echo "WORLD_SIZE=$WORLD_SIZE"

export BASHRCSOURCED="${BASHRCSOURCED:-}"
set +u
source /scratch/user/varshagoalla/.conda/etc/profile.d/conda.sh
set -u
conda activate selfrag_dpo

cd "$REPO_ROOT/retrieval_lm"

CMD=(
  python "$SCRIPT_PATH"
  --model_name "$MODEL_PATH"
  --input_file "$EVAL_FILE"
  --output_file "$OUTPUT_FILE"
  --mode adaptive_retrieval
  --max_new_tokens "$MAX_NEW_TOKENS"
  --threshold 0.2
  --metric "$METRIC"
  --ndocs "$NDOCS"
  --use_groundness
  --use_utility
  --use_seqscore
  --dtype half
  --device cuda
  --world_size "$WORLD_SIZE"
)

if [[ -n "$TASK" ]]; then
  CMD+=(--task "$TASK")
fi

echo "Running: ${CMD[*]}"
srun "${CMD[@]}"
