#!/usr/bin/env bash
# Train 8 seeds on Train_Set_clean_augmented.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$SCRIPT_DIR}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT/Train_Set_clean_augmented}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/cpa_output_clean_aug}"
LOG_DIR="${LOG_DIR:-$ROOT/train_logs_clean_aug}"
TRAIN_PY="${TRAIN_PY:-$ROOT/baseline/train.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"

BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-5}"
MAX_LENGTH="${MAX_LENGTH:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.15}"
PATIENCE="${PATIENCE:-4}"
VAL_RATIO="${VAL_RATIO:-0.1}"

SEEDS=(${SEEDS:-42 100 2025 12345 99 7 777 31337})
read -r -a GPUS <<< "$GPU_IDS"

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPU_IDS is empty. Example: GPU_IDS='0 1 2 3' bash step2_train_8seeds.sh" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "ROOT=$ROOT"
echo "TRAIN_DIR=$TRAIN_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "MODEL_NAME=$MODEL_NAME"
echo "SEEDS=${SEEDS[*]}"
echo "GPU_IDS=${GPUS[*]}"

"$PYTHON_BIN" "$ROOT/step1_prepare_0525.py"

run_one() {
  local gpu_id="$1"
  local seed="$2"
  local log="$LOG_DIR/train_seed${seed}.log"
  local seed_out="$OUTPUT_DIR/seed_${seed}"
  mkdir -p "$seed_out"
  echo "[GPU${gpu_id}] seed=${seed} -> ${log}"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  "$PYTHON_BIN" "$TRAIN_PY" \
    --train_dir "$TRAIN_DIR" \
    --output_dir "$seed_out" \
    --shortcut_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --epoch "$EPOCHS" \
    --lr "$LR" \
    --max_length "$MAX_LENGTH" \
    --random_seed "$seed" \
    --num_workers "$NUM_WORKERS" \
    --use_amp \
    --warmup_ratio "$WARMUP_RATIO" \
    --patience "$PATIENCE" \
    --val_ratio "$VAL_RATIO" \
    --device gpu \
    > "$log" 2>&1 &
  echo "  PID=$!"
}

idx=0
total=${#SEEDS[@]}
wave=1
while [[ $idx -lt $total ]]; do
  echo "============ WAVE $wave $(date '+%Y-%m-%d %H:%M:%S') ============"
  for gpu in "${GPUS[@]}"; do
    if [[ $idx -ge $total ]]; then
      break
    fi
    run_one "$gpu" "${SEEDS[$idx]}"
    idx=$((idx + 1))
  done
  wait
  echo "============ WAVE $wave done $(date '+%Y-%m-%d %H:%M:%S') ============"
  wave=$((wave + 1))
done

echo ""
echo "All seeds trained."
echo "Models under: $OUTPUT_DIR/seed_*/cpa_*/"
ls -d "$OUTPUT_DIR"/seed_*/cpa_*/ 2>/dev/null || true
