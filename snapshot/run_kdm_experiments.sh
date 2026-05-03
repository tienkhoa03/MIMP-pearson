#!/usr/bin/env bash
# Snapshot batch experiments for WiFi KDM
# Fixed window length: 6 (minutes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

DATASET="KDM"
WINDOW=6
BASES=("SAGE++DAC" "SAGE++DAMC")
EVAL_RATIOS=(0.1 0.3 0.5)
DELTAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

EPOCHS=200
NUM_ITER=5
K=10
STREAM=1

PREFIX_NO_PEARSON="no_pearson"
PREFIX_PEARSON="pearson"

TOTAL=$(( ${#BASES[@]} * ${#EVAL_RATIOS[@]} + ${#BASES[@]} * ${#EVAL_RATIOS[@]} * ${#DELTAS[@]} ))
CURRENT=0
START_TS=$(date +%s)

echo "========================================"
echo "Snapshot KDM Experiments"
echo "========================================"

for base in "${BASES[@]}"; do
  for eval_ratio in "${EVAL_RATIOS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL] dataset=$DATASET base=$base eval=$eval_ratio pearson=false"

    "$PYTHON_BIN" MPIN-plus.py \
      --dataset "$DATASET" \
      --window "$WINDOW" \
      --k "$K" \
      --epochs "$EPOCHS" \
      --prefix "$PREFIX_NO_PEARSON" \
      --num_of_iter "$NUM_ITER" \
      --base "$base" \
      --eval_ratio "$eval_ratio" \
      --stream "$STREAM" \
      --use_pearson false
  done
done

for base in "${BASES[@]}"; do
  for eval_ratio in "${EVAL_RATIOS[@]}"; do
    for delta in "${DELTAS[@]}"; do
      CURRENT=$((CURRENT + 1))
      echo "[$CURRENT/$TOTAL] dataset=$DATASET base=$base eval=$eval_ratio pearson=true delta=$delta"

      "$PYTHON_BIN" MPIN-plus.py \
        --dataset "$DATASET" \
        --window "$WINDOW" \
        --k "$K" \
        --epochs "$EPOCHS" \
        --prefix "$PREFIX_PEARSON" \
        --num_of_iter "$NUM_ITER" \
        --base "$base" \
        --eval_ratio "$eval_ratio" \
        --stream "$STREAM" \
        --use_pearson true \
        --delta "$delta"
    done
  done
done

END_TS=$(date +%s)
ELAPSED_MIN=$(awk "BEGIN { printf \"%.2f\", ($END_TS - $START_TS)/60 }")
echo "Completed $TOTAL experiments in ${ELAPSED_MIN} minutes."
