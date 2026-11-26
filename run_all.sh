#!/usr/bin/env bash
# Full pipeline on a Linux GPU box (Kaggle/Colab/cluster). Each stage can be rerun alone.
# Training resumes automatically if a last.pt exists.
set -euo pipefail
PY=${PY:-python}
MB=${MICRO_BATCH:-}          # e.g. MICRO_BATCH=32 on small GPUs; empty = full batch 128
MAXH=${MAX_HOURS:-}          # e.g. MAX_HOURS=11.3 on Kaggle (12 h session limit)
extra=()
[ -n "$MB" ] && extra+=(--micro-batch "$MB")
[ -n "$MAXH" ] && extra+=(--max-hours "$MAXH")

$PY -m pytest -q
for m in ann snn; do
  resume=()
  [ -f "${SNNGEO_OUT_ROOT:-.}/checkpoints/$m/last.pt" ] && resume=(--resume)
  [ -f "${SNNGEO_OUT_ROOT:-.}/checkpoints/$m/final.pt" ] || $PY train.py --model $m "${extra[@]}" "${resume[@]}"
done
# Stop here if a session limit interrupted training.
[ -f "${SNNGEO_OUT_ROOT:-.}/checkpoints/ann/final.pt" ] && [ -f "${SNNGEO_OUT_ROOT:-.}/checkpoints/snn/final.pt" ] || { echo "training not finished; rerun"; exit 0; }
$PY generate_corruptions.py
$PY extract_features.py --model ann
$PY extract_features.py --model snn
$PY extract_features.py --model snn --T 8
$PY analyze.py
$PY make_figures.py
