#!/usr/bin/env bash
# Pilot of the temporal-spike experiment. Run from anywhere; uses the paths in config.yaml
# or the TS_* env vars (TS_CHECKPOINT, TS_DATA_ROOT, TS_CORRUPTED_DIR, TS_OUT_ROOT).
# With two GPUs the T=8 extraction runs on GPU 1 while GPU 0 does T=4 and Experiment A.
set -euo pipefail
cd "$(dirname "$0")"
STAGE="${1:-pilot}"
python -m pytest tests -q
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$NGPU" -ge 2 ]; then
  CUDA_VISIBLE_DEVICES=1 python extract.py --stage "$STAGE" --T 8 > extract_T8.out 2>&1 &
  PID8=$!
  CUDA_VISIBLE_DEVICES=0 python extract.py --stage "$STAGE" --T 4
  CUDA_VISIBLE_DEVICES=0 python run_interventions.py --stage "$STAGE" --T 4
  wait $PID8 || { echo "T=8 extraction failed:"; tail -30 extract_T8.out; exit 1; }
else
  python extract.py --stage "$STAGE" --T 4
  python run_interventions.py --stage "$STAGE" --T 4
  python extract.py --stage "$STAGE" --T 8
fi
python analyze_capacity.py --stage "$STAGE"
python analyze_interventions.py --stage "$STAGE"
python analyze_detection.py --stage "$STAGE"
python make_figures.py --stage "$STAGE"
