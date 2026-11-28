#!/usr/bin/env bash
# Rate-monitor pilot (CPU). Needs RM_FEATURES (and RM_CHECKPOINT_DIR for the cost benchmark).
set -euo pipefail
cd "$(dirname "$0")"
STAGE="${1:-pilot}"
python -m pytest tests -q -W ignore
python -W ignore run_monitor.py --stage "$STAGE"
if [ -n "${RM_CHECKPOINT_DIR:-}" ]; then
  python benchmark_overhead.py --stage "$STAGE"
else
  echo "RM_CHECKPOINT_DIR not set: skipping the runtime benchmark"
fi
python make_figures.py --stage "$STAGE"
