"""Write synthetic features (exact on-disk layout of the real ones) for the smoke run.

    python make_synthetic.py OUT_DIR

The SNN gets a planted error signal in one layer3 channel; the ANN does not. The smoke run
should therefore report a positive SNN-minus-ANN gain difference and an L1-selected sparse
monitor that beats random channels. Nothing here is a result about real networks.
"""
import sys

from rmlib.data import load_cfg
from rmlib.synthetic import make_synthetic_features

if __name__ == "__main__":
    cfg = load_cfg(stage="smoke")
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_features"
    make_synthetic_features(out, cfg["run"]["corruptions"], cfg["run"]["severities"], n_test=600, n_train=800)
    print("synthetic features written to", out)
