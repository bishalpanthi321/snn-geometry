"""Whole pipeline on a random-weight model, small subsets (slow; opt-in).

    TS_RUN_SLOW=1 python -m pytest experiments/temporal_spikes/tests/test_smoke_end_to_end.py
Needs CIFAR-10 and the corrupted test sets on disk (default locations or TS_* env vars).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP = Path(__file__).resolve().parent.parent
PROJECT = EXP.parent.parent

pytestmark = pytest.mark.skipif(os.environ.get("TS_RUN_SLOW") != "1", reason="set TS_RUN_SLOW=1")


def run(args, out):
    r = subprocess.run([sys.executable, *args, "--stage", "smoke", "--out-root", str(out)],
                       cwd=EXP, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_pipeline_end_to_end(tmp_path):
    if not (PROJECT / "data" / "corrupted").exists() and not os.environ.get("TS_CORRUPTED_DIR"):
        pytest.skip("corrupted test sets not available")
    run(["extract.py", "--T", "4", "--random-init"], tmp_path)
    run(["extract.py", "--T", "8", "--random-init"], tmp_path)
    run(["run_interventions.py", "--T", "4", "--random-init"], tmp_path)
    # resumability: a second run finds everything done and recomputes nothing
    out = run(["extract.py", "--T", "4", "--random-init"], tmp_path)
    assert "skipping" in out
    for s in ["analyze_capacity.py", "analyze_interventions.py", "analyze_detection.py", "make_figures.py"]:
        run([s], tmp_path)
    res = tmp_path / "results" / "smoke"
    iv = pd.read_csv(res / "interventions.csv")
    cp = iv[iv.family != "count_change"]
    assert (cp.count_err_max == 0).all()
    assert (iv[iv.null_site].max_abs_dlogit <= 1e-4).all()
    ident = iv[iv.family == "identity"]
    assert (ident.flip == 0).all()
    cap = pd.read_csv(res / "capacity.csv")
    assert set(cap["T"]) == {4, 8}
    assert (cap.capacity_bits >= 0).all()
    assert len(list((res / "figures").glob("*.png"))) >= 8
    assert np.isfinite(pd.read_csv(res / "detection_loco.csv").auroc.dropna()).all()
