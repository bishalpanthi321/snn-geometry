"""Config, paths, and read-only access to the geometry experiment's saved features.

What the saved features are (from extract_features.py of the original experiment):
  ANN  {layer}.npy        [N, D] spatially averaged ReLU stage output
  SNN  {layer}.npy        [N, D] SEW stage output (a sum of spike trains, values 0..3 before
                          pooling) averaged over space AND over the T timesteps
       {layer}_spikes.npy [N, T, D] float16, the same stage output averaged over space at
                          each timestep. These are NOT binary spike trains.
  both logits.npy [N, 10], preds.npy, labels.npy, indices.npy (test-set image index)
  SNN  logits = fc(penultimate time-average); fc.npz holds the final layer.
Layout: <root>/<tag>/<split>/<corruption>/<severity>/<name>.npy, split in {train, fit, eval};
train holds clean training images only, and only time-averaged layers + logits.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

EXP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_ROOT.parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "experiments" / "temporal_spikes")):
    if p not in sys.path:
        sys.path.insert(0, p)


def add_common_args(ap):
    ap.add_argument("--stage", choices=["smoke", "pilot", "full"], default="pilot")
    ap.add_argument("--config", default=None)
    ap.add_argument("--features-root", default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-root", default=None)
    return ap


def _pick(flag, env, val, default):
    for v in (flag, os.environ.get(env), val):
        if v:
            return Path(v)
    return Path(default)


def load_cfg(args=None, **kw):
    a = args or argparse.Namespace(stage=kw.get("stage", "pilot"), config=kw.get("config"),
                                   features_root=kw.get("features_root"), checkpoint_dir=None,
                                   data_root=None, out_root=kw.get("out_root"))
    with open(a.config or EXP_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    st = dict(cfg["stages"][a.stage])
    if st["corruptions"] == "all":
        st["corruptions"] = list(cfg["corruptions"])
    cfg["stage"], cfg["run"] = a.stage, st
    out = _pick(a.out_root, "RM_OUT_ROOT", p.get("out_root"), EXP_ROOT)
    cfg["resolved"] = {
        "features_root": str(_pick(a.features_root, "RM_FEATURES", p.get("features_root"),
                                   PROJECT_ROOT / "features")),
        "checkpoint_dir": str(_pick(getattr(a, "checkpoint_dir", None), "RM_CHECKPOINT_DIR",
                                    p.get("checkpoint_dir"), PROJECT_ROOT / "checkpoints")),
        "data_root": str(_pick(getattr(a, "data_root", None), "RM_DATA_ROOT", p.get("data_root"),
                               PROJECT_ROOT / "data")),
        "results": str(out / "results" / a.stage),
        "logs": str(out / "logs" / a.stage),
    }
    for k in ("results", "logs"):
        r = Path(cfg["resolved"][k]).resolve()
        for forbidden in ("features", "results", "logs", "checkpoints"):
            if r == (PROJECT_ROOT / forbidden).resolve():
                raise ValueError(f"refusing to write into the original experiment: {r}")
    return cfg


def conditions(cfg):
    return [("clean", 0)] + [(c, s) for c in cfg["run"]["corruptions"] for s in cfg["run"]["severities"]]


class FeatureStore:
    """Read-only, memory-mapped access to one model's saved features."""

    def __init__(self, root, tag):
        self.root = Path(root) / tag
        self.tag = tag

    def path(self, split, corr, sev, name):
        return self.root / split / corr / str(sev) / f"{name}.npy"

    def has(self, split, corr, sev, name):
        return self.path(split, corr, sev, name).exists()

    def load(self, split, corr, sev, name, rows=None):
        a = np.load(self.path(split, corr, sev, name), mmap_mode="r", allow_pickle=False)
        return np.asarray(a if rows is None else a[rows])

    def fc(self):
        z = np.load(self.root / "fc.npz")
        return z["weight"], z["bias"]


def row_positions(n_rows, n_take, seed):
    """Positions (into a condition file of one split) to use; identical for every model and
    condition because all files of a split store the same images in the same order."""
    n_take = min(n_take, n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, n_take, replace=False))
