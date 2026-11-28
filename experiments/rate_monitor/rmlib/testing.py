"""Shared test helpers (kept out of conftest.py so other test suites can import them)."""
import argparse

from .data import load_cfg
from .synthetic import make_synthetic_features

CORR = ["gaussian_noise", "contrast", "brightness"]
SEV = [1, 3]


def synthetic_cfg(tmp_path, **kw):
    root = make_synthetic_features(tmp_path / "features", CORR, SEV, **kw)
    cfg = load_cfg(argparse.Namespace(stage="smoke", config=None, features_root=str(root),
                                      checkpoint_dir=None, data_root=None, out_root=str(tmp_path / "out")))
    return cfg, root
