import copy
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None, smoke=False, out_root=None):
    """Load config.yaml and resolve all paths.

    Smoke mode shrinks the run and writes everything (except the CIFAR-10
    download) under <out_root>/smoke_run so it can never overwrite real results.
    """
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    out_root = out_root or os.environ.get("SNNGEO_OUT_ROOT") or str(PROJECT_ROOT)
    out_root = Path(out_root)
    if smoke:
        s = cfg["smoke"]
        cfg["train"]["epochs"] = s["epochs"]
        cfg["corruptions"]["names"] = s["corruptions"]
        cfg["corruptions"]["severities"] = s["severities"]
        out_root = out_root / s["out_subdir"]
    cfg["smoke_mode"] = bool(smoke)
    cfg["n_test"] = cfg["smoke"]["n_test"] if smoke else None
    cfg["n_train_lda"] = cfg["smoke"]["n_train_lda"] if smoke else None
    cfg["n_train"] = cfg["smoke"]["n_train"] if smoke else None

    resolved = {}
    for key, rel in cfg["paths"].items():
        if key == "data_root":
            # The raw CIFAR-10 download is shared between smoke and full runs.
            base = Path(os.environ.get("SNNGEO_DATA_ROOT", PROJECT_ROOT / rel))
            resolved[key] = base
        else:
            resolved[key] = out_root / rel
    cfg["paths"] = {k: str(v) for k, v in resolved.items()}
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def env_report():
    """Versions that matter for reproducing the run."""
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
    }
    try:
        import spikingjelly  # noqa: F401
        from importlib.metadata import version
        info["spikingjelly"] = version("spikingjelly")
    except Exception as e:  # recorded, not fatal: the fallback LIF may be in use
        info["spikingjelly"] = f"unavailable ({type(e).__name__})"
    return info
