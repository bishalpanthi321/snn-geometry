"""Config, paths, model loading and image selection for the temporal-spike experiment.

Reuses the original repo's model, LIF neuron and data code (src/) without
modifying it. Everything written goes under <out_root>/{cache,results,logs}/<stage>.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

EXP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GPUNormalizer, corrupted_path, fit_eval_split, load_cifar10  # noqa: E402
from src.models import SEWResNet18  # noqa: E402
from src.utils import dump_json, env_report, get_logger  # noqa: E402

__all__ = ["EXP_ROOT", "PROJECT_ROOT", "GPUNormalizer", "dump_json", "env_report", "get_logger"]


# ------------------------------------------------------------------ config

def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--stage", choices=["smoke", "pilot", "full"], default="pilot")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--corrupted-dir", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--random-init", action="store_true",
                    help="random weights (smoke test only; no checkpoint needed)")
    return ap


def _pick(flag, env, file_value, default):
    for v in (flag, os.environ.get(env), file_value):
        if v:
            return Path(v)
    return Path(default)


def load_cfg(args=None, stage=None, config=None):
    """Config with resolved paths. `args` is an argparse namespace from add_common_args."""
    a = args or argparse.Namespace(stage=stage or "pilot", config=config, checkpoint=None,
                                   data_root=None, corrupted_dir=None, out_root=None,
                                   random_init=False)
    with open(a.config or EXP_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    out_root = _pick(a.out_root, "TS_OUT_ROOT", p.get("out_root"), EXP_ROOT)
    st = dict(cfg["stages"][a.stage])
    if st["corruptions"] == "all":
        st["corruptions"] = list(cfg["corruptions"])
    cfg["stage"] = a.stage
    cfg["run"] = st
    cfg["random_init"] = bool(getattr(a, "random_init", False))
    cfg["resolved"] = {
        "checkpoint": str(_pick(a.checkpoint, "TS_CHECKPOINT", p.get("checkpoint"),
                                PROJECT_ROOT / "checkpoints" / "snn" / "final.pt")),
        "data_root": str(_pick(a.data_root, "TS_DATA_ROOT", p.get("data_root"), PROJECT_ROOT / "data")),
        "corrupted_dir": str(_pick(a.corrupted_dir, "TS_CORRUPTED_DIR", p.get("corrupted_dir"),
                                   PROJECT_ROOT / "data" / "corrupted")),
        "cache": str(out_root / "cache" / a.stage),
        "results": str(out_root / "results" / a.stage),
        "logs": str(out_root / "logs" / a.stage),
    }
    # The original experiment must never be an output location.
    for k in ("cache", "results", "logs"):
        r = Path(cfg["resolved"][k]).resolve()
        for forbidden in ("checkpoints", "features", "results", "logs"):
            if r == (PROJECT_ROOT / forbidden).resolve():
                raise ValueError(f"refusing to write into the original experiment: {r}")
    return cfg


def conditions(cfg, severities=None):
    sev = severities or cfg["run"]["severities"]
    return [("clean", 0)] + [(c, s) for c in cfg["run"]["corruptions"] for s in sev]


def cond_name(corr, sev):
    return "clean" if corr == "clean" else f"{corr}_s{sev}"


# ------------------------------------------------------------------ runtime

def set_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def snn_hparams():
    """Architecture/neuron settings from the original config, so the model is identical."""
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)["snn"]


def randomize_bn(model, seed=0):
    """Random BN statistics so an untrained network actually fires (smoke/tests only)."""
    g = torch.Generator().manual_seed(seed)
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            n = mod.num_features
            mod.running_mean.copy_(torch.rand(n, generator=g) * 0.4 - 0.2)
            mod.running_var.copy_(torch.rand(n, generator=g) * 0.4 + 0.2)
            mod.weight.data.copy_(torch.rand(n, generator=g) * 0.8 + 0.8)
            mod.bias.data.copy_(torch.rand(n, generator=g) * 0.5)
    return model


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_snn(cfg, T, device):
    """The trained SEW-ResNet-18 at T timesteps (or random weights for the smoke test).
    The checkpoint is only read, never written."""
    s = snn_hparams()
    torch.manual_seed(cfg["seed"])
    model = SEWResNet18(T=T, tau=s["tau"], v_threshold=s["v_threshold"], v_reset=s["v_reset"],
                        detach_reset=s["detach_reset"], surrogate_alpha=s["surrogate_alpha"],
                        impl=s["impl"])
    if cfg["random_init"]:
        randomize_bn(model, cfg["seed"])
        info = {"checkpoint": "random-init (smoke test)", "epoch": None, "sha256": None}
    else:
        path = Path(cfg["resolved"]["checkpoint"])
        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {path}. Set --checkpoint or TS_CHECKPOINT. "
                "This experiment never trains a model.")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        info = {"checkpoint": str(path), "epoch": ck.get("epoch"), "sha256": file_sha256(path)}
    info.update({"T": T, "lif_impl": model.sn1.impl, "snn_hparams": s})
    return model.to(device).eval(), info


# ------------------------------------------------------------------ data

def _np_load(p):
    if not Path(p).exists():
        raise FileNotFoundError(f"{p} missing (set --corrupted-dir / TS_CORRUPTED_DIR)")
    return np.load(p, allow_pickle=False)


def load_test_set(cfg):
    x, y = load_cifar10(cfg["resolved"]["data_root"], train=False)
    return x, y


def condition_images(cfg, corr, sev, clean_x=None):
    if corr == "clean":
        return clean_x if clean_x is not None else load_test_set(cfg)[0]
    return _np_load(corrupted_path(cfg["resolved"]["corrupted_dir"], corr, sev))


def load_train_set(cfg):
    return load_cifar10(cfg["resolved"]["data_root"], train=True)


def select_indices(cfg, n_test=10000, n_train=50000):
    """Image subsets. fit and eval come from the geometry experiment's 50/50 test split,
    so they are disjoint; A images are a subset of eval; ref images are training images."""
    st = cfg["run"]
    fit_idx, eval_idx = fit_eval_split(n_test, cfg["split_seed"])
    rng = np.random.default_rng(cfg["seed"])
    fit_sel = np.sort(rng.choice(fit_idx, min(st["n_fit"], len(fit_idx)), replace=False))
    eval_sel = np.sort(rng.choice(eval_idx, min(st["n_eval"], len(eval_idx)), replace=False))
    a_sel = np.sort(rng.choice(eval_sel, min(st["n_A"], len(eval_sel)), replace=False))
    ref_sel = np.sort(rng.choice(n_train, min(st["n_ref"], n_train), replace=False))
    assert len(np.intersect1d(fit_sel, eval_sel)) == 0
    return {"fit": fit_sel, "eval": eval_sel, "A": a_sel, "ref": ref_sel}


def batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0


def write_meta(path, extra):
    meta = {"env": env_report(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta.update(extra)
    dump_json(meta, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)
