"""Measured CPU cost of attaching a monitor to the real networks.

    python benchmark_overhead.py --stage pilot [--checkpoint-dir DIR] [--random-init]

The monitor's cost is far below the run-to-run jitter of a forward pass, so it is NOT measured
as a difference of two forward timings. Instead (median over repetitions, after warm-up):
  base_forward  the plain forward pass
  full_B / sparse_k / full_D   the monitor step alone, applied to layer outputs captured from
                a real forward pass: spatial pooling of the (selected) channels, time-averaging
                for the SNN (except D), confidence features, standardisation, linear score.
                D also computes the per-timestep confidence trajectory.
overhead_pct_of_forward = monitor time / forward time.
Memory: pooled feature bytes per image (fp32) and monitor parameter bytes.
Which k channels are kept does not change the cost, so the first k channels are used.
Writes results/<stage>/overhead.csv.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from rmlib.data import PROJECT_ROOT, add_common_args, load_cfg
from src.models import build_model

LAYERS = ["layer1", "layer2", "layer3"]


class Capture:
    """Keeps the raw outputs of layer1..3 and of the pool (penultimate) for one forward pass."""

    def __init__(self, model):
        self.model, self.raw, self.handles = model, {}, []

    def __enter__(self):
        for n in LAYERS:
            self.handles.append(getattr(self.model, n).register_forward_hook(
                lambda m, i, o, n=n: self.raw.__setitem__(n, o)))
        self.handles.append(self.model.pool.register_forward_hook(
            lambda m, i, o: self.raw.__setitem__("penultimate", o)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def pool_features(raw, is_snn, select=None, per_t=False):
    """What a deployed monitor computes from the layer outputs: spatially pooled (selected)
    channels, time-averaged for the SNN unless per_t. Returns [N, d] and, for per_t, the
    per-timestep penultimate tensor."""
    feats, pen_t = [], None
    for name in LAYERS + ["penultimate"]:
        x = raw[name]
        sel = None if select is None else select.get(name)
        if sel is not None:
            x = x.index_select(-3 if name != "penultimate" else -1, sel)
        if name != "penultimate":
            x = x.mean(dim=(-2, -1))
        if is_snn:
            if per_t:
                if name == "penultimate":
                    pen_t = x
                x = x.transpose(0, 1).reshape(x.shape[1], -1)
            else:
                x = x.mean(0)
        feats.append(x)
    return torch.cat(feats, 1), pen_t


def split_select(k, dims):
    """First k channels overall, spread over layers in order (cost depends only on k)."""
    out, left = {}, k
    for name, d in dims.items():
        take = min(left, d)
        out[name] = torch.arange(take)
        left -= take
    return out


def time_it(fn, reps):
    fn()                                                             # warm-up
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--reps", type=int, default=10, help="forward-pass repetitions")
    ap.add_argument("--monitor-reps", type=int, default=200, help="monitor-only repetitions")
    args = ap.parse_args()
    cfg = load_cfg(args)
    res = Path(cfg["resolved"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    with open(PROJECT_ROOT / "config.yaml") as f:
        root_cfg = yaml.safe_load(f)
    torch.manual_seed(0)
    x = torch.randn(args.batch, 3, 32, 32)
    dims = {"layer1": 64, "layer2": 128, "layer3": 256, "penultimate": 512}
    n_ch = sum(dims.values())
    ks = sorted({max(1, int(round(f * n_ch))) for f in cfg["sparse"]["fractions"]})
    rows = []
    for tag in cfg["models"]:
        model = build_model(tag, root_cfg).eval()
        if not args.random_init:
            ck = Path(cfg["resolved"]["checkpoint_dir"]) / tag / "final.pt"
            model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
        is_snn = tag == "snn"
        rng = np.random.default_rng(0)

        w_cache = {}

        def monitor(raw, sel, per_t, logits):
            z, pen_t = pool_features(raw, is_snn, sel, per_t)
            p = logits.softmax(-1)
            conf = torch.stack([p.amax(-1), -(p * p.clamp_min(1e-12).log()).sum(-1),
                                logits.topk(2, -1).values.diff(dim=-1).squeeze(-1).abs()], 1)
            z = torch.cat([conf, z], 1)
            if per_t:                                                   # confidence trajectory
                lt = model.fc(pen_t)
                run = lt.cumsum(0) / torch.arange(1, lt.shape[0] + 1).view(-1, 1, 1)
                z = torch.cat([z, run.softmax(-1).amax(-1).T], 1)
            z = z.numpy().astype(np.float64)
            d = z.shape[1]
            if d not in w_cache:
                w_cache[d] = (np.zeros(d), np.ones(d), rng.normal(size=d))
            mu, sd, w = w_cache[d]
            return ((z - mu) / sd) @ w, d

        with torch.no_grad():
            base = time_it(lambda: model(x), args.reps)
            with Capture(model) as cap:
                logits = model(x)
            raw = cap.raw
            rows.append({"model": tag, "variant": "base_forward", "n_features": 0, "sec_per_batch": base,
                         "ms_per_image": 1e3 * base / args.batch, "monitor_ms_per_image": 0.0,
                         "overhead_pct_of_forward": 0.0, "feature_bytes_per_image": 0, "monitor_param_bytes": 0})
            variants = [("full_B", None, False)] + [(f"sparse_k{k}", split_select(k, dims), False) for k in ks]
            if is_snn:
                variants.append(("full_D", None, True))
            for name, sel, per_t in variants:
                _, d = monitor(raw, sel, per_t, logits)
                t = time_it(lambda: monitor(raw, sel, per_t, logits), args.monitor_reps)
                rows.append({"model": tag, "variant": name, "n_features": d, "sec_per_batch": t,
                             "ms_per_image": 1e3 * t / args.batch,
                             "monitor_ms_per_image": 1e3 * t / args.batch,
                             "overhead_pct_of_forward": 100 * t / base,
                             "feature_bytes_per_image": 4 * d,                  # pooled features, fp32
                             "monitor_param_bytes": 8 * 3 * d})                 # mean, scale, weight (fp64)
        print(tag, "done")
    df = pd.DataFrame(rows)
    df["torch_threads"] = torch.get_num_threads()
    df["batch"] = args.batch
    df.to_csv(res / "overhead.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
