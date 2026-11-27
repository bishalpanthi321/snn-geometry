"""Timing-information capacity of every recording site (CPU).

    python analyze_capacity.py --stage pilot

For each T, site and condition (clean training reference + every test condition):
  p(count k), fraction of neurons with more than one possible order (0 < k < T),
  capacity = sum_k p(k) log2 C(T,k) bits/neuron (binary sites), and the measured
  H(order | count) with the Miller-Madow correction, pooled over channels and per channel
  (the per-channel version is available for every binary site, and for integer sites on
  the reference set). Integer sites (SEW stage outputs, values 0..3) report the number of
  orderings of each train's multiset (reachable by permutation) and the number of trains
  with the same sum (reachable by resampling).
Writes results/<stage>/capacity.csv and timestep_rates.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tslib.common import add_common_args, load_cfg
from tslib.stats import capacity_metrics


def site_rows(npz, T, condition):
    sites = sorted({k.split("/")[0] for k in npz.files})
    rows, rate_rows = [], []
    for s in sites:
        base = int(npz[f"{s}/base"])
        m = capacity_metrics(npz[f"{s}/hist"], base, T)
        n = float(npz[f"{s}/n_neurons"])
        rate = npz[f"{s}/rate_t"] / (n / T) / (base - 1)      # mean value per neuron per t / vmax
        rows.append({"T": T, "condition": condition, "site": s,
                     "kind": "binary" if base == 2 else f"integer(0..{base - 1})",
                     "n_images": int(npz[f"{s}/n_images"]), **{k: (json.dumps(v) if isinstance(v, list) else v)
                                                              for k, v in m.items()}})
        for t, r in enumerate(rate):
            rate_rows.append({"T": T, "condition": condition, "site": s, "t": t + 1,
                              "mean_activity": float(r)})
    return rows, rate_rows


def main():
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_cfg(args)
    cache, res = Path(cfg["resolved"]["cache"]), Path(cfg["resolved"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    rows, rate_rows = [], []
    for T in cfg["run"]["T"]:
        root = cache / f"T{T}"
        if not root.exists():
            print(f"no cache for T={T}, skipping")
            continue
        r, rr = site_rows(np.load(root / "ref" / "ref_stats.npz"), T, "train_clean_ref")
        rows += r
        rate_rows += rr
        for cdir in sorted((root / "cond").iterdir()):
            if (cdir / "capacity.npz").exists():
                r, rr = site_rows(np.load(cdir / "capacity.npz"), T, cdir.name)
                rows += r
                rate_rows += rr
    df = pd.DataFrame(rows)
    df.to_csv(res / "capacity.csv", index=False)
    pd.DataFrame(rate_rows).to_csv(res / "timestep_rates.csv", index=False)
    show = df[df.condition.isin(["train_clean_ref", "clean"])]
    cols = ["T", "condition", "site", "frac_multi_order", "capacity_bits", "H_order_given_count_bits",
            "H_over_capacity"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(show[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
