"""Figures from results/<stage>/*.csv (CPU).

    python make_figures.py --stage pilot

fig1_gains            per held-out corruption: gain of B and C over A, SNN vs ANN, 95% CIs
fig2_gain_difference  SNN-minus-ANN difference of gains, per corruption and pooled (forest plot)
fig3_sparse           pooled AUROC vs number of channels: L1, univariate filter, random
fig4_overhead         measured monitor cost as % of a forward pass (if overhead.csv exists)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rmlib.data import add_common_args, load_cfg  # noqa: E402

MODEL_COLOR = {"snn": "#257de0", "ann": "#f47230"}
METHOD_COLOR = {"l1": "#00b894", "univariate": "#6a4ce2", "random": "#9c9c9c"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999999", "xtick.color": "#666666", "ytick.color": "#666666",
                     "axes.grid": True, "grid.color": "#eeeeee", "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def fig_gains(res, out, cfg):
    g = pd.read_csv(res / "gains.csv")
    scopes = list(cfg["run"]["corruptions"]) + ["pooled"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharey="row", constrained_layout=True)
    for r, gain in enumerate(["B-A", "C-A"]):
        for c, strat in enumerate([False, True]):
            ax = axes[r, c]
            d = g[(g.gain == gain) & (g.margin_stratified == strat)]
            for i, t in enumerate(cfg["models"]):
                dd = d[d.model == t].set_index("scope").reindex(scopes)
                x = np.arange(len(scopes)) + (i - 0.5) * 0.36
                ax.bar(x, dd.estimate, 0.34, color=MODEL_COLOR[t], edgecolor="white", label=t.upper())
                ax.errorbar(x, dd.estimate, yerr=[dd.estimate - dd.lo, dd.hi - dd.estimate],
                            fmt="none", color="#191919", lw=1, capsize=2)
            ax.axhline(0, color="#191919", lw=0.8)
            ax.set_xticks(range(len(scopes)), scopes, rotation=25, ha="right")
            ax.set_title(f"gain {gain} ({'margin-stratified' if strat else 'plain'} AUROC)", fontsize=9)
            if c == 0:
                ax.set_ylabel("AUROC gain")
    axes[0, 0].legend()
    fig.suptitle("Error-detection gain from internal activations, held-out corruption (95% paired bootstrap CI)")
    save(fig, out, "fig1_gains")


def fig_diff(res, out, cfg):
    d = pd.read_csv(res / "gain_diff.csv")
    scopes = list(cfg["run"]["corruptions"]) + ["pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, constrained_layout=True)
    for ax, strat in zip(axes, [False, True]):
        for j, (gain, col) in enumerate([("B-A", "#257de0"), ("C-A", "#00b894")]):
            dd = d[(d.gain == gain) & (d.margin_stratified == strat)].set_index("scope").reindex(scopes)
            y = np.arange(len(scopes)) + (j - 0.5) * 0.3
            ax.errorbar(dd.estimate, y, xerr=[dd.estimate - dd.lo, dd.hi - dd.estimate], fmt="o", ms=5,
                        color=col, capsize=2, label=f"gain {gain}")
        ax.axvline(0, color="#191919", lw=0.8)
        ax.set_yticks(range(len(scopes)), scopes)
        ax.set_xlabel("SNN gain minus ANN gain (AUROC)")
        ax.set_title("margin-stratified" if strat else "plain", fontsize=9)
    axes[0].legend(loc="lower left")
    fig.suptitle("Is the monitoring gain larger for the SNN? (paired over the same images; "
                 "each model scored on its own errors)")
    save(fig, out, "fig2_gain_difference")


def fig_sparse(res, out, cfg):
    s = pd.read_csv(res / "sparse.csv")
    fig, axes = plt.subplots(1, len(cfg["models"]), figsize=(5.2 * len(cfg["models"]), 3.4),
                             sharey=True, constrained_layout=True)
    for ax, t in zip(np.atleast_1d(axes), cfg["models"]):
        d = s[(s.model == t) & ~s.margin_stratified]
        for meth in ("l1", "univariate", "random"):
            dd = d[d.method == meth].sort_values("k")
            ax.plot(dd.k, dd.auroc, marker="o", ms=4, lw=1.8, color=METHOD_COLOR[meth], label=meth)
            ax.fill_between(dd.k, dd.lo, dd.hi, color=METHOD_COLOR[meth], alpha=0.12, lw=0)
        ax.axhline(d.auroc_A.iloc[0], color="#191919", ls=":", lw=1, label="A: confidence only")
        ax.axhline(d.auroc_B.iloc[0], color="#191919", ls="--", lw=1, label="B: all 960 channels")
        ax.set_xscale("log")
        ax.set_xlabel("channels kept (of 960)")
        ax.set_title(t.upper(), fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("pooled AUROC (held-out corruptions)")
    np.atleast_1d(axes)[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Sparse monitors: channels chosen on training rows only")
    save(fig, out, "fig3_sparse")


def fig_overhead(res, out, cfg):
    o = pd.read_csv(res / "overhead.csv")
    o = o[o.variant != "base_forward"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    variants = list(dict.fromkeys(o.variant))
    for i, t in enumerate(cfg["models"]):
        dd = o[o.model == t].set_index("variant").reindex(variants)
        ax.bar(np.arange(len(variants)) + (i - 0.5) * 0.38, dd.overhead_pct_of_forward, 0.36,
               color=MODEL_COLOR[t], edgecolor="white", label=t.upper())
    ax.set_xticks(range(len(variants)), variants, rotation=25, ha="right")
    ax.set_ylabel("monitor time, % of one forward pass")
    ax.legend()
    ax.set_title("Measured CPU cost of the monitor step", fontsize=9)
    save(fig, out, "fig4_overhead")


def main():
    ap = add_common_args(argparse.ArgumentParser())
    cfg = load_cfg(ap.parse_args())
    res = Path(cfg["resolved"]["results"])
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_gains, fig_diff, fig_sparse, fig_overhead):
        try:
            f(res, out, cfg)
        except FileNotFoundError as e:
            print("skipped (missing input):", e)


if __name__ == "__main__":
    main()
