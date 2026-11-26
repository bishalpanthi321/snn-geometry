"""Figures from results/*.csv (CPU only).

    python make_figures.py

Each corruption keeps the same colour and marker in every plot. Null baselines
are grey (random subspace) or dotted (clean-PCA). Severity 3 is the default
slice; if it was not run (smoke test), the highest available severity is used.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.utils import load_config  # noqa: E402

# Validated categorical palette (fixed order = fixed corruption identity).
COLORS = {"gaussian_noise": "#2a78d6", "shot_noise": "#eb6834", "impulse_noise": "#1baf7a",
          "gaussian_blur": "#eda100", "contrast": "#e87ba4", "brightness": "#008300"}
MARKERS = {"gaussian_noise": "o", "shot_noise": "s", "impulse_noise": "^",
           "gaussian_blur": "D", "contrast": "v", "brightness": "P"}
NULL_GREY = "#8a8a85"
LAYERS = ["layer1", "layer2", "layer3", "penultimate"]
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#f0efec", "#b52f2f"])

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "lines.linewidth": 2, "lines.markersize": 6, "font.size": 9, "legend.fontsize": 8})


def main_sev(df):
    sevs = sorted(s for s in df.severity.unique() if s > 0)
    return 3 if 3 in sevs else max(sevs)


def sel(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_energy(geo, out):
    for (model, layer), d in geo.groupby(["model", "layer"]):
        s = main_sev(d)
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for c in [c for c in COLORS if c in d.corruption.unique()]:
            x = sel(d, corruption=c, severity=s)
            h = sel(x, metric="heldout_energy").sort_values("k")
            n = sel(x, metric="null_cleanpca_energy").sort_values("k")
            ax.plot(h.k, h.value, color=COLORS[c], marker=MARKERS[c], label=c)
            ax.plot(n.k, n.value, color=COLORS[c], ls=":", lw=1.3)
        r = sel(d, metric="null_random_energy", severity=s).groupby("k").value.mean()
        ax.plot(r.index, r.values, color=NULL_GREY, ls="--", label="null: random subspace")
        ax.plot([], [], color="black", ls=":", lw=1.3, label="null: clean-PCA (dotted, per corruption)")
        ax.set_xscale("log")
        ax.set_xlabel("subspace dimension k (fit split)")
        ax.set_ylabel("held-out displacement energy captured")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{model} / {layer}: displacement energy vs k (severity {s})")
        ax.legend(loc="lower right")
        save(fig, out / f"energy_vs_k_{model}_{layer}.png")


def fig_angles(geo, out):
    for model, d in geo.groupby("model"):
        fig, axes = plt.subplots(2, len(LAYERS), figsize=(3.2 * len(LAYERS), 6), sharex=True)
        for j, layer in enumerate(LAYERS):
            dl = sel(d, layer=layer)
            for c in [c for c in COLORS if c in dl.corruption.unique()]:
                a = sel(dl, corruption=c, metric="angle_median").sort_values("severity")
                e = sel(dl, corruption=c, metric="energy_frac_W_over_chance").sort_values("severity")
                axes[0, j].plot(a.severity, a.value, color=COLORS[c], marker=MARKERS[c], label=c)
                axes[1, j].plot(e.severity, e.value, color=COLORS[c], marker=MARKERS[c], label=c)
            r = sel(dl, metric="null_random_angle_median").groupby("severity").value.mean()
            axes[0, j].plot(r.index, r.values, color=NULL_GREY, ls="--", label="null: random k-dim subspace")
            axes[0, j].axhline(60, color="black", lw=0.8, ls=":")
            axes[1, j].axhline(1, color=NULL_GREY, ls="--", label="chance (9/D)")
            axes[1, j].axhline(2, color="black", lw=0.8, ls=":", label="H2 threshold (2x chance)")
            axes[0, j].set_ylim(0, 92)
            axes[0, j].set_title(layer)
            axes[1, j].set_xlabel("severity")
        axes[0, 0].set_ylabel("median principal angle U_c vs W (deg)")
        axes[1, 0].set_ylabel("energy fraction in W / chance")
        axes[0, -1].legend(loc="lower right")
        axes[1, -1].legend(loc="upper right")
        fig.suptitle(f"{model}: orthogonality of corruption subspace to LDA subspace")
        save(fig, out / f"angles_vs_severity_{model}.png")


def fig_defense(dfn, out):
    for model, d in dfn.groupby("model"):
        s = main_sev(d)
        corrs = [c for c in COLORS if c in d.corruption.unique()]
        ncol = min(3, len(corrs))
        nrow = int(np.ceil(len(corrs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False, sharey=True)
        clean = sel(d, corruption="clean", metric="acc_clean").value.iloc[0]
        clean_u = sel(d, corruption="clean", metric="acc_clean_after_union").sort_values("k")
        for ax, c in zip(axes.flat, corrs):
            x = sel(d, corruption=c, severity=s)
            before = sel(x, metric="acc_corrupted").value.iloc[0]
            for metric, ls, lab in [("acc_corrupted_after_union", "-", "corrupted, union U"),
                                    ("acc_corrupted_after_oracle", "--", "corrupted, oracle U")]:
                v = sel(x, metric=metric).sort_values("k")
                ax.plot(v.k, v.value, color=COLORS[c], ls=ls, marker=MARKERS[c], label=lab)
            rnd = sel(x, metric="acc_corrupted_after_random").sort_values("k")
            ax.plot(rnd.k, rnd.value, color=NULL_GREY, ls="--", label="corrupted, random U")
            ax.axhline(before, color=COLORS[c], lw=1, ls=":", label="corrupted, no projection")
            ax.plot(clean_u.k, clean_u.value, color="black", marker=".", label="clean, union U")
            ax.axhline(clean, color="black", lw=1, ls=":", label="clean, no projection")
            ax.set_xscale("log")
            ax.set_title(f"{c} (severity {s})")
            ax.set_xlabel("k (directions removed)")
        for ax in axes[:, 0]:
            ax.set_ylabel("accuracy (eval split)")
        for ax in axes.flat[len(corrs):]:
            ax.axis("off")
        fig.suptitle(f"{model}: projection defense at the penultimate layer")
        # Legend below the panels so it never covers data; line styles are shared.
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
                   frameon=False, title="line style (colour = corruption)")
        fig.savefig(out / f"defense_vs_k_{model}.png", bbox_inches="tight")
        plt.close(fig)


def fig_transfer(tr, out):
    if tr.empty:
        return
    for model, d in tr.groupby("model"):
        s = main_sev(d)
        p = sel(d, severity=s).pivot_table(index="source_corruption", columns="corruption", values="gain") * 100
        order = [c for c in COLORS if c in p.index]
        p = p.loc[order, [c for c in COLORS if c in p.columns]]
        lim = max(1.0, np.nanmax(np.abs(p.values)))
        fig, ax = plt.subplots(figsize=(1.0 * p.shape[1] + 2.5, 0.8 * p.shape[0] + 1.8))
        im = ax.imshow(p.values, cmap=DIVERGING, vmin=-lim, vmax=lim)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                ax.text(j, i, f"{p.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(p.shape[1]), p.columns, rotation=35, ha="right")
        ax.set_yticks(range(p.shape[0]), p.index)
        ax.set_xlabel("tested on corruption")
        ax.set_ylabel("U fit on corruption (all severities)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="accuracy gain (points)")
        ax.set_title(f"{model}: transfer, k={int(d.k.iloc[0])}, severity {s}")
        save(fig, out / f"transfer_{model}.png")


def fig_temporal(tmp, out):
    if tmp is None:
        return
    metrics = [("rel_magnitude", "||d|| / ||h_clean||"), ("heldout_energy", "held-out energy, k=10"),
               ("energy_frac_W", "energy fraction in W"), ("acc_cumrate", "accuracy (cumulative rate)")]
    for model, d in tmp.groupby("model"):
        d = sel(d, layer="penultimate")
        s = main_sev(d)
        fig, axes = plt.subplots(2, len(metrics), figsize=(3.3 * len(metrics), 6), sharex=True)
        for i, kind in enumerate(["spikes", "cumrate"]):
            for j, (metric, lab) in enumerate(metrics):
                ax = axes[i, j]
                k_ = "cumrate" if metric == "acc_cumrate" else kind
                if metric == "acc_cumrate" and kind == "spikes":
                    ax.axis("off")
                    continue
                for c in [c for c in COLORS if c in d.corruption.unique()]:
                    v = sel(d, corruption=c, severity=s, metric=metric, kind=k_).sort_values("t")
                    ax.plot(v.t, v.value, color=COLORS[c], marker=MARKERS[c], label=c)
                if metric == "acc_cumrate":
                    v = sel(d, corruption="clean", metric=metric).sort_values("t")
                    ax.plot(v.t, v.value, color="black", marker=".", label="clean")
                if metric == "energy_frac_W":
                    # D after dropping constant dims, recovered from the stored k/D null.
                    kd = sel(d, metric="null_random_expected").iloc[0]
                    D = int(round(kd.k / kd.value))
                    ax.axhline(9 / D, color=NULL_GREY, ls="--", label=f"chance 9/D (D={D})")
                ax.set_ylabel(f"{lab}\n({'per-timestep spikes' if kind == 'spikes' else 'rate up to t'})")
                if i == 1:
                    ax.set_xlabel("timestep t")
        axes[1, -1].legend(loc="lower right")
        fig.suptitle(f"{model}: penultimate layer over time (severity {s})")
        save(fig, out / f"temporal_{model}.png")


def fig_training(cfg, out):
    logs = Path(cfg["paths"]["logs_dir"])
    files = [(m, logs / f"train_{m}.csv") for m in ["ann", "snn"] if (logs / f"train_{m}.csv").exists()]
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for (m, f), col in zip(files, ["#2a78d6", "#eb6834"]):
        d = pd.read_csv(f)
        axes[0].plot(d.epoch + 1, d.train_loss, color=col, label=m)
        axes[1].plot(d.epoch + 1, d.test_acc, color=col, label=f"{m} test")
        axes[1].plot(d.epoch + 1, d.train_acc, color=col, ls=":", label=f"{m} train (augmented)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].legend()
    save(fig, out / "training_curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke, out_root=args.out_root)
    res = Path(cfg["paths"]["results_dir"])
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    geo = pd.read_csv(res / "geometry.csv")
    dfn = pd.read_csv(res / "defense.csv")
    tr = pd.read_csv(res / "transfer.csv") if (res / "transfer.csv").stat().st_size > 1 else pd.DataFrame()
    tmp = pd.read_csv(res / "temporal.csv") if (res / "temporal.csv").exists() else None
    fig_energy(geo, out)
    fig_angles(geo, out)
    fig_defense(dfn, out)
    fig_transfer(tr, out)
    fig_temporal(tmp, out)
    fig_training(cfg, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
