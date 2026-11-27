"""Figures from the CSVs / caches of one stage (CPU).

    python make_figures.py --stage pilot

Every figure is built only from files written by the analysis scripts; a figure whose
inputs are missing is skipped with a message (e.g. T=8 interventions in the pilot).
Colors: one fixed categorical slot per intervention family / model (never cycled).
"""
import argparse
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from tslib import detection as dt  # noqa: E402
from tslib.common import add_common_args, load_cfg  # noqa: E402

SLOTS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
FAMILIES = ["same_count_swap", "shared_perm", "clean_resample", "uniform_perm", "circular_shift",
            "even_spread", "count_change", "identity"]
FAM_COLOR = {f: SLOTS[i] for i, f in enumerate(FAMILIES[:-1])} | {"identity": "#9c9c9c"}
FAM_LABEL = {"identity": "identity", "same_count_swap": "same-count swap (pop.-preserving)",
             "shared_perm": "shared permutation", "clean_resample": "clean resample (indep.)",
             "uniform_perm": "uniform permutation", "circular_shift": "circular shift",
             "even_spread": "even spread (rate only)", "count_change": "count change (+ control)"}
MODEL_COLOR = {"M0_conf": SLOTS[0], "M1_conf_rate_act": SLOTS[1], "M2_M1_plus_timing": SLOTS[2],
               "best_baseline": "#666666"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#257de0", "#e6e6e6", "#ec4647"])
TEXT2 = "#666666"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999999", "axes.labelcolor": "#191919",
                     "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True,
                     "grid.color": "#eeeeee", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# 1 -------------------------------------------------------------------------
def fig_rasters(cfg, cache, out, T=4, site="layer4.0.sn1"):
    cdir = cache / f"T{T}" / "cond"
    corrs = [c for c in ["gaussian_noise", "gaussian_blur", "contrast"] if c in cfg["run"]["corruptions"]]
    sev = max(cfg["run"]["severities"])
    clean = np.load(cdir / "clean" / "raster.npz")
    pairs = [(c, np.load(cdir / f"{c}_s{sev}" / "raster.npz")) for c in corrs
             if (cdir / f"{c}_s{sev}" / "raster.npz").exists()]
    raw = clean[f"raw/{site}"]                                         # [n_img, T, C, H, W]
    rng = np.random.default_rng(0)
    n_img = min(2, raw.shape[0])
    fig, axes = plt.subplots(n_img, 1 + len(pairs), figsize=(2.4 * (1 + len(pairs)), 3.3 * n_img),
                             squeeze=False, constrained_layout=True)
    for i in range(n_img):
        flat_c = raw[i].reshape(raw.shape[1], -1)
        active = np.where(flat_c.sum(0) > 0)[0]
        pick = np.sort(rng.choice(active, min(60, len(active)), replace=False)) if len(active) else []
        panels = [("clean", flat_c)] + [(f"{c} s{sev}", d[f"raw/{site}"][i].reshape(raw.shape[1], -1))
                                        for c, d in pairs]
        for j, (title, fl) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(fl[:, pick].T, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks(range(raw.shape[1]), [str(t + 1) for t in range(raw.shape[1])])
            ax.grid(False)
            ax.set_title(f"{title}\n{int(fl.sum())} spikes total", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"image {int(clean['indices'][i])}\n60 neurons (active in clean)")
            ax.set_yticks([])
            ax.set_xlabel("timestep")
    fig.suptitle(f"Spike rasters at {site} (binary LIF output), same neurons across conditions, T={T}")
    save(fig, out, "fig1_rasters")


# 2 -------------------------------------------------------------------------
def fig_timestep(cfg, res, out):
    df = pd.read_csv(res / "timestep_rates.csv")
    sites = list(cfg["detector_sites"]) + list(cfg["integer_sites"])
    sev = max(cfg["run"]["severities"])
    Ts = sorted(df["T"].unique())
    fig, axes = plt.subplots(len(Ts), len(sites), figsize=(2.6 * len(sites), 2.4 * len(Ts)), squeeze=False)
    conds = ["clean"] + [f"{c}_s{sev}" for c in cfg["run"]["corruptions"]]
    colors = {"clean": "#191919"} | {f"{c}_s{sev}": SLOTS[i] for i, c in enumerate(cfg["run"]["corruptions"])}
    for r, T in enumerate(Ts):
        for j, s in enumerate(sites):
            ax = axes[r, j]
            for c in conds:
                d = df[(df["T"] == T) & (df.site == s) & (df.condition == c)]
                if len(d):
                    ax.plot(d.t, d.mean_activity, marker="o", ms=3, lw=2 if c == "clean" else 1.2,
                            color=colors[c], label=c)
            ax.set_title(f"{s} (T={T})", fontsize=8)
            ax.set_xlabel("timestep")
            if j == 0:
                ax.set_ylabel("mean activity / vmax")
    axes[0, -1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Population activity at each timestep (clean vs highest severity)")
    save(fig, out, "fig2_timestep_stats")


# 3, 4 ----------------------------------------------------------------------
def fig_interventions(cfg, res, out):
    df = pd.read_csv(res / "interventions.csv")
    df = df[~df.null_site]
    for T in sorted(df["T"].unique()):
        d = df[df["T"] == T]
        sites = list(dict.fromkeys(d.site))
        fams = [f for f in FAMILIES if f in set(d.family)]
        # 3: accuracy after intervention, clean and mean over corrupted conditions
        fig, axes = plt.subplots(1, len(sites), figsize=(3.4 * len(sites), 3.2), squeeze=False, sharey=True)
        for j, s in enumerate(sites):
            ax = axes[0, j]
            ds = d[d.site == s]
            x = np.arange(len(fams))
            cl = ds[ds.condition == "clean"].set_index("family")
            co = ds[ds.condition != "clean"].groupby("family")[["acc_after", "acc_before"]].mean()
            ax.bar(x - 0.2, [cl.acc_after.get(f, np.nan) for f in fams], 0.38,
                   color=[FAM_COLOR[f] for f in fams], edgecolor="white", linewidth=1)
            ax.bar(x + 0.2, [co.acc_after.get(f, np.nan) for f in fams], 0.38,
                   color=[FAM_COLOR[f] for f in fams], alpha=0.45, edgecolor="white", linewidth=1,
                   hatch="///")
            ax.axhline(cl.acc_before.iloc[0], color="#191919", lw=1, ls="--")
            ax.axhline(co.acc_before.mean(), color="#191919", lw=1, ls=":")
            ax.set_xticks(x, [FAM_LABEL[f] for f in fams], rotation=60, ha="right", fontsize=7)
            ax.set_title(s, fontsize=9)
            if j == 0:
                ax.set_ylabel("accuracy after intervention")
        axes[0, 0].text(0.01, 0.02, "solid: clean; hatched: mean over corruptions\n"
                                    "dashed / dotted: before intervention", transform=axes[0, 0].transAxes,
                        fontsize=6, color=TEXT2)
        fig.suptitle(f"Accuracy vs intervention type (T={T})")
        save(fig, out, f"fig3_accuracy_vs_intervention_T{T}")
        # 4: flip rates of count-preserving interventions with 95% CIs, clean
        cp = [f for f in fams if f not in ("identity",)]
        fig, axes = plt.subplots(1, len(sites), figsize=(3.4 * len(sites), 3.2), squeeze=False, sharey=True)
        for j, s in enumerate(sites):
            ax = axes[0, j]
            ds = d[(d.site == s) & (d.condition == "clean")].set_index("family")
            for i, f in enumerate(cp):
                if f not in ds.index:
                    continue
                r = ds.loc[f]
                ax.bar(i, r.flip, 0.7, color=FAM_COLOR[f], edgecolor="white", linewidth=1,
                       hatch="///" if f == "count_change" else None)
                ax.errorbar(i, r.flip, yerr=[[r.flip - r.flip_lo], [r.flip_hi - r.flip]], color="#191919",
                            lw=1, capsize=2)
            ax.set_xticks(range(len(cp)), [FAM_LABEL[f] for f in cp], rotation=60, ha="right", fontsize=7)
            ax.set_title(s, fontsize=9)
            if j == 0:
                ax.set_ylabel("prediction-flip rate (clean)")
        fig.suptitle(f"Prediction flips under interventions, 95% bootstrap CI over images (T={T}); "
                     "hatched = changes counts")
        save(fig, out, f"fig4_flip_rates_T{T}")


# 5 -------------------------------------------------------------------------
def fig_score_dist(cfg, res, out, T=4):
    z = np.load(res / f"detection_scores_T{T}.npz")
    corr, sev, y = z["cleanonly/corruption"], z["cleanonly/severity"], z["cleanonly/y"]
    s = z["cleanonly/CO_timing_resid_maha"]
    conds = [("clean", 0)] + [(c, max(cfg["run"]["severities"])) for c in cfg["run"]["corruptions"]]
    fig, axes = plt.subplots(1, len(conds), figsize=(2.2 * len(conds), 2.4), squeeze=False, sharey=False)
    bins = np.quantile(np.log10(s), np.linspace(0, 1, 31))
    for j, (c, sv) in enumerate(conds):
        ax = axes[0, j]
        sel = (corr == c) & (sev == sv)
        for lab, col, name in [(0, SLOTS[0], "correct"), (1, SLOTS[7], "error")]:
            v = np.log10(s[sel & (y == lab)])
            if len(v):
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.5, color=col,
                        label=f"{name} (n={len(v)})")
        a = dt.auroc(y[sel], s[sel])
        ax.set_title(f"{c}{'' if c == 'clean' else f' s{sv}'}\nAUROC {a:.3f}", fontsize=8)
        ax.set_xlabel("log10 timing anomaly")
        ax.legend(fontsize=6)
    fig.suptitle(f"Clean-only timing anomaly score (count-residualized Mahalanobis), T={T}")
    save(fig, out, f"fig5_anomaly_distributions_T{T}")


def roc_xy(y, s):
    o = np.argsort(-s, kind="stable")
    y = y[o].astype(bool)
    tpr = np.r_[0, np.cumsum(y) / max(y.sum(), 1)]
    fpr = np.r_[0, np.cumsum(~y) / max((~y).sum(), 1)]
    return fpr, tpr


# 6 -------------------------------------------------------------------------
def fig_roc(cfg, res, out, T=4):
    z = np.load(res / f"detection_scores_T{T}.npz")
    loco = pd.read_csv(res / "detection_loco.csv")
    hs = [h for h in cfg["run"]["corruptions"] if f"{h}/y" in z.files]
    sev = sorted(set(cfg["run"]["severities"]))[len(cfg["run"]["severities"]) // 2]
    fig, axes = plt.subplots(1, len(hs), figsize=(2.4 * len(hs), 2.6), squeeze=False)
    for j, h in enumerate(hs):
        ax = axes[0, j]
        sel = z[f"{h}/severity"] == sev
        y = z[f"{h}/y"][sel]
        best = loco[(loco["T"] == T) & (loco.heldout == h) & loco.model.str.startswith("B")] \
            .groupby("model").train_auroc_cv.first().idxmax()
        for name, col in [("M0_conf", MODEL_COLOR["M0_conf"]), ("M1_conf_rate_act", MODEL_COLOR["M1_conf_rate_act"]),
                          ("M2_M1_plus_timing", MODEL_COLOR["M2_M1_plus_timing"]), (best, MODEL_COLOR["best_baseline"])]:
            fpr, tpr = roc_xy(y, z[f"{h}/{name}"][sel])
            ax.plot(fpr, tpr, lw=1.5, color=col, ls="--" if name == best else "-",
                    label=f"{name.split('_')[0]} {dt.auroc(y, z[f'{h}/{name}'][sel]):.3f}")
        ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.8)
        ax.set_title(f"held out: {h} s{sev}\nerrors {int(y.sum())}/{len(y)}", fontsize=8)
        ax.set_xlabel("FPR")
        if j == 0:
            ax.set_ylabel("TPR (errors caught)")
        ax.legend(fontsize=6, loc="lower right")
    fig.suptitle(f"Error-detection ROC on held-out corruptions (T={T}); dashed = strongest simple baseline")
    save(fig, out, f"fig6_roc_T{T}")


# 7 -------------------------------------------------------------------------
def fig_detection_grid(cfg, res, out):
    dl = pd.read_csv(res / "detection_deltas.csv")
    dl = dl[(dl.a == "M2_M1_plus_timing") & (dl.b == "M1_conf_rate_act")]
    loco = pd.read_csv(res / "detection_loco.csv")
    for T in sorted(dl["T"].unique()):
        fig, axes = plt.subplots(1, 3, figsize=(14, 0.5 * 6 + 1.8), constrained_layout=True)
        d = dl[dl["T"] == T]
        for ax, col, title in [(axes[0], "delta_auroc", "AUROC(M2) - AUROC(M1)"),
                               (axes[1], "delta_auroc_margin_strat", "same, margin-stratified")]:
            p = d.pivot(index="heldout", columns="severity", values=col)
            lim = max(0.02, float(np.nanmax(np.abs(p.values))))
            im = ax.imshow(p.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
            lo = d.pivot(index="heldout", columns="severity", values="lo" if col == "delta_auroc" else "lo_strat")
            hi = d.pivot(index="heldout", columns="severity", values="hi" if col == "delta_auroc" else "hi_strat")
            for i in range(p.shape[0]):
                for jj in range(p.shape[1]):
                    sig = "*" if (lo.values[i, jj] > 0 or hi.values[i, jj] < 0) else ""
                    ax.text(jj, i, f"{p.values[i, jj]:+.3f}{sig}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(p.shape[1]), [f"s{c}" for c in p.columns])
            ax.set_yticks(range(p.shape[0]), p.index)
            ax.grid(False)
            ax.set_title(title + "\n* = 95% CI excludes 0", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
        m2 = loco[(loco["T"] == T) & (loco.model == "M2_M1_plus_timing")].pivot(
            index="heldout", columns="severity", values="auroc")
        er = loco[(loco["T"] == T) & (loco.model == "M2_M1_plus_timing")].pivot(
            index="heldout", columns="severity", values="error_rate")
        im = axes[2].imshow(m2.values, cmap=LinearSegmentedColormap.from_list("seq", ["#e8f1fc", "#0b3f7a"]),
                            vmin=0.5, vmax=1.0, aspect="auto")
        for i in range(m2.shape[0]):
            for jj in range(m2.shape[1]):
                axes[2].text(jj, i, f"{m2.values[i, jj]:.3f}\nerr {er.values[i, jj]:.2f}", ha="center",
                             va="center", fontsize=6, color="white" if m2.values[i, jj] > 0.8 else "#191919")
        axes[2].set_xticks(range(m2.shape[1]), [f"s{c}" for c in m2.columns])
        axes[2].set_yticks(range(m2.shape[0]), m2.index)
        axes[2].grid(False)
        axes[2].set_title("AUROC of M2 (and error rate)", fontsize=8)
        fig.colorbar(im, ax=axes[2], shrink=0.8)
        fig.suptitle(f"Held-out-corruption error detection by corruption and severity (T={T})")
        save(fig, out, f"fig7_detection_by_condition_T{T}")


# 8 -------------------------------------------------------------------------
def network_order(site):
    """stem LIF first, then blocks in order; within a block sn1, sn2, downsample_sn."""
    if site == "sn1":
        return (0, 0, 0, 0)
    stage, block, neuron = site.split(".")
    return (1, int(stage[-1]), int(block), ["sn1", "sn2", "downsample_sn"].index(neuron))


def fig_T4_vs_T8(cfg, res, out):
    cap = pd.read_csv(res / "capacity.csv")
    c = cap[(cap.condition == "clean") & (cap.kind == "binary")]
    Ts = sorted(c["T"].unique())
    sites = sorted(dict.fromkeys(c[c["T"] == Ts[0]].site), key=network_order)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
    x = np.arange(len(sites))
    for i, T in enumerate(Ts):
        d = c[c["T"] == T].set_index("site").reindex(sites)
        col = SLOTS[0] if i == 0 else SLOTS[1]
        axes[0].plot(x, d.frac_multi_order, marker="o", ms=4, lw=1.5, color=col, label=f"T={T}")
        axes[1].plot(x, d.capacity_bits, marker="o", ms=4, lw=1.5, color=col, label=f"capacity T={T}")
        axes[1].plot(x, d.H_order_given_count_channel_bits, marker="s", ms=4, lw=1.5, ls="--", color=col,
                     label=f"measured H(order|count,channel) T={T}")
        axes[2].plot(x, d.H_order_given_count_channel_bits / d.capacity_bits, marker="o", ms=4, lw=1.5,
                     color=col, label=f"T={T}")
    for ax, yl in zip(axes, ["fraction of neurons with >1 possible order", "bits per neuron",
                             "measured / capacity"]):
        ax.set_xticks(x, sites, rotation=70, ha="right", fontsize=6)
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=6)
    axes[2].legend(fontsize=6)
    axes[1].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.62), ncol=2)
    fig.suptitle("Timing-information capacity at every binary LIF site, clean test images: T=4 vs T=8")
    save(fig, out, "fig8_capacity_T4_vs_T8")
    dl_path = res / "detection_deltas.csv"
    if dl_path.exists():
        dl = pd.read_csv(dl_path)
        dl = dl[(dl.a == "M2_M1_plus_timing") & (dl.b == "M1_conf_rate_act")]
        if dl["T"].nunique() > 1:
            fig, ax = plt.subplots(figsize=(5, 3))
            g = dl.groupby(["T", "heldout"]).delta_auroc_margin_strat.mean().unstack(0)
            for i, T in enumerate(g.columns):
                ax.bar(np.arange(len(g)) + (i - 0.5) * 0.38, g[T], 0.36, color=[SLOTS[0], SLOTS[1]][i],
                       edgecolor="white", label=f"T={T}")
            ax.axhline(0, color="#191919", lw=0.8)
            ax.set_xticks(range(len(g)), g.index, rotation=30, ha="right")
            ax.set_ylabel("margin-strat. AUROC(M2) - AUROC(M1)\n(mean over severities)")
            ax.legend()
            save(fig, out, "fig8b_detection_gain_T4_vs_T8")


def main():
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_cfg(args)
    cache, res = Path(cfg["resolved"]["cache"]), Path(cfg["resolved"]["results"])
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for f in [lambda: fig_rasters(cfg, cache, out), lambda: fig_timestep(cfg, res, out),
              lambda: fig_interventions(cfg, res, out), lambda: fig_score_dist(cfg, res, out),
              lambda: fig_roc(cfg, res, out), lambda: fig_detection_grid(cfg, res, out),
              lambda: fig_T4_vs_T8(cfg, res, out)]:
        try:
            f()
        except FileNotFoundError as e:
            print("skipped (missing input):", e)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
