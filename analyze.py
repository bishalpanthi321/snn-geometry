"""All analysis on saved features (CPU only, no torch).

    python analyze.py                    # every model found under features/
    python analyze.py --models ann snn

Subspaces are always fit on the fit split; every reported metric uses the eval
split unless the metric name says otherwise. Writes tidy CSVs to results/ with
columns model, layer, corruption, severity, metric, k, value.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import geometry as G
from src.utils import get_logger, load_config

LAYERS = ["layer1", "layer2", "layer3", "penultimate"]


class Store:
    """Reads features/{model}/{split}/{corruption}/{severity}/{name}.npy."""

    def __init__(self, root, model):
        self.root = Path(root) / model

    def load(self, split, corr, sev, name):
        return np.load(self.root / split / corr / str(sev) / f"{name}.npy", allow_pickle=False)

    def has(self, split, corr, sev, name="preds"):
        return (self.root / split / corr / str(sev) / f"{name}.npy").exists()


class Rows:
    def __init__(self):
        self.rows = []

    def add(self, model, layer, corr, sev, metric, value, k=np.nan, **extra):
        self.rows.append(dict(model=model, layer=layer, corruption=corr, severity=sev,
                              metric=metric, k=k, value=float(value), **extra))

    def df(self):
        return pd.DataFrame(self.rows)


def check_pairing(store, conds):
    """Every condition must hold the same test indices in the same order."""
    for split in ["fit", "eval"]:
        ref = store.load(split, "clean", 0, "indices")
        for c, s in conds:
            if not np.array_equal(store.load(split, c, s, "indices"), ref):
                raise RuntimeError(f"index mismatch {split} {c}/{s}: displacements would be unpaired")


def logits_acc(H, Wfc, bfc, y):
    return float(((H @ Wfc.T + bfc).argmax(1) == y).mean())


# ---------------------------------------------------------------- H1, H2

def geometry_for_layer(store, model, layer, conds, cfg, rows, sanity, rng):
    a = cfg["analysis"]
    Xtr = store.load("train", "clean", 0, layer).astype(np.float64)
    ytr = store.load("train", "clean", 0, "labels")
    Hc_fit = store.load("fit", "clean", 0, layer).astype(np.float64)
    Hc_eval = store.load("eval", "clean", 0, layer).astype(np.float64)
    D_full = Hc_fit.shape[1]
    mask = G.nonconstant_mask(Hc_fit)
    Dm = int(mask.sum())
    sanity.add(model, layer, "clean", 0, "dims_total", D_full)
    sanity.add(model, layer, "clean", 0, "dims_dropped_zero_variance", D_full - Dm)

    W, _ = G.lda_subspace(Xtr[:, mask], ytr)
    Vclean = G.clean_pca_basis(Hc_fit[:, mask])
    chance = W.shape[1] / Dm
    # Reference: how much of ordinary clean variation lies in W (W is fit to capture it).
    frac_clean, _ = G.energy_fraction(Hc_eval[:, mask] - Hc_fit[:, mask].mean(0), W)
    rows.add(model, layer, "clean", 0, "energy_frac_W_clean_variation", frac_clean)
    rows.add(model, layer, "clean", 0, "energy_frac_W_chance", chance)
    # Sanity: clean vs clean displacement must be exactly 0.
    mag0, _ = G.relative_magnitude(Hc_eval, Hc_eval.copy())
    sanity.add(model, layer, "clean", 0, "clean_vs_clean_rel_magnitude", mag0)

    ks = [k for k in a["ks_heldout"] if k < Dm]
    rand_bases = {k: [G.random_subspace(Dm, k, rng) for _ in range(a["n_random_subspaces"])] for k in ks}
    out = {"W": W, "mask": mask}

    for c, s in conds:
        Hx_fit = store.load("fit", c, s, layer).astype(np.float64)
        Hx_eval = store.load("eval", c, s, layer).astype(np.float64)
        Dfit_full, Deval_full = Hx_fit - Hc_fit, Hx_eval - Hc_eval
        Dfit, Deval = Dfit_full[:, mask], Deval_full[:, mask]
        tot = np.square(Deval_full).sum()
        # Dims dropped as constant on clean data can still move under corruption.
        sanity.add(model, layer, c, s, "eval_disp_energy_frac_in_dropped_dims",
                   np.square(Deval_full[:, ~mask]).sum() / tot if tot > 0 else np.nan)

        mag, n_used = G.relative_magnitude(Hc_eval, Hx_eval, a["min_norm"])
        rows.add(model, layer, c, s, "rel_magnitude", mag)
        rows.add(model, layer, c, s, "rel_magnitude_n_used", n_used)

        evals, V = G.displacement_spectrum(Dfit)
        k90 = G.k_for_energy(evals, a["energy_threshold"])
        rows.add(model, layer, c, s, "k90", k90)
        rows.add(model, layer, c, s, "k90_over_D", k90 / Dm)
        rows.add(model, layer, c, s, "participation_ratio", G.participation_ratio(evals))
        rows.add(model, layer, c, s, "D", Dm)
        ev_c, _ = G.displacement_spectrum(Dfit, center=True)
        k90c = G.k_for_energy(ev_c, a["energy_threshold"])
        rows.add(model, layer, c, s, "k90_centered", k90c)
        rows.add(model, layer, c, s, "k90_over_D_centered", k90c / Dm)
        rows.add(model, layer, c, s, "participation_ratio_centered", G.participation_ratio(ev_c))
        # Mean displacement vs spread: a large ratio means a shared shift dominates.
        mean_d = Dfit.mean(0)
        rows.add(model, layer, c, s, "mean_disp_energy_frac",
                 len(Dfit) * mean_d @ mean_d / np.square(Dfit).sum())

        for k in ks:
            rows.add(model, layer, c, s, "heldout_energy", G.captured_energy(Deval, V[:, :k]), k)
            rows.add(model, layer, c, s, "fit_energy", evals[:k].sum() / evals.sum(), k)
            rows.add(model, layer, c, s, "null_random_energy",
                     np.mean([G.captured_energy(Deval, B) for B in rand_bases[k]]), k)
            rows.add(model, layer, c, s, "null_random_expected", k / Dm, k)
            rows.add(model, layer, c, s, "null_cleanpca_energy", G.captured_energy(Deval, Vclean[:, :k]), k)

        # H2: principal angles between top-k displacement subspace and W.
        k = int(min(k90, a["k_cap"]))
        ang = G.principal_angles_deg(V[:, :k], W)
        rows.add(model, layer, c, s, "angle_k_used", k)
        # If k + 9 > D the subspaces must intersect: that many angles are 0 by counting.
        rows.add(model, layer, c, s, "angles_forced_zero_by_dimension", max(0, k + W.shape[1] - Dm))
        for i, v in enumerate(ang):
            rows.add(model, layer, c, s, f"principal_angle_{i + 1}", v, k)
        rows.add(model, layer, c, s, "angle_min", ang.min(), k)
        rows.add(model, layer, c, s, "angle_median", np.median(ang), k)
        rows.add(model, layer, c, s, "angle_frac_gt60", (ang > 60).mean(), k)
        # Null for angles: a random k-dim subspace in the same D.
        rang = [G.principal_angles_deg(G.random_subspace(Dm, k, rng), W) for _ in range(a["n_random_subspaces"])]
        rows.add(model, layer, c, s, "null_random_angle_median", np.mean([np.median(r) for r in rang]), k)
        rows.add(model, layer, c, s, "null_random_angle_min", np.mean([r.min() for r in rang]), k)
        rows.add(model, layer, c, s, "null_random_angle_frac_gt60", np.mean([(r > 60).mean() for r in rang]), k)

        frac, _ = G.energy_fraction(Deval, W, a["min_norm"])
        rows.add(model, layer, c, s, "energy_frac_W", frac)
        rows.add(model, layer, c, s, "energy_frac_W_chance", chance)
        rows.add(model, layer, c, s, "energy_frac_W_over_chance", frac / chance)
    return out


# ---------------------------------------------------------------- H3

def defense(store, model, conds, cfg, rows, transfer_rows, sanity, rng):
    a = cfg["analysis"]
    fc = np.load(store.root / "fc.npz")
    Wfc, bfc = fc["weight"], fc["bias"]
    layer = "penultimate"
    # Full feature space here (no dims dropped): the fc layer reads all 512 dims.
    Hc_fit = store.load("fit", "clean", 0, layer).astype(np.float64)
    Hc_eval = store.load("eval", "clean", 0, layer).astype(np.float64)
    y_fit = store.load("fit", "clean", 0, "labels")
    y_eval = store.load("eval", "clean", 0, "labels")
    m = Hc_fit.mean(0)
    Dfull = Hc_fit.shape[1]

    # fc(features) must reproduce the model's own predictions, else the defense is
    # evaluated on a different classifier than the one we trained.
    for split, H, y in [("fit", Hc_fit, y_fit), ("eval", Hc_eval, y_eval)]:
        agree = float(((H @ Wfc.T + bfc).argmax(1) == store.load(split, "clean", 0, "preds")).mean())
        sanity.add(model, layer, "clean", 0, f"fc_reproduces_model_preds_{split}", agree)

    data, spectra = {}, {}
    G_union = np.zeros((Dfull, Dfull))
    G_by_corr = {}
    for c, s in conds:
        # Kept as float32 (30 conditions x 10k x 512 in float64 would be ~1.2 GB);
        # project_out / logits_acc promote to float64 per call.
        Hx_fit = store.load("fit", c, s, layer)
        Hx_eval = store.load("eval", c, s, layer)
        Dfit = Hx_fit.astype(np.float64) - Hc_fit
        Gc = Dfit.T @ Dfit
        G_union += Gc
        G_by_corr[c] = G_by_corr.get(c, 0) + Gc
        spectra[(c, s)] = G.sorted_eig(Gc)[1]
        data[(c, s)] = (Hx_fit, Hx_eval)
    U_union = G.sorted_eig(G_union)[1]
    U_corr = {c: G.sorted_eig(Gm)[1] for c, Gm in G_by_corr.items()}

    clean_acc_eval = logits_acc(Hc_eval, Wfc, bfc, y_eval)
    clean_acc_fit = logits_acc(Hc_fit, Wfc, bfc, y_fit)
    rows.add(model, layer, "clean", 0, "acc_clean", clean_acc_eval)

    for k in a["ks_defense"]:
        Uu = U_union[:, :k]
        rows.add(model, layer, "clean", 0, "acc_clean_after_union", logits_acc(G.project_out(Hc_eval, m, Uu), Wfc, bfc, y_eval), k)
        rows.add(model, layer, "clean", 0, "acc_clean_after_union_fit", logits_acc(G.project_out(Hc_fit, m, Uu), Wfc, bfc, y_fit), k)
        rbases = [G.random_subspace(Dfull, k, rng) for _ in range(5)]
        rows.add(model, layer, "clean", 0, "acc_clean_after_random",
                 np.mean([logits_acc(G.project_out(Hc_eval, m, B), Wfc, bfc, y_eval) for B in rbases]), k)
        for c, s in conds:
            Hx_fit, Hx_eval = data[(c, s)]
            Uo = spectra[(c, s)][:, :k]
            if k == a["ks_defense"][0]:
                rows.add(model, layer, c, s, "acc_corrupted", logits_acc(Hx_eval, Wfc, bfc, y_eval))
                rows.add(model, layer, c, s, "acc_corrupted_fit", logits_acc(Hx_fit, Wfc, bfc, y_fit))
                rows.add(model, layer, c, s, "acc_clean", clean_acc_eval)
            rows.add(model, layer, c, s, "acc_corrupted_after_oracle", logits_acc(G.project_out(Hx_eval, m, Uo), Wfc, bfc, y_eval), k)
            rows.add(model, layer, c, s, "acc_clean_after_oracle", logits_acc(G.project_out(Hc_eval, m, Uo), Wfc, bfc, y_eval), k)
            rows.add(model, layer, c, s, "acc_corrupted_after_union", logits_acc(G.project_out(Hx_eval, m, Uu), Wfc, bfc, y_eval), k)
            rows.add(model, layer, c, s, "acc_corrupted_after_union_fit", logits_acc(G.project_out(Hx_fit, m, Uu), Wfc, bfc, y_fit), k)
            rows.add(model, layer, c, s, "acc_corrupted_after_random",
                     np.mean([logits_acc(G.project_out(Hx_eval, m, B), Wfc, bfc, y_eval) for B in rbases]), k)
    rows.add(model, layer, "clean", 0, "acc_clean_fit", clean_acc_fit)

    # Transfer: U fit on corruption A (all severities pooled), applied to B.
    kt = a["transfer_k"]
    for src, U in U_corr.items():
        Uk = U[:, :kt]
        clean_after = logits_acc(G.project_out(Hc_eval, m, Uk), Wfc, bfc, y_eval)
        for c, s in conds:
            Hx_eval = data[(c, s)][1]
            before = logits_acc(Hx_eval, Wfc, bfc, y_eval)
            after = logits_acc(G.project_out(Hx_eval, m, Uk), Wfc, bfc, y_eval)
            transfer_rows.append(dict(model=model, layer=layer, source_corruption=src, corruption=c,
                                      severity=s, k=kt, acc_before=before, acc_after=after,
                                      gain=after - before, clean_acc_after=clean_after,
                                      clean_drop=clean_acc_eval - clean_after))


# ---------------------------------------------------------------- H4

def temporal(store, model, conds, cfg, rows, geo_out, rng):
    a = cfg["analysis"]
    fc = np.load(store.root / "fc.npz")
    Wfc, bfc = fc["weight"], fc["bias"]
    k = a["temporal_k"]
    y_eval = store.load("eval", "clean", 0, "labels")
    for layer in LAYERS:
        W, mask = geo_out[layer]["W"], geo_out[layer]["mask"]
        Sc_fit = store.load("fit", "clean", 0, f"{layer}_spikes").astype(np.float64)
        Sc_eval = store.load("eval", "clean", 0, f"{layer}_spikes").astype(np.float64)
        T = Sc_fit.shape[1]

        def views(S, t):
            return {"spikes": S[:, t - 1], "cumrate": S[:, :t].mean(1)}

        if layer == "penultimate":
            for t in range(1, T + 1):
                rows.add(model, layer, "clean", 0, "acc_cumrate", logits_acc(views(Sc_eval, t)["cumrate"], Wfc, bfc, y_eval), t=t, kind="cumrate")
        for c, s in conds:
            Sx_fit = store.load("fit", c, s, f"{layer}_spikes").astype(np.float64)
            Sx_eval = store.load("eval", c, s, f"{layer}_spikes").astype(np.float64)
            for t in range(1, T + 1):
                vc_f, vx_f, vc_e, vx_e = views(Sc_fit, t), views(Sx_fit, t), views(Sc_eval, t), views(Sx_eval, t)
                for kind in ["spikes", "cumrate"]:
                    Dfit = (vx_f[kind] - vc_f[kind])[:, mask]
                    Deval = (vx_e[kind] - vc_e[kind])[:, mask]
                    mag, n_used = G.relative_magnitude(vc_e[kind], vx_e[kind], a["min_norm"])
                    rows.add(model, layer, c, s, "rel_magnitude", mag, t=t, kind=kind)
                    rows.add(model, layer, c, s, "rel_magnitude_n_used", n_used, t=t, kind=kind)
                    _, V = G.displacement_spectrum(Dfit)
                    rows.add(model, layer, c, s, "heldout_energy", G.captured_energy(Deval, V[:, :k]), k, t=t, kind=kind)
                    rows.add(model, layer, c, s, "null_random_expected", k / mask.sum(), k, t=t, kind=kind)
                    frac, _ = G.energy_fraction(Deval, W, a["min_norm"])
                    rows.add(model, layer, c, s, "energy_frac_W", frac, t=t, kind=kind)
                if layer == "penultimate":
                    rows.add(model, layer, c, s, "acc_cumrate", logits_acc(vx_e["cumrate"], Wfc, bfc, y_eval), t=t, kind="cumrate")
        if layer == "penultimate":
            Mc = store.load("eval", "clean", 0, "penultimate_membrane").astype(np.float64)
            for c, s in conds:
                Mx = store.load("eval", c, s, "penultimate_membrane").astype(np.float64)
                for t in range(1, T + 1):
                    mag, _ = G.relative_magnitude(Mc[:, t - 1], Mx[:, t - 1], a["min_norm"])
                    rows.add(model, layer, c, s, "membrane_rel_magnitude", mag, t=t, kind="membrane")


# ---------------------------------------------------------------- driver

def accuracy_table(store, model, conds):
    out = []
    for split in ["fit", "eval"]:
        for c, s in [("clean", 0)] + conds:
            acc = float((store.load(split, c, s, "preds") == store.load(split, c, s, "labels")).mean())
            out.append(dict(model=model, layer="output", corruption=c, severity=s,
                            metric=f"model_acc_{split}", k=np.nan, value=acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke, out_root=args.out_root)
    log = get_logger("analyze", cfg["paths"]["logs_dir"])
    feat_dir = Path(cfg["paths"]["features_dir"])
    res = Path(cfg["paths"]["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    models = args.models or sorted(p.name for p in feat_dir.iterdir() if (p / "meta.json").exists())
    rng = np.random.default_rng(cfg["seed"])

    geo, dfn, tmp, san = Rows(), Rows(), Rows(), Rows()
    transfer_rows, acc_rows = [], []
    for model in models:
        t0 = time.time()
        store = Store(feat_dir, model)
        conds = [(c, s) for c in cfg["corruptions"]["names"] for s in cfg["corruptions"]["severities"]
                 if store.has("eval", c, s)]
        missing = len(cfg["corruptions"]["names"]) * len(cfg["corruptions"]["severities"]) - len(conds)
        if missing:
            log.warning("%s: %d configured conditions have no features; skipped", model, missing)
        check_pairing(store, conds)
        meta = json.loads((store.root / "meta.json").read_text())
        for key, v in meta.get("clean_repeat_max_abs_diff", {}).items():
            san.add(model, key, "clean", 0, "clean_repeat_max_abs_diff", v)
        san.add(model, "penultimate", "all", -1, "max_fc_vs_logits_err",
                max(meta.get("max_fc_vs_logits_err", {"x": np.nan}).values()))
        acc_rows += accuracy_table(store, model, conds)

        geo_out = {}
        for layer in LAYERS:
            geo_out[layer] = geometry_for_layer(store, model, layer, conds, cfg, geo, san, rng)
            log.info("%s %s geometry done (%.0fs)", model, layer, time.time() - t0)
        defense(store, model, conds, cfg, dfn, transfer_rows, san, rng)
        log.info("%s defense done (%.0fs)", model, time.time() - t0)
        if model.startswith("snn"):
            temporal(store, model, conds, cfg, tmp, geo_out, rng)
            log.info("%s temporal done (%.0fs)", model, time.time() - t0)

    geo.df().to_csv(res / "geometry.csv", index=False)
    dfn.df().to_csv(res / "defense.csv", index=False)
    pd.DataFrame(transfer_rows).to_csv(res / "transfer.csv", index=False)
    if tmp.rows:
        tmp.df().to_csv(res / "temporal.csv", index=False)
    san.df().to_csv(res / "sanity.csv", index=False)
    pd.DataFrame(acc_rows).to_csv(res / "accuracy.csv", index=False)
    log.info("wrote CSVs to %s", res)

    from src.summary import write_summary
    write_summary(cfg, res, models)
    log.info("wrote %s", res / "summary.md")


if __name__ == "__main__":
    main()
