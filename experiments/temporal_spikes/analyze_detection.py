"""Experiment B analysis (CPU): do timing features predict classification errors beyond
confidence, firing rate and activation statistics?

    python analyze_detection.py --stage pilot [--T 4]

Feature families (per image, from the binary detector sites):
  conf    MSP, entropy, top-2 logit margin, and the confidence trajectory over t
          (MSP of the running-mean logits at each t, # prediction changes, final == first)
  rate    log total spikes per site, per-timestep population rate, count-class fractions,
          log per-channel counts
  act     Mahalanobis distance of the penultimate rate to the nearest clean class mean
  timing  timing GIVEN count: deviations of mean spike time, first-spike time and (1,1)
          transitions from their uniform-order expectation within each count class;
          lag-1 cross-channel covariance in a clean PCA basis; per-channel mean-spike-time
          deviation at the last site
Models (L2 logistic regression, C by grouped CV on training rows):
  M0 = conf, M1 = conf + rate + act, M2 = M1 + timing.   Key quantity: AUROC(M2) - AUROC(M1).
Protocols:
  LOCO        train on fit-split images of clean + 5 corruptions, test on eval-split images
              of the held-out corruption (disjoint images, nothing fit on test rows)
  clean-only  Mahalanobis anomaly score of count-residualized timing features, fit on clean
              TRAINING images only, no error labels used
Controls: every metric within one corruption x severity cell; margin-stratified and
firing-count-stratified AUROC (decile edges from training rows); timing-only score
residualized on margin, MSP and firing counts.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression, Ridge

from tslib import detection as dt
from tslib.common import add_common_args, dump_json, load_cfg
from tslib.stats import expected_centroid, expected_first_spike, expected_trans11


# ------------------------------------------------------------------ features

def conf_features(logits, logits_t):
    p = softmax(logits, 1)
    srt = np.sort(logits, 1)
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
    T = logits_t.shape[1]
    run = np.cumsum(logits_t, 1) / np.arange(1, T + 1)[None, :, None]
    msp_t = softmax(run, 2).max(2)                                         # [N, T]
    pred_t = run.argmax(2)
    changes = (pred_t[:, 1:] != pred_t[:, :-1]).sum(1)
    same = (pred_t[:, 0] == pred_t[:, -1]).astype(float)
    return {"msp": p.max(1), "entropy": ent, "margin": srt[:, -1] - srt[:, -2],
            "X": np.c_[p.max(1), ent, srt[:, -1] - srt[:, -2], msp_t, changes, same]}


def site_features(d, s, T, last):
    nk = d[f"{s}/n_k"].astype(float)                                       # [N, T+1]
    tot_neurons = nk.sum(1, keepdims=True)
    chan = d[f"{s}/chan_counts"].astype(float)
    total = chan.sum(1)
    rate = np.c_[np.log1p(total), d[f"{s}/pop_t"], nk / tot_neurons]
    rate_full = np.c_[rate, np.log1p(chan)]

    def dev(key, expect, ks):
        sm = d[f"{s}/{key}"]
        cols = []
        for k in ks:
            n = nk[:, k]
            cols.append(np.where(n > 0, sm[:, k] / np.maximum(n, 1) - expect(T, k), 0.0))
        return np.stack(cols, 1)

    timing = [dev("sum_centroid_k", expected_centroid, range(1, T)),
              dev("sum_first_k", expected_first_spike, range(1, T)),
              dev("sum_trans_k", expected_trans11, range(2, T)),
              d[f"{s}/lag1"].reshape(len(nk), -1)]
    if last:
        nm = d[f"{s}/chan_n_multi"].astype(float)
        timing.append(np.where(nm > 0, d[f"{s}/chan_centroid_sum"] / np.maximum(nm, 1)
                               - (T - 1) / 2.0, 0.0))
    return {"rate_small": rate, "rate_full": rate_full, "timing": np.concatenate(timing, 1),
            "total": total, "pop_t": d[f"{s}/pop_t"]}


class Mahalanobis:
    """Distance to the nearest class mean with a shared Ledoit-Wolf covariance (Lee et al. 2018
    style), or to the global mean if labels is None."""

    def __init__(self, X, labels=None):
        X = np.asarray(X, dtype=np.float64)
        if labels is None:
            self.means = X.mean(0, keepdims=True)
            R = X - self.means
        else:
            cls = np.unique(labels)
            self.means = np.stack([X[labels == c].mean(0) for c in cls])
            R = X - self.means[np.searchsorted(cls, labels)]
        self.prec = LedoitWolf().fit(R).precision_

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        d = [np.einsum("ij,jk,ik->i", X - m, self.prec, X - m) for m in self.means]
        return np.min(np.stack(d, 1), 1)


def build(d, cfg, T, penult_maha):
    sites = cfg["detector_sites"]
    c = conf_features(d["logits"], d["logits_t"])
    sf = [site_features(d, s, T, s == sites[-1]) for s in sites]
    f = {"conf": c["X"], "msp": c["msp"], "entropy": c["entropy"], "margin": c["margin"],
         "rate_small": np.concatenate([x["rate_small"] for x in sf], 1),
         "rate_full": np.concatenate([x["rate_full"] for x in sf], 1),
         "timing": np.concatenate([x["timing"] for x in sf], 1),
         "pop_t": np.concatenate([x["pop_t"] for x in sf], 1),
         "log_total": np.log1p(np.stack([x["total"] for x in sf], 1)),
         "maha_penult": penult_maha.score(d["penult_rate"].astype(np.float64))}
    f["total_all"] = f["log_total"].sum(1)
    f["layer_counts"] = np.c_[f["log_total"], np.concatenate(
        [x["rate_small"][:, 1 + T:] for x in sf], 1)]                     # totals + count classes
    f["act"] = f["maha_penult"][:, None]
    return f


MODELS = {                                                  # name: feature families
    "M0_conf": ["conf"],
    "M1_conf_rate_act": ["conf", "rate_full", "act"],
    "M2_M1_plus_timing": ["conf", "rate_full", "act", "timing"],
    "timing_only": ["timing"],
    "B5_rate_t_vector": ["pop_t"],
    "B6_layer_count_stats": ["layer_counts"],
    "B8_rate_plus_conf": ["conf", "pop_t", "log_total"],
}
SINGLE = {"B1_neg_msp": lambda f: -f["msp"], "B2_entropy": lambda f: f["entropy"],
          "B3_neg_margin": lambda f: -f["margin"], "B4_total_rate": lambda f: f["total_all"],
          "B7_maha_penult": lambda f: f["maha_penult"]}


def stack(F, fams):
    return np.concatenate([F[k] if F[k].ndim == 2 else F[k][:, None] for k in fams], 1)


# ------------------------------------------------------------------ main

def load_T(cfg, T):
    root = Path(cfg["resolved"]["cache"]) / f"T{T}"
    ref = dict(np.load(root / "ref" / "features.npz"))
    penult = Mahalanobis(ref["penult_rate"].astype(np.float64), ref["labels"])
    F_ref = build(ref, cfg, T, penult)
    parts = []
    for cdir in sorted((root / "cond").iterdir()):
        p = cdir / "features.npz"
        if not p.exists():
            continue
        d = dict(np.load(p))
        name = cdir.name
        corr, sev = ("clean", 0) if name == "clean" else (name.rsplit("_s", 1)[0], int(name.rsplit("_s", 1)[1]))
        F = build(d, cfg, T, penult)
        n = len(d["labels"])
        F["meta"] = pd.DataFrame({"corruption": corr, "severity": sev, "split": d["split"],
                                  "index": d["indices"], "label": d["labels"],
                                  "pred": d["logits"].argmax(1)})
        parts.append(F)
    keys = [k for k in parts[0] if k != "meta"]
    F_all = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    meta = pd.concat([p["meta"] for p in parts], ignore_index=True)
    meta["error"] = (meta.pred != meta.label).astype(int)
    return F_all, meta, F_ref, ref


def orient(score_tr, y_tr):
    """Sign for single-score baselines, chosen on training rows only."""
    return 1.0 if dt.auroc(y_tr, score_tr) >= 0.5 else -1.0


def run_T(cfg, T, res, log_rows):
    F, meta, F_ref, ref = load_T(cfg, T)
    y = meta.error.values
    corr = np.asarray(meta.corruption.to_numpy(dtype=str))
    split = np.asarray(meta.split.to_numpy(dtype=str))
    sev = meta.severity.to_numpy(dtype=np.int64)
    groups = meta["index"].values
    dc, bs = cfg["detection"], cfg["bootstrap"]
    rows, deltas, resid_rows, rel_rows, score_store = [], [], [], [], {}
    for h in cfg["run"]["corruptions"]:
        tr, te = dt.protocol_masks(corr, split, h)
        if te.sum() == 0:
            continue
        edges_m = dt.quantile_edges(F["margin"][tr], dc["n_strata"])
        edges_c = dt.quantile_edges(F["total_all"][tr], dc["n_strata"])
        scores, cv = {}, {}
        for name, fams in MODELS.items():
            X = stack(F, fams)
            m = dt.fit_logistic(X[tr], y[tr], groups[tr], dc["C_grid"], dc["cv_folds"], cfg["seed"])
            scores[name] = m.decision_function(X)
            cv[name] = (m.chosen_C_, m.cv_auroc_)
            if name in ("M1_conf_rate_act", "M2_M1_plus_timing"):
                pr = m.predict_proba(X[te])[:, 1]
                bins = np.minimum((pr * 10).astype(int), 9)
                for b in range(10):
                    sel = bins == b
                    if sel.any():
                        rel_rows.append({"T": T, "heldout": h, "model": name, "bin": b,
                                         "mean_pred": float(pr[sel].mean()),
                                         "frac_error": float(y[te][sel].mean()), "n": int(sel.sum())})
        for name, fn in SINGLE.items():
            s = fn(F)
            scores[name] = orient(s[tr], y[tr]) * s
            cv[name] = (None, dt.auroc(y[tr], scores[name][tr]))
        # strongest simple baseline, chosen on TRAINING rows (CV AUROC / train AUROC)
        base_names = [n for n in scores if n.startswith("B")]
        best_base = max(base_names, key=lambda n: cv[n][1])
        # residual test: timing-only score with margin, MSP and firing counts regressed out
        Z = np.c_[F["margin"], F["msp"], F["log_total"]]
        lr = LinearRegression().fit(Z[tr], scores["timing_only"][tr])
        resid = scores["timing_only"] - lr.predict(Z)
        for s_ in sorted(np.unique(sev[te])):
            cell = te & (sev == s_)
            yc = y[cell]
            ms, cs = dt.strata_from_edges(F["margin"][cell], edges_m), dt.strata_from_edges(F["total_all"][cell], edges_c)
            base = {"T": T, "heldout": h, "severity": int(s_), "n": int(cell.sum()),
                    "n_errors": int(yc.sum()), "error_rate": float(yc.mean())}
            for name, sc in scores.items():
                rows.append({**base, "model": name, "chosen_C": cv[name][0], "train_auroc_cv": cv[name][1],
                             **dt.all_metrics(yc, sc[cell], ms, cs)})
            for a, b in [("M2_M1_plus_timing", "M1_conf_rate_act"), ("M2_M1_plus_timing", best_base),
                         ("M1_conf_rate_act", "M0_conf")]:
                p, lo, hi = dt.paired_bootstrap_delta(yc, scores[a][cell], scores[b][cell], bs["n"], bs["seed"])
                pm, lom, him = dt.paired_bootstrap_delta(yc, scores[a][cell], scores[b][cell], bs["n"],
                                                         bs["seed"], strata=ms)
                deltas.append({**base, "a": a, "b": b, "delta_auroc": p, "lo": lo, "hi": hi,
                               "delta_auroc_margin_strat": pm, "lo_strat": lom, "hi_strat": him})
            resid_rows.append({**base, "auroc_timing_only": dt.auroc(yc, scores["timing_only"][cell]),
                               "auroc_residual": dt.auroc(yc, resid[cell]),
                               "auroc_residual_margin_strat": dt.stratified_auroc(yc, resid[cell], ms),
                               "auroc_residual_count_strat": dt.stratified_auroc(yc, resid[cell], cs)})
        score_store[h] = {k: v[te].astype(np.float32) for k, v in scores.items()}
        score_store[h]["y"], score_store[h]["severity"] = y[te], sev[te]
        log_rows.append(f"T={T} held-out {h}: best simple baseline (train) = {best_base}")
        print(log_rows[-1], flush=True)

    # ---- clean-only protocol: nothing but clean training images, no error labels
    ridge = Ridge(alpha=1.0).fit(F_ref["rate_small"], F_ref["timing"])
    t_res_ref = F_ref["timing"] - ridge.predict(F_ref["rate_small"])
    maha_t = Mahalanobis(t_res_ref)
    maha_r = Mahalanobis(F_ref["rate_small"])
    edges_m = dt.quantile_edges(F_ref["margin"], dc["n_strata"])
    co = {"CO_timing_resid_maha": maha_t.score(F["timing"] - ridge.predict(F["rate_small"])),
          "CO_rate_maha": maha_r.score(F["rate_small"]), "CO_neg_msp": -F["msp"],
          "CO_maha_penult": F["maha_penult"]}
    co_rows = []
    ev = split == "eval"
    for (c_, s_), _ in meta[ev].groupby(["corruption", "severity"]):
        cell = ev & (corr == c_) & (sev == s_)
        yc = y[cell]
        ms = dt.strata_from_edges(F["margin"][cell], edges_m)
        for name, sc in co.items():
            co_rows.append({"T": T, "corruption": c_, "severity": int(s_), "n": int(cell.sum()),
                            "n_errors": int(yc.sum()), "model": name, **dt.all_metrics(yc, sc[cell], ms)})
    np.savez(res / f"detection_scores_T{T}.npz",
             **{f"{h}/{k}": v for h, d in score_store.items() for k, v in d.items()},
             **{f"cleanonly/{k}": v[ev].astype(np.float32) for k, v in co.items()},
             **{"cleanonly/y": y[ev], "cleanonly/corruption": corr[ev], "cleanonly/severity": sev[ev]})
    return rows, deltas, resid_rows, rel_rows, co_rows


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--T", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args)
    res = Path(cfg["resolved"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    out = {k: [] for k in ["loco", "deltas", "residual", "reliability", "clean_only"]}
    log_rows = []
    for T in args.T or cfg["run"]["T"]:
        if not (Path(cfg["resolved"]["cache"]) / f"T{T}" / "ref" / "features.npz").exists():
            continue
        for k, v in zip(out, run_T(cfg, T, res, log_rows)):
            out[k] += v
    for k, v in out.items():
        pd.DataFrame(v).to_csv(res / f"detection_{k}.csv", index=False)
    # ---- pilot decision: margin-stratified M2 - M1 >= 0.01 in >= 2 held-out corruptions
    dl = pd.DataFrame(out["deltas"])
    dl = dl[(dl.a == "M2_M1_plus_timing") & (dl.b == "M1_conf_rate_act")]
    per = dl.groupby(["T", "heldout"]).delta_auroc_margin_strat.mean().reset_index()
    dec = {"criterion": "mean over severities of margin-stratified AUROC(M2) - AUROC(M1) >= 0.01 "
                        "in at least 2 of the 6 held-out corruptions (per T)",
           "per_heldout": per.to_dict("records"), "notes": log_rows}
    dec["go_by_T"] = {int(t): bool((g.delta_auroc_margin_strat >= 0.01).sum() >= 2)
                      for t, g in per.groupby("T")}
    dec["go"] = any(dec["go_by_T"].values())
    dump_json(dec, res / "pilot_decision_B.json")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(per.round(4).to_string(index=False))
    print("Experiment B go:", dec["go_by_T"])


if __name__ == "__main__":
    main()
