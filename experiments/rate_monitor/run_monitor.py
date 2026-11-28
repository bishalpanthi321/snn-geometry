"""Rate-monitor analysis (CPU). Is the error-detection gain from internal activations larger
for the SNN than for the ANN, and can a sparse monitor keep it?

    python run_monitor.py --stage pilot [--features-root DIR]

Protocol (per held-out corruption h):
  train rows  fit-split images of clean + every other corruption
  test rows   eval-split images of h (disjoint images; nothing is fit on them)
  For each model separately: Mahalanobis on clean TRAINING images, standardisation, C by
  grouped CV, channel rankings and margin-decile edges, all from training rows only.
Outputs in results/<stage>/:
  feature_check.json   what was verified in the saved features
  cells.csv            AUROC / margin-stratified AUROC / AUPRC / FPR95 per model, set and cell
  gains.csv            per-cell gains (B-A, C-A, D-C) per model with paired-bootstrap CIs
  gain_diff.csv        SNN-minus-ANN difference of gains: per cell, per held-out corruption
                       (pooled over severities) and pooled over everything
  sparse.csv           sparse monitors (L1 / univariate filter / random) vs number of channels
  sparse_compare.csv   pooled paired comparisons at each k
  cost.csv             measured CPU time and memory of applying each monitor
  decision.json        the exploratory screening rules
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rmlib.data import FeatureStore, add_common_args, load_cfg
from rmlib.features import SETS, build, fit_mahalanobis, stack, verify
from rmlib.stats import BootCache, fit_fixed_C, monitor_cost, random_subsets, rank_l1, rank_univariate
from tslib.detection import all_metrics, fit_logistic, protocol_masks, quantile_edges, strata_from_edges


def log_to(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def fmt(p, lo, hi):
    return {"estimate": p, "lo": lo, "hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)}


def fit_for_heldout(cfg, Ft, y, tr, groups, ks, n_ch):
    """Everything fitted for one model and one held-out corruption. Uses ONLY rows `tr`
    (training rows) for: standardisation, C selection, coefficients, margin-decile edges and
    channel rankings. Scores are then computed for all rows."""
    dc, sp, seed = cfg["detection"], cfg["sparse"], cfg["seed"]
    out = {"scores": {}, "chosen": {}, "cost": [], "rank": {}}
    for s in [s for s in SETS if all(p in Ft for p in SETS[s])]:
        X = stack(Ft, SETS[s])
        m = fit_logistic(X[tr], y[tr], groups[tr], dc["C_grid"], dc["cv_folds"], seed, dc["max_cv_rows"])
        out["scores"][s] = m.decision_function(X)
        out["chosen"][s] = m.chosen_C_
        out["cost"].append({"set": s, "n_features": X.shape[1],
                            **monitor_cost(m.named_steps["standardscaler"], m.named_steps["logisticregression"],
                                           X.shape[1])})
    out["edges"] = quantile_edges(Ft["conf"][tr, 2], dc["n_strata"])             # top-2 logit margin
    out["strata"] = strata_from_edges(Ft["conf"][:, 2], out["edges"])
    Xc, Xch = Ft["conf"], Ft["channels"]
    # rankings use at most max_cv_rows TRAINING rows, sampled by whole image (for speed)
    tr_rows = np.flatnonzero(tr)
    if len(tr_rows) > dc["max_cv_rows"]:
        ug = np.unique(groups[tr_rows])
        keep = np.random.default_rng(seed).permutation(ug)[: int(len(ug) * dc["max_cv_rows"] / len(tr_rows))]
        tr_rows = tr_rows[np.isin(groups[tr_rows], keep)]
    out["rank"] = {"l1": rank_l1(Xc[tr_rows], Xch[tr_rows], y[tr_rows], sp["l1_path_points"], seed),
                   "univariate": rank_univariate(Xch[tr_rows], y[tr_rows])}
    for k in ks:
        subsets = {"l1": [out["rank"]["l1"][:k]], "univariate": [out["rank"]["univariate"][:k]],
                   "random": random_subsets(n_ch, k, sp["n_random"], seed + k)}
        for meth, subs in subsets.items():
            for j, sub in enumerate(subs):
                X = np.c_[Xc, Xch[:, sub]]
                sc, mdl = fit_fixed_C(X[tr], y[tr], out["chosen"]["B"], seed)
                out["scores"][f"sparse_{meth}_{k}_{j}"] = mdl.decision_function(sc.transform(X))
                if j == 0:
                    out["cost"].append({"set": f"sparse_{meth}_k{k}", "n_features": X.shape[1],
                                        **monitor_cost(sc, mdl, X.shape[1])})
    return out


def main():
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_cfg(args)
    res = Path(cfg["resolved"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    log = log_to(Path(cfg["resolved"]["logs"]) / "run_monitor.log")
    t_start = time.time()
    dc, sp = cfg["detection"], cfg["sparse"]
    n_boot = cfg["run"].get("n_boot", cfg["bootstrap"]["n"])
    seed = cfg["seed"]

    # ---- 0. verify the saved features
    rep = verify(cfg, log)
    json.dump(rep | {"stage": cfg["stage"], "features_root": cfg["resolved"]["features_root"]},
              open(res / "feature_check.json", "w"), indent=2)

    # ---- 1. features for both models, aligned row by row
    meta, F = {}, {}
    for tag in cfg["models"]:
        st = FeatureStore(cfg["resolved"]["features_root"], tag)
        maha = fit_mahalanobis(st, cfg["layers"], seed=seed)
        meta[tag], F[tag] = build(cfg, st, tag, maha)
        log(f"{tag}: {len(meta[tag]['label'])} rows, error rate {meta[tag]['error'].mean():.3f} [{time.time()-t_start:.0f}s]")
    m0, m1 = cfg["models"]
    for k in ("corruption", "severity", "split", "index", "label"):
        assert np.array_equal(meta[m0][k], meta[m1][k]), f"rows not aligned across models: {k}"
    corr, split, sev = meta[m0]["corruption"], meta[m0]["split"], meta[m0]["severity"]
    groups = meta[m0]["index"]
    n_ch = F[m0]["channels"].shape[1]
    ks = sorted({max(1, int(round(f * n_ch))) for f in sp["fractions"]})

    cells, cell_rows, sparse_rows, cost_rows = [], [], [], []
    for h in cfg["run"]["corruptions"]:
        tr, te = protocol_masks(corr, split, h)
        scores = {t: {} for t in cfg["models"]}
        strata = {}
        for tag in cfg["models"]:
            y = meta[tag]["error"]
            fit = fit_for_heldout(cfg, F[tag], y, tr, groups, ks, n_ch)
            scores[tag], strata[tag], chosen = fit["scores"], fit["strata"], fit["chosen"]
            if h == cfg["run"]["corruptions"][0]:
                cost_rows += [{"model": tag, **r} for r in fit["cost"]]
            log(f"held-out {h} / {tag}: C = {chosen} [{time.time()-t_start:.0f}s]")
        for s_ in sorted(np.unique(sev[te])):
            rows = np.where(te & (sev == s_))[0]
            cell = {"heldout": h, "severity": int(s_), "n": len(rows),
                    "y": {t: meta[t]["error"][rows] for t in cfg["models"]},
                    "strata": {t: strata[t][rows] for t in cfg["models"]},
                    "scores": {t: {k: v[rows] for k, v in scores[t].items()} for t in cfg["models"]}}
            cells.append(cell)
            for t in cfg["models"]:
                for s in [k for k in SETS if k in scores[t]]:
                    cell_rows.append({"model": t, "set": s, "heldout": h, "severity": int(s_), "n": len(rows),
                                      "n_errors": int(cell["y"][t].sum()), "error_rate": float(cell["y"][t].mean()),
                                      **all_metrics(cell["y"][t], cell["scores"][t][s], cell["strata"][t])})
    pd.DataFrame(cell_rows).to_csv(res / "cells.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(res / "cost.csv", index=False)
    log(f"all monitors fit [{time.time()-t_start:.0f}s]; bootstrapping")

    # ---- 2. gains and SNN-minus-ANN differences (paired bootstrap over the same images)
    bc = BootCache(cells, n_boot, cfg["bootstrap"]["seed"])
    log(f"bootstrap replicates drawn ({n_boot}) [{time.time()-t_start:.0f}s]")
    idx_of = {"pooled": list(range(len(cells)))} | {
        h: [i for i, c in enumerate(cells) if c["heldout"] == h] for h in cfg["run"]["corruptions"]}

    def g(ci, t, a, b, strat):
        return bc.vec(ci, t, b, strat) - bc.vec(ci, t, a, strat)

    gain_rows, diff_rows = [], []
    for scope, cis in idx_of.items():
        for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
            for strat in (False, True):
                for t in cfg["models"]:
                    p_, lo, hi, _ = bc.summarize([g(ci, t, a, b, strat) for ci in cis])
                    gain_rows.append({"scope": scope, "model": t, "gain": f"{b}-{a}", "margin_stratified": strat,
                                      **fmt(p_, lo, hi), "n_cells": len(cis)})
                p_, lo, hi, per = bc.summarize([g(ci, m0, a, b, strat) - g(ci, m1, a, b, strat) for ci in cis])
                diff_rows.append({"scope": scope, "gain": f"{b}-{a}", "margin_stratified": strat,
                                  "diff": f"{m0}-{m1}", **fmt(p_, lo, hi), "n_cells": len(cis)})
                if scope != "pooled":
                    continue
                for ci in cis:                                     # per-cell estimate and CI
                    p_, lo, hi, _ = bc.summarize([g(ci, m0, a, b, strat) - g(ci, m1, a, b, strat)])
                    c = cells[ci]
                    diff_rows.append({"scope": f"cell:{c['heldout']}_s{c['severity']}", "gain": f"{b}-{a}",
                                      "margin_stratified": strat, "diff": f"{m0}-{m1}", **fmt(p_, lo, hi),
                                      "n_cells": 1})
        if "D" in cells[0]["scores"].get("snn", {}):
            for strat in (False, True):
                p_, lo, hi, _ = bc.summarize([g(ci, "snn", "C", "D", strat) for ci in cis])
                gain_rows.append({"scope": scope, "model": "snn", "gain": "D-C", "margin_stratified": strat,
                                  **fmt(p_, lo, hi), "n_cells": len(cis)})
    gains = pd.DataFrame(gain_rows)
    diffs = pd.DataFrame(diff_rows)
    gains.to_csv(res / "gains.csv", index=False)
    diffs.to_csv(res / "gain_diff.csv", index=False)
    log(f"gains done [{time.time()-t_start:.0f}s]")

    # ---- 3. sparse monitors: pooled AUROC, gain retention, paired comparisons
    def sv(ci, t, meth, k, strat):
        n = sp["n_random"] if meth == "random" else 1
        return np.nanmean(np.stack([bc.vec(ci, t, f"sparse_{meth}_{k}_{j}", strat) for j in range(n)]), 0)

    all_ci = idx_of["pooled"]
    sp_rows, cmp_rows = [], []
    for t in cfg["models"]:
        for strat in (False, True):
            aA = bc.summarize([bc.vec(ci, t, "A", strat) for ci in all_ci])[0]
            aB = bc.summarize([bc.vec(ci, t, "B", strat) for ci in all_ci])[0]
            for k in ks:
                for meth in ("l1", "univariate", "random"):
                    v, lo, hi, _ = bc.summarize([sv(ci, t, meth, k, strat) for ci in all_ci])
                    sp_rows.append({"model": t, "margin_stratified": strat, "k": k, "fraction": k / n_ch,
                                    "method": meth, "auroc": v, "lo": lo, "hi": hi, "auroc_A": aA, "auroc_B": aB,
                                    "gain_retained": (v - aA) / (aB - aA) if aB != aA else np.nan})
                for a_, b_ in (("l1", "random"), ("l1", "univariate"), ("l1", "full_B")):
                    vecs = [sv(ci, t, a_, k, strat) - (bc.vec(ci, t, "B", strat) if b_ == "full_B"
                                                       else sv(ci, t, b_, k, strat)) for ci in all_ci]
                    p_, lo, hi, _ = bc.summarize(vecs)
                    cmp_rows.append({"model": t, "margin_stratified": strat, "k": k, "a": a_, "b": b_,
                                     **fmt(p_, lo, hi)})
    spd = pd.DataFrame(sp_rows)
    spd.to_csv(res / "sparse.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(res / "sparse_compare.csv", index=False)
    log(f"sparse done [{time.time()-t_start:.0f}s]")

    # ---- 4. exploratory screening rules (fixed before the run; not significance tests)
    per_h = diffs[(diffs.gain == "B-A") & diffs.scope.isin(cfg["run"]["corruptions"])]
    dD = gains[(gains.gain == "D-C") & gains.scope.isin(cfg["run"]["corruptions"]) & gains.margin_stratified]
    k_ret = max(1, int(round(sp["retention_fraction"] * n_ch)))
    k_ret = min(ks, key=lambda k: abs(k - k_ret))
    ret = spd[(spd.model == m0) & (spd.k == k_ret) & (spd.method == "l1") & ~spd.margin_stratified]
    l1r = pd.DataFrame(cmp_rows)
    l1r = l1r[(l1r.k == k_ret) & (l1r.a == "l1") & (l1r.b == "random") & ~l1r.margin_stratified]
    decision = {
        "note": "Exploratory screening rules written before the run. They involve many comparisons "
                "without multiple-testing correction and are not significance tests. A positive SNN-ANN "
                "difference would not by itself show a spiking-specific mechanism.",
        "snn_specific_rule": "B-A gain difference (SNN-ANN) CI > 0 in >= 3 of 6 held-out corruptions "
                             "(plain or margin-stratified), or D-C >= 0.01 margin-stratified in >= 2 of 6",
        "n_heldout_diff_ci_positive_plain": int(((per_h.lo > 0) & ~per_h.margin_stratified).sum()),
        "n_heldout_diff_ci_positive_strat": int(((per_h.lo > 0) & per_h.margin_stratified).sum()),
        "n_heldout_D_minus_C_ge_0.01_strat": int((dD.estimate >= 0.01).sum()),
        "sparse_rule": f"L1 monitor with {k_ret} channels ({k_ret/n_ch:.0%}) keeps >= 90% of the B-A gain "
                       "(pooled, plain AUROC) and beats random subsets (pooled CI > 0), per model",
        "sparse": {t: {"gain_retained": float(spd[(spd.model == t) & (spd.k == k_ret) & (spd.method == "l1")
                                                  & ~spd.margin_stratified].gain_retained.iloc[0]),
                       "l1_minus_random": l1r[l1r.model == t][["estimate", "lo", "hi"]].iloc[0].to_dict()}
                   for t in cfg["models"]},
        "runtime_s": time.time() - t_start,
    }
    pos = max(decision["n_heldout_diff_ci_positive_plain"], decision["n_heldout_diff_ci_positive_strat"])
    decision["snn_specific_screen"] = bool(pos >= 3 or decision["n_heldout_D_minus_C_ge_0.01_strat"] >= 2)
    decision["sparse_screen"] = {t: bool(v["gain_retained"] >= 0.9 and v["l1_minus_random"]["lo"] > 0)
                                 for t, v in decision["sparse"].items()}
    json.dump(decision, open(res / "decision.json", "w"), indent=2, default=float)
    pooled = diffs[diffs.scope == "pooled"]
    log("pooled SNN-ANN gain differences:\n" + pooled[["gain", "margin_stratified", "estimate", "lo", "hi"]]
        .round(4).to_string(index=False))
    log("pooled gains:\n" + gains[gains.scope == "pooled"][["model", "gain", "margin_stratified", "estimate", "lo", "hi"]]
        .round(4).to_string(index=False))
    log(f"screens: snn_specific={decision['snn_specific_screen']} sparse={decision['sparse_screen']} "
        f"[{time.time()-t_start:.0f}s]")


if __name__ == "__main__":
    main()
