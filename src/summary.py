"""Write results/summary.md from the CSVs analyze.py produced.

Every number in the summary is read from a results CSV (the file is named next
to each table); nothing is typed in by hand. Verdict rules are the thresholds
from the experiment spec, made precise where the spec was qualitative:

  H1 per condition: k90/D < 0.1 AND held-out energy at k=10 >= 1.5x the larger
     null (random, clean-PCA) AND at least 0.10 above it ("clearly above").
  H2 per condition: energy fraction in W < 2x chance AND > 50% of principal
     angles above 60 degrees.
  Per model: HOLDS if >= 80% of conditions pass, FAILS if <= 20%, else MIXED.
  H3: union projection, k chosen on the FIT split (largest mean gain at
     severity >= 3 with fit clean drop < 1 point); HOLDS if on the EVAL split the
     mean gain over severity >= 3 conditions is >= 2 points and the clean drop
     is < 1 point.
  H4: slope of each metric vs t (penultimate, severity 3, cumulative rate);
     "consistent" if the sign is the same for every corruption.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY = "penultimate"


def _verdict(frac):
    if np.isnan(frac):
        return "NO DATA"
    return "HOLDS" if frac >= 0.8 else ("FAILS" if frac <= 0.2 else "MIXED")


def _pivot(df, model, layer, metrics, k=None):
    d = df[(df.model == model) & (df.layer == layer) & df.metric.isin(metrics) & (df.severity > 0)]
    if k is not None:
        d = d[(d.k == k) | d.k.isna()]
    return d.pivot_table(index=["corruption", "severity"], columns="metric", values="value")


def h1_table(geo, model, layer):
    p = _pivot(geo, model, layer, ["k90", "k90_over_D", "participation_ratio", "D"])
    q = _pivot(geo, model, layer, ["heldout_energy", "null_random_energy", "null_cleanpca_energy"], k=10)
    t = p.join(q)
    null = t[["null_random_energy", "null_cleanpca_energy"]].max(axis=1)
    t["pass"] = (t.k90_over_D < 0.1) & (t.heldout_energy >= 1.5 * null) & (t.heldout_energy - null >= 0.10)
    return t


def h2_table(geo, model, layer):
    t = _pivot(geo, model, layer, ["energy_frac_W", "energy_frac_W_chance", "angle_frac_gt60",
                                   "angle_median", "angle_min", "angle_k_used",
                                   "null_random_angle_median", "angles_forced_zero_by_dimension"])
    t["pass"] = (t.energy_frac_W < 2 * t.energy_frac_W_chance) & (t.angle_frac_gt60 > 0.5)
    return t


def h3(dfn, model):
    d = dfn[dfn.model == model]

    def get(metric, c=None, s=None, k=None):
        x = d[d.metric == metric]
        if c is not None:
            x = x[(x.corruption == c) & (x.severity == s)]
        if k is not None:
            x = x[x.k == k]
        return x

    conds = sorted({(r.corruption, r.severity) for r in get("acc_corrupted").itertuples()})
    hi = [(c, s) for c, s in conds if s >= 3]
    ks = sorted(get("acc_corrupted_after_union").k.dropna().unique())
    clean_fit = get("acc_clean_fit").value.iloc[0]
    clean_eval = get("acc_clean").value.iloc[0]
    sel = []
    for k in ks:
        gain_fit = np.mean([get("acc_corrupted_after_union_fit", c, s, k).value.iloc[0]
                            - get("acc_corrupted_fit", c, s).value.iloc[0] for c, s in hi]) if hi else np.nan
        drop_fit = clean_fit - get("acc_clean_after_union_fit", k=k).value.iloc[0]
        sel.append((k, gain_fit, drop_fit))
    ok = [x for x in sel if x[2] < 0.01]
    k_sel = max(ok, key=lambda x: x[1])[0] if ok else min(sel, key=lambda x: x[2])[0]
    rows = []
    for c, s in conds:
        before = get("acc_corrupted", c, s).value.iloc[0]
        rows.append(dict(corruption=c, severity=s, acc_corrupted=before,
                         union=get("acc_corrupted_after_union", c, s, k_sel).value.iloc[0],
                         oracle=get("acc_corrupted_after_oracle", c, s, k_sel).value.iloc[0],
                         random=get("acc_corrupted_after_random", c, s, k_sel).value.iloc[0]))
    t = pd.DataFrame(rows).set_index(["corruption", "severity"])
    t["union_gain"] = t.union - t.acc_corrupted
    clean_after = get("acc_clean_after_union", k=k_sel).value.iloc[0]
    hi_t = t[t.index.get_level_values("severity") >= 3]
    mean_gain = hi_t.union_gain.mean() if len(hi_t) else np.nan
    holds = (mean_gain >= 0.02) and (clean_eval - clean_after < 0.01)
    return dict(table=t, k=k_sel, selection=sel, clean=clean_eval, clean_after=clean_after,
                mean_gain_hi=mean_gain, n_hi_pass=int((hi_t.union_gain >= 0.02).sum()), n_hi=len(hi_t),
                verdict="HOLDS" if holds else "FAILS")


def h4(tmp, model):
    d = tmp[(tmp.model == model) & (tmp.layer == PRIMARY) & (tmp.kind == "cumrate") & (tmp.severity == 3)]
    if d.empty:
        d = tmp[(tmp.model == model) & (tmp.layer == PRIMARY) & (tmp.kind == "cumrate") & (tmp.severity > 0)]
        d = d[d.severity == d.severity.max()]
    out = {}
    for metric in ["rel_magnitude", "energy_frac_W", "heldout_energy", "acc_cumrate"]:
        x = d[d.metric == metric]
        slopes = {c: float(np.polyfit(g.t, g.value, 1)[0]) for c, g in x.groupby("corruption") if len(g) > 1}
        signs = {np.sign(v) for v in slopes.values()}
        out[metric] = dict(slopes=slopes, consistent=len(signs) == 1 and 0 not in signs,
                           severity=int(x.severity.iloc[0]) if len(x) else None)
    return out


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def df_md(df, nd=3):
    df = df.reset_index()
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v, nd) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_summary(cfg, res, models):
    res = Path(res)
    geo = pd.read_csv(res / "geometry.csv")
    dfn = pd.read_csv(res / "defense.csv")
    acc = pd.read_csv(res / "accuracy.csv")
    san = pd.read_csv(res / "sanity.csv")
    tmp = pd.read_csv(res / "temporal.csv") if (res / "temporal.csv").exists() else None
    L = ["# Results summary", ""]
    if cfg.get("smoke_mode"):
        L += ["**SMOKE RUN: 2 epochs, 500 test images, 2 corruptions, 2 severities. These numbers say "
              "nothing about the hypotheses; they only show that every stage runs.**", ""]
    L += ["Generated automatically by `analyze.py` (src/summary.py) from the CSVs in this folder. "
          "Verdict rules are documented at the top of src/summary.py.", ""]

    L += ["## Models", ""]
    ck = Path(cfg["paths"]["ckpt_dir"])
    for m in ["ann", "snn"]:
        p = ck / m / "train_summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            L.append(f"- {m}: final test acc {fmt(s['final_test_acc'], 4)}, best {fmt(s['best_test_acc'], 4)}, "
                     f"{s['epochs']} epochs (`{p.as_posix()}`)")
            if s.get("firing_rates_test"):
                L.append(f"  - LIF firing rate per stage on test set: "
                         + ", ".join(f"{k} {v:.4f}" for k, v in s["firing_rates_test"].items()))
    for m in models:
        a = acc[(acc.model == m) & (acc.metric == "model_acc_eval") & (acc.corruption == "clean")].value
        L.append(f"- {m}: clean accuracy on eval split {fmt(a.iloc[0], 4)} (`results/accuracy.csv`)")
    L.append("")

    L += ["## Sanity checks (`results/sanity.csv`)", ""]
    for metric in ["clean_vs_clean_rel_magnitude", "clean_repeat_max_abs_diff", "max_fc_vs_logits_err",
                   "fc_reproduces_model_preds_eval", "dims_dropped_zero_variance"]:
        x = san[san.metric == metric]
        if len(x):
            L.append(f"- {metric}: " + ", ".join(f"{r.model}/{r.layer}={r.value:.3g}" for r in x.itertuples()))
    x = san[san.metric == "eval_disp_energy_frac_in_dropped_dims"]
    if len(x):
        L.append(f"- max displacement energy fraction in dropped (clean-constant) dims: {x.value.max():.3g}")
    L.append("")

    for m in models:
        L += [f"## {m}", ""]
        L += ["### H1 (low rank) — `results/geometry.csv`", ""]
        for layer in ["layer1", "layer2", "layer3", PRIMARY]:
            t = h1_table(geo, m, layer)
            L.append(f"- {layer}: {int(t['pass'].sum())}/{len(t)} conditions pass -> **{_verdict(t['pass'].mean())}**"
                     f" (median k90/D {fmt(t.k90_over_D.median())}, median held-out@10 {fmt(t.heldout_energy.median())}"
                     f" vs clean-PCA null {fmt(t.null_cleanpca_energy.median())}, random {fmt(t.null_random_energy.median())})")
        L += ["", f"Penultimate detail:", "", df_md(h1_table(geo, m, PRIMARY)), ""]

        L += ["### H2 (orthogonality to LDA subspace W) — `results/geometry.csv`", ""]
        for layer in ["layer1", "layer2", "layer3", PRIMARY]:
            t = h2_table(geo, m, layer)
            L.append(f"- {layer}: {int(t['pass'].sum())}/{len(t)} pass -> **{_verdict(t['pass'].mean())}** "
                     f"(median energy frac in W {fmt(t.energy_frac_W.median())} vs chance {fmt(t.energy_frac_W_chance.median())}; "
                     f"median angle {fmt(t.angle_median.median(), 1)} deg vs random-subspace {fmt(t.null_random_angle_median.median(), 1)})")
        L += ["", "Penultimate detail:", "", df_md(h2_table(geo, m, PRIMARY)), ""]

        r = h3(dfn, m)
        L += ["### H3 (projection defense, penultimate) — `results/defense.csv`", "",
              f"k selected on fit split: **{int(r['k'])}** (fit-split gain/clean-drop per k: "
              + ", ".join(f"k={int(k)}: {fmt(g)}/{fmt(dr)}" for k, g, dr in r["selection"]) + ")",
              f"- clean acc {fmt(r['clean'], 4)} -> {fmt(r['clean_after'], 4)} after union projection "
              f"(drop {fmt(100 * (r['clean'] - r['clean_after']), 2)} points)",
              f"- mean union gain at severity >= 3: {fmt(100 * r['mean_gain_hi'], 2)} points "
              f"({r['n_hi_pass']}/{r['n_hi']} conditions gain >= 2 points) -> **{r['verdict']}**", "",
              df_md(r["table"], 4), ""]

        if tmp is not None and m.startswith("snn") and (tmp.model == m).any():
            res4 = h4(tmp, m)
            L += ["### H4 (temporal, penultimate, cumulative rate) — `results/temporal.csv`", ""]
            for metric, v in res4.items():
                L.append(f"- {metric} (severity {v['severity']}): slope vs t per corruption "
                         + ", ".join(f"{c} {s:+.4f}" for c, s in v["slopes"].items())
                         + f" -> {'consistent sign' if v['consistent'] else 'no consistent trend'}")
            L.append("")
    L += ["## Interpretation", "", "_To be written by hand after reviewing the full run; must only cite the numbers above._", ""]
    (res / "summary.md").write_text("\n".join(L), encoding="utf-8")
