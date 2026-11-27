"""Experiment A analysis (CPU): effect of spike-timing interventions on classification.

    python analyze_interventions.py --stage pilot

Per T, site, condition and intervention family (averaged over its seeds/permutations):
  accuracy before/after, prediction-flip rate (all / initially correct / initially wrong),
  change of the true-class margin, spread across seeds, count-preservation error.
Paired comparisons on the same images (bootstrap over images):
  uniform_perm - clean_resample   damage from shuffled (unseen) patterns beyond realistic timing
  clean_resample - same_count_swap   damage from breaking correlations / own timing
  count_change - uniform_perm     counts vs timing at the same perturbation size
Margin-matched corrupted - clean flip-rate difference (deciles of the baseline margin).
Null sites: max |logit change| must be ~0.
Writes results/<stage>/interventions*.csv and pilot_decision_A.json.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tslib.common import add_common_args, dump_json, load_cfg

FAMILY_ORDER = ["identity", "same_count_swap", "shared_perm", "clean_resample", "uniform_perm",
                "circular_shift", "even_spread", "count_change"]


def true_margin(logits, y):
    zt = logits[np.arange(len(y)), y]
    other = logits.copy()
    other[np.arange(len(y)), y] = -np.inf
    return zt - other.max(1)


def boot_mean_ci(v, n, rng):
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return np.nan, np.nan
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def per_image(d):
    """Per-image, per-family arrays from one interventions npz."""
    y = d["labels"]
    base = d["base_logits"]
    p0 = base.argmax(1)
    m0 = true_margin(base, y)
    fam = d["run_family"]
    out = {"y": y, "p0": p0, "m0": m0, "correct0": p0 == y, "fam": {}}
    for f in dict.fromkeys(fam):
        rs = np.where(fam == f)[0]
        lg = d["logits"][rs]                                           # [R, N, 10]
        pr = lg.argmax(2)
        flips = (pr != p0[None]).astype(float)
        acc_runs = (pr == y[None]).mean(1)
        dm = np.stack([true_margin(l, y) for l in lg]) - m0[None]
        out["fam"][f] = {"flip": flips.mean(0), "flip_runs": flips.mean(1), "acc_runs": acc_runs,
                         "dmargin": dm.mean(0), "count_err": float(d["count_err"][rs].max()),
                         "l1": float(d["l1"][rs].mean()), "n_fallback": int(d["n_fallback"][rs].sum()),
                         "n_runs": len(rs), "max_abs_dlogit": float(np.abs(lg - base[None]).max())}
    return out


def margin_matched_diff(clean, corr, fam, n_strata, n_boot, rng):
    """sum_d w_d (flip_corr_d - flip_clean_d), w_d = share of corrupted images in margin
    decile d (deciles of the pooled baseline margins). Deciles without clean images are
    dropped (reported as coverage)."""
    m_all = np.concatenate([clean["m0"], corr["m0"]])
    edges = np.unique(np.quantile(m_all, np.linspace(0, 1, n_strata + 1)[1:-1]))
    sc, sk = np.searchsorted(edges, clean["m0"]), np.searchsorted(edges, corr["m0"])
    fc, fk = clean["fam"][fam]["flip"], corr["fam"][fam]["flip"]

    def stat(ic, ik):
        num, w, cov = 0.0, 0.0, 0
        for g in np.unique(sk[ik]):
            a, b = ik[sk[ik] == g], ic[sc[ic] == g]
            if len(b) == 0:
                continue
            num += len(a) * (fk[a].mean() - fc[b].mean())
            w += len(a)
            cov += len(a)
        return (num / w if w else np.nan), cov / len(ik)

    point, coverage = stat(np.arange(len(fc)), np.arange(len(fk)))
    bs = [stat(rng.integers(0, len(fc), len(fc)), rng.integers(0, len(fk), len(fk)))[0]
          for _ in range(n_boot)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return point, lo, hi, coverage


def main():
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_cfg(args)
    cache, res = Path(cfg["resolved"]["cache"]), Path(cfg["resolved"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    nb = cfg["bootstrap"]["n"]
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    rows, comp_rows, mm_rows, null_rows = [], [], [], []
    decision = {"criteria": "go if, at an upstream site, the clean flip rate under clean_resample "
                            "or uniform_perm has a 95% CI excluding 0 AND differs from "
                            "same_count_swap (CI of the difference excludes 0), OR a margin-matched "
                            "corrupted-minus-clean flip difference has a CI excluding 0.",
                "evidence": []}
    sites = cfg["interventions"]["sites"]
    null_sites = cfg["interventions"]["null_sites"]
    for T in cfg["run"]["T"]:
        root = cache / f"T{T}" / "interv"
        if not root.exists():
            continue
        conds = sorted(p.name for p in root.iterdir() if p.is_dir())
        for site in sites + null_sites:
            data = {c: per_image(np.load(root / c / f"{site}.npz")) for c in conds
                    if (root / c / f"{site}.npz").exists()}
            for c, pi in data.items():
                acc0 = float(pi["correct0"].mean())
                for f in [f for f in FAMILY_ORDER if f in pi["fam"]]:
                    q = pi["fam"][f]
                    lo, hi = boot_mean_ci(q["flip"], nb, rng)
                    row = {"T": T, "site": site, "null_site": site in null_sites, "condition": c,
                           "family": f, "n_runs": q["n_runs"], "n_images": len(pi["y"]),
                           "acc_before": acc0, "acc_after": float(q["acc_runs"].mean()),
                           "acc_after_sd_runs": float(q["acc_runs"].std()),
                           "flip": float(q["flip"].mean()), "flip_lo": lo, "flip_hi": hi,
                           "flip_sd_runs": float(q["flip_runs"].std()),
                           "flip_initially_correct": float(q["flip"][pi["correct0"]].mean()),
                           "flip_initially_wrong": float(q["flip"][~pi["correct0"]].mean())
                           if (~pi["correct0"]).any() else np.nan,
                           "dmargin_mean": float(q["dmargin"].mean()),
                           "dmargin_correct": float(q["dmargin"][pi["correct0"]].mean()),
                           "count_err_max": q["count_err"], "mean_l1_per_image": q["l1"],
                           "n_fallback": q["n_fallback"], "max_abs_dlogit": q["max_abs_dlogit"]}
                    rows.append(row)
                    if site in null_sites:
                        null_rows.append({k: row[k] for k in ["T", "site", "condition", "family",
                                                               "max_abs_dlogit", "flip"]})
                if site in null_sites:
                    continue
                for a, b in [("uniform_perm", "clean_resample"), ("clean_resample", "same_count_swap"),
                             ("uniform_perm", "same_count_swap"), ("shared_perm", "same_count_swap"),
                             ("count_change", "uniform_perm")]:
                    if a in pi["fam"] and b in pi["fam"]:
                        dlt = pi["fam"][a]["flip"] - pi["fam"][b]["flip"]
                        lo, hi = boot_mean_ci(dlt, nb, rng)
                        comp_rows.append({"T": T, "site": site, "condition": c, "a": a, "b": b,
                                          "flip_a_minus_b": float(dlt.mean()), "lo": lo, "hi": hi})
                if c == "clean":
                    for f in ["clean_resample", "uniform_perm"]:
                        if f not in pi["fam"]:
                            continue
                        lo, hi = boot_mean_ci(pi["fam"][f]["flip"], nb, rng)
                        dlt = pi["fam"][f]["flip"] - pi["fam"]["same_count_swap"]["flip"]
                        dlo, dhi = boot_mean_ci(dlt, nb, rng)
                        ok = lo > 0 and (dlo > 0 or dhi < 0)
                        decision["evidence"].append({"T": T, "site": site, "family": f, "test": "clean flip",
                                                     "flip": float(pi["fam"][f]["flip"].mean()),
                                                     "ci": [lo, hi], "minus_swap": float(dlt.mean()),
                                                     "minus_swap_ci": [dlo, dhi], "pass": bool(ok)})
            if "clean" in data and site not in null_sites:
                for c in conds:
                    if c == "clean" or c not in data:
                        continue
                    for f in ["shared_perm", "clean_resample", "uniform_perm", "count_change"]:
                        if f not in data[c]["fam"]:
                            continue
                        p, lo, hi, cov = margin_matched_diff(data["clean"], data[c], f,
                                                             cfg["detection"]["n_strata"], nb, rng)
                        mm_rows.append({"T": T, "site": site, "condition": c, "family": f,
                                        "raw_diff": float(data[c]["fam"][f]["flip"].mean()
                                                          - data["clean"]["fam"][f]["flip"].mean()),
                                        "margin_matched_diff": p, "lo": lo, "hi": hi,
                                        "coverage": cov})
                        if f != "count_change":
                            decision["evidence"].append({"T": T, "site": site, "family": f, "condition": c,
                                                         "test": "margin-matched corrupted-clean",
                                                         "diff": p, "ci": [lo, hi],
                                                         "pass": bool(lo > 0 or hi < 0)})
    pd.DataFrame(rows).to_csv(res / "interventions.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(res / "interventions_comparisons.csv", index=False)
    pd.DataFrame(mm_rows).to_csv(res / "interventions_margin_matched.csv", index=False)
    pd.DataFrame(null_rows).to_csv(res / "interventions_null_sites.csv", index=False)
    decision["go"] = any(e["pass"] for e in decision["evidence"])
    dump_json(decision, res / "pilot_decision_A.json")
    df = pd.DataFrame(rows)
    if len(df):
        show = df[(df.condition == "clean") & ~df.null_site]
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(show[["T", "site", "family", "acc_before", "acc_after", "flip", "flip_lo", "flip_hi",
                        "count_err_max"]].round(4).to_string(index=False))
        nd = pd.DataFrame(null_rows)
        print("null sites: max |dlogit| =", float(nd.max_abs_dlogit.max()) if len(nd) else None)
    print("Experiment A go:", decision["go"])


if __name__ == "__main__":
    main()
