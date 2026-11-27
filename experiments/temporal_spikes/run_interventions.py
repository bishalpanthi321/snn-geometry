"""Experiment A (GPU): spike-timing interventions at defined sites, rest of the network rerun.

    python run_interventions.py --stage pilot --T 4

Needs <cache>/T{T}/ref/ref_stats.npz from extract.py (clean-resampling tables).

For every condition, site and batch:
  1. baseline logits from the unmodified full forward pass
  2. prefix run to the site (state reset first), then for each intervention the modified
     tensor is passed through the remaining network (state reset again)
  3. identity must reproduce the baseline logits exactly; count-preserving interventions
     must preserve every neuron's count exactly; values must stay in {0..vmax}.
     Any violation stops the run.
Outputs: <cache>/T{T}/interv/<condition>/<site>.npz (resumable per condition x site).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from tslib import interventions as iv
from tslib.common import (GPUNormalizer, Timer, add_common_args, batches, build_snn, cond_name,
                          condition_images, conditions, dump_json, get_device, get_logger,
                          load_cfg, load_test_set, read_json, select_indices, set_determinism)
from tslib.injection import is_binary_site, run_from_site, run_to_site

COUNT_PRESERVING = ["identity", "same_count_swap", "shared_perm", "clean_resample",
                    "uniform_perm", "circular_shift", "even_spread"]


def site_vmax(cfg, site):
    if is_binary_site(site):
        return 1
    if site in cfg["integer_sites"]:
        return cfg["integer_sites"][site]
    return None                                     # layer4_out: no bound needed (null site)


def run_plan(cfg, T, site, is_null, has_table):
    n = cfg["interventions"]["n_seeds"]
    nmax = cfg["interventions"]["n_shared_perms_max"]
    nmax = nmax[T] if isinstance(nmax, dict) else nmax
    perms = iv.all_shared_perms(T, nmax, seed=cfg["seed"])
    plan = [("identity", 0)]
    plan += [("same_count_swap", s) for s in range(n)]
    plan += [("shared_perm", j) for j in range(len(perms))]
    if has_table:
        plan += [("clean_resample", s) for s in range(n)]
    plan += [("uniform_perm", s) for s in range(n)]
    plan += [("circular_shift", s) for s in range(n)]
    plan += [("even_spread", 0)]
    if not is_null:
        plan += [("count_change", s) for s in range(n)]      # after uniform_perm (needs its L1)
    return plan, perms


def seed_for(cfg, family_i, s, batch_i, site_i):
    return (cfg["seed"] * 1_000_003 + family_i * 100_003 + s * 10_007 + batch_i * 101 + site_i) % (2**62)


@torch.no_grad()
def run_site(cfg, model, site, site_i, x_u8, dev, table, T, is_null, log):
    vmax = site_vmax(cfg, site)
    plan, perms = run_plan(cfg, T, site, is_null, table is not None)
    fam_index = {f: i for i, f in enumerate(dict.fromkeys(p[0] for p in plan))}
    norm = GPUNormalizer(dev)
    R = len(plan)
    logits = [[] for _ in range(R)]
    l1 = [[] for _ in range(R)]
    count_err = np.zeros(R)
    n_fb = np.zeros(R, dtype=np.int64)
    base_all = []
    ident_exact, ident_maxdiff = True, 0.0
    for bi, sl in enumerate(batches(len(x_u8), cfg["run"]["batch_size"])):
        xb = norm(torch.from_numpy(x_u8[sl]).to(dev))
        base = model(xb)
        base_all.append(base.cpu().numpy())
        state = run_to_site(model, xb, site)
        z = state["z"]
        if vmax is not None:
            assert float(z.max()) <= vmax and float(z.min()) >= 0, (site, float(z.max()))
        c0 = iv.counts(z)
        uni_l1 = {}
        for r, (fam, s) in enumerate(plan):
            sd = seed_for(cfg, fam_index[fam], s, bi, site_i)
            fb = 0
            if fam == "identity":
                z2 = z
            elif fam == "same_count_swap":
                z2 = iv.same_count_swap(z, sd)
            elif fam == "shared_perm":
                z2 = iv.shared_perm(z, perms[s])
            elif fam == "clean_resample":
                z2, fb = table.sample(z, sd)
            elif fam == "uniform_perm":
                z2 = iv.uniform_perm(z, sd)
            elif fam == "circular_shift":
                z2 = iv.circular_shift(z, sd)
            elif fam == "even_spread":
                z2 = iv.even_spread(z)
            elif fam == "count_change":
                z2 = iv.count_change(z, uni_l1[s], vmax, sd)
            else:
                raise ValueError(fam)
            d = iv.l1_per_image(z, z2)
            if fam == "uniform_perm":
                uni_l1[s] = d
            cerr = float((iv.counts(z2) - c0).abs().max())
            if fam in COUNT_PRESERVING and cerr != 0:
                raise AssertionError(f"{fam} changed spike counts at {site} (max {cerr})")
            if vmax is not None and (float(z2.min()) < 0 or float(z2.max()) > vmax
                                     or not torch.equal(z2, z2.round())):
                raise AssertionError(f"{fam} left the value support at {site}")
            out = run_from_site(model, state, z2)
            if not torch.isfinite(out).all():
                raise FloatingPointError(f"non-finite logits after {fam} at {site}")
            if fam == "identity":
                ident_exact &= bool(torch.equal(out, base))
                ident_maxdiff = max(ident_maxdiff, float((out - base).abs().max()))
            logits[r].append(out.cpu().numpy())
            l1[r].append(d.cpu().numpy())
            count_err[r] = max(count_err[r], cerr)
            n_fb[r] += fb
    if not ident_exact:
        log.warning("identity not bit-exact at %s (max diff %.2e)", site, ident_maxdiff)
    return {
        "run_family": np.array([p[0] for p in plan]), "run_index": np.array([p[1] for p in plan]),
        "shared_perms": np.array(perms), "logits": np.stack([np.concatenate(v) for v in logits]),
        "l1": np.stack([np.concatenate(v) for v in l1]).astype(np.float32),
        "count_err": count_err, "n_fallback": n_fb, "base_logits": np.concatenate(base_all),
        "identity_exact": np.bool_(ident_exact), "identity_max_diff": np.float64(ident_maxdiff),
        "vmax": np.int64(vmax if vmax is not None else -1),
    }


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--T", type=int, default=4)
    args = ap.parse_args()
    cfg = load_cfg(args)
    T = args.T
    root = Path(cfg["resolved"]["cache"]) / f"T{T}"
    log = get_logger(f"interventions_T{T}", cfg["resolved"]["logs"])
    set_determinism()
    dev = get_device()
    model, info = build_snn(cfg, T, dev)
    ref_path = root / "ref" / "ref_stats.npz"
    if not ref_path.exists():
        raise FileNotFoundError(f"{ref_path} missing: run extract.py --T {T} first")
    ref = np.load(ref_path)
    sites = list(cfg["interventions"]["sites"])
    null_sites = list(cfg["interventions"]["null_sites"])
    tables = {}
    for s in sites + null_sites:
        if f"{s}/hist" in ref.files and ref[f"{s}/hist"].ndim == 2:
            tables[s] = iv.ResampleTable(ref[f"{s}/hist"], int(ref[f"{s}/base"]), T, dev)
    idx = select_indices(cfg)["A"]
    x_clean, y = load_test_set(cfg)
    timer = Timer()
    meta_path = root / "interv" / "meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {"model": info, "timings": {}}
    for corr, sev in conditions(cfg, severities=cfg["run"]["A_severities"]):
        name = cond_name(corr, sev)
        x = condition_images(cfg, corr, sev, clean_x=x_clean)[idx]
        for si, site in enumerate(sites + null_sites):
            path = root / "interv" / name / f"{site}.npz"
            if path.exists():
                continue
            t0 = timer()
            res = run_site(cfg, model, site, si, x, dev, tables.get(site), T,
                           site in null_sites, log)
            res.update({"labels": y[idx], "indices": idx})
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **res)
            tmp.replace(path)
            dt = timer() - t0
            meta["timings"][f"{name}/{site}"] = {"seconds": dt, "n_images": len(idx),
                                                 "n_runs": int(len(res["run_family"])),
                                                 "identity_exact": bool(res["identity_exact"]),
                                                 "max_fallback": int(res["n_fallback"].max())}
            log.info("%s %s: %d runs x %d images, identity exact=%s [%.0fs]", name, site,
                     len(res["run_family"]), len(idx), bool(res["identity_exact"]), dt)
            dump_json(meta, meta_path)
    log.info("done [%.0fs]", timer())


if __name__ == "__main__":
    main()
