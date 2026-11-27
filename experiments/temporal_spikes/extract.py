"""GPU pass: timing-capacity statistics, clean reference tables, and per-image features.

    python extract.py --stage pilot --T 4 [--checkpoint PATH] [--data-root DIR] [--corrupted-dir DIR]

Steps (each is skipped if its output already exists, so the script resumes):
  0. Re-measure clean test accuracy of the checkpoint on all 10k test images and fail if
     it differs from the value measured by the original experiment (wrong checkpoint).
  1. Reference pass over clean TRAINING images:
       - per-channel train-code histograms at every binary LIF site (20) and both integer
         sites -> capacity on clean data and the clean-resampling tables of Experiment A
       - channel moments at the detector sites -> PCA basis for lag-1 covariance
  2. Second reference pass: per-image detector features of the same training images
     (clean reference distribution for the clean-only detector).
  3. Every condition (clean + corruptions x severities) on the fit+eval test images:
       per-image detector features, logits, per-timestep logits, time-averaged penultimate
       rate, and per-condition capacity statistics at every site. Raw spikes for the first
       few eval images (raster figures).

Layout: <cache>/T{T}/ref/*.npz, <cache>/T{T}/cond/<condition>/*.npz, <cache>/T{T}/meta.json
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from tslib.common import (GPUNormalizer, Timer, add_common_args, batches, build_snn, cond_name,
                          condition_images, conditions, dump_json, get_device, get_logger,
                          load_cfg, load_test_set, load_train_set, read_json, select_indices,
                          set_determinism, write_meta)
from tslib.stats import MomentAccumulator, SiteAccumulator, detector_features
from src.neurons import MultiStepLIF


class Recorder:
    """Forward hooks on every LIF module, on the inputs of layer3/layer4 (integer sites)
    and on the output of layer4 (per-timestep logits). Statistics are computed inside the
    hooks so no full spike tensor is kept."""

    def __init__(self, model, cfg, T, mode, pca=None, raw_sites=()):
        self.model, self.cfg, self.T, self.mode = model, cfg, T, mode
        self.lif = {n: m for n, m in model.named_modules() if isinstance(m, MultiStepLIF)}
        self.int_sites = dict(cfg["integer_sites"])
        self.det = list(cfg["detector_sites"])
        self.pca = pca or {}
        self.raw_sites = set(raw_sites)
        per_ch_int = mode == "ref1"
        self.acc = {}
        if mode in ("ref1", "cond"):
            self.acc.update({n: SiteAccumulator(T, 2, per_channel=True) for n in self.lif})
            self.acc.update({n: SiteAccumulator(T, v + 1, per_channel=per_ch_int)
                             for n, v in self.int_sites.items()})
        self.mom = {n: MomentAccumulator() for n in self.det} if mode == "ref1" else {}
        self.batch = {}
        self.handles = []

    def _site(self, name, z):
        if name in self.acc:
            self.acc[name].update(z)
        if name in self.mom:
            self.mom[name].update(z)
        if self.mode in ("ref2", "cond") and name in self.det:
            f = detector_features(z, self.pca.get(name), per_channel_timing=(name == self.det[-1]))
            for k, v in f.items():
                self.batch[f"{name}/{k}"] = v
        if name in self.raw_sites:
            self.batch[f"raw/{name}"] = z.transpose(0, 1).to(torch.uint8).cpu().numpy()

    def __enter__(self):
        for n, m in self.lif.items():
            self.handles.append(m.register_forward_hook(lambda mod, i, o, n=n: self._site(n, o)))
        for n in self.int_sites:
            stage = getattr(self.model, n.split("_")[0])
            self.handles.append(stage.register_forward_pre_hook(lambda mod, a, n=n: self._site(n, a[0])))

        def head(mod, i, out):
            lt = self.model.fc(self.model.pool(out))                     # [T, N, 10]
            self.batch["logits_t"] = lt.transpose(0, 1).float().cpu().numpy()
            self.batch["penult_rate"] = self.model.pool(out).mean(0).half().cpu().numpy()
        self.handles.append(self.model.layer4.register_forward_hook(head))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_images(model, rec, x_u8, dev, bs, log_every=None, log=None):
    norm = GPUNormalizer(dev)
    out = {}
    max_err = 0.0
    with rec:
        for sl in batches(len(x_u8), bs):
            rec.batch = {}
            logits = model(norm(torch.from_numpy(x_u8[sl]).to(dev)))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite logits")
            b = rec.batch
            if "logits_t" in b:
                max_err = max(max_err, float(np.abs(b["logits_t"].mean(1) - logits.cpu().numpy()).max()))
            b["logits"] = logits.float().cpu().numpy()
            for k, v in b.items():
                out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}, max_err


def accuracy_check(cfg, model, T, dev, log):
    if cfg["random_init"]:
        return {"skipped": "random-init"}
    x, y = load_test_set(cfg)
    norm = GPUNormalizer(dev)
    correct = 0
    with torch.no_grad():
        for sl in batches(len(x), cfg["run"]["batch_size"]):
            correct += int((model(norm(torch.from_numpy(x[sl]).to(dev))).argmax(1).cpu().numpy()
                            == y[sl]).sum())
    acc = correct / len(y)
    exp = cfg["expected_clean_acc"].get(T)
    res = {"clean_test_acc": acc, "expected": exp, "n": len(y)}
    log.info("clean test accuracy (10k) at T=%d: %.4f (original run: %s)", T, acc, exp)
    if exp is not None and abs(acc - exp) > cfg["accuracy_tolerance"]:
        raise RuntimeError(f"accuracy {acc:.4f} differs from {exp} by more than "
                           f"{cfg['accuracy_tolerance']}: wrong checkpoint?")
    return res


def save_npz(path, d):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(path)                     # atomic: a crash never leaves a half-written file


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--T", type=int, required=True)
    args = ap.parse_args()
    cfg = load_cfg(args)
    T = args.T
    root = Path(cfg["resolved"]["cache"]) / f"T{T}"
    log = get_logger(f"extract_T{T}", cfg["resolved"]["logs"])
    set_determinism()
    dev = get_device()
    model, info = build_snn(cfg, T, dev)
    bs = cfg["run"]["batch_size"]
    timer = Timer()
    log.info("model: %s", info)

    meta_path = root / "meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    if "accuracy_check" not in meta:
        meta["accuracy_check"] = accuracy_check(cfg, model, T, dev, log)
    meta.update({"model": info, "stage": cfg["stage"], "resolved_paths": cfg["resolved"],
                 "run": cfg["run"]})
    write_meta(meta_path, meta)

    idx = select_indices(cfg)
    ref_dir = root / "ref"
    # ---- 1. reference statistics on clean training images
    if not (ref_dir / "ref_stats.npz").exists():
        xtr, ytr = load_train_set(cfg)
        rec = Recorder(model, cfg, T, "ref1")
        run_images(model, rec, xtr[idx["ref"]], dev, bs)
        d = {}
        for n, a in rec.acc.items():
            d.update(a.to_dict(n))
        save_npz(ref_dir / "ref_stats.npz", d)
        pca = {}
        for n, m in rec.mom.items():
            mu, P, ev = m.basis(cfg["lag_pca_dim"])
            pca[f"{n}/mu"], pca[f"{n}/P"], pca[f"{n}/eig"] = mu.cpu().numpy(), P.cpu().numpy(), ev.cpu().numpy()
        save_npz(ref_dir / "pca.npz", pca)
        log.info("reference stats done (%d train images) [%.0fs]", len(idx["ref"]), timer())
    pz = np.load(ref_dir / "pca.npz")
    pca = {n: (torch.from_numpy(pz[f"{n}/mu"]).to(dev), torch.from_numpy(pz[f"{n}/P"]).to(dev))
           for n in cfg["detector_sites"]}

    # ---- 2. per-image features of the reference images
    if not (ref_dir / "features.npz").exists():
        xtr, ytr = load_train_set(cfg)
        rec = Recorder(model, cfg, T, "ref2", pca=pca)
        out, err = run_images(model, rec, xtr[idx["ref"]], dev, bs)
        out.update({"labels": ytr[idx["ref"]], "indices": idx["ref"]})
        save_npz(ref_dir / "features.npz", out)
        log.info("reference features done, max |mean_t logits_t - logits| = %.2e [%.0fs]", err, timer())

    # ---- 3. test conditions
    x_clean, y = load_test_set(cfg)
    rows = np.concatenate([idx["fit"], idx["eval"]])
    split = np.array(["fit"] * len(idx["fit"]) + ["eval"] * len(idx["eval"]))
    order = np.argsort(rows)
    rows, split = rows[order], split[order]
    raster_idx = idx["eval"][: cfg["raster_images"]]
    timings = meta.get("timings", {})
    for corr, sev in conditions(cfg):
        name = cond_name(corr, sev)
        cdir = root / "cond" / name
        if (cdir / "features.npz").exists():
            log.info("exists, skipping %s", name)
            continue
        t0 = timer()
        x = condition_images(cfg, corr, sev, clean_x=x_clean)
        assert len(x) == len(y)
        rec = Recorder(model, cfg, T, "cond", pca=pca)
        out, err = run_images(model, rec, x[rows], dev, bs)
        out.update({"labels": y[rows], "indices": rows, "split": split})
        acc = float((out["logits"].argmax(1) == y[rows]).mean())
        cap = {}
        for n, a in rec.acc.items():
            cap.update(a.to_dict(n))
        rrec = Recorder(model, cfg, T, "raw", raw_sites=["layer3.1.sn2", "layer4.0.sn1", "layer4.1.sn2"])
        raw, _ = run_images(model, rrec, x[raster_idx], dev, bs)
        raw = {k: v for k, v in raw.items() if k.startswith("raw/")}
        raw.update({"indices": raster_idx, "labels": y[raster_idx]})
        save_npz(cdir / "capacity.npz", cap)
        save_npz(cdir / "raster.npz", raw)
        save_npz(cdir / "features.npz", out)           # written last: marks the condition done
        timings[name] = {"seconds": timer() - t0, "n_images": int(len(rows)), "acc": acc,
                         "max_logit_t_err": err}
        log.info("%s: acc %.4f on %d images, logit check %.1e [%.0fs]", name, acc, len(rows), err,
                 timer() - t0)
        meta["timings"] = timings
        dump_json(meta, meta_path)
    log.info("done [%.0fs]", timer())


if __name__ == "__main__":
    main()
