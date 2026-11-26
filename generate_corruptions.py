"""Generate the corrupted CIFAR-10 test sets once, with a fixed seed.

    python generate_corruptions.py                 # all 6 corruptions x 5 severities
    python generate_corruptions.py --cifar10c      # also download CIFAR-10-C and cross-check

Output: data/corrupted/{corruption}/{severity}.npy, uint8 [10000,32,32,3], same
order as the torchvision test set, plus manifest.json with checksums. Both
models read these exact files, so they see identical corrupted images.
"""
import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.corruptions import PARAMS, corrupt
from src.data import corrupted_path, load_cifar10
from src.utils import dump_json, get_logger, load_config

CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_cifar10c(dest_dir, names, log):
    """Download the official tar (~2.9 GB) and extract only the files we need.

    Members are written by name into dest_dir; nothing else in the archive is
    extracted, so a malformed archive cannot write outside dest_dir.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"CIFAR-10-C/{n}.npy" for n in names} | {"CIFAR-10-C/labels.npy"}
    if all((dest_dir / Path(w).name).exists() for w in wanted):
        return True
    tar_path = dest_dir / "CIFAR-10-C.tar"
    if not tar_path.exists():
        log.info("downloading CIFAR-10-C from %s", CIFAR10C_URL)
        tmp = tar_path.with_suffix(".part")
        with urllib.request.urlopen(CIFAR10C_URL, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        tmp.replace(tar_path)
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if member.name in wanted and member.isfile():
                with tf.extractfile(member) as src, open(dest_dir / Path(member.name).name, "wb") as dst:
                    dst.write(src.read())
    return all((dest_dir / Path(w).name).exists() for w in wanted)


def crosscheck(cfg, x_clean, y_clean, log):
    names, sevs = cfg["corruptions"]["names"], cfg["corruptions"]["severities"]
    d = Path(cfg["paths"]["cifar10c_dir"])
    try:
        ok = fetch_cifar10c(d, names, log)
    except Exception as e:
        log.warning("CIFAR-10-C download/extract failed (%s); continuing with generated set only", e)
        return
    if not ok:
        log.warning("CIFAR-10-C archive did not contain the expected files")
        return
    labels = np.load(d / "labels.npy", allow_pickle=False)
    rows = [{"corruption": "labels", "severity": 0, "metric": "labels_match_test_order",
             "value": float((labels[:10000] == y_clean).all())}]
    xc = x_clean.astype(np.float64)
    for name in names:
        off_all = np.load(d / f"{name}.npy", allow_pickle=False, mmap_mode="r")
        for s in sevs:
            off = np.asarray(off_all[(s - 1) * 10000: s * 10000]).astype(np.float64)
            mine = np.load(corrupted_path(cfg["paths"]["corrupted_dir"], name, s)).astype(np.float64)
            # Deterministic corruptions should match pixel-wise up to rounding
            # (the original truncates); noise ones can only match in distribution.
            rows += [
                {"corruption": name, "severity": s, "metric": "mean_abs_diff_mine_vs_official",
                 "value": float(np.abs(mine - off).mean())},
                {"corruption": name, "severity": s, "metric": "frac_pixels_diff_gt1",
                 "value": float((np.abs(mine - off) > 1).mean())},
                {"corruption": name, "severity": s, "metric": "rms_change_from_clean_mine",
                 "value": float(np.sqrt(((mine - xc) ** 2).mean()))},
                {"corruption": name, "severity": s, "metric": "rms_change_from_clean_official",
                 "value": float(np.sqrt(((off - xc) ** 2).mean()))},
            ]
    out = Path(cfg["paths"]["results_dir"]) / "cifar10c_crosscheck.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("CIFAR-10-C cross-check written to %s", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--force", action="store_true", help="regenerate even if files exist")
    ap.add_argument("--cifar10c", action="store_true", help="download CIFAR-10-C and cross-check")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke, out_root=args.out_root)
    log = get_logger("generate_corruptions", cfg["paths"]["logs_dir"])

    # Always the full 10k test set: the smoke subset is taken later, at load time.
    x, y = load_cifar10(cfg["paths"]["data_root"], train=False)
    seed = cfg["corruptions"]["seed"]
    out_dir = Path(cfg["paths"]["corrupted_dir"])
    manifest = {"seed": seed, "params": PARAMS, "files": {}}
    for name in cfg["corruptions"]["names"]:
        for s in cfg["corruptions"]["severities"]:
            p = corrupted_path(out_dir, name, s)
            if p.exists() and not args.force:
                log.info("exists, skipping %s", p)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                xc = corrupt(x, name, s, seed)
                assert xc.shape == x.shape and xc.dtype == np.uint8
                np.save(p, xc)
                log.info("%s s%d: mean |change| %.2f levels", name, s,
                         np.abs(xc.astype(np.int16) - x).mean())
            manifest["files"][f"{name}/{s}"] = sha256(p)
    np.save(out_dir / "labels.npy", y)
    dump_json(manifest, out_dir / "manifest.json")
    if args.cifar10c:
        crosscheck(cfg, x, y, log)


if __name__ == "__main__":
    main()
