"""Train the ANN or the SNN on CIFAR-10 from scratch.

    python train.py --model ann
    python train.py --model snn --micro-batch 32        # 4 GB GPU
    python train.py --model snn --resume                # continue from last.pt
    python train.py --model snn --max-hours 11.5        # stop cleanly before a session limit

Writes checkpoints/{model}/{last,best,final}.pt and logs/train_{model}.csv.
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import GPUNormalizer, augment_batch, load_cifar10
from src.models import build_model, stage_lifs
from src.utils import dump_json, env_report, get_device, get_logger, load_config, seed_everything


@torch.no_grad()
def evaluate(model, x_u8, y, norm, batch_size, is_snn):
    """Clean test accuracy in fp32, plus per-stage LIF firing rates for the SNN."""
    model.eval()
    groups = stage_lifs(model) if is_snn else {}
    for mods in groups.values():
        for m in mods:
            m.reset_rate()
            m.track_rate = True
    correct = 0
    for i in range(0, len(x_u8), batch_size):
        logits = model(norm(x_u8[i:i + batch_size]))
        correct += (logits.argmax(1) == y[i:i + batch_size]).sum().item()
    rates = {}
    for st, mods in groups.items():
        s = sum(m.spike_sum for m in mods)
        c = sum(m.spike_count for m in mods)
        rates[st] = s / max(c, 1)
        for m in mods:
            m.track_rate = False
    return correct / len(x_u8), rates


def save_ckpt(path, **state):
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic on the same filesystem: a crash never leaves a half file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ann", "snn"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke, out_root=args.out_root)
    tc = cfg["train"]
    if args.micro_batch:
        tc["micro_batch"] = args.micro_batch
    if args.epochs:
        tc["epochs"] = args.epochs
    seed = cfg["seed"]
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True  # speed; makes runs not bit-reproducible

    dev = get_device()
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"]) / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["paths"]["logs_dir"])
    log = get_logger(f"train_{args.model}", log_dir)
    log.info("env: %s", env_report())
    if dev.type != "cuda":
        log.warning("CUDA not available: training on CPU will be extremely slow")

    is_snn = args.model == "snn"
    model = build_model(args.model, cfg).to(dev)
    if is_snn:
        log.info("LIF implementation: %s", model.sn1.impl)
    log.info("params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    x_tr, y_tr = load_cifar10(cfg["paths"]["data_root"], train=True)
    x_te, y_te = load_cifar10(cfg["paths"]["data_root"], train=False)
    if cfg.get("n_train"):
        x_tr, y_tr = x_tr[: cfg["n_train"]], y_tr[: cfg["n_train"]]
    if cfg.get("n_test"):
        x_te, y_te = x_te[: cfg["n_test"]], y_te[: cfg["n_test"]]
    # Whole dataset lives on the GPU as uint8 (150 MB); avoids DataLoader workers on Windows.
    x_tr = torch.from_numpy(x_tr).to(dev)
    y_tr = torch.from_numpy(y_tr).to(dev)
    x_te = torch.from_numpy(x_te).to(dev)
    y_te = torch.from_numpy(y_te).to(dev)
    norm = GPUNormalizer(dev)

    bs = tc["batch_size"]
    mb = tc["micro_batch"] or bs
    assert bs % mb == 0, "batch_size must be a multiple of micro_batch"
    epochs = tc["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = tc["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.CrossEntropyLoss()
    eval_bs = 200 if is_snn else 500

    csv_path = log_dir / f"train_{args.model}.csv"
    start_epoch, best_acc = 0, -1.0
    last_path = ckpt_dir / "last.pt"
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume given but {last_path} does not exist")
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
        if ck["config"]["train"]["epochs"] != epochs:
            raise ValueError("epochs differs from the checkpoint; the cosine schedule would break")
        # Drop CSV rows logged after the checkpoint (crash between log and save).
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df[df.epoch < start_epoch].to_csv(csv_path, index=False)
        log.info("resumed from epoch %d (best %.4f)", ck["epoch"], best_acc)
    elif csv_path.exists():
        csv_path.unlink()  # fresh run: start a fresh log

    dump_json({"config": cfg, "env": env_report(), "model": args.model}, ckpt_dir / "run_info.json")
    n = len(x_tr)
    steps_per_epoch = n // bs  # drop the last partial batch
    t_start = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        # Per-epoch generator: resuming reproduces the same shuffles/augmentations.
        g = torch.Generator().manual_seed(seed * 100003 + epoch)
        perm = torch.randperm(n, generator=g).to(dev)
        tot_loss, tot_correct, tot_seen, bad = 0.0, 0, 0, 0
        for step in range(steps_per_epoch):
            idx = perm[step * bs:(step + 1) * bs]
            xb = norm.normalize(augment_batch(norm.to_float01(x_tr[idx]), g))
            yb = y_tr[idx]
            opt.zero_grad(set_to_none=True)
            for j in range(0, bs, mb):
                xm, ym = xb[j:j + mb], yb[j:j + mb]
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(xm)
                    loss = crit(logits.float(), ym)
                if not torch.isfinite(loss):
                    bad += 1
                scaler.scale(loss * (len(xm) / bs)).backward()
                tot_loss += loss.item() * len(xm)
                tot_correct += (logits.argmax(1) == ym).sum().item()
                tot_seen += len(xm)
            scaler.step(opt)
            scaler.update()
        sched.step()
        if bad > 10:
            raise RuntimeError(f"{bad} non-finite losses in epoch {epoch}: AMP unstable, set train.amp: false")
        train_time = time.time() - t0

        test_acc, rates = evaluate(model, x_te, y_te, norm, eval_bs, is_snn)
        row = {"epoch": epoch, "train_loss": tot_loss / tot_seen, "train_acc": tot_correct / tot_seen,
               "test_acc": test_acc, "lr": opt.param_groups[0]["lr"], "epoch_time_s": train_time,
               "nonfinite_losses": bad}
        row.update({f"fr_{k}": v for k, v in rates.items()})
        if is_snn and any(v < 1e-3 for v in rates.values()):
            log.warning("very low firing rate in some stage (spikes vanishing?): %s", rates)

        is_best = test_acc > best_acc
        best_acc = max(best_acc, test_acc)
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     scaler=scaler.state_dict(), epoch=epoch, best_acc=best_acc, config=cfg, seed=seed)
        save_ckpt(last_path, **state)
        if is_best:
            shutil.copyfile(last_path, ckpt_dir / "best.pt")
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        log.info("epoch %d/%d loss %.4f train %.4f test %.4f (%.0fs, %.1f img/s) %s", epoch + 1, epochs,
                 row["train_loss"], row["train_acc"], test_acc, train_time, tot_seen / train_time,
                 {k: round(v, 4) for k, v in rates.items()})

        if args.max_hours is not None and (time.time() - t_start) / 3600 > args.max_hours and epoch + 1 < epochs:
            log.info("max-hours reached after epoch %d; rerun with --resume to continue", epoch)
            return

    shutil.copyfile(last_path, ckpt_dir / "final.pt")
    final_acc, rates = evaluate(model, x_te, y_te, norm, eval_bs, is_snn)
    summary = {"model": args.model, "final_test_acc": final_acc, "best_test_acc": best_acc,
               "epochs": epochs, "firing_rates_test": rates, "smoke": cfg["smoke_mode"]}
    dump_json(summary, ckpt_dir / "train_summary.json")
    if is_snn:
        res = Path(cfg["paths"]["results_dir"])
        res.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"model": "snn", "layer": k, "corruption": "clean", "severity": 0,
                       "metric": "lif_firing_rate", "value": v} for k, v in rates.items()]
                     ).to_csv(res / "firing_rates.csv", index=False)
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
