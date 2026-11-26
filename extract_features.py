"""Extract features once so the analysis can rerun on CPU.

    python extract_features.py --model ann
    python extract_features.py --model snn            # T from config (4)
    python extract_features.py --model snn --T 8      # inference-only, saved as snn_T8

Layout: features/{model}/{split}/{corruption}/{severity}/{name}.npy with split in
{train, fit, eval}; train holds only clean features (for LDA), fit/eval are the
two halves of the CIFAR-10 test set. Every condition stores the same input
indices in the same order, so displacements are pairwise row-by-row.

Files per condition:
  ANN: layer1..3.npy, penultimate.npy [N, D] float32
  SNN: same names = time-averaged rates [N, D] float32,
       {layer}_spikes.npy [N, T, D] float16 (pooled per-timestep outputs; exact in
       fp16 since they are multiples of 1/(H*W) <= 2), penultimate_membrane.npy
       [N, T, 512] float16
  all: logits.npy [N, 10] float32, preds.npy, labels.npy, indices.npy
layer4 is not saved separately: its pooled output IS the penultimate feature
(checked with an exact equality assert on every batch).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.data import GPUNormalizer, fit_eval_split, load_cifar10, load_test_condition, test_labels
from src.features import FeatureRecorder
from src.models import build_model
from src.utils import dump_json, env_report, get_device, get_logger, load_config

SAVE_LAYERS = ["layer1", "layer2", "layer3", "penultimate"]


def model_tag(model, T, cfg):
    if model == "snn" and T != cfg["snn"]["T"]:
        return f"snn_T{T}"
    return model


@torch.no_grad()
def run(model, x_u8, is_snn, norm, dev, batch_size):
    out = {}
    rec = FeatureRecorder(model, is_snn=is_snn)
    max_fc_err = 0.0
    with rec:
        for i in range(0, len(x_u8), batch_size):
            xb = norm(torch.from_numpy(x_u8[i:i + batch_size]).to(dev))
            logits = model(xb)
            f = rec.collect()
            sfx = "_spikes" if is_snn else ""
            assert torch.equal(f["layer4" + sfx], f["penultimate" + sfx]), "layer4 GAP != penultimate"
            batch = {"logits": logits.float()}
            for layer in SAVE_LAYERS:
                v = f[layer + sfx].float()
                if is_snn:
                    batch[layer + "_spikes"] = v.half()
                    batch[layer] = v.mean(1)              # time-averaged rate
                else:
                    batch[layer] = v
            if is_snn:
                batch["penultimate_membrane"] = f["penultimate_membrane"].half()
            # The defense relies on logits == fc(penultimate rate); check on real data.
            max_fc_err = max(max_fc_err, (model.fc(batch["penultimate"]) - logits).abs().max().item())
            for k, v in batch.items():
                out.setdefault(k, []).append(v.cpu().numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["preds"] = out["logits"].argmax(1)
    return out, max_fc_err


def save_condition(root, arrays, labels, indices):
    root.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(root / f"{k}.npy", v[indices] if indices is not None else v)
    np.save(root / "labels.npy", labels)
    np.save(root / "indices.npy", indices if indices is not None else np.arange(len(labels)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ann", "snn"], required=True)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--checkpoint", default=None, help="final | best (default from config)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke, out_root=args.out_root)
    is_snn = args.model == "snn"
    T = (args.T or cfg["snn"]["T"]) if is_snn else None
    tag = model_tag(args.model, T, cfg)
    log = get_logger(f"extract_{tag}", cfg["paths"]["logs_dir"])

    # Feature extraction should be deterministic so that clean-vs-clean is exactly 0.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = get_device()

    which = args.checkpoint or cfg["train"]["checkpoint_for_analysis"]
    ck_path = Path(cfg["paths"]["ckpt_dir"]) / args.model / f"{which}.pt"
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    model = build_model(args.model, cfg, T=T).to(dev).eval()
    model.load_state_dict(ck["model"])
    log.info("loaded %s (epoch %d), tag=%s, T=%s", ck_path, ck["epoch"], tag, T)
    norm = GPUNormalizer(dev)
    bs = cfg["extract"]["batch_size"] if not is_snn else max(10, 400 // T)

    feat_root = Path(cfg["paths"]["features_dir"]) / tag
    feat_root.mkdir(parents=True, exist_ok=True)
    # The final layer, so the projection defense runs on CPU without torch.
    np.savez(feat_root / "fc.npz", weight=model.fc.weight.detach().cpu().double().numpy(),
             bias=model.fc.bias.detach().cpu().double().numpy())
    y_test = test_labels(cfg)
    fit_idx, eval_idx = fit_eval_split(len(y_test), cfg["split"]["seed"])
    meta = {"model": args.model, "tag": tag, "T": T, "checkpoint": str(ck_path), "epoch": ck["epoch"],
            "env": env_report(), "n_fit": len(fit_idx), "n_eval": len(eval_idx),
            "lif_impl": model.sn1.impl if is_snn else None, "max_fc_vs_logits_err": {}, "accuracy": {}}

    conditions = [("clean", 0)] + [(c, s) for c in cfg["corruptions"]["names"]
                                   for s in cfg["corruptions"]["severities"]]
    t0 = time.time()
    clean_out = None
    for corr, sev in conditions:
        marker = feat_root / "eval" / corr / str(sev) / "preds.npy"
        if marker.exists() and not args.force and corr != "clean":
            log.info("exists, skipping %s/%s", corr, sev)
            continue
        x = load_test_condition(cfg, corr, sev)
        assert len(x) == len(y_test)
        out, err = run(model, x, is_snn, norm, dev, bs)
        meta["max_fc_vs_logits_err"][f"{corr}/{sev}"] = err
        meta["accuracy"][f"{corr}/{sev}"] = float((out["preds"] == y_test).mean())
        for split, idx in [("fit", fit_idx), ("eval", eval_idx)]:
            save_condition(feat_root / split / corr / str(sev), out, y_test[idx], idx)
        log.info("%s s%s acc %.4f (fc err %.2e) [%.0fs]", corr, sev, meta["accuracy"][f"{corr}/{sev}"],
                 err, time.time() - t0)
        if corr == "clean":
            clean_out = out

    # Determinism check: the same clean images extracted again must give identical features.
    n_rep = min(cfg["extract"]["repeat_check_n"], len(y_test))
    x = load_test_condition(cfg, "clean", 0)[:n_rep]
    rep, _ = run(model, x, is_snn, norm, dev, bs)
    meta["clean_repeat_max_abs_diff"] = {k: float(np.abs(rep[k] - clean_out[k][:n_rep]).max())
                                         for k in SAVE_LAYERS}
    log.info("clean repeat max |diff|: %s", meta["clean_repeat_max_abs_diff"])

    # Clean training features for LDA (rates only; no augmentation).
    x_tr, y_tr = load_cifar10(cfg["paths"]["data_root"], train=True)
    if cfg.get("n_train_lda"):
        x_tr, y_tr = x_tr[: cfg["n_train_lda"]], y_tr[: cfg["n_train_lda"]]
    tr_root = feat_root / "train" / "clean" / "0"
    if not (tr_root / "penultimate.npy").exists() or args.force:
        out, _ = run(model, x_tr, is_snn, norm, dev, bs)
        keep = {k: out[k] for k in SAVE_LAYERS + ["logits", "preds"]}
        save_condition(tr_root, keep, y_tr, None)
        meta["accuracy"]["train_clean"] = float((out["preds"] == y_tr).mean())
    meta["extract_time_s"] = time.time() - t0
    dump_json(meta, feat_root / f"meta_{int(time.time())}.json")
    dump_json(meta, feat_root / "meta.json")
    log.info("done in %.0fs", meta["extract_time_s"])


if __name__ == "__main__":
    main()
