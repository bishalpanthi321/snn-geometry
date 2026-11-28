"""Synthetic features in the exact on-disk layout of the geometry experiment, for tests and the
smoke run. A known signal is planted so the pipeline can be checked end to end:

  - logits = fc(penultimate) for both models (the SNN file also gets fc.npz)
  - errors come from the logits; noise grows with severity
  - SNN only: channel 0 of layer3 carries error information beyond the logits
    (planted_snn_signal > 0), so the SNN's channel gain should exceed the ANN's
  - SNN per-timestep values average exactly (up to fp16) to the time-averaged values
"""
from pathlib import Path

import numpy as np

from src.data import fit_eval_split

DIMS = {"layer1": 64, "layer2": 128, "layer3": 256, "penultimate": 512}


def _save(d, **arrays):
    d.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(d / f"{k}.npy", v)


def make_synthetic_features(root, corruptions, severities, n_test=400, n_train=600, T=4,
                            planted_snn_signal=2.0, seed=0, split_seed=0):
    root = Path(root)
    rng = np.random.default_rng(seed)
    y_test = rng.integers(0, 10, n_test)
    y_train = rng.integers(0, 10, n_train)
    proto = {l: np.abs(rng.normal(size=(10, d))) for l, d in DIMS.items()}
    # a linear read-out aligned with the class prototypes, so clean accuracy is high and
    # errors increase with the noise level (severity)
    W = proto["penultimate"] - proto["penultimate"].mean(0, keepdims=True)
    W = W / np.linalg.norm(W, axis=1, keepdims=True) * 0.8
    b = np.zeros(10)
    fit_idx, eval_idx = fit_eval_split(n_test, split_seed)
    conds = [("clean", 0)] + [(c, s) for c in corruptions for s in severities]
    for tag in ("snn", "ann"):
        (root / tag).mkdir(parents=True, exist_ok=True)
        if tag == "snn":
            np.savez(root / tag / "fc.npz", weight=W, bias=b)
    # shared image-level noise so both models see "the same images"
    base_noise = {l: rng.normal(size=(n_test, d)) for l, d in DIMS.items()}

    def layers_for(y, noise_scale, n, extra=None):
        X = {}
        for l, d in DIMS.items():
            nz = (extra[l] if extra is not None else rng.normal(size=(n, d)))
            X[l] = np.abs(proto[l][y] + noise_scale * nz).astype(np.float32)
        return X

    for tag in ("snn", "ann"):
        r = np.random.default_rng(seed + (1 if tag == "snn" else 2))
        # train split (clean training images)
        Xtr = layers_for(y_train, 1.2, n_train)
        lg = (Xtr["penultimate"].astype(np.float64) @ W.T + b).astype(np.float32)
        _save(root / tag / "train" / "clean" / "0", **Xtr, logits=lg, preds=lg.argmax(1),
              labels=y_train, indices=np.arange(n_train))
        for corr, sev in conds:
            scale = 1.2 + 0.5 * sev + (0.4 if corr == "contrast" else 0.0)
            X = layers_for(y_test, scale, n_test,
                           extra={l: base_noise[l] * (1.0 if tag == "snn" else 0.9)
                                  + 0.3 * r.normal(size=base_noise[l].shape) for l in DIMS})
            lg = (X["penultimate"].astype(np.float64) @ W.T + b).astype(np.float32)
            err = lg.argmax(1) != y_test
            if tag == "snn" and planted_snn_signal:
                X["layer3"][:, 0] = (planted_snn_signal * err + r.normal(size=n_test)).astype(np.float32) + 3
            out = {}
            if tag == "snn":
                for l in DIMS:
                    dev = r.normal(size=(n_test, T, DIMS[l])) * 0.05
                    dev -= dev.mean(1, keepdims=True)
                    sp = (X[l][:, None, :] + dev).astype(np.float16)
                    X[l] = sp.astype(np.float32).mean(1)          # exact time average of what is stored
                    out[f"{l}_spikes"] = sp
                lg = (X["penultimate"].astype(np.float64) @ W.T + b).astype(np.float32)
            for split, idx in (("fit", fit_idx), ("eval", eval_idx)):
                _save(root / tag / split / corr / str(sev),
                      **{k: v[idx] for k, v in X.items()}, **{k: v[idx] for k, v in out.items()},
                      logits=lg[idx], preds=lg[idx].argmax(1), labels=y_test[idx], indices=idx)
    return root
