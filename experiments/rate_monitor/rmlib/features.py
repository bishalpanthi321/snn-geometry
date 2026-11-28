"""Feature verification and monitor feature sets.

Feature sets (identical construction for both models, except D which only the SNN has):
  A  confidence: MSP, entropy, top-2 logit margin
  B  A + per-channel activations: log1p of the 960 spatially pooled channel values of
     layer1..3 + penultimate (SNN: time-averaged)
  C  B + activation statistics: per-layer Mahalanobis distance to the nearest clean class
     mean (shared Ledoit-Wolf covariance, fit on clean TRAINING images)
  D  (SNN only) C + per-timestep pooled channel values (log1p, T x 960) + the confidence
     trajectory (MSP of the running-mean logits at each t, number of prediction changes)
"""
import numpy as np
from scipy.special import softmax
from sklearn.covariance import LedoitWolf

from .data import FeatureStore, conditions, row_positions


# ------------------------------------------------------------------ verification

def verify(cfg, log=print):
    """Check that every file the analysis needs exists and is consistent. Returns a report
    dict; raises AssertionError on any inconsistency."""
    root, layers, T = cfg["resolved"]["features_root"], cfg["layers"], cfg["T"]
    st = {t: FeatureStore(root, t) for t in cfg["models"]}
    rep = {"checked": [], "max_snn_time_avg_err": 0.0, "max_snn_fc_err": 0.0}
    need = ["logits", "preds", "labels", "indices"] + layers
    ref_idx, ref_lab = {}, {}
    for split in ("fit", "eval"):
        for corr, sev in conditions(cfg):
            for tag, s in st.items():
                miss = [n for n in need + ([f"{l}_spikes" for l in layers] if tag == "snn" else [])
                        if not s.has(split, corr, sev, n)]
                assert not miss, f"{tag}/{split}/{corr}/{sev} missing {miss}"
                idx = s.load(split, corr, sev, "indices")
                lab = s.load(split, corr, sev, "labels")
                key = split
                if key not in ref_idx:
                    ref_idx[key], ref_lab[key] = idx, lab
                assert np.array_equal(idx, ref_idx[key]), f"image order differs: {tag}/{split}/{corr}/{sev}"
                assert np.array_equal(lab, ref_lab[key]), f"labels differ: {tag}/{split}/{corr}/{sev}"
                lg = s.load(split, corr, sev, "logits")
                assert np.array_equal(lg.argmax(1), s.load(split, corr, sev, "preds")), \
                    f"preds != argmax(logits): {tag}/{split}/{corr}/{sev}"
                for l in layers:
                    x = s.load(split, corr, sev, l)
                    assert x.shape == (len(idx), cfg["layer_dims"][l]), (tag, l, x.shape)
                if tag == "snn":
                    rows = slice(0, min(500, len(idx)))
                    for l in layers:
                        sp = s.load(split, corr, sev, f"{l}_spikes", rows).astype(np.float64)
                        assert sp.shape[1] == T, sp.shape
                        err = np.abs(sp.mean(1) - s.load(split, corr, sev, l, rows)).max()
                        rep["max_snn_time_avg_err"] = max(rep["max_snn_time_avg_err"], float(err))
                    W, b = s.fc()
                    pen = s.load(split, corr, sev, "penultimate", rows).astype(np.float64)
                    err = np.abs(pen @ W.T + b - lg[rows]).max()
                    rep["max_snn_fc_err"] = max(rep["max_snn_fc_err"], float(err))
            rep["checked"].append(f"{split}/{corr}/{sev}")
    split_overlap = np.intersect1d(ref_idx["fit"], ref_idx["eval"])
    assert len(split_overlap) == 0, "fit and eval splits share images"
    for tag, s in st.items():
        for n in ["labels", "logits"] + layers:
            assert s.has("train", "clean", 0, n), f"{tag} train/clean missing {n}"
        assert np.array_equal(s.load("train", "clean", 0, "labels"),
                              st[cfg["models"][0]].load("train", "clean", 0, "labels"))
    # fp16 storage of per-timestep values: multiples of 1/(H*W) up to 3 round at ~1e-3
    assert rep["max_snn_time_avg_err"] < 5e-3, rep["max_snn_time_avg_err"]
    assert rep["max_snn_fc_err"] < 1e-3, rep["max_snn_fc_err"]
    rep["n_fit_rows_available"] = int(len(ref_idx["fit"]))
    rep["n_eval_rows_available"] = int(len(ref_idx["eval"]))
    log(f"features verified: {len(rep['checked'])} split/conditions, "
        f"SNN time-average err {rep['max_snn_time_avg_err']:.1e}, fc err {rep['max_snn_fc_err']:.1e}")
    return rep


# ------------------------------------------------------------------ feature construction

def conf_features(logits):
    p = softmax(logits.astype(np.float64), 1)
    srt = np.sort(logits, 1)
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
    return np.c_[p.max(1), ent, srt[:, -1] - srt[:, -2]]


def trajectory_features(penult_t, W, b):
    """penult_t [N, T, 512] per-timestep pooled penultimate values -> MSP of the running-mean
    logits at each t, and the number of prediction changes over t."""
    lt = penult_t.astype(np.float64) @ W.T + b                          # [N, T, 10]
    T = lt.shape[1]
    run = np.cumsum(lt, 1) / np.arange(1, T + 1)[None, :, None]
    msp = softmax(run, 2).max(2)
    pred = run.argmax(2)
    return np.c_[msp, (pred[:, 1:] != pred[:, :-1]).sum(1)]


class NearestClassMahalanobis:
    def __init__(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.cls = np.unique(y)
        self.means = np.stack([X[y == c].mean(0) for c in self.cls])
        self.prec = LedoitWolf().fit(X - self.means[np.searchsorted(self.cls, y)]).precision_

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.min(np.stack([np.einsum("ij,jk,ik->i", X - m, self.prec, X - m)
                                for m in self.means], 1), 1)


def fit_mahalanobis(store, layers, max_train=None, seed=0):
    """Per-layer Mahalanobis models fit on clean TRAINING images only."""
    y = store.load("train", "clean", 0, "labels")
    rows = None
    if max_train and len(y) > max_train:
        rows = np.sort(np.random.default_rng(seed).choice(len(y), max_train, replace=False))
        y = y[rows]
    return {l: NearestClassMahalanobis(store.load("train", "clean", 0, l, rows), y) for l in layers}


def build(cfg, store, tag, maha):
    """All rows (fit + eval subsets) of every condition for one model.

    Returns (meta, F): meta has corruption, severity, split, index, label, pred, error;
    F maps feature-set name -> matrix. Row order is identical for every model."""
    layers, run = cfg["layers"], cfg["run"]
    is_snn = tag == "snn"
    W, b = store.fc() if is_snn else (None, None)
    metas, A, chans, mah, pt, traj = [], [], [], [], [], []
    for split, n_take in (("fit", run["n_fit"]), ("eval", run["n_eval"])):
        n_rows = len(store.load(split, "clean", 0, "indices"))
        rows = row_positions(n_rows, n_take, cfg["seed"] + (0 if split == "fit" else 1))
        for corr, sev in conditions(cfg):
            lg = store.load(split, corr, sev, "logits", rows).astype(np.float64)
            lab = store.load(split, corr, sev, "labels", rows)
            metas.append({"corruption": np.full(len(rows), corr), "severity": np.full(len(rows), sev),
                          "split": np.full(len(rows), split),
                          "index": store.load(split, corr, sev, "indices", rows),
                          "label": lab, "pred": lg.argmax(1)})
            A.append(conf_features(lg))
            X = {l: store.load(split, corr, sev, l, rows).astype(np.float64) for l in layers}
            chans.append(np.log1p(np.clip(np.concatenate([X[l] for l in layers], 1), 0, None)))
            mah.append(np.stack([maha[l].score(X[l]) for l in layers], 1))
            if is_snn:
                sp = [store.load(split, corr, sev, f"{l}_spikes", rows) for l in layers]
                pt.append(np.log1p(np.clip(np.concatenate(sp, 2).astype(np.float32), 0, None))
                          .reshape(len(rows), -1))
                traj.append(trajectory_features(sp[-1], W, b))
    meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}
    meta["error"] = (meta["pred"] != meta["label"]).astype(int)
    F = {"conf": np.concatenate(A), "channels": np.concatenate(chans),
         "maha": np.log1p(np.concatenate(mah))}
    if is_snn:
        F["per_t"] = np.concatenate(pt)
        F["trajectory"] = np.concatenate(traj)
    return meta, F


SETS = {
    "A": ["conf"],
    "B": ["conf", "channels"],
    "C": ["conf", "channels", "maha"],
    "D": ["conf", "channels", "maha", "per_t", "trajectory"],
}


def stack(F, parts):
    return np.concatenate([F[p] for p in parts], 1)
