"""Paired, pooled bootstrap over images, and channel selection for sparse monitors.

Every comparison between models or feature sets uses the SAME evaluation images. Each model
keeps its own error labels (an image can be an error for one model and not the other), and
each AUROC is computed with that model's labels. A bootstrap replicate resamples image rows
within every cell (corruption x severity) and applies the same resampled rows to every model
and feature set, so the pairing is preserved.
"""
import time

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tslib.detection import auroc  # reused from the temporal-spike experiment

# scikit-learn >= 1.8 deprecates penalty= in favour of l1_ratio=; both select pure L1 here.
_SK = tuple(int(v) for v in sklearn.__version__.split(".")[:2])
L1_KW = {"l1_ratio": 1.0} if _SK >= (1, 8) else {"penalty": "l1"}


def weighted_auc(score, y, Wm, strata=None):
    """AUROC (ties = 1/2) of `score` for error labels `y` under each row of weights Wm [B, n].
    Row b of Wm holds the multiplicity of every image in bootstrap replicate b (all ones gives
    the ordinary AUROC). With `strata`, returns the stratified AUROC: within-stratum AUROCs
    combined with weights n_pos * n_neg. Undefined replicates are NaN."""
    Wm = np.atleast_2d(np.asarray(Wm, dtype=np.float64))
    y = np.asarray(y).astype(bool)
    if strata is None:
        num, den = _num_den(np.asarray(score, dtype=np.float64), y, Wm)
    else:
        num = np.zeros(Wm.shape[0])
        den = np.zeros(Wm.shape[0])
        for g in np.unique(strata):
            sel = strata == g
            n_, d_ = _num_den(np.asarray(score, dtype=np.float64)[sel], y[sel], Wm[:, sel])
            num += n_
            den += d_
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _num_den(s, y, Wm):
    o = np.argsort(s, kind="mergesort")
    s, y, Wm = s[o], y[o], Wm[:, o]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]           # tie groups (sorted)
    G0 = np.add.reduceat(Wm * ~y, starts, axis=1)
    G1 = np.add.reduceat(Wm * y, starts, axis=1)
    below0 = np.cumsum(G0, 1) - G0                                    # negatives strictly below
    num = (G1 * (below0 + 0.5 * G0)).sum(1)
    den = G1.sum(1) * G0.sum(1)
    return num, den


class BootCache:
    """Shared bootstrap replicates. For each cell, B resamples of its image rows are drawn
    once and used for EVERY model, feature set and statistic, so all comparisons are paired.
    vec(...) returns an array [1 + B]: element 0 is the estimate on the original rows."""

    def __init__(self, cells, n_boot, seed):
        rng = np.random.default_rng(seed)
        self.cells = cells
        self.W = []
        for c in cells:
            draws = rng.integers(0, c["n"], (n_boot, c["n"]))
            Wm = np.zeros((n_boot + 1, c["n"]))
            Wm[0] = 1.0
            for b in range(n_boot):
                Wm[b + 1] = np.bincount(draws[b], minlength=c["n"])
            self.W.append(Wm)
        self._cache = {}

    def vec(self, ci, model, fset, strat):
        key = (ci, model, fset, strat)
        if key not in self._cache:
            c = self.cells[ci]
            self._cache[key] = weighted_auc(c["scores"][model][fset], c["y"][model], self.W[ci],
                                            c["strata"][model] if strat else None)
        return self._cache[key]

    def summarize(self, per_cell_vecs):
        """Mean over cells of per-cell statistic vectors -> estimate, 95% CI, per-cell estimates."""
        V = np.stack(per_cell_vecs)                                       # [cells, 1 + B]
        with np.errstate(all="ignore"):
            pooled = np.nanmean(V, 0)
        reps = pooled[1:][np.isfinite(pooled[1:])]
        lo, hi = np.percentile(reps, [2.5, 97.5]) if len(reps) else (np.nan, np.nan)
        return float(pooled[0]), float(lo), float(hi), V[:, 0]


# ------------------------------------------------------------------ channel selection

def rank_l1(X_conf, X_ch, y, n_points, seed=0):
    """Rank channels by the order in which they enter an L1-regularised logistic path
    (strongest regularisation first). Confidence features are always in the model.
    Fit on TRAINING rows only (the caller passes only training rows)."""
    Xs = StandardScaler().fit_transform(np.c_[X_conf, X_ch])
    n_conf = X_conf.shape[1]
    entry = np.full(X_ch.shape[1], np.inf)
    last = None
    n_nonzero_first = None
    for i, c in enumerate(np.logspace(-4, 0, n_points)):
        m = LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=seed,
                               **L1_KW).fit(Xs, y)
        coef = np.abs(m.coef_[0, n_conf:])
        if n_nonzero_first is None:
            n_nonzero_first = int((coef > 0).sum())
            # guard: the strongest penalty must actually zero out channels (L1 really active)
            if n_nonzero_first >= X_ch.shape[1]:
                raise RuntimeError("L1 penalty not applied: all channels nonzero at C=1e-4")
        newly = (coef > 0) & ~np.isfinite(entry)
        entry[newly] = i - coef[newly] / (coef.max() + 1e-12)   # earlier and larger first
        last = coef
    never = ~np.isfinite(entry)
    entry[never] = n_points + 1 - last[never]                   # by |coef| at weakest penalty
    return np.argsort(entry, kind="stable")


def rank_univariate(X_ch, y):
    """Supervised filter baseline: rank channels by |AUROC - 0.5| against the error label
    on TRAINING rows."""
    return np.argsort(-np.array([abs(auroc(y, X_ch[:, j]) - 0.5) for j in range(X_ch.shape[1])]),
                      kind="stable")


def random_subsets(n_ch, k, n, seed):
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_ch, k, replace=False)) for _ in range(n)]


def fit_fixed_C(X, y, C, seed=0):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=5000, random_state=seed).fit(sc.transform(X), y)
    return sc, m


# ------------------------------------------------------------------ runtime / memory of a monitor

def monitor_cost(sc, m, n_features, n_rows=10000, reps=20, seed=0):
    """Measured CPU time per image and parameter memory of applying a fitted monitor
    (standardise + linear score) to already-pooled features."""
    X = np.random.default_rng(seed).normal(size=(n_rows, n_features))
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sc.transform(X) @ m.coef_[0] + m.intercept_[0]
        t.append(time.perf_counter() - t0)
    params = sc.mean_.nbytes + sc.scale_.nbytes + m.coef_.nbytes + m.intercept_.nbytes
    return {"us_per_image": 1e6 * float(np.median(t)) / n_rows, "param_bytes": int(params),
            "feature_bytes_per_image_fp32": 4 * n_features}
