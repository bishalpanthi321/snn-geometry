"""Error-detection metrics, stratified AUROC, paired bootstrap, and nested logistic models.

Convention: `score` is higher for inputs predicted to be ERRORS; `y` is 1 for an error.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def auroc(y, score):
    """Mann-Whitney AUROC with ties counted as 1/2. NaN if a class is missing."""
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, score):
    y = np.asarray(y).astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    return float(average_precision_score(y, score))


def fpr_at_tpr(y, score, tpr=0.95):
    """Fraction of correct inputs flagged when the threshold catches >= `tpr` of errors."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    s1 = np.sort(score[y])
    # largest threshold that still flags at least tpr of the errors (score >= thr)
    k = int(np.floor((1 - tpr) * len(s1)))
    thr = s1[k]
    return float((score[~y] >= thr).mean())


def strata_from_edges(values, edges):
    return np.searchsorted(edges, values, side="right")


def quantile_edges(values, n):
    """Inner edges of n quantile bins (fit on training rows only)."""
    qs = np.quantile(values, np.linspace(0, 1, n + 1)[1:-1])
    return np.unique(qs)


def stratified_auroc(y, score, strata):
    """AUROC computed within strata and combined with weights n_pos * n_neg (pairs).
    A score that only re-encodes the stratifying variable gets ~0.5."""
    y = np.asarray(y).astype(bool)
    num, den = 0.0, 0.0
    for g in np.unique(strata):
        sel = strata == g
        n1, n0 = y[sel].sum(), (~y[sel]).sum()
        if n1 == 0 or n0 == 0:
            continue
        w = float(n1) * float(n0)
        num += w * auroc(y[sel], score[sel])
        den += w
    return num / den if den > 0 else float("nan")


def all_metrics(y, score, margin_strata=None, count_strata=None):
    out = {"auroc": auroc(y, score), "auprc": auprc(y, score), "fpr95": fpr_at_tpr(y, score)}
    if margin_strata is not None:
        out["auroc_margin_strat"] = stratified_auroc(y, score, margin_strata)
    if count_strata is not None:
        out["auroc_count_strat"] = stratified_auroc(y, score, count_strata)
    return out


def paired_bootstrap_delta(y, s_a, s_b, n_boot, seed, strata=None):
    """Bootstrap over inputs of AUROC(s_a) - AUROC(s_b) on the same inputs.
    Returns (point estimate, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    f = (lambda yy, s, st: stratified_auroc(yy, s, st)) if strata is not None else \
        (lambda yy, s, st: auroc(yy, s))
    point = f(y, s_a, strata) - f(y, s_b, strata)
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        st = strata[i] if strata is not None else None
        v = f(y[i], s_a[i], st) - f(y[i], s_b[i], st)
        if np.isfinite(v):
            d.append(v)
    if not d:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def fit_logistic(X, y, groups, C_grid, folds, seed=0, max_cv_rows=8000):
    """Standardize + L2 logistic regression; C chosen by grouped CV AUROC on the training
    rows only (groups = image index, so the same image never sits in two folds). For speed
    the CV uses at most `max_cv_rows` rows, sampled by whole groups; the final model is fit
    on all training rows."""
    X = np.asarray(X, dtype=np.float64)
    best, best_c = -np.inf, C_grid[0]
    ug = np.unique(groups)
    cv_rows = np.arange(len(y))
    if len(y) > max_cv_rows:
        keep = np.random.default_rng(seed).permutation(ug)[: max(2, int(len(ug) * max_cv_rows / len(y)))]
        cv_rows = np.where(np.isin(groups, keep))[0]
    Xc, yc, gc = X[cv_rows], y[cv_rows], groups[cv_rows]
    k = min(folds, len(np.unique(gc)))
    if k >= 2 and len(np.unique(yc)) == 2:
        gkf = GroupKFold(n_splits=k)
        for c in C_grid:
            aucs = []
            for tr, va in gkf.split(Xc, yc, gc):
                if len(np.unique(yc[tr])) < 2:
                    continue
                m = _model(c, seed).fit(Xc[tr], yc[tr])
                aucs.append(auroc(yc[va], m.decision_function(Xc[va])))
            a = np.nanmean(aucs) if aucs else -np.inf
            if a > best:
                best, best_c = a, c
    m = _model(best_c, seed).fit(X, y)
    m.chosen_C_ = best_c
    m.cv_auroc_ = best
    return m


def _model(c, seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=c, max_iter=5000, random_state=seed))


def protocol_masks(corruption, split, heldout):
    """Leave-one-corruption-out rows. Train: fit-split images of every condition except the
    held-out corruption (clean included). Test: eval-split images of the held-out corruption.
    fit and eval are disjoint image sets, so no clean/corrupted pair spans train and test."""
    corruption, split = np.asarray(corruption), np.asarray(split)
    train = (split == "fit") & (corruption != heldout)
    test = (split == "eval") & (corruption == heldout)
    return train, test
