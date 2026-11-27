"""Detection metrics, stratification controls, leakage-free protocol, and bootstrap."""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from tslib import detection as dt
from tslib.common import load_cfg, select_indices


def rng(seed=0):
    return np.random.default_rng(seed)


def test_auroc_matches_sklearn():
    r = rng()
    y = r.integers(0, 2, 500)
    s = r.normal(size=500) + y * 0.7
    s[::7] = 0.0                                # ties
    assert np.isclose(dt.auroc(y, s), roc_auc_score(y, s))
    assert np.isnan(dt.auroc(np.zeros(5), np.arange(5)))


def test_fpr95():
    y = np.array([1] * 20 + [0] * 10)
    s = np.concatenate([np.arange(20) + 10.0, np.arange(10) + 0.0])   # perfect separation
    assert dt.fpr_at_tpr(y, s) == 0.0
    s2 = np.concatenate([np.arange(20) + 0.0, np.arange(10) + 0.0])   # overlap
    # threshold keeps >= 95% of errors: s >= 1 -> negatives 1..9 flagged = 9/10
    assert np.isclose(dt.fpr_at_tpr(y, s2), 0.9)


def test_stratified_auroc_null():
    """A score that is margin + noise predicts errors overall, but carries no information
    once margin is controlled for."""
    r = rng(1)
    n = 20000
    margin = r.normal(size=n)
    y = (r.random(n) < 1 / (1 + np.exp(3 * margin))).astype(int)
    score = -margin + 0.05 * r.normal(size=n)
    strata = dt.strata_from_edges(margin, dt.quantile_edges(margin, 50))
    assert dt.auroc(y, score) > 0.8
    assert abs(dt.stratified_auroc(y, score, strata) - 0.5) < 0.05


def test_planted_signal_detected():
    """A feature with information beyond margin gives a positive nested AUROC gain."""
    r = rng(2)
    n = 6000
    margin, f, junk = r.normal(size=n), r.normal(size=n), r.normal(size=n)
    y = (r.random(n) < 1 / (1 + np.exp(2 * margin - 1.5 * f))).astype(int)
    groups = np.arange(n)
    tr, te = slice(0, 4000), slice(4000, None)
    X0 = np.c_[margin, junk]
    X1 = np.c_[margin, junk, f]
    m0 = dt.fit_logistic(X0[tr], y[tr], groups[tr], [0.1, 1.0], 3)
    m1 = dt.fit_logistic(X1[tr], y[tr], groups[tr], [0.1, 1.0], 3)
    a0 = dt.auroc(y[te], m0.decision_function(X0[te]))
    a1 = dt.auroc(y[te], m1.decision_function(X1[te]))
    assert a1 - a0 > 0.03


def test_scaler_fit_on_training_rows_only():
    r = rng(3)
    X = r.normal(size=(300, 4)) + 5
    y = r.integers(0, 2, 300)
    m = dt.fit_logistic(X[:200], y[:200], np.arange(200), [1.0], 2)
    assert np.allclose(m.named_steps["standardscaler"].mean_, X[:200].mean(0))


def test_splits_disjoint_no_leak():
    for stage in ["smoke", "pilot", "full"]:
        cfg = load_cfg(argparse.Namespace(stage=stage, config=None, checkpoint=None, data_root=None,
                                          corrupted_dir=None, out_root=None, random_init=True))
        idx = select_indices(cfg)
        assert len(np.intersect1d(idx["fit"], idx["eval"])) == 0
        assert np.all(np.isin(idx["A"], idx["eval"]))
        assert len(idx["fit"]) == cfg["run"]["n_fit"] and len(idx["eval"]) == cfg["run"]["n_eval"]
    # protocol masks: held-out corruption never in training rows, test rows only eval split
    corr = np.array(["clean", "gaussian_noise", "contrast", "gaussian_noise", "contrast", "clean"])
    split = np.array(["fit", "fit", "fit", "eval", "eval", "eval"])
    tr, te = dt.protocol_masks(corr, split, "gaussian_noise")
    assert tr.tolist() == [True, False, True, False, False, False]
    assert te.tolist() == [False, False, False, True, False, False]


def test_paired_bootstrap_identical_models():
    r = rng(4)
    y = r.integers(0, 2, 400)
    s = r.normal(size=400) + y
    point, lo, hi = dt.paired_bootstrap_delta(y, s, s.copy(), 200, 0)
    assert point == 0.0 and lo == 0.0 and hi == 0.0
    p2, lo2, hi2 = dt.paired_bootstrap_delta(y, s + 2 * y, s, 200, 0)
    assert p2 > 0 and lo2 > 0
