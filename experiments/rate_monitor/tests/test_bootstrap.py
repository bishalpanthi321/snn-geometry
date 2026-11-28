"""Weighted AUROC equals the ordinary one, and paired bootstrap differences behave."""
import numpy as np

from rmlib.stats import BootCache, weighted_auc
from tslib.detection import auroc, stratified_auroc


def rng(s=0):
    return np.random.default_rng(s)


def test_weighted_auc_matches_plain_and_stratified():
    r = rng()
    y = r.integers(0, 2, 300)
    s = np.round(r.normal(size=300) + y, 1)                          # ties on purpose
    st = r.integers(0, 4, 300)
    assert np.isclose(weighted_auc(s, y, np.ones(300))[0], auroc(y, s))
    assert np.isclose(weighted_auc(s, y, np.ones(300), st)[0], stratified_auroc(y, s, st))


def test_weights_equal_explicit_resample():
    r = rng(1)
    y = r.integers(0, 2, 200)
    s = r.normal(size=200) + 0.5 * y
    st = r.integers(0, 3, 200)
    idx = r.integers(0, 200, 200)
    w = np.bincount(idx, minlength=200)
    assert np.isclose(weighted_auc(s, y, w)[0], auroc(y[idx], s[idx]))
    assert np.isclose(weighted_auc(s, y, w, st)[0], stratified_auroc(y[idx], s[idx], st[idx]))


def make_cell(y_s, y_a, sc_s, sc_a):
    n = len(y_s)
    st = np.zeros(n, int)
    return {"n": n, "y": {"snn": y_s, "ann": y_a}, "strata": {"snn": st, "ann": st},
            "scores": {"snn": sc_s, "ann": sc_a}}


def test_identical_models_give_exact_zero_difference():
    r = rng(2)
    y = r.integers(0, 2, 400)
    A, B = r.normal(size=400), r.normal(size=400) + y
    c = make_cell(y, y, {"A": A, "B": B}, {"A": A, "B": B})
    bc = BootCache([c, c], 200, 0)

    def g(t, ci):
        return bc.vec(ci, t, "B", False) - bc.vec(ci, t, "A", False)

    vs = [g("snn", ci) - g("ann", ci) for ci in (0, 1)]
    assert all(np.all(v == 0) for v in vs)                           # paired: same resamples
    p, lo, hi, _ = bc.summarize(vs)
    assert p == lo == hi == 0


def test_planted_snn_advantage_detected_with_own_labels():
    r = rng(3)
    n = 1500
    y_s, y_a = r.integers(0, 2, n), r.integers(0, 2, n)               # different error sets
    A_s, A_a = r.normal(size=n), r.normal(size=n)
    c = make_cell(y_s, y_a, {"A": A_s, "B": A_s + 2 * y_s}, {"A": A_a, "B": A_a + 0.1 * r.normal(size=n)})
    bc = BootCache([c], 300, 0)
    v = (bc.vec(0, "snn", "B", False) - bc.vec(0, "snn", "A", False)) - \
        (bc.vec(0, "ann", "B", False) - bc.vec(0, "ann", "A", False))
    p, lo, hi, _ = bc.summarize([v])
    assert p > 0.2 and lo > 0.15
    # scored against the wrong model's labels the planted gain disappears: labels matter
    c2 = make_cell(y_a, y_a, c["scores"]["snn"], c["scores"]["ann"])
    bc2 = BootCache([c2], 100, 0)
    v2 = bc2.vec(0, "snn", "B", False) - bc2.vec(0, "snn", "A", False)
    assert abs(bc2.summarize([v2])[0]) < 0.05


def test_summarize_pools_mean_over_cells():
    a = np.r_[0.1, np.full(50, 0.1)]
    b = np.r_[0.3, np.full(50, 0.3)]
    p, lo, hi, per = BootCache([], 0, 0).summarize([a, b])
    assert np.isclose(p, 0.2) and np.isclose(lo, 0.2) and np.allclose(per, [0.1, 0.3])
