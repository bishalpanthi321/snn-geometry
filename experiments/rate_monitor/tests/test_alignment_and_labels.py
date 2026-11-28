"""The saved features line up across models and conditions, and error labels are right."""
import numpy as np
import pytest

from rmlib.data import FeatureStore
from rmlib.features import build, fit_mahalanobis, verify
from rmlib.testing import synthetic_cfg


def quiet(_):
    pass


def test_verify_passes_on_consistent_features(tmp_path):
    cfg, _ = synthetic_cfg(tmp_path)
    rep = verify(cfg, log=quiet)
    assert rep["max_snn_fc_err"] < 1e-3 and rep["max_snn_time_avg_err"] < 5e-3


@pytest.mark.parametrize("breakage", ["order", "labels", "preds", "fc", "time_average", "missing"])
def test_verify_detects_inconsistency(tmp_path, breakage):
    cfg, root = synthetic_cfg(tmp_path)
    tag = "ann" if breakage in ("order", "labels", "preds", "missing") else "snn"
    d = root / tag / "eval" / "contrast" / "3"
    if breakage == "order":
        np.save(d / "indices.npy", np.load(d / "indices.npy")[::-1])
    elif breakage == "labels":
        np.save(d / "labels.npy", (np.load(d / "labels.npy") + 1) % 10)
    elif breakage == "preds":
        np.save(d / "preds.npy", (np.load(d / "preds.npy") + 1) % 10)
    elif breakage == "fc":
        np.save(d / "penultimate.npy", np.load(d / "penultimate.npy") + 1)
    elif breakage == "time_average":
        np.save(d / "layer2.npy", np.load(d / "layer2.npy") + 0.1)
    elif breakage == "missing":
        (d / "layer3.npy").unlink()
    with pytest.raises(AssertionError):
        verify(cfg, log=quiet)


def test_rows_aligned_and_labels_per_model(tmp_path):
    cfg, root = synthetic_cfg(tmp_path)
    out = {}
    for tag in ("snn", "ann"):
        st = FeatureStore(root, tag)
        out[tag] = build(cfg, st, tag, fit_mahalanobis(st, cfg["layers"]))
    (ms, Fs), (ma, Fa) = out["snn"], out["ann"]
    for k in ("corruption", "severity", "split", "index", "label"):
        assert np.array_equal(ms[k], ma[k])
    # each model's error label comes from its own logits
    for tag, (m, F) in out.items():
        st = FeatureStore(root, tag)
        sel = (m["split"] == "eval") & (m["corruption"] == "contrast") & (m["severity"] == 3)
        pos = np.searchsorted(st.load("eval", "contrast", 3, "indices"), m["index"][sel])
        lg = st.load("eval", "contrast", 3, "logits")[pos]
        assert np.array_equal(m["error"][sel], (lg.argmax(1) != m["label"][sel]).astype(int))
    assert not np.array_equal(ms["error"], ma["error"])          # the two models do differ
    assert Fs["channels"].shape[1] == Fa["channels"].shape[1] == 960
    assert "per_t" in Fs and "per_t" not in Fa
    assert Fs["per_t"].shape[1] == 4 * 960
    fit_i, ev_i = ms["index"][ms["split"] == "fit"], ms["index"][ms["split"] == "eval"]
    assert len(np.intersect1d(fit_i, ev_i)) == 0
