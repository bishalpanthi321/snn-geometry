"""Nothing fitted for a held-out corruption may depend on evaluation rows."""
import numpy as np

from rmlib.data import FeatureStore
from rmlib.features import build, fit_mahalanobis
from rmlib.testing import synthetic_cfg
from run_monitor import fit_for_heldout
from tslib.detection import protocol_masks


def test_fitted_quantities_ignore_test_rows(tmp_path):
    cfg, root = synthetic_cfg(tmp_path)
    st = FeatureStore(root, "snn")
    meta, F = build(cfg, st, "snn", fit_mahalanobis(st, cfg["layers"]))
    tr, te = protocol_masks(meta["corruption"], meta["split"], "contrast")
    assert not (tr & te).any()
    assert not np.isin(meta["index"][tr], meta["index"][te]).any()      # no image in both
    assert (meta["corruption"][tr] != "contrast").all() and (meta["split"][te] == "eval").all()
    ks = [10, 96]
    a = fit_for_heldout(cfg, F, meta["error"], tr, meta["index"], ks, 960)
    # scramble every non-training row: features and labels
    rng = np.random.default_rng(1)
    F2 = {k: v.copy() for k, v in F.items()}
    for k in F2:
        F2[k][~tr] = rng.normal(size=F2[k][~tr].shape) * 10
    y2 = meta["error"].copy()
    y2[~tr] = rng.integers(0, 2, int((~tr).sum()))
    b = fit_for_heldout(cfg, F2, y2, tr, meta["index"], ks, 960)
    assert a["chosen"] == b["chosen"]
    assert np.array_equal(a["edges"], b["edges"])
    for m in ("l1", "univariate"):
        assert np.array_equal(a["rank"][m], b["rank"][m])
    for k in a["scores"]:
        assert np.allclose(a["scores"][k][tr], b["scores"][k][tr])       # identical fitted model
        assert not np.allclose(a["scores"][k][~tr], b["scores"][k][~tr])


def test_mahalanobis_uses_training_split_only(tmp_path):
    cfg, root = synthetic_cfg(tmp_path)
    st = FeatureStore(root, "ann")
    m1 = fit_mahalanobis(st, cfg["layers"])
    d = root / "ann" / "fit" / "clean" / "0"
    np.save(d / "layer1.npy", np.load(d / "layer1.npy") * 100)
    m2 = fit_mahalanobis(st, cfg["layers"])
    assert np.array_equal(m1["layer1"].means, m2["layer1"].means)
