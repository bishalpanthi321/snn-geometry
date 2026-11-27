"""Temporal statistics and capacity bounds on hand-built and synthetic spike trains."""
import math

import numpy as np
import pytest
import torch

from tslib import interventions as iv
from tslib import stats as ts
from tslib.testing import binary_z


def test_capacity_bound():
    assert [round(2 ** ts.log2_binom(4, k)) for k in range(5)] == [1, 4, 6, 4, 1]
    assert math.isclose(ts.log2_binom(8, 4), math.log2(70))
    T, K = 4, 16
    hist = np.zeros(K)
    hist[0b0000] = 2          # k=0
    hist[0b1111] = 2          # k=4
    hist[0b0011] = 3          # k=2
    hist[0b0001] = 3          # k=1
    m = ts.capacity_metrics(hist, 2, T)
    assert math.isclose(m["frac_multi_order"], 0.6)
    assert math.isclose(m["capacity_bits"], 0.3 * math.log2(6) + 0.3 * math.log2(4))
    assert math.isclose(m["capacity_max_bits"], math.log2(6))
    # every count class has a single observed order -> no timing entropy
    assert m["H_order_given_count_bits"] == 0.0


def test_integer_capacity_counts_orderings():
    T, base = 4, 4
    s, multinom_bits, n_seq = ts.code_table(base, T)
    code = 3 + 3 * 4                          # train [3, 3, 0, 0]: 4!/(2!2!) = 6 orderings
    assert math.isclose(2 ** multinom_bits[code], 6)
    assert n_seq.sum() == base ** T and n_seq[0] == 1 and n_seq[12] == 1


def test_pattern_codes_bits():
    z = torch.zeros(4, 1, 2, dtype=torch.float64)
    z[0, 0, 0] = 1                            # spike at t=0 -> code 1
    z[3, 0, 1] = 1                            # spike at t=3 -> code 8
    assert iv.encode(z, 2).flatten().tolist() == [1, 8]
    assert torch.equal(iv.decode(iv.encode(z, 2), 2, 4).double(), z)


def test_neuron_timing_hand_built():
    s = torch.tensor([[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float64)
    s = s.view(4, 1, 1, 3)                    # three neurons: 0110, 1101, 0000
    k, cen, first, trans = ts.neuron_timing(s)
    assert k.flatten().tolist() == [2, 3, 0]
    assert cen.flatten()[:2].tolist() == [1.5, (0 + 1 + 3) / 3]
    assert first.flatten()[:2].tolist() == [1, 0]
    assert trans.flatten().tolist() == [1, 1, 0]


def test_transition_and_first_spike_expectations_under_uniform_order():
    """Monte Carlo: uniform timing given count reproduces the closed-form expectations,
    so the deviation features are zero-mean when timing carries no information."""
    T, n = 8, 200_000
    g = torch.Generator().manual_seed(0)
    for k in [1, 3, 5]:
        keys = torch.rand(T, n, generator=g)
        s = (keys.argsort(0).argsort(0) < k).double().view(T, 1, 1, n)
        kk, cen, first, trans = ts.neuron_timing(s)
        assert torch.all(kk == k)
        assert abs(float(cen.mean()) - ts.expected_centroid(T, k)) < 0.02
        assert abs(float(first.mean()) - ts.expected_first_spike(T, k)) < 0.02
        assert abs(float(trans.mean()) - ts.expected_trans11(T, k)) < 0.02


def test_lag_cov_planted():
    """Channel 1 copies channel 0 one step later: lag1[0, 1] is large; time-shuffling
    each neuron destroys it."""
    T, N, L = 8, 50, 64
    g = torch.Generator().manual_seed(0)
    s = torch.zeros(T, N, 3, L, dtype=torch.float64)
    s[:, :, 0] = (torch.rand(T, N, L, generator=g) < 0.4).double()
    s[1:, :, 1] = s[:-1, :, 0]
    s[:, :, 2] = (torch.rand(T, N, L, generator=g) < 0.4).double()
    mu = s.mean(dim=(0, 1, 3))
    P = torch.eye(3, dtype=torch.float64)
    f = ts.detector_features(s.view(T, N, 3, 8, 8), pca=(mu, P))
    lag = f["lag1"].mean(0)
    shuf = ts.detector_features(iv.uniform_perm(s, 0).view(T, N, 3, 8, 8), pca=(mu, P))["lag1"].mean(0)
    assert lag[0, 1] > 0.15
    assert abs(shuf[0, 1]) < 0.05
    assert abs(lag[1, 0]) < 0.05 and abs(lag[0, 2]) < 0.05


def test_detector_features_consistent():
    z = binary_z(T=4, N=3, C=5)
    f = ts.detector_features(z, per_channel_timing=True)
    k = z.sum(0)
    assert np.array_equal(f["chan_counts"], k.flatten(2).sum(-1).numpy())
    assert np.array_equal(f["n_k"].sum(1), np.full(3, 5 * 16))
    assert np.allclose(f["pop_t"], z.mean(dim=(2, 3, 4)).T.numpy())
    multi = ((k > 0) & (k < 4)).flatten(2).sum(-1).numpy()
    assert np.array_equal(f["chan_n_multi"], multi)


@pytest.mark.parametrize("T", [4, 8])
def test_cond_entropy_le_capacity(T):
    """Uniform timing given count saturates the bound; the estimate never exceeds it
    (beyond the small Miller-Madow correction)."""
    z = binary_z(T=T, N=400, C=8, H=8, W=8, p=0.5)
    acc = ts.SiteAccumulator(T, 2, per_channel=True)
    acc.update(z)
    m = ts.capacity_metrics(acc.hist, 2, T)
    assert m["H_order_given_count_bits"] <= m["capacity_bits"] + 0.01
    assert m["H_order_given_count_bits"] > 0.97 * m["capacity_bits"]
    # a site where order is a fixed function of count has ~zero timing entropy
    zs = (torch.arange(T).view(T, 1, 1, 1, 1) < z.sum(0, keepdim=True)).double()
    acc2 = ts.SiteAccumulator(T, 2, per_channel=True)
    acc2.update(zs)
    assert ts.capacity_metrics(acc2.hist, 2, T)["H_order_given_count_bits"] < 1e-3


def test_accumulator_rejects_out_of_support():
    acc = ts.SiteAccumulator(4, 2)
    with pytest.raises(ValueError):
        acc.update(torch.full((4, 1, 1, 2, 2), 2.0))
