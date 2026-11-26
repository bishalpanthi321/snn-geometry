"""Synthetic data with known answers for every piece of math in analyze.py."""
import numpy as np
import pytest

from src import geometry as G


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_known_3d_subspace_gives_k90_le_3_and_zero_angles(rng):
    D, n = 128, 2000
    B = G.random_subspace(D, 3, rng)                       # true subspace
    coeff = rng.standard_normal((n, 3)) * np.array([3.0, 2.0, 1.0])
    disp = coeff @ B.T + 1e-4 * rng.standard_normal((n, D))  # tiny isotropic noise
    evals, V = G.displacement_spectrum(disp)
    assert G.k_for_energy(evals, 0.9) <= 3
    angles = G.principal_angles_deg(V[:, :3], B)
    assert np.all(angles < 1.0)
    # held-out: a fresh draw from the same subspace is captured almost entirely
    disp_eval = rng.standard_normal((n, 3)) @ B.T
    assert G.captured_energy(disp_eval, V[:, :3]) > 0.999
    # participation ratio of 3 unequal axes is between 1 and 3
    assert 1.0 < G.participation_ratio(evals) < 3.0


def test_orthogonal_subspaces_give_90_degrees(rng):
    D = 64
    Q = G.random_subspace(D, 20, rng)
    A, B = Q[:, :9], Q[:, 9:20]
    angles = G.principal_angles_deg(A, B)
    assert len(angles) == 9
    np.testing.assert_allclose(angles, 90.0, atol=1e-6)
    # and a subspace with itself gives 0
    np.testing.assert_allclose(G.principal_angles_deg(A, A), 0.0, atol=1e-5)


def test_projection_removes_subspace_component(rng):
    D, n, k = 100, 500, 7
    U = G.random_subspace(D, k, rng)
    H = rng.standard_normal((n, D)) * 5 + 2
    m = rng.standard_normal(D)
    Hp = G.project_out(H, m, U)
    np.testing.assert_allclose((Hp - m) @ U, 0.0, atol=1e-10)
    # the orthogonal complement is untouched
    P_perp = np.eye(D) - U @ U.T
    np.testing.assert_allclose((Hp - m) @ P_perp, (H - m) @ P_perp, atol=1e-10)


def test_isotropic_energy_fraction_is_dimW_over_D(rng):
    D, n = 256, 20000
    W = G.random_subspace(D, 9, rng)
    disp = rng.standard_normal((n, D))
    frac, used = G.energy_fraction(disp, W)
    assert used == n
    assert abs(frac - 9 / D) < 0.003


def test_random_subspace_null_is_k_over_D(rng):
    D, n, k = 200, 5000, 10
    disp = rng.standard_normal((n, D))
    U = G.random_subspace(D, k, rng)
    assert abs(G.captured_energy(disp, U) - k / D) < 0.005


def test_k_for_energy_edge_cases():
    assert G.k_for_energy(np.array([1.0, 0, 0]), 0.9) == 1
    assert G.k_for_energy(np.array([0.5, 0.4, 0.1]), 0.9) == 2   # exactly 0.9 counts
    assert G.k_for_energy(np.ones(10), 0.9) == 9
    assert np.isnan(G.k_for_energy(np.zeros(4), 0.9))
    np.testing.assert_allclose(G.participation_ratio(np.ones(10)), 10.0)


def test_clean_vs_clean_magnitude_is_exactly_zero(rng):
    H = rng.standard_normal((50, 32))
    mag, used = G.relative_magnitude(H, H.copy())
    assert mag == 0.0 and used == 50


def test_lda_recovers_class_mean_subspace(rng):
    # 10 classes whose means lie in a known 9-dim subspace, isotropic noise:
    # LDA directions then span exactly that subspace.
    # Well separated + many samples: with weak separation the weakest LDA direction
    # is tilted by sampling noise (checked: 24 deg at scale 4/400 per class, 2.4 here).
    D, per = 60, 2000
    B = G.random_subspace(D, 9, rng)
    means = rng.standard_normal((10, 9)) * 10 @ B.T
    X = np.concatenate([means[c] + rng.standard_normal((per, D)) for c in range(10)])
    y = np.repeat(np.arange(10), per)
    W, _ = G.lda_subspace(X, y)
    assert W.shape == (D, 9)
    np.testing.assert_allclose(W.T @ W, np.eye(9), atol=1e-8)
    assert np.max(G.principal_angles_deg(W, B)) < 5.0
