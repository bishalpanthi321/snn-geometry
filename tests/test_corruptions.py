import numpy as np
import pytest

from src.corruptions import ALL_CORRUPTIONS, PARAMS, corrupt
from src.data import fit_eval_split


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(0)
    # Smooth-ish random images so blur and contrast have something to act on.
    base = rng.integers(0, 256, size=(64, 8, 8, 3)).astype(np.float64)
    return np.clip(np.kron(base, np.ones((1, 4, 4, 1))), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", ALL_CORRUPTIONS)
def test_deterministic_shape_dtype_and_monotone(name, images):
    a = corrupt(images, name, 3, seed=7)
    b = corrupt(images, name, 3, seed=7)
    assert a.dtype == np.uint8 and a.shape == images.shape
    assert np.array_equal(a, b)
    dist = [np.abs(corrupt(images, name, s, seed=7).astype(float) - images).mean() for s in range(1, 6)]
    assert dist[0] > 0
    # Severity should not decrease the change (noise draws make it ~monotone, allow 2% slack).
    assert all(dist[i + 1] >= dist[i] * 0.98 for i in range(4)), dist


@pytest.mark.parametrize("name", ALL_CORRUPTIONS)
def test_index_alignment(name, images):
    """Corrupted image i must be closest to clean image i."""
    c = corrupt(images, name, 2, seed=7).reshape(len(images), -1).astype(float)
    x = images.reshape(len(images), -1).astype(float)
    d = ((c[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    assert np.array_equal(d.argmin(1), np.arange(len(images)))


def test_subset_reproduces_full_run(images):
    # A per-(corruption, severity) RNG stream means chunking/subsets don't change pixels.
    full = corrupt(images, "gaussian_noise", 2, seed=7, chunk=64)
    chunked = corrupt(images, "gaussian_noise", 2, seed=7, chunk=16)
    assert np.array_equal(full, chunked)


def test_contrast_and_brightness_match_closed_form(images):
    x = images.astype(np.float64) / 255
    c = PARAMS["contrast"][2]
    m = x.mean(axis=(1, 2), keepdims=True)
    expect = np.rint(np.clip((x - m) * c + m, 0, 1) * 255)
    assert np.abs(corrupt(images, "contrast", 3, 0).astype(float) - expect).max() == 0
    # brightness raises V = max(r,g,b) by c (clipped)
    b = corrupt(images, "brightness", 1, 0).astype(float) / 255
    v_expect = np.clip(x.max(-1) + PARAMS["brightness"][0], 0, 1)
    assert np.abs(b.max(-1) - v_expect).max() <= 0.5 / 255 + 1e-9


def test_fit_eval_split_is_disjoint_and_fixed():
    f, e = fit_eval_split(10000, 0)
    assert len(f) == len(e) == 5000
    assert len(np.intersect1d(f, e)) == 0
    f2, _ = fit_eval_split(10000, 0)
    assert np.array_equal(f, f2)
