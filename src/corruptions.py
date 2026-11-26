"""Six CIFAR-10-C corruptions, re-implemented with numpy/scipy/matplotlib.

Severity constants are copied from Hendrycks' make_cifar_c.py (the CIFAR
version, not ImageNet-C). Differences from that script, all deliberate:
  * vectorised over a batch [N, H, W, 3] instead of one PIL image at a time;
  * skimage is replaced by equivalents: scipy.ndimage.gaussian_filter with
    mode='nearest', truncate=4 (what skimage.filters.gaussian calls), matplotlib's
    rgb_to_hsv / hsv_to_rgb, and the same per-element salt-and-pepper rule as
    skimage.util.random_noise(mode='s&p', salt_vs_pepper=0.5);
  * the float result is ROUNDED to uint8. The original truncates with np.uint8,
    which shifts every pixel down by 0.5 on average; that shift would show up as
    a small corruption-independent displacement, so we avoid it.
"""
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from scipy.ndimage import gaussian_filter

# Fixed order: used to derive a per-(corruption, severity) RNG stream, so a
# subset (smoke test) produces exactly the same images as the full run.
ALL_CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise",
                   "gaussian_blur", "contrast", "brightness"]

PARAMS = {
    "gaussian_noise": [0.04, 0.06, 0.08, 0.09, 0.10],
    "shot_noise": [500, 250, 100, 75, 50],
    "impulse_noise": [0.01, 0.02, 0.03, 0.05, 0.07],
    "gaussian_blur": [0.4, 0.6, 0.7, 0.8, 1.0],
    "contrast": [0.75, 0.5, 0.4, 0.3, 0.15],
    "brightness": [0.05, 0.1, 0.15, 0.2, 0.3],
}


def _gaussian_noise(x, c, rng):
    return np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1)


def _shot_noise(x, c, rng):
    return np.clip(rng.poisson(x * c) / c, 0, 1)


def _impulse_noise(x, c, rng):
    out = x.copy()
    flipped = rng.random(x.shape) <= c
    salted = rng.random(x.shape) <= 0.5
    out[flipped & salted] = 1.0
    out[flipped & ~salted] = 0.0
    return out


def _gaussian_blur(x, c, rng):
    # Blur spatial axes only; batch and channel axes get sigma 0.
    return np.clip(gaussian_filter(x, sigma=(0, c, c, 0), mode="nearest", truncate=4.0), 0, 1)


def _contrast(x, c, rng):
    means = x.mean(axis=(1, 2), keepdims=True)    # per image, per channel
    return np.clip((x - means) * c + means, 0, 1)


def _brightness(x, c, rng):
    hsv = rgb_to_hsv(x)
    hsv[..., 2] = np.clip(hsv[..., 2] + c, 0, 1)
    return np.clip(hsv_to_rgb(hsv), 0, 1)


_FUNCS = {
    "gaussian_noise": _gaussian_noise,
    "shot_noise": _shot_noise,
    "impulse_noise": _impulse_noise,
    "gaussian_blur": _gaussian_blur,
    "contrast": _contrast,
    "brightness": _brightness,
}


def corrupt(images_uint8, corruption, severity, seed, chunk=2500):
    """Corrupt a uint8 [N,32,32,3] array. Output keeps the input order (index i -> i)."""
    if corruption not in _FUNCS:
        raise ValueError(corruption)
    if not 1 <= severity <= 5:
        raise ValueError(severity)
    c = PARAMS[corruption][severity - 1]
    rng = np.random.default_rng([seed, ALL_CORRUPTIONS.index(corruption), severity])
    out = np.empty_like(images_uint8)
    for start in range(0, len(images_uint8), chunk):
        x = images_uint8[start:start + chunk].astype(np.float64) / 255.0
        y = _FUNCS[corruption](x, c, rng)
        out[start:start + chunk] = np.rint(np.clip(y, 0, 1) * 255.0).astype(np.uint8)
    return out
