"""CIFAR-10 as in-memory uint8 arrays.

Everything is kept as uint8 NHWC numpy arrays in the original torchvision order,
so test index i always refers to the same image in the clean set and in every
corrupted set. Normalisation and augmentation happen on the GPU per batch.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(data_root, train):
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=str(data_root), train=train, download=True)
    x = np.asarray(ds.data, dtype=np.uint8)          # [N, 32, 32, 3]
    y = np.asarray(ds.targets, dtype=np.int64)
    assert x.shape[1:] == (32, 32, 3)
    return x, y


def corrupted_path(corrupted_dir, corruption, severity):
    return Path(corrupted_dir) / corruption / f"{severity}.npy"


def load_test_condition(cfg, corruption, severity):
    """Images for one test condition, in clean test-set order.

    corruption == "clean" returns the clean test set.
    """
    if corruption == "clean":
        x, _ = load_cifar10(cfg["paths"]["data_root"], train=False)
    else:
        p = corrupted_path(cfg["paths"]["corrupted_dir"], corruption, severity)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run generate_corruptions.py first")
        x = np.load(p, allow_pickle=False)
    if cfg.get("n_test"):
        x = x[: cfg["n_test"]]
    return x


def test_labels(cfg):
    _, y = load_cifar10(cfg["paths"]["data_root"], train=False)
    if cfg.get("n_test"):
        y = y[: cfg["n_test"]]
    return y


def fit_eval_split(n, seed):
    """50/50 split of test indices, fixed by seed. Returns (fit_idx, eval_idx), sorted."""
    perm = np.random.default_rng(seed).permutation(n)
    half = n // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


class GPUNormalizer:
    def __init__(self, device):
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)

    @staticmethod
    def to_float01(x_uint8_nhwc):
        """uint8 [N,H,W,3] tensor -> float [N,3,H,W] in [0,1]."""
        return x_uint8_nhwc.permute(0, 3, 1, 2).float().div_(255.0)

    def normalize(self, x01):
        return (x01 - self.mean) / self.std

    def __call__(self, x_uint8_nhwc):
        """uint8 [N,H,W,3] tensor -> normalised float [N,3,H,W]."""
        return self.normalize(self.to_float01(x_uint8_nhwc))


def augment_batch(x, generator):
    """Random crop 32 with zero padding 4 + random horizontal flip.

    x is a float [N,3,32,32] batch in [0,1] (before normalisation) so that the
    zero padding matches torchvision's RandomCrop(padding=4) on PIL images.
    """
    n = x.shape[0]
    dev = x.device
    padded = F.pad(x, (4, 4, 4, 4))
    oi = torch.randint(0, 9, (n,), generator=generator, device="cpu").to(dev)
    oj = torch.randint(0, 9, (n,), generator=generator, device="cpu").to(dev)
    ar = torch.arange(32, device=dev)
    rows = (oi[:, None] + ar)[:, None, :, None]          # [N,1,32,1]
    cols = (oj[:, None] + ar)[:, None, None, :]          # [N,1,1,32]
    nidx = torch.arange(n, device=dev)[:, None, None, None]
    cidx = torch.arange(3, device=dev)[None, :, None, None]
    out = padded[nidx, cidx, rows, cols]
    flip = torch.rand(n, generator=generator, device="cpu").to(dev) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out
