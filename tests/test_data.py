import torch

from src.data import augment_batch


def test_augment_is_a_padded_crop_or_its_mirror():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(16, 3, 32, 32)
    out = augment_batch(x, g)
    padded = torch.nn.functional.pad(x, (4, 4, 4, 4))
    n_flipped = 0
    for i in range(len(x)):
        windows = [padded[i, :, a:a + 32, b:b + 32] for a in range(9) for b in range(9)]
        plain = any(torch.equal(out[i], w) for w in windows)
        mirrored = any(torch.equal(out[i].flip(2), w) for w in windows)
        assert plain or mirrored
        n_flipped += int(mirrored and not plain)
    assert 0 < n_flipped < len(x)   # flips happen, but not always


def test_augment_reproducible_with_same_seed():
    x = torch.rand(8, 3, 32, 32)
    a = augment_batch(x, torch.Generator().manual_seed(3))
    b = augment_batch(x, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)
