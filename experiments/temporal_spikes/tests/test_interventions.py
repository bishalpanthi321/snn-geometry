"""Every intervention does exactly what it claims, on binary and integer-valued tensors."""
import numpy as np
import pytest
import torch

from tslib import interventions as iv
from tslib.testing import binary_z, integer_z

CASES = [("binary", 1), ("integer", 3)]


def make(kind, T, seed=0):
    return binary_z(T=T, seed=seed) if kind == "binary" else integer_z(T=T, seed=seed)


def ref_table(kind, T, seed=10):
    vmax = 1 if kind == "binary" else 3
    zr = binary_z(T=T, N=40, seed=seed) if kind == "binary" else integer_z(T=T, N=40, seed=seed)
    base = vmax + 1
    code = iv.encode(zr.reshape(T, zr.shape[1], zr.shape[2], -1), base)       # [N, C, L]
    C, K = zr.shape[2], base ** T
    hist = np.zeros((C, K), dtype=np.int64)
    for c in range(C):
        hist[c] = np.bincount(code[:, c].reshape(-1).numpy(), minlength=K)
    return iv.ResampleTable(hist, base, T, "cpu"), hist, base


def count_preserving(z, kind, T):
    table, _, _ = ref_table(kind, T)
    out = {"uniform_perm": iv.uniform_perm(z, 0), "circular_shift": iv.circular_shift(z, 1),
           "even_spread": iv.even_spread(z), "same_count_swap": iv.same_count_swap(z, 2),
           "clean_resample": table.sample(z, 3)[0]}
    for j, p in enumerate(iv.all_shared_perms(T, 3)):
        out[f"shared_perm_{j}"] = iv.shared_perm(z, p)
    return out


@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("kind,vmax", CASES)
def test_counts_preserved(kind, vmax, T):
    z = make(kind, T)
    for name, z2 in count_preserving(z, kind, T).items():
        assert torch.equal(iv.counts(z2), iv.counts(z)), name


@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("kind,vmax", CASES)
def test_support_preserved(kind, vmax, T):
    z = make(kind, T)
    items = dict(count_preserving(z, kind, T))
    items["count_change"] = iv.count_change(z, torch.full((z.shape[1],), 7), vmax, 0)
    for name, z2 in items.items():
        assert z2.shape == z.shape, name
        assert torch.equal(z2, z2.round()), name
        assert float(z2.min()) >= 0 and float(z2.max()) <= vmax, name


@pytest.mark.parametrize("kind,vmax", CASES)
def test_uniform_perm_keeps_each_trains_values(kind, vmax):
    z = make(kind, 4)
    z2 = iv.uniform_perm(z, 0)
    assert torch.equal(z.sort(0).values, z2.sort(0).values)
    assert not torch.equal(z, z2)


@pytest.mark.parametrize("T", [4, 8])
def test_shared_perm_coactivation(T):
    z = binary_z(T=T)
    for p in iv.all_shared_perms(T, 5):
        z2 = iv.shared_perm(z, p)
        for t in range(T):
            assert torch.equal(z2[t], z[p[t]])


def test_shared_perms_enumerated_T4():
    perms = iv.all_shared_perms(4, 23)
    assert len(perms) == 23 and len(set(perms)) == 23
    assert (0, 1, 2, 3) not in perms
    p8 = iv.all_shared_perms(8, 10, seed=0)
    assert len(p8) == 10 and len(set(p8)) == 10 and tuple(range(8)) not in p8
    assert p8 == iv.all_shared_perms(8, 10, seed=0)


@pytest.mark.parametrize("T", [4, 8])
def test_circular_shift_inverse(T):
    z = integer_z(T=T)
    s = iv.random_shifts(z, 5)
    assert int(s.min()) >= 1 and int(s.max()) <= T - 1
    z2 = iv.circular_shift(z, shifts=s)
    assert torch.equal(iv.circular_shift(z2, shifts=T - s), z)
    assert torch.equal(iv.circular_shift(z, seed=5), z2)


@pytest.mark.parametrize("kind,vmax", CASES)
def test_same_count_swap_preserves_channel_profile(kind, vmax):
    T = 4
    z = make(kind, T)
    z2 = iv.same_count_swap(z, 0)
    zz, zz2 = z.reshape(T, 3, 5, -1), z2.reshape(T, 3, 5, -1)
    assert torch.equal(zz.sum(-1), zz2.sum(-1))                  # channel count at every t
    assert torch.equal(zz.sum(0), zz2.sum(0))                    # every neuron's count
    c1, c2 = iv.encode(zz, vmax + 1), iv.encode(zz2, vmax + 1)
    assert torch.equal(c1.sort(-1).values, c2.sort(-1).values)   # same trains, relocated
    assert not torch.equal(z, z2)


@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("kind,vmax", CASES)
def test_clean_resample_support(kind, vmax, T):
    table, hist, base = ref_table(kind, T)
    z = make(kind, T, seed=3)
    z2, n_fb = table.sample(z, 0)
    code = iv.encode(z2.reshape(T, 3, 5, -1), base)
    S = iv.counts(z).reshape(3, 5, -1)
    seen = torch.as_tensor(hist)[torch.arange(5).view(1, 5, 1), code] > 0
    # every non-fallback train is one that occurred in the reference for that channel
    tot = table.tot_cs[torch.arange(5).view(1, 5, 1), S.long()]
    assert bool(seen[tot > 0].all())
    assert n_fb == int((tot == 0).sum())


def test_clean_resample_follows_reference_and_falls_back():
    T, base = 4, 2
    K = base ** T
    hist = np.zeros((2, K), dtype=np.int64)
    hist[0, 0b0011] = 5          # channel 0, count 2: only "spikes at t=0,1"
    hist[0, 0b0001] = 5          # channel 0, count 1: only "spike at t=0"
    hist[1, 0b1000] = 3          # channel 1, count 1: only "spike at t=3"
    table = iv.ResampleTable(hist, base, T, "cpu")
    z = torch.zeros(T, 1, 2, 1, 2, dtype=torch.float64)
    z[[2, 3], 0, 0, 0, 0] = 1    # ch0 pos0: count 2 -> must become 0011
    z[1, 0, 0, 0, 1] = 1         # ch0 pos1: count 1 -> must become 0001
    z[0, 0, 1, 0, 0] = 1         # ch1 pos0: count 1 -> must become 1000
    z[[0, 2, 3], 0, 1, 0, 1] = 1  # ch1 pos1: count 3 never seen -> fallback (count kept)
    z2, n_fb = table.sample(z, 0)
    assert z2[:, 0, 0, 0, 0].tolist() == [1, 1, 0, 0]
    assert z2[:, 0, 0, 0, 1].tolist() == [1, 0, 0, 0]
    assert z2[:, 0, 1, 0, 0].tolist() == [0, 0, 0, 1]
    assert n_fb == 1              # only ch1 pos1 (count 3 never seen in channel 1)
    assert torch.equal(iv.counts(z2), iv.counts(z))


@pytest.mark.parametrize("kind,vmax", CASES)
def test_count_change_hamming(kind, vmax):
    z = make(kind, 4)
    d = torch.tensor([0, 5, 17])
    z2 = iv.count_change(z, d, vmax, 0)
    assert torch.equal(iv.l1_per_image(z, z2), d.double())
    assert not torch.equal(iv.counts(z2), iv.counts(z))
    assert torch.equal(z2[:, 0], z[:, 0])


@pytest.mark.parametrize("kind,vmax", CASES)
def test_seed_reproducible(kind, vmax):
    z = make(kind, 4)
    table, _, _ = ref_table(kind, 4)
    fns = [lambda s: iv.uniform_perm(z, s), lambda s: iv.circular_shift(z, s),
           lambda s: iv.same_count_swap(z, s), lambda s: table.sample(z, s)[0],
           lambda s: iv.count_change(z, torch.full((3,), 9), vmax, s)]
    for f in fns:
        assert torch.equal(f(1), f(1))
        assert not torch.equal(f(1), f(2))


def test_even_spread():
    z = torch.zeros(4, 1, 1, 1, 3, dtype=torch.float64)
    z[[2, 3], 0, 0, 0, 0] = 1                 # binary count 2
    z[:, 0, 0, 0, 1] = torch.tensor([3., 3, 0, 1])   # integer count 7
    z[3, 0, 0, 0, 2] = 1                      # count 1
    e = iv.even_spread(z)
    assert e[:, 0, 0, 0, 0].tolist() == [0, 1, 0, 1]
    assert e[:, 0, 0, 0, 1].tolist() == [2, 1, 2, 2]   # extras at int((j+.5)*4/3) = 0, 2, 3
    assert e[:, 0, 0, 0, 2].tolist() == [0, 0, 1, 0]
    assert torch.equal(iv.even_spread(z), e)
    zi = integer_z(T=8)
    ei = iv.even_spread(zi)
    assert float((ei.max(0).values - ei.min(0).values).max()) <= 1
