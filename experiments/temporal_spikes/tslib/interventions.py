"""Spike-timing interventions on a tensor z of shape [T, N, C, H, W].

A "neuron" is one (image, channel, spatial position); its train is z[:, n, c, h, w].
Values are integers stored as floats: 0/1 at binary LIF sites, 0..vmax at SEW stage
outputs (integer sites). "Count" = sum over t of the train.

All interventions except count_change preserve every neuron's count exactly
(tested, and asserted on every batch at run time).

  uniform_perm     independent random permutation of each neuron's timesteps
  shared_perm      the same permutation of t for every neuron (keeps co-activation)
  circular_shift   independent random cyclic shift (1..T-1) per neuron; spikes that move
                   past T wrap around to the start (artificial; documented)
  even_spread      rate-only: the count spread as evenly as possible over t, deterministic
  same_count_swap  population-preserving control: within each image and channel, spike
                   trains are swapped between spatial positions that have the same count.
                   Keeps every neuron's count AND each channel's count at every timestep;
                   destroys which position carries which ordering (and alignment across
                   channels at a position).
  clean_resample   each neuron's train is redrawn from the clean-training distribution of
                   trains of that channel with the same count. Keeps counts and the clean
                   per-channel ordering distribution; destroys correlations between
                   neurons (they are drawn independently) and this image's own timing.
  count_change     positive control: changes the counts. Per image, exactly d[n] entries
                   (t, neuron) are moved by +-1 within [0, vmax]; d is matched to the L1
                   size of a count-preserving intervention.
"""
import itertools

import numpy as np
import torch


def _gen(z, seed):
    g = torch.Generator(device=z.device)
    g.manual_seed(int(seed))
    return g


def counts(z):
    return z.sum(0)


def l1_per_image(z, z2):
    return (z2 - z).abs().flatten(2).sum(dim=(0, 2)) if z.dim() > 2 else (z2 - z).abs().sum(0)


def uniform_perm(z, seed):
    T = z.shape[0]
    flat = z.reshape(T, -1)
    keys = torch.rand(flat.shape, generator=_gen(z, seed), device=z.device)
    return flat.gather(0, keys.argsort(0)).view_as(z)


def all_shared_perms(T, n_max, seed=0):
    """Non-identity permutations of range(T): all of them if there are <= n_max,
    otherwise n_max distinct ones sampled with a fixed seed."""
    ident = tuple(range(T))
    if T <= 5:
        perms = [p for p in itertools.permutations(range(T)) if p != ident]
        if len(perms) <= n_max:
            return perms
    rng = np.random.default_rng(seed)
    out = set()
    while len(out) < n_max:
        p = tuple(int(i) for i in rng.permutation(T))
        if p != ident:
            out.add(p)
    return sorted(out)


def shared_perm(z, perm):
    """new[t] = z[perm[t]] for every neuron."""
    return z[torch.as_tensor(perm, device=z.device)]


def random_shifts(z, seed):
    """One shift in 1..T-1 per neuron (flattened neuron order)."""
    T = z.shape[0]
    return torch.randint(1, T, (z[0].numel(),), generator=_gen(z, seed), device=z.device)


def circular_shift(z, seed=None, shifts=None):
    """new[t] = z[(t - s) mod T] per neuron, with s from `shifts` or drawn from `seed`."""
    T = z.shape[0]
    flat = z.reshape(T, -1)
    s = random_shifts(z, seed) if shifts is None else shifts
    idx = (torch.arange(T, device=z.device)[:, None] - s[None, :]) % T
    return flat.gather(0, idx).view_as(z)


def even_spread(z):
    """Each neuron's count S spread as evenly as possible: floor(S/T) everywhere plus one
    extra at r = S mod T positions floor((j + 0.5) T / r), j < r (distinct positions)."""
    T = z.shape[0]
    S = counts(z).long().reshape(-1)
    base = S // T
    r = S % T
    # r < T is small, so build the extra positions explicitly for each value of r.
    extra = torch.zeros(T, S.numel(), device=z.device, dtype=torch.long)
    for rv in range(1, T):
        sel = r == rv
        if sel.any():
            pos = torch.tensor([int((j + 0.5) * T / rv) for j in range(rv)], device=z.device)
            col = torch.zeros(T, device=z.device, dtype=torch.long)
            col[pos] = 1
            extra[:, sel] = col[:, None]
    out = base[None, :] + extra
    return out.to(z.dtype).view_as(z)


def same_count_swap(z, seed):
    T, N, C = z.shape[:3]
    zz = z.reshape(T, N, C, -1)
    L = zz.shape[-1]
    S = zz.sum(0)                                                   # [N, C, L]
    u = torch.rand(S.shape, generator=_gen(z, seed), device=z.device)
    rand_order = (S * 2 + u).argsort(-1)                            # grouped by S, random within
    pos = torch.arange(L, device=z.device, dtype=z.dtype) / L
    stable_order = (S * 2 + pos).argsort(-1)                        # grouped by S, by position
    src = zz.gather(-1, rand_order.unsqueeze(0).expand(T, -1, -1, -1))
    out = torch.empty_like(zz).scatter_(-1, stable_order.unsqueeze(0).expand(T, -1, -1, -1), src)
    return out.view_as(z)


# ------------------------------------------------------------------ clean resampling

def encode(z, base):
    """Integer code of each neuron's train: sum_t z_t * base**t. Returns [N, C, ...] long."""
    T = z.shape[0]
    w = (base ** torch.arange(T, device=z.device, dtype=torch.long)).view(T, *([1] * (z.dim() - 1)))
    return (z.long() * w).sum(0)


def decode(codes, base, T):
    """[...] long codes -> [T, ...] values."""
    w = base ** torch.arange(T, device=codes.device, dtype=torch.long)
    return (codes.unsqueeze(0) // w.view(T, *([1] * codes.dim()))) % base


class ResampleTable:
    """Per-channel distribution of clean trains, grouped by count, for fast sampling.

    hist: [C, base**T] int counts of each train code per channel (clean training images).
    """

    def __init__(self, hist, base, T, device):
        hist = torch.as_tensor(np.asarray(hist), dtype=torch.long, device=device)
        C, K = hist.shape
        assert K == base ** T, (K, base, T)
        self.C, self.K, self.base, self.T = C, K, base, T
        code_sum = decode(torch.arange(K, device=device), base, T).sum(0)        # [K]
        self.order = torch.argsort(code_sum, stable=True)                        # codes by sum
        sums_sorted = code_sum[self.order]
        smax = (base - 1) * T
        svals = torch.arange(smax + 1, device=device)
        self.g0 = torch.searchsorted(sums_sorted, svals, right=False)             # group starts
        self.g1 = torch.searchsorted(sums_sorted, svals, right=True)              # group ends
        cum = hist[:, self.order].cumsum(1)                                       # [C, K]
        self.cum = cum
        tot = cum[:, -1]
        self.chan_off = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                                tot[:-1] + 1]), 0)                 # strictly increasing
        self.cumflat = (cum + self.chan_off[:, None]).reshape(-1)
        pad = torch.cat([torch.zeros(C, 1, dtype=torch.long, device=device), cum], 1)  # [C, K+1]
        # base and group total for each (channel, sum)
        self.base_cs = pad.gather(1, self.g0.unsqueeze(0).expand(C, -1))          # [C, smax+1]
        self.tot_cs = pad.gather(1, self.g1.unsqueeze(0).expand(C, -1)) - self.base_cs

    def sample(self, z, seed):
        """Returns (z_new, n_fallback). Neurons whose (channel, count) never occurred in
        the clean reference fall back to a uniform permutation of their own train."""
        T = z.shape[0]
        assert T == self.T
        C = z.shape[2]
        assert C == self.C
        S = counts(z).long()                                                     # [N, C, H, W]
        cidx = torch.arange(C, device=z.device).view(1, C, *([1] * (S.dim() - 2))).expand_as(S)
        b = self.base_cs[cidx, S]
        tot = self.tot_cs[cidx, S]
        g = _gen(z, seed)
        u = torch.rand(S.shape, generator=g, device=z.device, dtype=torch.float64)
        r = torch.minimum((u * tot.double()).long(), (tot - 1).clamp(min=0))
        target = self.chan_off[cidx] + b + r
        i = torch.searchsorted(self.cumflat, target.reshape(-1), right=True).view_as(S)
        local = i - cidx * self.K
        code = self.order[local.clamp(0, self.K - 1)]
        new = decode(code, self.base, T).to(z.dtype)
        missing = tot == 0
        n_fb = int(missing.sum())
        if n_fb:
            fb = uniform_perm(z, int(seed) + 7919)
            new = torch.where(missing.unsqueeze(0), fb, new)
        return new, n_fb


# ------------------------------------------------------------------ positive control

def count_change(z, d, vmax, seed):
    """Move exactly d[n] randomly chosen entries of image n by +-1 (within [0, vmax]).
    Entries at 0 go up, entries at vmax go down, others move up or down at random."""
    T, N = z.shape[:2]
    flat = z.transpose(0, 1).reshape(N, -1)                         # [N, P]
    g = _gen(z, seed)
    keys = torch.rand(flat.shape, generator=g, device=z.device)
    rank = keys.argsort(1).argsort(1)
    d = torch.as_tensor(d, device=z.device).long().clamp(max=flat.shape[1])
    sel = rank < d[:, None]
    coin = torch.rand(flat.shape, generator=g, device=z.device) < 0.5
    step = torch.where(coin, 1.0, -1.0).to(z.dtype)
    step = torch.where(flat <= 0, torch.ones_like(step), step)
    step = torch.where(flat >= vmax, -torch.ones_like(step), step)
    out = flat + sel.to(z.dtype) * step
    return out.view(N, T, *z.shape[2:]).transpose(0, 1).contiguous()
