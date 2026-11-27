"""Temporal spike statistics: GPU accumulators during extraction, numpy metrics after.

Terminology
  count k        number of spikes of one neuron (one channel at one spatial position) over T
  pattern/code   the full train, encoded as sum_t s_t * base**t (base 2 for binary sites)
  capacity       sum_k p(k) log2 C(T, k): the most information (bits per neuron) that the
                 ORDER of spikes could carry once the count is known. Neurons with k=0 or
                 k=T have a single possible order and contribute 0.
"""
import math

import numpy as np
import torch

from .interventions import decode, encode


# ------------------------------------------------------------------ expectations under uniform order

def expected_centroid(T, k):
    """Mean spike time (0-based) of k spikes placed uniformly at random among T slots."""
    return (T - 1) / 2.0


def expected_first_spike(T, k):
    """E[min] of a uniform random k-subset of {0..T-1}."""
    return (T - k) / (k + 1.0)


def expected_trans11(T, k):
    """E[# adjacent (1,1) pairs] for a uniform random k-subset of {0..T-1}."""
    return k * (k - 1) / float(T)


# ------------------------------------------------------------------ per-neuron quantities (torch)

def neuron_timing(s):
    """s: [T, N, C, L] binary. Returns k, centroid, first spike, trans11, each [N, C, L].
    centroid / first spike are only meaningful where k > 0 (0 is returned otherwise)."""
    T = s.shape[0]
    t = torch.arange(T, device=s.device, dtype=s.dtype).view(T, 1, 1, 1)
    k = s.sum(0)
    kk = k.clamp(min=1)
    centroid = (t * s).sum(0) / kk
    first = s.argmax(0).to(s.dtype)            # index of the first 1 (first maximum)
    trans = (s[:-1] * s[1:]).sum(0)
    return k, centroid, first, trans


def per_image_by_count(values, k, T):
    """Sum of `values` over the neurons of each image with each count k. [N, T+1]."""
    N = values.shape[0]
    idx = (torch.arange(N, device=values.device).view(N, *([1] * (values.dim() - 1))) * (T + 1)
           + k.long()).reshape(-1)
    out = torch.zeros(N * (T + 1), device=values.device, dtype=torch.float64)
    out.index_add_(0, idx, values.reshape(-1).double())
    return out.view(N, T + 1)


# ------------------------------------------------------------------ accumulators

class SiteAccumulator:
    """Per-condition statistics of one site, accumulated over batches.

    binary: count histogram [T+1], per-timestep spike sum [T], per-channel code
            histogram [C, 2**T].
    integer: per-timestep sum [T], code histogram per channel [C, base**T] if
             per_channel else pooled [base**T].
    """

    def __init__(self, T, base, per_channel=True):
        self.T, self.base, self.per_channel = T, base, per_channel
        self.n_neurons = 0                     # neuron-samples (images x channels x positions)
        self.n_images = 0
        self.rate_t = np.zeros(T)
        self.hist = None
        self.max_value = 0.0

    @torch.no_grad()
    def update(self, z):
        T, N, C = z.shape[:3]
        zz = z.reshape(T, N, C, -1)
        self.max_value = max(self.max_value, float(zz.max()))
        if self.max_value > self.base - 1:
            raise ValueError(f"value {self.max_value} exceeds vmax {self.base - 1}")
        self.n_images += N
        self.n_neurons += N * C * zz.shape[-1]
        self.rate_t += zz.double().sum(dim=(1, 2, 3)).cpu().numpy()
        code = encode(zz, self.base)                                  # [N, C, L]
        K = self.base ** T
        if self.per_channel:
            cidx = torch.arange(C, device=z.device).view(1, C, 1)
            h = torch.bincount((cidx * K + code).reshape(-1), minlength=C * K).view(C, K)
        else:
            h = torch.bincount(code.reshape(-1), minlength=K)
        h = h.cpu().numpy()
        self.hist = h if self.hist is None else self.hist + h

    def to_dict(self, prefix):
        return {f"{prefix}/hist": self.hist.astype(np.int64), f"{prefix}/rate_t": self.rate_t,
                f"{prefix}/n_neurons": np.int64(self.n_neurons),
                f"{prefix}/n_images": np.int64(self.n_images), f"{prefix}/base": np.int64(self.base),
                f"{prefix}/T": np.int64(self.T), f"{prefix}/max_value": np.float64(self.max_value)}


class MomentAccumulator:
    """Sum and second moment over channels at a binary site (for the lag-PCA basis)."""

    def __init__(self):
        self.s1 = None
        self.s2 = None
        self.n = 0

    @torch.no_grad()
    def update(self, z):
        T, N, C = z.shape[:3]
        zz = z.reshape(T, N, C, -1).double()
        s1 = zz.sum(dim=(0, 1, 3))
        s2 = torch.einsum("tncl,tndl->cd", zz, zz)
        self.s1 = s1 if self.s1 is None else self.s1 + s1
        self.s2 = s2 if self.s2 is None else self.s2 + s2
        self.n += T * N * zz.shape[-1]

    def basis(self, q):
        mu = self.s1 / self.n
        cov = self.s2 / self.n - torch.outer(mu, mu)
        evals, evecs = torch.linalg.eigh(cov)
        idx = torch.argsort(evals, descending=True)[:q]
        return mu.float(), evecs[:, idx].float(), evals[idx].float()


# ------------------------------------------------------------------ per-image detector features

@torch.no_grad()
def detector_features(z, pca=None, per_channel_timing=False):
    """Per-image features of a binary site z [T, N, C, H, W]. Returns dict of numpy arrays.

    counts (rate) features:
      chan_counts [N, C]       spikes per channel (summed over t and space)
      pop_t       [N, T]       fraction of neurons spiking at each t
      n_k         [N, T+1]     number of neurons with each count k
    timing features (raw sums; deviations from the uniform-order expectation are formed
    in the analysis, so they carry no count information by construction):
      sum_centroid_k, sum_first_k, sum_trans_k  [N, T+1]
      chan_centroid_sum, chan_n_multi  [N, C]   (if per_channel_timing) over 0<k<T neurons
      lag1 [N, q, q]           lag-1 cross-channel covariance in the clean PCA basis
    """
    T, N, C = z.shape[:3]
    s = z.reshape(T, N, C, -1)
    L = s.shape[-1]
    k, cen, first, trans = neuron_timing(s)
    out = {
        "chan_counts": k.sum(-1).cpu().numpy().astype(np.int32),
        "pop_t": s.mean(dim=(2, 3)).transpose(0, 1).cpu().numpy().astype(np.float32),
        "n_k": per_image_by_count(torch.ones_like(k), k, T).cpu().numpy().astype(np.int32),
        "sum_centroid_k": per_image_by_count(cen, k, T).cpu().numpy(),
        "sum_first_k": per_image_by_count(first, k, T).cpu().numpy(),
        "sum_trans_k": per_image_by_count(trans, k, T).cpu().numpy(),
    }
    if per_channel_timing:
        multi = ((k > 0) & (k < T)).to(s.dtype)
        out["chan_centroid_sum"] = (cen * multi).sum(-1).cpu().numpy().astype(np.float32)
        out["chan_n_multi"] = multi.sum(-1).cpu().numpy().astype(np.int32)
    if pca is not None:
        mu, P = pca
        y = torch.einsum("tncl,cq->tnql", s - mu.view(1, 1, C, 1), P)
        lag = torch.einsum("tnql,tnrl->nqr", y[:-1], y[1:]) / ((T - 1) * L)
        out["lag1"] = lag.cpu().numpy().astype(np.float32)
    return out


# ------------------------------------------------------------------ capacity / entropy (numpy)

def log2_binom(T, k):
    return math.log2(math.comb(T, k))


def entropy_mm_bits(counts):
    """Plug-in entropy with the Miller-Madow bias correction, in bits."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    h = -(p * np.log(p)).sum() + (len(p) - 1) / (2.0 * n)
    return float(h / np.log(2))


def conditional_entropy_bits(hist, cond):
    """H(code | cond) = sum_g p(g) H(code | g), Miller-Madow per group.

    hist: [..., K] counts (leading dims are extra conditioning, e.g. channel).
    cond: [K] group label of each code (e.g. its count).
    """
    h = np.asarray(hist, dtype=np.float64).reshape(-1, hist.shape[-1])
    total = h.sum()
    out = 0.0
    for g in np.unique(cond):
        sel = cond == g
        for row in h[:, sel]:
            n = row.sum()
            if n > 0:
                out += n / total * entropy_mm_bits(row)
    return out


def code_table(base, T):
    codes = torch.arange(base ** T)
    digits = decode(codes, base, T).numpy()                         # [T, K]
    s = digits.sum(0)
    # number of distinct orderings of each code's multiset of values
    lg = math.lgamma
    multinom = np.array([lg(T + 1) - sum(lg(int((digits[:, i] == v).sum()) + 1) for v in range(base))
                         for i in range(base ** T)]) / math.log(2)
    # number of sequences in {0..base-1}^T with each sum
    n_seq = np.bincount(s, minlength=(base - 1) * T + 1)
    return s, multinom, n_seq


def capacity_metrics(hist, base, T):
    """Timing capacity of a site from its code histogram ([C, K] or [K])."""
    hist = np.asarray(hist, dtype=np.float64)
    pooled = hist.reshape(-1, hist.shape[-1]).sum(0)
    n = pooled.sum()
    s, multinom_bits, n_seq = code_table(base, T)
    p_code = pooled / n
    p_k = np.bincount(s, weights=p_code, minlength=(base - 1) * T + 1)
    out = {"n_neuron_samples": int(n), "p_count": p_k.tolist()}
    if base == 2:
        cap = sum(p_k[k] * log2_binom(T, k) for k in range(T + 1))
        out["frac_multi_order"] = float(p_k[1:T].sum())
    else:
        cap = float((p_code * multinom_bits).sum())                  # reachable by permutation
        out["frac_multi_order"] = float(p_code[multinom_bits > 1e-9].sum())
        out["capacity_sum_bits"] = float(sum(p_k[v] * math.log2(n_seq[v]) for v in range(len(p_k))))
    out["capacity_bits"] = float(cap)
    out["capacity_max_bits"] = float(max(log2_binom(T, k) for k in range(T + 1))) if base == 2 \
        else float(multinom_bits.max())
    out["H_order_given_count_bits"] = conditional_entropy_bits(pooled, s)
    if hist.ndim == 2:
        out["H_order_given_count_channel_bits"] = conditional_entropy_bits(hist, s)
    ref = out["capacity_bits"] if base == 2 else out["capacity_sum_bits"]
    out["H_over_capacity"] = float(out["H_order_given_count_bits"] / ref) if ref > 0 else float("nan")
    return out
