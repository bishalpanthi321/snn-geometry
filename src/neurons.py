"""Multi-step LIF neuron.

The primary implementation is spikingjelly's LIFNode. A plain-PyTorch LIF with
identical dynamics is kept as a fallback (and as an independent reference that
the unit tests compare against spikingjelly).

Dynamics (decay_input=False, hard reset), per timestep t:
    h_t = v_{t-1} - (v_{t-1} - v_reset) / tau + x_t     (charge; 0.5 decay for tau=2)
    s_t = Heaviside(h_t - v_th)                         (ATan surrogate gradient)
    v_t = h_t * (1 - s_t) + v_reset * s_t               (hard reset)
With v_reset = 0 and tau = 2 this is v_t = 0.5 v_{t-1} + x_t, i.e. beta = 0.5.
decay_input=False is chosen so the input is not halved; BatchNorm precedes every
neuron so either choice is reachable, but this one matches the "beta = 0.5" form.
"""
import math

import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import neuron as sj_neuron
    from spikingjelly.activation_based import surrogate as sj_surrogate
    SPIKINGJELLY_AVAILABLE = True
except Exception:  # import errors, broken CUDA extensions, etc.
    SPIKINGJELLY_AVAILABLE = False


class ATanSpike(torch.autograd.Function):
    """Heaviside forward, ATan surrogate backward (same formula as spikingjelly)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        sg = (a / 2) / (1 + (math.pi / 2 * a * x).pow(2))
        return grad_out * sg, None


class MultiStepLIF(nn.Module):
    """LIF over a [T, N, ...] sequence. State lives only inside one forward call
    for the plain implementation; the spikingjelly node keeps state until reset(),
    which the SNN model calls at the start of every forward."""

    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=True,
                 surrogate_alpha=2.0, impl="auto"):
        super().__init__()
        if impl == "auto":
            impl = "spikingjelly" if SPIKINGJELLY_AVAILABLE else "plain"
        if impl == "spikingjelly" and not SPIKINGJELLY_AVAILABLE:
            raise RuntimeError("spikingjelly requested but not importable")
        self.impl = impl
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.v_reset = float(v_reset)
        self.detach_reset = detach_reset
        self.alpha = float(surrogate_alpha)
        # Recording switches, used by feature extraction / firing-rate logging only.
        self.record_membrane = False
        self.track_rate = False
        self.h_seq = None
        self.spike_sum = 0.0
        self.spike_count = 0
        if impl == "spikingjelly":
            self.node = sj_neuron.LIFNode(
                tau=self.tau, decay_input=False, v_threshold=self.v_threshold,
                v_reset=self.v_reset, surrogate_function=sj_surrogate.ATan(alpha=self.alpha),
                detach_reset=detach_reset, step_mode="m", store_v_seq=False)

    def reset(self):
        if self.impl == "spikingjelly":
            self.node.reset()
        self.h_seq = None

    def reset_rate(self):
        self.spike_sum = 0.0
        self.spike_count = 0

    def _plain_forward(self, x_seq):
        v = torch.full_like(x_seq[0], self.v_reset)
        spikes, hs = [], []
        for t in range(x_seq.shape[0]):
            h = v - (v - self.v_reset) / self.tau + x_seq[t]
            s = ATanSpike.apply(h - self.v_threshold, self.alpha)
            s_r = s.detach() if self.detach_reset else s
            v = h * (1.0 - s_r) + self.v_reset * s_r
            spikes.append(s)
            if self.record_membrane:
                hs.append(h.detach())
        if self.record_membrane:
            self.h_seq = torch.stack(hs)
        return torch.stack(spikes)

    def forward(self, x_seq):
        if self.impl == "plain":
            s = self._plain_forward(x_seq)
        else:
            self.node.store_v_seq = self.record_membrane
            s = self.node(x_seq)
            if self.record_membrane:
                # spikingjelly stores the post-reset potential; rebuild the pre-spike
                # potential h_t from it so both implementations expose the same thing.
                v_post = self.node.v_seq.detach()
                v_prev = torch.cat([torch.full_like(v_post[:1], self.v_reset), v_post[:-1]])
                self.h_seq = v_prev - (v_prev - self.v_reset) / self.tau + x_seq.detach()
        if self.track_rate:
            self.spike_sum += float(s.detach().sum())
            self.spike_count += s.numel()
        return s
