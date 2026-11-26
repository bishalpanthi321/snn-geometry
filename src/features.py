"""Forward hooks that capture stage outputs and the penultimate layer.

Stage outputs are globally average-pooled over space right inside the hook, so
only [N, C] (ANN) or [N, T, C] (SNN) tensors are kept.

Keys returned by collect():
  ANN: layer1..layer4, penultimate                      -> [N, D]
  SNN: layer1_spikes..layer4_spikes, penultimate_spikes -> [N, T, D]
       penultimate_membrane                             -> [N, T, 512]
For the SNN, "spikes" at a stage output are SEW sums of spikes (values 0..2
before pooling). penultimate_membrane is the pre-spike potential h_t of the
last LIF in layer4 (the residual-branch neuron sn2 of the final block), pooled
over space: there is no neuron after the global pool itself.
"""
import torch

LAYERS = ["layer1", "layer2", "layer3", "layer4"]


class FeatureRecorder:
    def __init__(self, model, is_snn, record_membrane=True):
        self.model = model
        self.is_snn = is_snn
        self.record_membrane = record_membrane and is_snn
        self.handles = []
        self.store = {}

    def _hook(self, name):
        sfx = "_spikes" if self.is_snn else ""

        def fn(module, inp, out):
            if name == "penultimate":
                f = out                                    # already pooled
            else:
                f = out.mean(dim=(-2, -1))                 # global average pool over H, W
            if self.is_snn:
                f = f.transpose(0, 1)                      # [T, N, D] -> [N, T, D]
            self.store[name + sfx] = f.detach()
        return fn

    def __enter__(self):
        for name in LAYERS:
            self.handles.append(getattr(self.model, name).register_forward_hook(self._hook(name)))
        self.handles.append(self.model.pool.register_forward_hook(self._hook("penultimate")))
        if self.record_membrane:
            self.model.layer4[-1].sn2.record_membrane = True
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.record_membrane:
            self.model.layer4[-1].sn2.record_membrane = False

    def collect(self):
        out = dict(self.store)
        if self.record_membrane:
            h = self.model.layer4[-1].sn2.h_seq              # [T, N, C, H, W]
            out["penultimate_membrane"] = h.mean(dim=(-2, -1)).transpose(0, 1)
        return out
