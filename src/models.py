"""CIFAR-style ResNet-18 (ANN) and SEW-ResNet-18 (SNN) with the same layout.

Shared layout: 3x3 conv stem (stride 1, no max-pool), four stages of two basic
blocks with 64/128/256/512 channels (stride 1/2/2/2), global average pool,
linear layer to 10 classes. Module names (layer1..layer4, pool, fc) are the same
in both models so the feature hooks are identical.

Known difference beyond the neuron type: the ANN is the standard ResNet (ReLU
after the residual sum), while SEW puts a spiking neuron on each branch and adds
spikes afterwards (Fang et al. 2021, ADD connect function). That is what
"SEW-ResNet" means; a ReLU after the sum has no SEW equivalent.
"""
import torch
import torch.nn as nn

from .neurons import MultiStepLIF


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


def conv1x1(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 1, stride=stride, bias=False)


class GlobalAvgPool(nn.Module):
    """Mean over the two trailing spatial dims; works for [N,C,H,W] and [T,N,C,H,W]."""

    def forward(self, x):
        return x.mean(dim=(-2, -1))


# ---------------------------------------------------------------- ANN

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(conv1x1(cin, cout, stride), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._stage(64, 64, 1)
        self.layer2 = self._stage(64, 128, 2)
        self.layer3 = self._stage(128, 256, 2)
        self.layer4 = self._stage(256, 512, 2)
        self.pool = GlobalAvgPool()
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _stage(cin, cout, stride):
        return nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.pool(x))


# ---------------------------------------------------------------- SNN

class SeqConvBN(nn.Module):
    """Conv + BN applied to every timestep by folding T into the batch.
    BN statistics are therefore computed over T*N samples (as spikingjelly's
    SeqToANNContainer does)."""

    def __init__(self, conv, bn):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x_seq):
        T, N = x_seq.shape[:2]
        y = self.bn(self.conv(x_seq.flatten(0, 1)))
        return y.view(T, N, *y.shape[1:])


class SEWBlock(nn.Module):
    def __init__(self, cin, cout, stride, lif):
        super().__init__()
        self.cb1 = SeqConvBN(conv3x3(cin, cout, stride), nn.BatchNorm2d(cout))
        self.sn1 = lif()
        self.cb2 = SeqConvBN(conv3x3(cout, cout), nn.BatchNorm2d(cout))
        self.sn2 = lif()
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = SeqConvBN(conv1x1(cin, cout, stride), nn.BatchNorm2d(cout))
            self.downsample_sn = lif()

    def forward(self, x_seq):
        out = self.sn2(self.cb2(self.sn1(self.cb1(x_seq))))
        identity = x_seq if self.downsample is None else self.downsample_sn(self.downsample(x_seq))
        # ADD connect function: outputs are sums of spikes, so values are in {0, 1, 2}
        # (identity may itself be a block output; after the stem it is binary).
        return out + identity


class SEWResNet18(nn.Module):
    def __init__(self, T=4, num_classes=10, tau=2.0, v_threshold=1.0, v_reset=0.0,
                 detach_reset=True, surrogate_alpha=2.0, impl="auto"):
        super().__init__()
        self.T = T

        def lif():
            return MultiStepLIF(tau=tau, v_threshold=v_threshold, v_reset=v_reset,
                                detach_reset=detach_reset, surrogate_alpha=surrogate_alpha,
                                impl=impl)

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.sn1 = lif()
        self.layer1 = nn.Sequential(SEWBlock(64, 64, 1, lif), SEWBlock(64, 64, 1, lif))
        self.layer2 = nn.Sequential(SEWBlock(64, 128, 2, lif), SEWBlock(128, 128, 1, lif))
        self.layer3 = nn.Sequential(SEWBlock(128, 256, 2, lif), SEWBlock(256, 256, 1, lif))
        self.layer4 = nn.Sequential(SEWBlock(256, 512, 2, lif), SEWBlock(512, 512, 1, lif))
        self.pool = GlobalAvgPool()
        self.fc = nn.Linear(512, num_classes)

    def lif_modules(self):
        return [m for m in self.modules() if isinstance(m, MultiStepLIF)]

    def reset(self):
        for m in self.lif_modules():
            m.reset()

    def forward_unreset(self, x):
        """Forward pass WITHOUT resetting neuron state. Only for the state-leak test."""
        # Direct encoding: the same image at every timestep. Conv+BN of identical
        # copies is identical, so compute it once and repeat (BN batch statistics
        # over T identical copies equal those over the N images).
        stem = self.bn1(self.conv1(x))
        x_seq = stem.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x_seq = self.sn1(x_seq)
        x_seq = self.layer4(self.layer3(self.layer2(self.layer1(x_seq))))
        pooled = self.pool(x_seq)                    # [T, N, 512] pooled spikes
        return self.fc(pooled).mean(0)               # per-timestep logits, averaged over T

    def forward(self, x):
        # Reset at the start of every call so state can never leak between batches.
        self.reset()
        return self.forward_unreset(x)


def build_model(name, cfg, T=None):
    if name == "ann":
        return ResNet18()
    if name == "snn":
        s = cfg["snn"]
        return SEWResNet18(T=T or s["T"], tau=s["tau"], v_threshold=s["v_threshold"],
                           v_reset=s["v_reset"], detach_reset=s["detach_reset"],
                           surrogate_alpha=s["surrogate_alpha"], impl=s["impl"])
    raise ValueError(name)


STAGES = ["layer1", "layer2", "layer3", "layer4"]


def stage_lifs(model):
    """{stage_name: [MultiStepLIF, ...]} including the stem neuron."""
    out = {"stem": [model.sn1]}
    for st in STAGES:
        out[st] = [m for m in getattr(model, st).modules() if isinstance(m, MultiStepLIF)]
    return out
