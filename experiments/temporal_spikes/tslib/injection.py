"""Run the SEW-ResNet-18 up to a site, then continue from a (possibly modified) tensor.

The network is the sequence: stem (conv+BN repeated T times, LIF) -> 8 SEW blocks
(layer1.0 ... layer4.1) -> global average pool -> fc -> mean over t.

Sites
-----
  "layerK_in"      input of stage K (= output of stage K-1), K in 2..4. Integer-valued:
                   SEW ADD outputs are sums of binary spike trains, values 0..3.
                   The modified tensor feeds BOTH the residual branch and the identity
                   path of the next block, exactly as the original tensor did.
  "layer4_out"     output of the last block. Downstream is pool -> fc -> mean_t, which is
                   linear and symmetric in t: count-preserving changes cannot alter the
                   logits (null site).
  "layerK.B.sn1"   binary output of the first LIF of block B in stage K. Downstream:
                   cb2 -> sn2 (+ unchanged identity path) -> later blocks.
  "layerK.B.sn2"   binary output of the second LIF. Downstream: + identity -> later
                   blocks. For layer4.1.sn2 this is a null site.

run_to_site() resets every neuron and runs the prefix. run_from_site() resets every
neuron again and runs only the suffix, so no LIF state from the prefix or from an
earlier suffix call can leak into the result (tested).
"""
import re

import torch

BLOCK_NAMES = [f"layer{s}.{b}" for s in (1, 2, 3, 4) for b in (0, 1)]


def blocks(model):
    return [getattr(model, f"layer{s}")[b] for s in (1, 2, 3, 4) for b in (0, 1)]


def parse_site(site):
    m = re.fullmatch(r"layer([2-4])_in", site)
    if m:
        return ("input", BLOCK_NAMES.index(f"layer{m.group(1)}.0"), None)
    if site == "layer4_out":
        return ("input", len(BLOCK_NAMES), None)
    m = re.fullmatch(r"(layer[1-4]\.[01])\.(sn1|sn2)", site)
    if m:
        return ("internal", BLOCK_NAMES.index(m.group(1)), m.group(2))
    raise ValueError(f"unknown site {site!r}")


def is_binary_site(site):
    return parse_site(site)[0] == "internal"


def stem(model, x):
    s = model.bn1(model.conv1(x))
    return model.sn1(s.unsqueeze(0).repeat(model.T, 1, 1, 1, 1))


def head(model, z):
    """Logits from the last block output: mean over t of fc(pool(z_t))."""
    return model.fc(model.pool(z)).mean(0)


def head_per_t(model, z):
    """[T, N, 10] per-timestep logits fc(pool(z_t)); their mean over t is the logit."""
    return model.fc(model.pool(z))


@torch.no_grad()
def run_to_site(model, x, site):
    """Returns a state dict with 'z' (the tensor at the site, [T,N,C,H,W]) and, for
    internal sites, 'idt' (the block's identity-path output, which is not modified)."""
    kind, k, which = parse_site(site)
    model.reset()
    z = stem(model, x)
    bl = blocks(model)
    for blk in bl[:k]:
        z = blk(z)
    if kind == "input":
        return {"site": site, "z": z}
    blk = bl[k]
    a = blk.sn1(blk.cb1(z))
    if which == "sn2":
        a = blk.sn2(blk.cb2(a))
    idt = z if blk.downsample is None else blk.downsample_sn(blk.downsample(z))
    return {"site": site, "z": a, "idt": idt}


@torch.no_grad()
def run_from_site(model, state, z, per_t=False):
    """Continue from tensor z at state['site']. Returns logits [N,10] (and [T,N,10])."""
    kind, k, which = parse_site(state["site"])
    model.reset()
    bl = blocks(model)
    if kind == "internal":
        blk = bl[k]
        a = blk.sn2(blk.cb2(z)) if which == "sn1" else z
        out = a + state["idt"]
        start = k + 1
    else:
        out = z
        start = k
    for blk in bl[start:]:
        out = blk(out)
    lt = head_per_t(model, out)
    return (lt.mean(0), lt) if per_t else lt.mean(0)
