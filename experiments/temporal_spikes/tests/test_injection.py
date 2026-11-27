"""Running the network up to a site and on from it must equal the plain forward pass."""
import pytest
import torch

from src.neurons import SPIKINGJELLY_AVAILABLE
from tslib import interventions as iv
from tslib.injection import run_from_site, run_to_site
from tslib.testing import small_snn

IMPLS = ["plain"] + (["spikingjelly"] if SPIKINGJELLY_AVAILABLE else [])
ALL_SITES = ["layer2_in", "layer3_in", "layer4_in", "layer4_out",
             "layer1.0.sn1", "layer2.0.sn2", "layer3.1.sn1", "layer4.0.sn1", "layer4.1.sn2"]
UPSTREAM_SITES = ["layer3_in", "layer4_in", "layer4.0.sn1"]
NULL_SITES = ["layer4_out", "layer4.1.sn2"]


def x_batch(n=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 32, 32, generator=g).double()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("site", ALL_SITES)
def test_prefix_suffix_equals_forward(site, T, impl):
    m = small_snn(T=T, impl=impl)
    x = x_batch()
    ref = m(x)
    st = run_to_site(m, x, site)
    out = run_from_site(m, st, st["z"])
    assert torch.equal(out, ref)


@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("site", UPSTREAM_SITES)
def test_identity_intervention_exact(site, T):
    m = small_snn(T=T)
    x = x_batch(n=4)
    ref = m(x)
    st = run_to_site(m, x, site)
    z_id = st["z"].clone()
    out = run_from_site(m, st, z_id)
    assert torch.equal(out, ref)
    assert torch.equal(out.argmax(1), ref.argmax(1))


@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("site", NULL_SITES)
def test_final_site_null(site, T):
    """Downstream of these sites is pool -> fc -> mean over t: count-preserving changes can
    only change the logits by floating-point reordering of the time sum."""
    m = small_snn(T=T)
    x = x_batch()
    st = run_to_site(m, x, site)
    z = st["z"]
    ref = run_from_site(m, st, z)
    variants = [iv.uniform_perm(z, 0), iv.uniform_perm(z, 1), iv.circular_shift(z, 2),
                iv.even_spread(z), iv.same_count_swap(z, 3)]
    variants += [iv.shared_perm(z, p) for p in iv.all_shared_perms(T, 5)]
    for z2 in variants:
        assert (run_from_site(m, st, z2) - ref).abs().max() <= 1e-9   # 1e-5 is the fp32 budget


@pytest.mark.parametrize("site", UPSTREAM_SITES)
def test_upstream_perm_has_effect(site):
    """At sites upstream of a LIF the suffix really runs neuron dynamics, so reordering
    spikes changes the logits (otherwise Experiment A would be vacuous)."""
    m = small_snn(T=4)
    x = x_batch()
    st = run_to_site(m, x, site)
    z = st["z"]
    assert 0 < float(z.mean()) < float(z.max())          # neurons fire, not all saturated
    ref = run_from_site(m, st, z)
    out = run_from_site(m, st, iv.uniform_perm(z, 0))
    assert (out - ref).abs().max() > 1e-6


@pytest.mark.parametrize("impl", IMPLS)
def test_suffix_state_reset(impl):
    """No state leaks between suffix calls: B after A equals B alone, repeats are identical."""
    m = small_snn(T=4, impl=impl)
    xa, xb = x_batch(seed=1), x_batch(seed=2)
    for site in ["layer3_in", "layer4.0.sn1"]:
        sa = run_to_site(m, xa, site)
        sa_z = sa["z"].clone()
        sb = run_to_site(m, xb, site)
        b_alone = run_from_site(m, sb, sb["z"])
        run_from_site(m, sa, iv.uniform_perm(sa_z, 0))
        b_after = run_from_site(m, sb, sb["z"])
        assert torch.equal(b_alone, b_after)
        assert torch.equal(run_from_site(m, sb, sb["z"]), b_alone)
        assert torch.equal(b_alone, m(xb))
