"""SNN correctness: neuron dynamics, state reset, logits = fc(mean rate)."""
import pytest
import torch

from src.features import FeatureRecorder
from src.models import SEWResNet18
from src.neurons import SPIKINGJELLY_AVAILABLE, MultiStepLIF

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def small_snn(impl="auto", T=4):
    torch.manual_seed(0)
    m = SEWResNet18(T=T, impl=impl).double().eval()
    # Random BN statistics so that neurons actually fire in an untrained net.
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.2, 0.6)
            mod.weight.data.uniform_(0.8, 1.6)
            mod.bias.data.uniform_(0.0, 0.5)
    return m


def test_plain_lif_dynamics_by_hand():
    lif = MultiStepLIF(impl="plain")
    lif.record_membrane = True
    x = torch.tensor([[0.6], [0.6], [0.6], [2.0]]).double()  # [T=4, N=1]
    s = lif(x)
    # v: 0.6 -> 0.3+0.6=0.9 -> 0.45+0.6=1.05 (spike, reset 0) -> 0+2.0 (spike)
    assert s.flatten().tolist() == [0.0, 0.0, 1.0, 1.0]
    torch.testing.assert_close(lif.h_seq.flatten(), torch.tensor([0.6, 0.9, 1.05, 2.0]).double())


@pytest.mark.skipif(not SPIKINGJELLY_AVAILABLE, reason="spikingjelly not installed")
def test_spikingjelly_matches_plain_lif_forward_backward_and_membrane():
    torch.manual_seed(0)
    x = (torch.randn(6, 8, 5, 3) * 1.5).double()
    outs = {}
    for impl in ["plain", "spikingjelly"]:
        lif = MultiStepLIF(impl=impl)
        lif.record_membrane = True
        xi = x.clone().requires_grad_(True)
        s = lif(xi)
        (s * torch.linspace(0, 1, s.numel()).view_as(s).double()).sum().backward()
        outs[impl] = (s.detach(), lif.h_seq.clone(), xi.grad.clone())
    for a, b in zip(outs["plain"], outs["spikingjelly"]):
        torch.testing.assert_close(a, b)
    s, h, _ = outs["spikingjelly"]
    assert torch.equal(s, (h >= 1.0).double())      # reconstructed membrane agrees with spikes
    assert 0.05 < s.mean() < 0.95                  # the test is not trivially all-0/all-1


@pytest.mark.parametrize("impl", ["plain"] + (["spikingjelly"] if SPIKINGJELLY_AVAILABLE else []))
def test_state_reset_no_leak_between_batches(impl):
    m = small_snn(impl)
    torch.manual_seed(1)
    a, b = torch.randn(3, 3, 32, 32).double(), torch.randn(3, 3, 32, 32).double()
    rec = FeatureRecorder(m, is_snn=True)
    with torch.no_grad(), rec:
        m(b)
        fb_alone = {k: v.clone() for k, v in rec.collect().items()}
        m(a)
        m(b)
        fb_after = rec.collect()
    for k in fb_alone:
        torch.testing.assert_close(fb_alone[k], fb_after[k], rtol=0, atol=0)
    assert fb_alone["penultimate_spikes"].abs().sum() > 0   # something actually fired

    # Control: without the reset spikingjelly's state DOES leak, so the test above is
    # meaningful. (The plain LIF keeps state only inside one call, so it cannot leak.)
    if impl == "plain":
        return
    with torch.no_grad():
        m(a)
        leaky = m.forward_unreset(b)
        m.reset()
        clean = m.forward_unreset(b)
    assert not torch.allclose(leaky, clean)


def test_logits_equal_fc_of_time_averaged_rate():
    m = small_snn()
    x = torch.randn(4, 3, 32, 32).double()
    rec = FeatureRecorder(m, is_snn=True)
    with torch.no_grad(), rec:
        logits = m(x)
        f = rec.collect()
    rate = f["penultimate_spikes"].mean(1)               # [N, T, D] -> [N, D]
    torch.testing.assert_close(logits, m.fc(rate), rtol=0, atol=1e-12)
    assert f["penultimate_spikes"].shape == (4, 4, 512)


def test_layer4_gap_equals_penultimate():
    m = small_snn()
    rec = FeatureRecorder(m, is_snn=True)
    with torch.no_grad(), rec:
        m(torch.randn(2, 3, 32, 32).double())
        f = rec.collect()
    torch.testing.assert_close(f["layer4_spikes"], f["penultimate_spikes"], rtol=0, atol=0)
