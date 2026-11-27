"""Helpers shared by the unit tests (kept here, not in conftest.py, so that the
repository-level test run, which has its own conftest.py, can import them too)."""
import torch


def small_snn(T=4, impl="plain", seed=0):
    """Random-weight SEW-ResNet-18 in float64 with BN statistics that make neurons fire."""
    from src.models import SEWResNet18
    from .common import randomize_bn
    torch.manual_seed(seed)
    m = SEWResNet18(T=T, impl=impl)
    randomize_bn(m, seed)
    return m.double().eval()


def binary_z(T=4, N=3, C=5, H=4, W=4, p=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(T, N, C, H, W, generator=g) < p).double()


def integer_z(T=4, N=3, C=5, H=4, W=4, vmax=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vmax + 1, (T, N, C, H, W), generator=g).double()
