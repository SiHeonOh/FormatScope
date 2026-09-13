"""Tests for quant/fakequant.py against the S4.3 "Done when" criteria."""

import copy

import torch
import torch.nn as nn
import pytest

from quant import formats
from quant.fakequant import convert_model


class _Toy(nn.Module):
    """conv+bn -> conv+bn -> linear, small enough to test fast, big enough
    that the reduction dim isn't a multiple of 32 anywhere (exercises padding)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 5, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 7, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(7)
        self.fc = nn.Linear(7 * 4 * 4, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return self.fc(x.flatten(1))


def _fresh_model(seed=0):
    torch.manual_seed(seed)
    m = _Toy()
    # give BN nontrivial folded stats instead of the untrained default (0,1)
    with torch.no_grad():
        m.bn1.running_mean.copy_(torch.randn(5) * 0.1)
        m.bn1.running_var.copy_(torch.rand(5) + 0.5)
        m.bn1.weight.copy_(torch.rand(5) + 0.5)
        m.bn1.bias.copy_(torch.randn(5) * 0.1)
        m.bn2.running_mean.copy_(torch.randn(7) * 0.1)
        m.bn2.running_var.copy_(torch.rand(7) + 0.5)
        m.bn2.weight.copy_(torch.rand(7) + 0.5)
        m.bn2.bias.copy_(torch.randn(7) * 0.1)
    m.eval()
    return m


def test_none_format_reproduces_fp32_exactly():
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8, 8)
    ref = _fresh_model()
    out_ref = ref(x)

    folded = copy.deepcopy(ref)
    convert_model(folded, fmt=None)
    out_folded = folded(x)

    assert torch.allclose(out_ref, out_folded, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("fmt", formats.FORMAT_IDS)
def test_each_format_runs_end_to_end_and_trains(fmt):
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8, 8)
    target = torch.randint(0, 10, (4,))

    model = _fresh_model()
    convert_model(model, fmt=fmt)
    model.train()

    out = model(x)
    assert out.shape == (4, 10)
    assert torch.isfinite(out).all()

    loss = nn.functional.cross_entropy(out, target)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_int8_is_much_closer_to_fp32_than_int4():
    """Sanity check mirroring S4.5's broken-quantizer check, one level down:
    a format with more codes should reconstruct activations more faithfully."""
    torch.manual_seed(0)
    x = torch.randn(8, 3, 8, 8)

    ref = _fresh_model()
    out_ref = ref(x)

    m8 = copy.deepcopy(ref)
    convert_model(m8, fmt="int8")
    m4 = copy.deepcopy(ref)
    convert_model(m4, fmt="int4")

    err8 = (m8(x) - out_ref).abs().mean().item()
    err4 = (m4(x) - out_ref).abs().mean().item()
    assert err8 < err4
