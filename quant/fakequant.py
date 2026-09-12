"""QConv2d / QLinear wrappers: BN folding, fake quantization, STE (build-plan.md S4.3).

For MX formats (mxint8, mxfp4, int4_b32) the true reduction dimension of a
conv is the flattened (in_channels * kh * kw) patch, not just the channel
axis (build-plan.md S1.3/S4.1 watch-out: blocking along the wrong axis
"still works" and silently produces wrong accuracy). So MX-format conv
layers go through im2col (F.unfold) and quantize each patch along that
flattened axis, then compute the output as a matmul against the
identically-blocked, flattened weight -- rather than reconstructing a
quantized (N,C,H,W) activation and calling F.conv2d, which can't represent
one activation pixel being quantized differently in each overlapping patch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant import formats

_BLOCK_FORMATS = ("mxint8", "mxfp4", "int4_b32")
_BLOCK_SIZE = formats.BLOCK_SIZE
_ELEMENT_MAX = {**formats._FMT_MAX, **formats._BLOCK_ELEMENT_MAX}


def _fold_bn(weight, bias, bn):
    """Fold eval-mode BatchNorm stats into the preceding conv/linear's weight+bias."""
    with torch.no_grad():
        std = torch.sqrt(bn.running_var + bn.eps)
        gamma_over_std = bn.weight / std
        shape = [-1] + [1] * (weight.ndim - 1)
        w = weight * gamma_over_std.view(*shape)
        b0 = bias if bias is not None else torch.zeros_like(bn.running_mean)
        b = (b0 - bn.running_mean) * gamma_over_std + bn.bias
    return w.clone(), b.clone()


def _elementwise_scale(x_np, fmt, axis):
    amax = np.max(np.abs(x_np), axis=axis, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    return amax / _ELEMENT_MAX[fmt]


def _block_scale_pow2(scale, fmt):
    exp = scale.astype(np.float64) if fmt == "int4_b32" else scale.astype(np.float64) - 127.0
    return np.exp2(exp)


class _STEElementwise(torch.autograd.Function):
    """Fake-quant with a fixed scale; backward is identity inside the
    representable range and zero outside (build-plan.md S4.3 STE)."""

    @staticmethod
    def forward(ctx, x, fmt, scale_np):
        x_np = x.detach().cpu().double().numpy()
        codes = formats.encode(fmt, x_np, scale_np)
        recon = formats.decode(fmt, codes, scale_np)
        bound_np = _ELEMENT_MAX[fmt] * np.broadcast_to(scale_np, x_np.shape)
        mask = torch.as_tensor(x_np.__abs__() <= bound_np, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(mask)
        return torch.as_tensor(recon, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None, None


def _fake_quant_elementwise(x, fmt, scale_np):
    return _STEElementwise.apply(x, fmt, scale_np)


class _STEBlock(torch.autograd.Function):
    """Fake-quant along the last axis in blocks of BLOCK_SIZE, picking a
    fresh shared scale per block every call (no calibration for MX)."""

    @staticmethod
    def forward(ctx, x, fmt):
        x_np = x.detach().cpu().double().numpy()
        codes, scale = formats.quantize(fmt, x_np)
        recon = formats.dequantize(fmt, codes, scale).reshape(x_np.shape)
        scale_pow2 = _block_scale_pow2(scale, fmt)
        bound_per_block = _ELEMENT_MAX[fmt] * scale_pow2
        bound_np = np.repeat(bound_per_block, _BLOCK_SIZE, axis=-1)
        mask = torch.as_tensor(np.abs(x_np) <= bound_np, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(mask)
        return torch.as_tensor(recon, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None


def _fake_quant_block(x, fmt):
    return _STEBlock.apply(x, fmt)


def _pad_last_to_block(x):
    pad = (-x.shape[-1]) % _BLOCK_SIZE
    return (F.pad(x, (0, pad)) if pad else x), pad


class QConv2d(nn.Module):
    def __init__(self, conv, bn, fmt):
        super().__init__()
        self.stride, self.padding = conv.stride, conv.padding
        self.dilation, self.groups = conv.dilation, conv.groups
        self.fmt = fmt
        if bn is not None:
            weight, bias = _fold_bn(conv.weight, conv.bias, bn)
        else:
            weight = conv.weight.detach().clone()
            bias = (conv.bias.detach().clone() if conv.bias is not None
                    else torch.zeros(conv.out_channels))
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self._act_scale = None  # set by quant.calibrate for elementwise formats only

    def forward(self, x):
        if self.fmt is None:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                             self.dilation, self.groups)
        if self.fmt in _BLOCK_FORMATS:
            return self._forward_block(x)

        out_ch = self.weight.shape[0]
        w_np = self.weight.detach().cpu().double().numpy()
        w_scale = _elementwise_scale(w_np.reshape(out_ch, -1), self.fmt, axis=1).reshape(out_ch, 1, 1, 1)
        qw = _fake_quant_elementwise(self.weight, self.fmt, w_scale)

        if self._act_scale is not None:
            a_scale = self._act_scale
        else:
            a_scale = _elementwise_scale(x.detach().cpu().double().numpy(), self.fmt, axis=None)
        qx = _fake_quant_elementwise(x, self.fmt, a_scale)

        return F.conv2d(qx, qw, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def _forward_block(self, x):
        n, _, h, w = x.shape
        out_ch, in_ch, kh, kw = self.weight.shape
        out_h = (h + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
        out_w = (w + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1

        patches = F.unfold(x, (kh, kw), dilation=self.dilation, padding=self.padding, stride=self.stride)
        patches = patches.transpose(1, 2)                 # (N, L, in_ch*kh*kw)
        patches, pad = _pad_last_to_block(patches)
        q_patches = _fake_quant_block(patches, self.fmt)  # (N, L, red+pad)

        w_flat = self.weight.reshape(out_ch, -1)           # (out_ch, in_ch*kh*kw)
        w_flat, _ = _pad_last_to_block(w_flat)
        q_w = _fake_quant_block(w_flat, self.fmt)           # (out_ch, red+pad)

        out = torch.einsum("nlk,ok->nol", q_patches, q_w) + self.bias.view(1, -1, 1)
        return out.reshape(n, out_ch, out_h, out_w)


class QLinear(nn.Module):
    def __init__(self, linear, fmt):
        super().__init__()
        self.fmt = fmt
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone() if linear.bias is not None
                                  else torch.zeros(linear.out_features))
        self._act_scale = None

    def forward(self, x):
        if self.fmt is None:
            return F.linear(x, self.weight, self.bias)
        if self.fmt in _BLOCK_FORMATS:
            xp, _ = _pad_last_to_block(x)
            wp, _ = _pad_last_to_block(self.weight)
            qx = _fake_quant_block(xp, self.fmt)
            qw = _fake_quant_block(wp, self.fmt)
            return F.linear(qx, qw, self.bias)

        w_np = self.weight.detach().cpu().double().numpy()
        w_scale = _elementwise_scale(w_np, self.fmt, axis=1)
        qw = _fake_quant_elementwise(self.weight, self.fmt, w_scale)

        if self._act_scale is not None:
            a_scale = self._act_scale
        else:
            a_scale = _elementwise_scale(x.detach().cpu().double().numpy(), self.fmt, axis=None)
        qx = _fake_quant_elementwise(x, self.fmt, a_scale)

        return F.linear(qx, qw, self.bias)


def _convert_recursive(module, fmt, skip, prefix):
    children = list(module.named_children())
    i = 0
    while i < len(children):
        name, child = children[i]
        qualified = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and qualified not in skip:
            bn, consumed = None, 1
            if i + 1 < len(children) and isinstance(children[i + 1][1], nn.BatchNorm2d):
                bn = children[i + 1][1]
                consumed = 2
            setattr(module, name, QConv2d(child, bn, fmt))
            if bn is not None:
                setattr(module, children[i + 1][0], nn.Identity())
            i += consumed
            continue
        if isinstance(child, nn.Linear) and qualified not in skip:
            setattr(module, name, QLinear(child, fmt))
            i += 1
            continue
        _convert_recursive(child, fmt, skip, qualified)
        i += 1


def convert_model(model, fmt, first_last=True):
    """Replace every Conv2d/Linear (+ any immediately-following BatchNorm2d,
    folded in) with QConv2d/QLinear for `fmt`. Mutates `model` in place and
    returns it. `fmt=None` folds BN but quantizes nothing (S4.3 done-when
    check: must reproduce FP32 accuracy exactly). `first_last=False` would
    skip the first and last quantizable layer; this project always leaves
    it True ("quantize everything, for honesty").
    """
    names = [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    skip = {names[0], names[-1]} if (not first_last and names) else set()
    _convert_recursive(model, fmt, skip, prefix="")
    return model
