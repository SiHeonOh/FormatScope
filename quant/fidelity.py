"""PyTorch path vs DP32 hardware path on one real layer (build-plan.md S4.7).

Layer: block2.conv1 (16->32 channels, stride 2), the "recommended" layer in
S4.7 -- reduction length in_ch*kh*kw = 16*3*3 = 144 = 4.5 blocks, padded to 5.
int4_b32 has no hardware unit (S1.3) so it is excluded here.

Path (a), "PyTorch": dequantize both operands once and accumulate the whole
144(->160)-element dot product in FP32, exactly like fakequant.py does.
Path (b), "DP32": tb/refs.py's per-format reference, called once per 32-wide
block with the running accumulator threaded through -- bit-for-bit what the
RTL computes (one fused rounding per 32 products, not one per full reduction).
"""

import csv
import datetime
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "models"))

from data import get_loaders  # noqa: E402
from resnet8 import ResNet8  # noqa: E402

from quant import formats  # noqa: E402
from quant.fakequant import _fold_bn, _BLOCK_FORMATS  # noqa: E402
from tb import refs  # noqa: E402

_CKPT_PATH = os.path.join(_REPO_ROOT, "models", "checkpoints", "fp32.pt")
_CSV_PATH = os.path.join(_REPO_ROOT, "results", "fidelity.csv")
_CSV_COLUMNS = ["format", "layer", "n_outputs", "max_abs_diff", "mean_abs_diff",
                "max_rel_diff", "layer_output_change", "exact", "date"]

_HW_FORMATS = ("int4", "int8", "fp8e4m3", "mxint8", "mxfp4", "int4_b32")
_ALIGN_W = 24
_LAYER_NAME = "block2.conv1"
_BATCH_SIZE = 8
_MAX_PAIRS = 300  # sampled (output, out_channel) pairs per format -- see _quantize_and_run


def _select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _layer_patches_and_weight(device):
    """Run the trained model up to block2.conv1's input, fold its BN, and
    im2col both the activation and the folded weight to (n, reduction)."""
    model = ResNet8(num_classes=10)
    model.load_state_dict(torch.load(_CKPT_PATH, map_location="cpu", weights_only=True))
    model.eval().to(device)

    _, test_loader = get_loaders(batch_size=_BATCH_SIZE)
    images, _ = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        x = torch.relu(model.stem_bn(model.stem_conv(images)))
        x = model.block1(x)  # real activation input to block2.conv1

        conv, bn = model.block2.conv1, model.block2.bn1
        weight, bias = _fold_bn(conv.weight, conv.bias, bn)
        out_ch, in_ch, kh, kw = weight.shape

        patches = F.unfold(x, (kh, kw), padding=conv.padding, stride=conv.stride)
        patches = patches.transpose(1, 2).reshape(-1, in_ch * kh * kw)

    patches = patches.cpu().double().numpy()
    w_np = weight.reshape(out_ch, -1).cpu().double().numpy()

    pad = (-patches.shape[1]) % formats.BLOCK_SIZE
    if pad:
        patches = np.pad(patches, ((0, 0), (0, pad)))
        w_np = np.pad(w_np, ((0, 0), (0, pad)))
    return patches, w_np


def _quantize_and_run(fmt, patches, w_np, ideal, pairs):
    """Return (path_a, path_b, ideal_sampled), each a flat array over `pairs`
    ((output, out_channel) index pairs). The reference DP32 functions in
    tb/refs.py are pure Python (one call per 32-wide block, meant for
    per-vector RTL comparison, not bulk sweeps), so we sample a bounded
    number of outputs rather than compute the full (n_outputs x out_ch) grid.
    """
    n_outputs, red = patches.shape
    out_ch = w_np.shape[0]
    n_blocks = red // formats.BLOCK_SIZE

    if fmt in _BLOCK_FORMATS:
        p_codes, p_scale = formats.quantize(fmt, patches)      # (N, nblk, 32), (N, nblk)
        w_codes, w_scale = formats.quantize(fmt, w_np)          # (C, nblk, 32), (C, nblk)
        p_dq = formats.dequantize(fmt, p_codes, p_scale).reshape(n_outputs, red)
        w_dq = formats.dequantize(fmt, w_codes, w_scale).reshape(out_ch, red)
    else:
        p_scale = np.max(np.abs(patches)) / formats._FMT_MAX[fmt]
        p_scale = np.array(1.0 if p_scale == 0 else p_scale)
        w_scale = np.max(np.abs(w_np), axis=1, keepdims=True) / formats._FMT_MAX[fmt]
        w_scale = np.where(w_scale == 0, 1.0, w_scale)
        p_codes = formats.encode(fmt, patches, p_scale)
        w_codes = formats.encode(fmt, w_np, w_scale)
        p_dq = formats.decode(fmt, p_codes, p_scale)
        w_dq = formats.decode(fmt, w_codes, w_scale)

    path_a_full = p_dq @ w_dq.T
    path_a = np.array([path_a_full[o, c] for o, c in pairs])
    ideal_sampled = np.array([ideal[o, c] for o, c in pairs])

    # INT/FP8 requantization (weight_scale * activation_scale) happens *after*
    # the accumulator and is excluded from the DP32 unit itself (S1.4/S1.5), so
    # refs.dp32_int/dp32_fp8 return a raw, unscaled result -- rescale it here to
    # compare against path_a (already dequantized) in the same real-value units.
    # MX needs no such step: the E8M0 block scale is already inside dp32_mx.
    path_b = np.zeros(len(pairs), dtype=np.float64)
    if fmt in ("int4", "int8"):
        w_bits = 4 if fmt == "int4" else 8
        for i, (o, c) in enumerate(pairs):
            acc = 0
            for blk in range(n_blocks):
                sl = slice(blk * formats.BLOCK_SIZE, (blk + 1) * formats.BLOCK_SIZE)
                acc = refs.dp32_int(p_codes[o, sl], w_codes[c, sl], acc, w_bits)
            path_b[i] = refs.to_signed32(acc) * float(p_scale) * float(w_scale[c, 0])
    elif fmt == "fp8e4m3":
        for i, (o, c) in enumerate(pairs):
            acc_bits = 0
            for blk in range(n_blocks):
                sl = slice(blk * formats.BLOCK_SIZE, (blk + 1) * formats.BLOCK_SIZE)
                acc_bits, _ = refs.dp32_fp8(p_codes[o, sl], w_codes[c, sl], acc_bits, _ALIGN_W)
            path_b[i] = float(refs.fp32_value(acc_bits)) * float(p_scale) * float(w_scale[c, 0])
    else:  # mxint8, mxfp4, int4_b32
        # formats.py keeps the int4_b32 block scale as a raw exponent; the hardware
        # takes it as E8M0 (raw + 127), like the MX formats (tb/refs.py).
        bias = 127 if fmt == "int4_b32" else 0
        for i, (o, c) in enumerate(pairs):
            acc_bits = 0
            for blk in range(n_blocks):
                acc_bits, _ = refs.dp32_mx(fmt, p_codes[o, blk], w_codes[c, blk],
                                            int(p_scale[o, blk]) + bias, int(w_scale[c, blk]) + bias,
                                            acc_bits, _ALIGN_W)
            path_b[i] = float(refs.fp32_value(acc_bits))

    return path_a, path_b, ideal_sampled


def _append_csv(fmt, n_outputs, max_abs, mean_abs, max_rel, layer_change, exact):
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)
    is_new = not os.path.exists(_CSV_PATH) or os.path.getsize(_CSV_PATH) == 0
    with open(_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(_CSV_COLUMNS)
        writer.writerow([fmt, _LAYER_NAME, n_outputs, f"{max_abs:.8g}", f"{mean_abs:.8g}",
                          f"{max_rel:.8g}", f"{layer_change:.8g}", exact,
                          datetime.date.today().isoformat()])


def run_fidelity(formats_to_run=_HW_FORMATS, device=None):
    device = device or _select_device()
    print(f"Device: {device}")

    if not os.path.exists(_CKPT_PATH):
        raise FileNotFoundError(f"{_CKPT_PATH} not found -- train the FP32 baseline first")

    patches, w_np = _layer_patches_and_weight(device)
    ideal = patches @ w_np.T
    n_outputs, out_ch = ideal.shape
    print(f"layer={_LAYER_NAME}  n_outputs={n_outputs}  out_ch={out_ch}  "
          f"reduction_padded={patches.shape[1]}")

    rng = np.random.default_rng(0)
    n_pairs = min(_MAX_PAIRS, n_outputs * out_ch)
    flat = rng.choice(n_outputs * out_ch, size=n_pairs, replace=False)
    pairs = [(int(i) // out_ch, int(i) % out_ch) for i in flat]
    print(f"sampling {n_pairs} of {n_outputs * out_ch} (output, out_channel) pairs, seed 0")

    results = {}
    for fmt in formats_to_run:
        path_a, path_b, ideal_sampled = _quantize_and_run(fmt, patches, w_np, ideal, pairs)
        abs_diff = np.abs(path_a - path_b)
        denom = np.where(np.abs(path_a) > 1e-12, np.abs(path_a), 1e-12)
        rel_diff = abs_diff / denom
        layer_change = float(np.linalg.norm(path_b - ideal_sampled) / np.linalg.norm(ideal_sampled))
        max_abs, mean_abs, max_rel = float(abs_diff.max()), float(abs_diff.mean()), float(rel_diff.max())
        # int4/int8 are exact integer sums (S4.7); allow only float64 rounding
        # noise from *our* comparison's final rescale-by-float-scale step, which
        # isn't part of the hardware's integer arithmetic.
        exact = max_abs < 1e-9

        print(f"{fmt:>10s}  max_abs={max_abs:.6g}  mean_abs={mean_abs:.6g}  "
              f"max_rel={max_rel:.6g}  layer_change={layer_change:.6g}  exact={exact}")
        _append_csv(fmt, n_pairs, max_abs, mean_abs, max_rel, layer_change, exact)
        results[fmt] = dict(max_abs=max_abs, mean_abs=mean_abs, max_rel=max_rel,
                             layer_change=layer_change, exact=exact)

    for fmt in ("int4", "int8"):
        if fmt in results and not results[fmt]["exact"]:
            print(f"WARNING: {fmt} fidelity gap is not exactly zero "
                  f"(max_abs={results[fmt]['max_abs']:.3g}) -- S4.7 expects exact equality.")
    return results


if __name__ == "__main__":
    run_fidelity()
