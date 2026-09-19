"""Checks on tb/refs.py itself, in pure Python (no simulator).

The RTL is only as trustworthy as its reference, so the reference is checked
against quant/formats.py (decoders) and against exact rational arithmetic
(the fused stage).
"""

import math
from fractions import Fraction

import numpy as np
import pytest

import refs
from quant import formats


def _same_float(x, y):
    return x == y and math.copysign(1.0, x) == math.copysign(1.0, y)


def test_fp8_decoder_matches_formats():
    table = formats.decode_table("fp8e4m3")
    for code in range(256):
        sign, exp, sig, is_nan = refs.decode_fp8(code)
        if table[code] == "nan":
            assert is_nan, f"code 0x{code:02x}"
            continue
        assert not is_nan, f"code 0x{code:02x}"
        value = (-1.0 if sign else 1.0) * sig * 2.0 ** (exp - 9)
        assert _same_float(value, table[code]), f"code 0x{code:02x}: {value} vs {table[code]}"


def test_e2m1_decoder_matches_formats():
    table = formats.decode_table("e2m1")
    for code in range(16):
        sign, mag = refs.decode_e2m1(code)
        assert _same_float((-1.0 if sign else 1.0) * mag / 2, table[code]), f"code 0x{code:x}"


def test_e8m0_decoder_matches_formats():
    table = formats.decode_table("e8m0")
    for code in range(256):
        exp, is_nan = refs.decode_e8m0(code)
        assert is_nan == (table[code] == "nan"), f"code 0x{code:02x}"
        if not is_nan:
            assert 2.0 ** (exp - 127) == table[code], f"code 0x{code:02x}"


def round_fp32(x):
    """Independent round-to-nearest-even of an exact Fraction into FP32 register bits."""
    if x == 0:
        return 0
    sign = int(x < 0)
    m = abs(x)
    e = m.numerator.bit_length() - m.denominator.bit_length()
    if m < Fraction(2) ** e:
        e -= 1
    if m >= Fraction(2) ** (e + 1):
        e += 1
    sig = round(m / Fraction(2) ** (e - 23))       # Fraction.__round__ is half-even
    if sig == 1 << 24:
        sig >>= 1
        e += 1
    return (sign << 31) | ((e + 127) << 23) | (sig & 0x7FFFFF)


def test_fp8_one_times_one():
    one = 0x38                                      # E=7, M=0
    bits, nan = refs.dp32_fp8([one] + [0] * 31, [one] + [0] * 31, 0, 24)
    assert (bits, nan) == (0x3F800000, False)


def test_fp8_nan_lane_contributes_zero():
    one = 0x38
    bits, nan = refs.dp32_fp8([0x7F, one] + [0] * 30, [one, one] + [0] * 30, 0, 24)
    assert (bits, nan) == (0x3F800000, True)


@pytest.mark.parametrize("align_w", [24, 32])
def test_fused_stage_against_exact_arithmetic(align_w):
    """No dropped bits: result is exactly RNE of the true sum. Otherwise: within the documented bound."""
    rng = np.random.default_rng(0)
    acc = 0
    exact_hits = bounded_hits = 0
    for step in range(3000):
        mode = rng.random()
        if mode < 0.6:
            a, b = rng.integers(0, 256, size=(2, 32))
        elif mode < 0.8:
            a, b = rng.integers(0, 24, size=(2, 32)) | (rng.integers(0, 2, size=(2, 32)) << 7)
        else:
            a, b = rng.integers(96, 127, size=(2, 32)) | (rng.integers(0, 2, size=(2, 32)) << 7)
        if rng.random() < 0.05:
            acc = 0
        info = {}
        new, _ = refs.dp32_fp8(a, b, acc, align_w, info)
        exact = refs.fp32_value(acc) + refs.fp8_product_sum(a, b)
        assert not info["saturate"] and not info["underflow"], f"step {step}: FP8 left FP32 range"
        hit = _check_against_exact(new, exact, info, n_terms=33, align_w=align_w, where=f"step {step}")
        exact_hits += hit == "exact"
        bounded_hits += hit == "bounded"
        acc = new
    assert exact_hits > 100 and bounded_hits > 100


def _check_against_exact(new, exact, info, n_terms, align_w, where):
    """'exact' (no dropped bits: correctly rounded), 'bounded', or 'range' (saturated / underflowed)."""
    if info["saturate"]:
        assert new & 0x7FFFFFFF == refs.FP32_MAX_FINITE and abs(exact) > Fraction(2) ** 127, where
        return "range"
    if info["underflow"]:
        assert new == 0 and abs(exact) < Fraction(2) ** -125, where
        return "range"
    if not info["term_sticky"] and not info["rest_sticky"]:
        assert new == round_fp32(exact), f"{where}: 0x{new:08x} vs 0x{round_fp32(exact):08x}"
        return "exact"
    got = refs.fp32_value(new)
    ulp = Fraction(2) ** (((new >> 23) & 0xFF) - 127 - 23) if new else Fraction(0)
    bound = ulp + n_terms * Fraction(2) ** (info["t_max"] - (align_w + 2))
    assert abs(got - exact) <= bound, f"{where}: error {float(got - exact)} > {float(bound)}"
    return "bounded"


def test_mx_block_sums_match_formats():
    """The integer block sums times their scale equal formats.decode products, summed."""
    rng = np.random.default_rng(1)
    for fmt, w in (("mxint8", 8), ("mxfp4", 4), ("int4_b32", 4)):
        # formats.py holds the int4_b32 block scale as a raw exponent; the hardware
        # reference takes it as E8M0 (raw + 127), like the MX formats.
        bias = 127 if fmt == "int4_b32" else 0
        for _ in range(200):
            a, b = rng.integers(0, 1 << w, size=(2, 32))
            sa, sb = (int(s) for s in rng.integers(100, 155, size=2))
            ours = refs.mx_block_value(fmt, a, b, sa, sb)
            va = formats.decode(fmt, a, np.array(sa - bias))
            vb = formats.decode(fmt, b, np.array(sb - bias))
            theirs = sum((Fraction(float(x)) * Fraction(float(y)) for x, y in zip(va.ravel(), vb.ravel())),
                         Fraction(0))
            assert ours == theirs, f"{fmt}: {ours} vs {theirs}"


def test_mx_extreme_block_sums():
    assert refs.mx_block_sum("mxint8", [0x80] * 32, [0x80] * 32) == 1 << 19
    assert refs.mx_block_sum("mxint8", [0x80] * 32, [0x7F] * 32) == -128 * 127 * 32
    assert refs.mx_block_sum("mxfp4", [0x7] * 32, [0x7] * 32) == 144 * 32
    assert refs.mx_block_sum("mxfp4", [0xF] * 32, [0x7] * 32) == -144 * 32
    assert refs.mx_block_sum("int4_b32", [0x8] * 32, [0x8] * 32) == 1 << 11
    assert refs.mx_block_sum("int4_b32", [0x8] * 32, [0x7] * 32) == -8 * 7 * 32


def test_mx_nan_scale_contributes_zero():
    bits, nan = refs.dp32_mx("mxint8", [64] * 32, [64] * 32, 0xFF, 127, 0x3F800000, 24)
    assert (bits, nan) == (0x3F800000, True)


@pytest.mark.parametrize("fmt", ["mxint8", "mxfp4", "int4_b32"])
@pytest.mark.parametrize("align_w", [24, 32])
def test_mx_against_exact_arithmetic(fmt, align_w):
    rng = np.random.default_rng(0)
    w = refs.MX[fmt]["w"]
    acc = 0
    hits = {"exact": 0, "bounded": 0, "range": 0}
    for step in range(3000):
        a, b = rng.integers(0, 1 << w, size=(2, 32))
        mode = rng.random()
        if mode < 0.85:
            sa, sb = np.clip(np.rint(rng.normal(127, 8, size=2)), 0, 254).astype(int)
        elif mode < 0.97:
            sa, sb = rng.integers(0, 255, size=2)
        else:
            sa, sb = rng.integers(0, 256, size=2)
        if rng.random() < 0.05:
            acc = 0
        info = {}
        new, _ = refs.dp32_mx(fmt, a, b, int(sa), int(sb), acc, align_w, info)
        exact = refs.fp32_value(acc) + refs.mx_block_value(fmt, a, b, int(sa), int(sb))
        hits[_check_against_exact(new, exact, info, n_terms=2, align_w=align_w, where=f"{fmt} step {step}")] += 1
        acc = new
    assert hits["exact"] > 100 and hits["bounded"] > 100 and hits["range"] > 10, hits
