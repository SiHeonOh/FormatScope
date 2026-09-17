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
        if not info["term_sticky"] and not info["rest_sticky"]:
            assert new == round_fp32(exact), f"step {step}: 0x{new:08x} vs 0x{round_fp32(exact):08x}"
            exact_hits += 1
        else:
            got = refs.fp32_value(new)
            ulp = Fraction(2) ** (((new >> 23) & 0xFF) - 127 - 23) if new else Fraction(0)
            bound = ulp + 33 * Fraction(2) ** (info["t_max"] - (align_w + 2))
            assert abs(got - exact) <= bound, f"step {step}: error {float(got - exact)} > {float(bound)}"
            bounded_hits += 1
        acc = new
    assert exact_hits > 100 and bounded_hits > 100
