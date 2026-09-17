"""Bit-exact reference models for the DP32 units (build-plan.md S6.2).

Each dp32_* function returns the exact 32-bit accumulator register value the
RTL must hold after one accumulation. Code decoding comes from quant/formats.py
so the quantization and hardware lanes cannot drift apart.
"""

from fractions import Fraction

import numpy as np

from quant import formats

BLOCK = formats.BLOCK_SIZE
ACC_MASK = (1 << 32) - 1
FP32_BIAS = 127
EXP_W = 10                              # signed exponent width inside the fused stage
EXP_FLOOR = -(1 << (EXP_W - 1))         # exponent given to zero terms in the max


def clog2(n):
    """Verilog $clog2."""
    return (n - 1).bit_length()


def _codes(codes, w):
    codes = np.asarray(codes, dtype=np.int64)
    if codes.shape != (BLOCK,):
        raise ValueError(f"expected {BLOCK} codes, got shape {codes.shape}")
    if codes.min() < 0 or codes.max() >= (1 << w):
        raise ValueError(f"codes out of range for {w} bits")
    return codes


def pack(codes, w):
    """Flatten BLOCK w-bit codes into one integer, element 0 in the LSBs (a_flat/b_flat)."""
    packed = 0
    for i, code in enumerate(_codes(codes, w)):
        packed |= int(code) << (i * w)
    return packed


def to_signed32(value):
    value &= ACC_MASK
    return value - (1 << 32) if value >> 31 else value


def dp32_int(a_codes, b_codes, acc, w):
    """INT4/INT8 (S5.3): acc + sum(a_i * b_i), wrapping at 32 bits (D13).

    Codes are raw two's-complement bit patterns, including the most-negative
    code the quantizer never emits (D1); the hardware still has to handle it.
    """
    a = formats._int_code_to_value(_codes(a_codes, w), w).astype(np.int64)
    b = formats._int_code_to_value(_codes(b_codes, w), w).astype(np.int64)
    return (int(acc) + int(np.dot(a, b))) & ACC_MASK


# ---------------------------------------------------------------------------
# Decoders (rtl/common/decode_*.v). Bit-field versions of quant/formats.py;
# tb/test_refs.py checks every code against formats.decode_table.
# ---------------------------------------------------------------------------

def decode_fp8(code):
    """(sign, exp, sig, is_nan) with value = sig * 2^(exp - 9) (docs/fused-stage.md)."""
    sign, e, m = (code >> 7) & 1, (code >> 3) & 0xF, code & 0x7
    if e:
        return sign, e - 1, 8 | m, e == 0xF and m == 0x7
    return sign, 0, m, False


def decode_e2m1(code):
    """(sign, mag) with mag = element value * 2, in {0, 1, 2, 3, 4, 6, 8, 12}."""
    return (code >> 3) & 1, (0, 1, 2, 3, 4, 6, 8, 12)[code & 0x7]


def decode_e8m0(code):
    """(exp, is_nan): exp is the raw code, value = 2^(exp - 127), 0xFF is NaN."""
    return code, code == 0xFF


_FP8_DECODE = [decode_fp8(c) for c in range(256)]


# ---------------------------------------------------------------------------
# FP32-format register helpers
# ---------------------------------------------------------------------------

def fp32_value(bits):
    """Exact value of an FP32-format register (zero or normal only) as a Fraction."""
    sign, eb, frac = bits >> 31, (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if eb == 0:
        if frac:
            raise ValueError(f"subnormal register value 0x{bits:08x}")
        return Fraction(0)
    if eb == 0xFF:
        raise ValueError(f"inf/NaN register value 0x{bits:08x}")
    value = Fraction((1 << 23) | frac) * Fraction(2) ** (eb - FP32_BIAS - 23)
    return -value if sign else value


# ---------------------------------------------------------------------------
# Fused stage (docs/fused-stage.md, steps 1-8)
# ---------------------------------------------------------------------------

def fused_stage(terms, acc_bits, align_w, n_prod, info=None):
    """Accumulate product terms into an FP32-format register value.

    terms: n_prod tuples (sign, t, mag, sig_w) with value = (-1)^sign * mag * 2^(t - sig_w).
    If `info` is a dict it receives the intermediate decisions (for coverage and debugging).
    """
    if len(terms) != n_prod or n_prod & (n_prod - 1):
        raise ValueError(f"need a power-of-two number of terms, got {len(terms)} for n_prod={n_prod}")
    ww = align_w + 2
    k = clog2(n_prod)
    mag_w = ww + 1 + k
    rest_w = mag_w - 26

    # 1. Unpack the accumulator.
    acc_eb = (acc_bits >> 23) & 0xFF
    if acc_eb:
        acc_term = ((acc_bits >> 31) & 1, acc_eb - 126, (1 << 23) | (acc_bits & 0x7FFFFF), 24)
    else:
        acc_term = (0, 0, 0, 24)
    all_terms = list(terms) + [acc_term]

    # 2. Max exponent over nonzero terms.
    t_max = max((t if mag else EXP_FLOOR) for _, t, mag, _ in all_terms)

    # 3-5. Align with per-term sticky, negate, sum.
    total = 0
    term_sticky = False
    for sign, t, mag, sig_w in all_terms:
        if not mag:
            continue
        if not 0 < mag < (1 << sig_w) or sig_w > ww:
            raise ValueError(f"term magnitude {mag} does not fit {sig_w} bits in a {ww}-bit window")
        top = mag << (ww - sig_w)
        d = min(t_max - t, ww)
        aligned = top >> d
        term_sticky |= (aligned << d) != top
        total += -aligned if sign else aligned

    if info is not None:
        info.update(t_max=t_max, term_sticky=term_sticky, zero=total == 0,
                    guard=0, round=0, rest_sticky=False, increment=False, overflow=False)
    if total == 0:
        return 0

    # 6. Sign-magnitude.
    neg = total < 0
    mag = -total if neg else total
    assert mag < (1 << mag_w), "sum magnitude exceeds MAG_W"

    # 7. Normalize, round once (RNE).
    lzc = mag_w - mag.bit_length()
    norm = mag << lzc
    sig = norm >> (rest_w + 2)
    guard = (norm >> (rest_w + 1)) & 1
    rnd = (norm >> rest_w) & 1
    rest_sticky = (norm & ((1 << rest_w) - 1)) != 0
    sticky = term_sticky or rest_sticky
    e = t_max + k - lzc
    increment = bool(guard and (rnd or sticky or sig & 1))
    sig += increment
    overflow = sig >> 24 == 1
    if overflow:
        sig >>= 1
        e += 1

    # 8. Pack.
    eb = e + FP32_BIAS
    if not 1 <= eb <= 254:
        raise AssertionError(f"result exponent {e} outside FP32 normal range")
    if info is not None:
        info.update(guard=guard, round=rnd, rest_sticky=rest_sticky,
                    increment=increment, overflow=overflow)
    return (neg << 31) | (eb << 23) | (sig & 0x7FFFFF)


def fp8_terms(a_codes, b_codes):
    """Per-lane (sign, t, mag, 8) product terms and whether any lane saw a NaN code."""
    terms = []
    nan = False
    for ca, cb in zip(_codes(a_codes, 8), _codes(b_codes, 8)):
        sa, ea, ga, na = _FP8_DECODE[ca]
        sb, eb, gb, nb = _FP8_DECODE[cb]
        if na or nb:
            nan = True
            terms.append((0, 0, 0, 8))
        else:
            terms.append((sa ^ sb, ea + eb - 10, ga * gb, 8))
    return terms, nan


def dp32_fp8(a_codes, b_codes, acc_bits, align_w, info=None):
    """FP8-E4M3 DP32 (S5.5): (new register value, NaN seen this accumulation)."""
    terms, nan = fp8_terms(a_codes, b_codes)
    return fused_stage(terms, acc_bits, align_w, BLOCK, info), nan


def fp8_product_sum(a_codes, b_codes):
    """Exact sum of the non-NaN lane products, as a Fraction (for checking the reference)."""
    terms, _ = fp8_terms(a_codes, b_codes)
    return sum((Fraction(-mag if sign else mag) * Fraction(2) ** (t - sig_w)
                for sign, t, mag, sig_w in terms), Fraction(0))
