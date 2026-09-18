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
FP32_EMAX = 127                         # largest unbiased exponent of a normal FP32 value
FP32_EMIN = -126                        # smallest
FP32_MAX_FINITE = 0x7F7FFFFF
EXP_W = 10                             # signed exponent width inside the fused stage
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
    # Plain Python ints throughout: numpy integers would overflow the shifts below.
    terms = [tuple(int(x) for x in term) for term in terms]
    acc_bits = int(acc_bits)
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
                    guard=0, round=0, rest_sticky=False, increment=False, overflow=False,
                    saturate=False, underflow=False)
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

    # 8. Pack. Out of FP32's normal range: saturate to the largest finite value,
    # or return +0 below the smallest normal (no infinities, no subnormals).
    saturate = e > FP32_EMAX
    underflow = e < FP32_EMIN
    if info is not None:
        info.update(guard=guard, round=rnd, rest_sticky=rest_sticky,
                    increment=increment, overflow=overflow, saturate=saturate, underflow=underflow)
    if underflow:
        return 0
    if saturate:
        return (neg << 31) | FP32_MAX_FINITE
    return (neg << 31) | ((e + FP32_BIAS) << 23) | (sig & 0x7FFFFF)


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


# ---------------------------------------------------------------------------
# MX units (S5.6, S5.7): exact integer block sum, then a normalized block term
# into the fused stage with n_prod = 1.
# ---------------------------------------------------------------------------

# Per format: element width, magnitude field width of the block sum, and the
# offset in value = S * 2^(scale_a + scale_b - offset).
#   mxint8: elements are code/64 -> products carry 2^-12; offset = 254 + 12.
#   mxfp4:  mag = element * 2    -> products carry 2^-2;  offset = 254 + 2.
#   int4_b32: elements are plain INT4 integers            -> offset = 254.
#     formats.py keeps this format's block scale as a raw exponent e; the
#     hardware carries it as E8M0 (code = e + 127), the same as the MX units.
#     |S| <= 32 * (-8)^2 = 2^11, so the magnitude needs 12 bits.
MX = {
    "mxint8": {"w": 8, "sig_w": 20, "offset": 266},
    "mxfp4": {"w": 4, "sig_w": 13, "offset": 256},
    "int4_b32": {"w": 4, "sig_w": 12, "offset": 254},
}
MX_INT_ELEMENTS = ("mxint8", "int4_b32")    # two's-complement elements; mxfp4 is E2M1


def mx_block_sum(fmt, a_codes, b_codes):
    """Exact signed integer block sum S of the element products."""
    w = MX[fmt]["w"]
    a, b = _codes(a_codes, w), _codes(b_codes, w)
    if fmt in MX_INT_ELEMENTS:
        av = formats._int_code_to_value(a, w).astype(np.int64)
        bv = formats._int_code_to_value(b, w).astype(np.int64)
        return int(np.dot(av, bv))
    total = 0
    for ca, cb in zip(a, b):
        sa, ma = decode_e2m1(int(ca))
        sb, mb = decode_e2m1(int(cb))
        total += -(ma * mb) if sa ^ sb else ma * mb
    return total


def mx_block_term(fmt, block_sum, scale_a, scale_b):
    """(term, nan): the normalized (sign, t, mag, sig_w) term for one block.

    The block sum is normalized (one leading-zero count) so the fused stage's
    window starts at its leading one; a NaN scale makes the term zero (D4).
    """
    sig_w, offset = MX[fmt]["sig_w"], MX[fmt]["offset"]
    ea, nan_a = decode_e8m0(int(scale_a))
    eb, nan_b = decode_e8m0(int(scale_b))
    nan = nan_a or nan_b
    mag = abs(block_sum)
    if not mag < (1 << sig_w):
        raise ValueError(f"{fmt} block sum {block_sum} exceeds {sig_w} bits")
    if nan or mag == 0:
        return (0, 0, 0, sig_w), nan
    lz = sig_w - mag.bit_length()
    return (int(block_sum < 0), ea + eb - offset + sig_w - lz, mag << lz, sig_w), nan


def dp32_mx(fmt, a_codes, b_codes, scale_a, scale_b, acc_bits, align_w, info=None):
    """MXINT8 / MXFP4 / INT4-b32 DP32: (new register value, NaN scale seen this accumulation)."""
    term, nan = mx_block_term(fmt, mx_block_sum(fmt, a_codes, b_codes), scale_a, scale_b)
    return fused_stage([term], acc_bits, align_w, 1, info), nan


def mx_block_value(fmt, a_codes, b_codes, scale_a, scale_b):
    """Exact block value as a Fraction (zero for a NaN scale), for checking the reference."""
    scale_a, scale_b = int(scale_a), int(scale_b)
    if scale_a == 0xFF or scale_b == 0xFF:
        return Fraction(0)
    return Fraction(mx_block_sum(fmt, a_codes, b_codes)) * Fraction(2) ** (scale_a + scale_b - MX[fmt]["offset"])
