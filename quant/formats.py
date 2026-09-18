"""Single source of truth for FormatScope's number formats (build-plan.md S4.1).

Six top-level format IDs: int4, int8, fp8e4m3, mxint8, mxfp4, int4_b32.
Five underlying code types (what decode_table() enumerates, and what the RTL
decoders in rtl/common/decode_*.v are checked against): int4, int8, fp8e4m3,
e2m1, e8m0. int4/int8/fp8e4m3 double as both a code type and a top-level
format because those formats have no separate element/scale split.

Reconstruction convention throughout: value ~= code * scale.
"""

import json
from pathlib import Path

import numpy as np

BLOCK_SIZE = 32
FORMAT_IDS = ("int4", "int8", "fp8e4m3", "mxint8", "mxfp4", "int4_b32")
CODE_TYPES = ("int4", "int8", "fp8e4m3", "e2m1", "e8m0")

FP8_BIAS = 7
FP8_MAX = 448.0
FP8_MIN_SUBNORMAL = 2.0 ** -9

_E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

_FMT_MAX = {"int4": 7.0, "int8": 127.0, "fp8e4m3": FP8_MAX}

# Exponent of the largest normal element value, used by the MX shared-scale
# rule (build-plan.md S1.3): floor(log2(element_max_magnitude)).
_BLOCK_ELEMENT_MAX = {"mxint8": 127.0 / 64.0, "mxfp4": 6.0, "int4_b32": 7.0}


def _rne(x):
    return np.round(np.asarray(x, dtype=np.float64))


# ---------------------------------------------------------------------------
# Element/code-level codecs: raw code <-> represented value, no scale.
# ---------------------------------------------------------------------------

def _int_value_to_code(values, nbits):
    qmax = (1 << (nbits - 1)) - 1
    signed = np.clip(_rne(values), -qmax, qmax).astype(np.int64)
    return np.where(signed < 0, signed + (1 << nbits), signed).astype(np.int64)


def _int_code_to_value(codes, nbits):
    codes = np.asarray(codes, dtype=np.int64)
    half = 1 << (nbits - 1)
    signed = np.where(codes >= half, codes - (1 << nbits), codes)
    return signed.astype(np.float64)


def _fp8e4m3_value_to_code(values):
    x = np.asarray(values, dtype=np.float64)
    sign = (x < 0).astype(np.int64)
    mag = np.abs(x)
    nonzero = mag > 0
    safe_mag = np.where(nonzero, mag, 1.0)

    frac, exp2 = np.frexp(safe_mag)          # safe_mag == (2*frac) * 2**(exp2-1), 2*frac in [1,2)
    unbiased = (exp2 - 1).astype(np.float64)
    sig_normal = 2.0 * frac
    normal_exp_field = unbiased + FP8_BIAS

    m_n = _rne((sig_normal - 1.0) * 8.0)
    e_n = normal_exp_field.copy()
    overflow_n = m_n >= 8
    e_n = np.where(overflow_n, e_n + 1.0, e_n)
    m_n = np.where(overflow_n, 0.0, m_n)

    m_s = _rne(mag * 512.0)                  # subnormal: value = mant * 2**-9
    promote = m_s >= 8
    e_s = np.where(promote, 1.0, 0.0)
    m_s = np.where(promote, 0.0, m_s)

    is_normal = nonzero & (normal_exp_field >= 1)
    exp_field = np.where(~nonzero, 0.0, np.where(is_normal, e_n, e_s))
    mant_field = np.where(~nonzero, 0.0, np.where(is_normal, m_n, m_s))

    # Saturate to the largest finite value (exp field 15, mant 6) rather than
    # ever landing on the S.1111.111 NaN slot (overflow policy: saturate).
    exceeds = (exp_field > 15) | ((exp_field == 15) & (mant_field >= 7))
    exp_field = np.where(exceeds, 15.0, exp_field)
    mant_field = np.where(exceeds, 6.0, mant_field)

    code = (sign << 7) | (exp_field.astype(np.int64) << 3) | mant_field.astype(np.int64)
    return code.astype(np.int64)


def _fp8e4m3_code_to_value(codes):
    codes = np.asarray(codes, dtype=np.int64)
    sign = (codes >> 7) & 0x1
    exp = (codes >> 3) & 0xF
    mant = codes & 0x7
    is_nan = (exp == 0xF) & (mant == 0x7)
    normal = exp != 0
    significand = np.where(normal, 1.0 + mant / 8.0, mant / 8.0)
    exponent = np.where(normal, exp.astype(np.float64) - FP8_BIAS, 1.0 - FP8_BIAS)
    value = np.where(sign == 1, -1.0, 1.0) * significand * np.exp2(exponent)
    return np.where(is_nan, np.nan, value)


_E2M1_MIDPOINTS = (_E2M1_MAGNITUDES[:-1] + _E2M1_MAGNITUDES[1:]) / 2.0  # 8 magnitudes -> 7 thresholds


def _e2m1_value_to_code(values):
    # Closed-form cascade instead of a generic table search: E2M1 has only 8
    # fixed magnitude levels, and this is called on every MX activation/weight
    # block every forward pass, so the naive per-element search was the
    # dominant cost of MX-format training (profiled: >80% of a QAT step).
    x = np.asarray(values, dtype=np.float64)
    sign = (x < 0).astype(np.int64)
    mag = np.abs(x)
    idx = np.zeros(mag.shape, dtype=np.int64)
    for level, threshold in enumerate(_E2M1_MIDPOINTS, start=1):
        idx = np.where(mag >= threshold, level, idx)
    return ((sign << 3) | idx).astype(np.int64)


def _e2m1_code_to_value(codes):
    codes = np.asarray(codes, dtype=np.int64)
    sign = (codes >> 3) & 0x1
    mag = _E2M1_MAGNITUDES[codes & 0x7]
    return np.where(sign == 1, -mag, mag)


def _e8m0_from_exponent(shared_exp):
    return np.clip(np.asarray(shared_exp, dtype=np.int64) + 127, 0, 254).astype(np.int64)


def _e8m0_code_to_value(codes):
    codes = np.asarray(codes, dtype=np.int64)
    is_nan = codes == 0xFF
    value = np.exp2((codes - 127).astype(np.float64))
    return np.where(is_nan, np.nan, value)


_ELEMENT_CODECS = {
    "int4": (lambda v: _int_value_to_code(v, 4), lambda c: _int_code_to_value(c, 4), 4),
    "int8": (lambda v: _int_value_to_code(v, 8), lambda c: _int_code_to_value(c, 8), 8),
    "fp8e4m3": (_fp8e4m3_value_to_code, _fp8e4m3_code_to_value, 8),
    "e2m1": (_e2m1_value_to_code, _e2m1_code_to_value, 4),
}


def decode_table(code_type):
    """{code: exact_value_or_"nan"} for every raw code of one of CODE_TYPES."""
    if code_type == "e8m0":
        codes = np.arange(256)
        values = _e8m0_code_to_value(codes)
    else:
        encode_fn, decode_fn, nbits = _ELEMENT_CODECS[code_type]
        codes = np.arange(1 << nbits)
        values = decode_fn(codes)
    return {int(c): ("nan" if np.isnan(v) else float(v)) for c, v in zip(codes, values)}


# ---------------------------------------------------------------------------
# MX block quantization (mxint8, mxfp4, int4_b32): one shared scale per
# BLOCK_SIZE elements along the last axis (build-plan.md S1.3 shared-scale rule).
# ---------------------------------------------------------------------------

def _emax_elem(fmt):
    return int(np.floor(np.log2(_BLOCK_ELEMENT_MAX[fmt])))


def _block_shared_exp(fmt, values):
    if values.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError(f"last axis ({values.shape[-1]}) is not a multiple of {BLOCK_SIZE}")
    blocks = values.reshape(values.shape[:-1] + (-1, BLOCK_SIZE))
    max_abs = np.max(np.abs(blocks), axis=-1)
    emax_elem = _emax_elem(fmt)
    with np.errstate(divide="ignore"):
        exp = np.floor(np.log2(np.where(max_abs == 0, 1.0, max_abs))) - emax_elem
    exp = np.where(max_abs == 0, 0.0, exp).astype(np.int64)
    return exp, blocks


def _mx_encode_elements(fmt, blocks, shared_exp):
    scale_pow2 = np.exp2(shared_exp.astype(np.float64))[..., None]
    normalized = blocks / scale_pow2
    if fmt == "mxint8":
        return _int_value_to_code(normalized * 64.0, 8)
    if fmt == "mxfp4":
        return _e2m1_value_to_code(normalized)
    if fmt == "int4_b32":
        return _int_value_to_code(normalized, 4)
    raise ValueError(fmt)


def _mx_decode_elements(fmt, codes):
    if fmt == "mxint8":
        return _int_code_to_value(codes, 8) / 64.0
    if fmt == "mxfp4":
        return _e2m1_code_to_value(codes)
    if fmt == "int4_b32":
        return _int_code_to_value(codes, 4)
    raise ValueError(fmt)


# ---------------------------------------------------------------------------
# Public API, dispatched by top-level format ID.
# ---------------------------------------------------------------------------

def _pick_elementwise_scale(fmt, values, axis):
    amax = np.max(np.abs(values), axis=axis, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    return amax / _FMT_MAX[fmt]


def encode(fmt, values, scale):
    """values, scale -> codes. scale already chosen (see quantize() to pick one)."""
    values = np.asarray(values, dtype=np.float64)
    if fmt in ("int4", "int8", "fp8e4m3"):
        encode_fn, _, _ = _ELEMENT_CODECS[fmt]
        return encode_fn(values / scale)
    if fmt in ("mxint8", "mxfp4", "int4_b32"):
        blocks = values.reshape(values.shape[:-1] + (-1, BLOCK_SIZE))
        shared_exp = np.asarray(scale, dtype=np.int64) if fmt == "int4_b32" else (
            np.asarray(scale, dtype=np.int64) - 127
        )
        return _mx_encode_elements(fmt, blocks, shared_exp)
    raise ValueError(f"unknown format {fmt!r}")


def decode(fmt, codes, scale):
    """codes, scale -> values."""
    if fmt in ("int4", "int8", "fp8e4m3"):
        _, decode_fn, _ = _ELEMENT_CODECS[fmt]
        return decode_fn(codes) * scale
    if fmt in ("mxint8", "mxfp4", "int4_b32"):
        elem_vals = _mx_decode_elements(fmt, codes)
        shared_exp = np.asarray(scale, dtype=np.float64)
        if fmt != "int4_b32":
            shared_exp = shared_exp - 127.0
        scale_pow2 = np.exp2(shared_exp)[..., None]
        return elem_vals * scale_pow2
    raise ValueError(f"unknown format {fmt!r}")


# dequantize is decode under the name callers reach for once they already
# hold a scale (from quantize(), or from a peer's RTL/scale channel) rather
# than one they're picking themselves.
dequantize = decode


def quantize(fmt, values, axis=None):
    """values -> (codes, scale), choosing the scale (per S4.1 rules)."""
    values = np.asarray(values, dtype=np.float64)
    if fmt in ("int4", "int8", "fp8e4m3"):
        scale = _pick_elementwise_scale(fmt, values, axis)
        return encode(fmt, values, scale), scale
    if fmt in ("mxint8", "mxfp4", "int4_b32"):
        shared_exp, blocks = _block_shared_exp(fmt, values)
        codes = _mx_encode_elements(fmt, blocks, shared_exp)
        scale = shared_exp if fmt == "int4_b32" else _e8m0_from_exponent(shared_exp)
        return codes, scale
    raise ValueError(f"unknown format {fmt!r}")


def dump_tables(out_dir=None):
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for code_type in CODE_TYPES:
        with open(out_dir / f"{code_type}.json", "w") as f:
            json.dump(decode_table(code_type), f, indent=2, sort_keys=True)
    return out_dir


if __name__ == "__main__":
    path = dump_tables()
    print(f"wrote {len(CODE_TYPES)} tables to {path}")
