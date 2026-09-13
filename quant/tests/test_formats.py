"""Tests for the build-plan.md S4.1 "Done when" criteria on quant/formats.py."""

import numpy as np
import pytest

from quant import formats

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# (i) decode(encode(v)) is the nearest representable value, per format.
#
# Checked as a global-nearest-neighbor property against decode_table's own
# per-code values, rather than a literal second formula: this catches both
# formula bugs and vectorization/broadcasting bugs, and doesn't just
# re-assert the implementation against itself.
# ---------------------------------------------------------------------------

def _assert_globally_nearest(code_type, values, atol=1e-9):
    encode_fn, decode_fn, _ = formats._ELEMENT_CODECS[code_type]
    codes = encode_fn(values)
    decoded = decode_fn(codes)

    table = formats.decode_table(code_type)
    finite_values = np.array([v for v in table.values() if v != "nan"])
    if code_type in ("int4", "int8"):
        # D1: the quantizer never emits the most-negative code, so its
        # representable range is the symmetric subset, not the full table.
        finite_values = finite_values[finite_values != finite_values.min()]
    best_dist = np.min(np.abs(finite_values[None, :] - values[:, None]), axis=1)
    actual_dist = np.abs(decoded - values)
    assert np.all(actual_dist <= best_dist + atol)


def test_int4_roundtrip_is_nearest():
    _assert_globally_nearest("int4", RNG.uniform(-10, 10, size=100_000))


def test_int8_roundtrip_is_nearest():
    _assert_globally_nearest("int8", RNG.uniform(-160, 160, size=100_000))


def test_fp8e4m3_roundtrip_is_nearest():
    values = np.concatenate([
        RNG.uniform(-500, 500, size=90_000),
        RNG.uniform(-0.03, 0.03, size=10_000),  # dense near the subnormal range
    ])
    _assert_globally_nearest("fp8e4m3", values)


def test_e2m1_roundtrip_is_nearest():
    _assert_globally_nearest("e2m1", RNG.uniform(-10, 10, size=100_000))


# ---------------------------------------------------------------------------
# (ii) every table value matches an independent (scalar, non-vectorized)
# reimplementation of the format's formula.
# ---------------------------------------------------------------------------

def _ref_int(code, nbits):
    half = 1 << (nbits - 1)
    return float(code - (1 << nbits)) if code >= half else float(code)


def _ref_fp8e4m3(code):
    sign = (code >> 7) & 1
    exp = (code >> 3) & 0xF
    mant = code & 0x7
    if exp == 0xF and mant == 0x7:
        return "nan"
    val = (1 + mant / 8.0) * 2.0 ** (exp - 7) if exp else (mant / 8.0) * 2.0 ** (1 - 7)
    return -val if sign else val


def _ref_e2m1(code):
    sign = (code >> 3) & 1
    mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    val = mags[code & 0x7]
    return -val if sign else val


def _ref_e8m0(code):
    return "nan" if code == 0xFF else 2.0 ** (code - 127)


@pytest.mark.parametrize("code_type,nbits,ref", [
    ("int4", 4, lambda c: _ref_int(c, 4)),
    ("int8", 8, lambda c: _ref_int(c, 8)),
    ("fp8e4m3", 8, _ref_fp8e4m3),
    ("e2m1", 4, _ref_e2m1),
    ("e8m0", 8, _ref_e8m0),
])
def test_decode_table_matches_independent_formula(code_type, nbits, ref):
    table = formats.decode_table(code_type)
    assert len(table) == 1 << nbits
    for code, value in table.items():
        expected = ref(code)
        if expected == "nan":
            assert value == "nan", f"code {code}"
        else:
            assert value == pytest.approx(expected), f"code {code}"


# ---------------------------------------------------------------------------
# (iii) FP8 table: max 448, min subnormal 2**-9, exactly one NaN code per sign.
# ---------------------------------------------------------------------------

def test_fp8e4m3_table_bounds():
    table = formats.decode_table("fp8e4m3")
    finite = {c: v for c, v in table.items() if v != "nan"}
    assert max(finite.values()) == 448.0
    positive_subnormals = [v for v in finite.values() if 0 < v < 2 ** -6]
    assert min(positive_subnormals) == pytest.approx(2.0 ** -9)
    nan_codes = [c for c, v in table.items() if v == "nan"]
    assert nan_codes == [0x7F, 0xFF]


# ---------------------------------------------------------------------------
# (iv) E2M1 table equals +/-{0, 0.5, 1, 1.5, 2, 3, 4, 6}.
# ---------------------------------------------------------------------------

def test_e2m1_table_values():
    table = formats.decode_table("e2m1")
    values = sorted(set(table.values()))
    expected = sorted({0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5,
                        2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0})
    assert values == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Watch-outs called out in S4.1: MX block axis and E8M0 bias off-by-one.
# ---------------------------------------------------------------------------

def test_mx_block_axis_known_vector():
    values = np.zeros(64)
    values[:32] = 1.0
    values[32:] = 100.0
    codes, scale = formats.quantize("mxfp4", values.reshape(1, 64))
    assert codes.shape == (1, 2, 32)
    assert scale.shape == (1, 2)
    assert scale[0, 0] != scale[0, 1]
    recon = formats.dequantize("mxfp4", codes, scale).reshape(1, 64)
    assert np.allclose(recon[0, :32], 1.0, atol=0.1)
    assert np.allclose(recon[0, 32:], 100.0, atol=5.0)


def test_e8m0_bias_off_by_one():
    assert formats._e8m0_from_exponent(np.array([0]))[0] == 127
    table = formats.decode_table("e8m0")
    assert table[127] == 1.0
    assert table[255] == "nan"


def test_mx_all_zero_block_gets_scale_127():
    values = np.zeros((1, 32))
    codes, scale = formats.quantize("mxint8", values)
    assert scale[0, 0] == 127
    assert np.all(codes == 0)


# ---------------------------------------------------------------------------
# D5: MXINT8 (and plain INT8) quantizer never emits the most-negative code.
# ---------------------------------------------------------------------------

def test_int8_never_emits_most_negative_code():
    codes, _ = formats.quantize("int8", RNG.uniform(-1000, 1000, size=50_000))
    assert not np.any(codes == 128)  # raw byte for -128


def test_mxint8_never_emits_most_negative_element_code():
    values = RNG.uniform(-1000, 1000, size=(10, 64))
    codes, _ = formats.quantize("mxint8", values)
    assert not np.any(codes == 128)


# ---------------------------------------------------------------------------
# Full quantize -> dequantize sanity for all six top-level format IDs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", formats.FORMAT_IDS)
def test_quantize_dequantize_shapes_and_sanity(fmt):
    values = RNG.normal(0, 1, size=(4, 64)).astype(np.float64)
    codes, scale = formats.quantize(fmt, values)
    recon = formats.dequantize(fmt, codes, scale).reshape(values.shape)
    assert recon.shape == values.shape
    assert np.max(np.abs(recon - values)) < 2.0  # loose: these are lossy formats
