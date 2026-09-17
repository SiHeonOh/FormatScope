"""MXINT8 and MXFP4 DP32 units against tb/refs.py at ALIGN_W 24 and 32 (build-plan.md S5.6, S5.7, S6.4).

Same approach as the FP8 tests: every cycle is checked against the reference,
and every constructed corner first asserts in Python that the reference takes
the path the corner is named after.
"""

import os
from collections import Counter

import numpy as np
import pytest

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

import refs
from conftest import run

N_VEC_CI = 300
N_VEC_FULL = 10_000
ALIGN_WS = (24, 32)
COMMON = ["rtl/common/adder_tree.v", "rtl/common/lzc.v", "rtl/common/fused_stage.v", "rtl/common/decode_e8m0.v"]
SOURCES = {
    "mxint8": COMMON + ["rtl/mxint8/dp32_mxint8.v"],
    "mxfp4": COMMON + ["rtl/common/decode_e2m1.v", "rtl/mxfp4/dp32_mxfp4.v"],
}

N = refs.BLOCK
NAN_SCALE = 0xFF
UNIT = 127                 # E8M0 code for 2^0


# ---------------------------------------------------------------------------
# Blocks with an exactly chosen nonnegative block sum
# ---------------------------------------------------------------------------

_E2M1_MAGS = (0, 1, 2, 3, 4, 6, 8, 12)
_E2M1_PRODUCTS = sorted({ma * mb: (ia, ib)
                         for ia, ma in enumerate(_E2M1_MAGS) for ib, mb in enumerate(_E2M1_MAGS)
                         if ma * mb}.items(), reverse=True)


def block_with_sum(fmt, s):
    """(a, b) codes whose block sum is exactly s >= 0."""
    a, b = [0] * N, [0] * N
    lane = 0
    if fmt == "mxint8":
        while s >= 127 * 127:
            a[lane], b[lane] = 127, 127
            s -= 127 * 127
            lane += 1
        q, r = divmod(s, 127)
        for x, y in ((127, q), (r, 1)):
            if y and x:
                a[lane], b[lane] = x, y
                lane += 1
    else:
        while s:
            p, (ia, ib) = next((p, pair) for p, pair in _E2M1_PRODUCTS if p <= s)
            a[lane], b[lane] = ia, ib
            s -= p
            lane += 1
    assert lane <= N, f"{fmt}: block sum needs {lane} lanes"
    return a, b


def scales_for(fmt, exponent):
    """(scale_a, scale_b) so a block's value is S * 2^exponent."""
    sb = exponent + refs.MX[fmt]["offset"] - UNIT
    assert 0 <= sb <= 254, f"exponent {exponent} not reachable with scale_a = {UNIT}"
    return UNIT, sb


class Harness:
    def __init__(self, dut):
        self.dut = dut
        self.fmt = os.environ["FORMATSCOPE_FMT"]
        self.w = refs.MX[self.fmt]["w"]
        self.align_w = int(os.environ["FORMATSCOPE_ALIGN_W"])
        self.acc = 0
        self.nan = 0
        self.coverage = Counter()

    async def start(self):
        Clock(self.dut.clk, 10, unit="ns").start()
        await self.reset()

    async def reset(self):
        dut = self.dut
        dut.rst_n.value = 0
        dut.en.value = 0
        dut.clear.value = 0
        dut.a_flat.value = 0
        dut.b_flat.value = 0
        dut.scale_a.value = UNIT
        dut.scale_b.value = UNIT
        for _ in range(2):
            await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        dut.rst_n.value = 1
        self.acc, self.nan = 0, 0
        self.check("after reset")

    def predict(self, a, b, sa, sb):
        info = {}
        acc, nan = refs.dp32_mx(self.fmt, a, b, sa, sb, self.acc, self.align_w, info)
        return acc, nan, info

    async def step(self, a, b, sa=UNIT, sb=UNIT, en=1, clear=0, label=""):
        dut = self.dut
        dut.a_flat.value = refs.pack(a, self.w)
        dut.b_flat.value = refs.pack(b, self.w)
        dut.scale_a.value = int(sa)
        dut.scale_b.value = int(sb)
        dut.en.value = en
        dut.clear.value = clear
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        before = self.acc
        if clear:
            self.acc, self.nan = 0, 0
        elif en:
            self.acc, nan, info = self.predict(a, b, sa, sb)
            self.nan |= nan
            self.coverage.update(k for k in ("term_sticky", "rest_sticky", "increment", "overflow",
                                             "zero", "saturate", "underflow") if info[k])
        self.check(label, a, b, sa, sb, before, en, clear)

    def check(self, label, a=None, b=None, sa=None, sb=None, before=None, en=None, clear=None):
        got = self.dut.acc_out.value.to_unsigned()
        got_nan = int(self.dut.flag_nan.value)
        if got != self.acc or got_nan != self.nan:
            digits = self.w // 4
            detail = "" if a is None else (
                f"\n  a = {refs.pack(a, self.w):0{N * digits}x}\n  b = {refs.pack(b, self.w):0{N * digits}x}"
                f"\n  scale_a={sa} scale_b={sb} en={en} clear={clear} acc_before=0x{before:08x}"
                f" ({float(refs.fp32_value(before))!r})"
            )
            raise AssertionError(
                f"{self.fmt} a{self.align_w} {label}: expected acc 0x{self.acc:08x} "
                f"({float(refs.fp32_value(self.acc))!r}) flag_nan {self.nan}, "
                f"got 0x{got:08x} flag_nan {got_nan}{detail}"
            )


@cocotb.test()
async def dp32_mx_random(dut):
    """Seeded random elements; realistic scales, occasional extreme and NaN scales."""
    n = int(os.environ.get("FORMATSCOPE_NVEC", N_VEC_CI))
    h = Harness(dut)
    await h.start()
    rng = np.random.default_rng(0)
    for k in range(n):
        a, b = rng.integers(0, 1 << h.w, size=(2, N))
        mode = rng.random()
        if mode < 0.85:
            sa, sb = np.clip(np.rint(rng.normal(UNIT, 8, size=2)), 0, 254).astype(int)
        elif mode < 0.97:
            sa, sb = rng.integers(0, 255, size=2)
        else:
            sa, sb = (NAN_SCALE, UNIT) if rng.random() < 0.5 else (UNIT, NAN_SCALE)
        en = int(rng.random() >= 0.1)
        clear = int(rng.random() < 0.02)
        await h.step(list(a), list(b), sa, sb, en=en, clear=clear, label=f"vector {k}")
    dut._log.info(f"{h.fmt} a{h.align_w} coverage over {n} vectors: {dict(h.coverage)}")


@cocotb.test()
async def dp32_mx_corners(dut):
    """Block-sum extremes, element pairing, extreme scales, saturation, underflow."""
    h = Harness(dut)
    await h.start()
    if h.fmt == "mxint8":
        pos_max, neg_max, minus_one = 0x7F, 0x80, 0xFF
    else:
        pos_max, neg_max, minus_one = 0x7, 0xF, 0x9          # +6, -6, -0.5
    zeros = [0] * N

    cases = [
        ("all-zero block", zeros, zeros, UNIT, UNIT),
        ("max * max", [pos_max] * N, [pos_max] * N, UNIT, UNIT),
        ("most negative squared", [neg_max] * N, [neg_max] * N, UNIT, UNIT),
        ("max * most negative", [pos_max] * N, [neg_max] * N, UNIT, UNIT),
        ("minus one squared", [minus_one] * N, [minus_one] * N, UNIT, UNIT),
        ("alternating signs cancel", [pos_max if i % 2 == 0 else neg_max for i in range(N)],
         [pos_max] * N, UNIT, UNIT),
        ("scale 0 with scale 254", [pos_max] * N, [pos_max] * N, 0, 254),
        ("scale 254 with scale 0", [neg_max] * N, [pos_max] * N, 254, 0),
    ]
    if h.fmt == "mxint8":
        # The alternating case above does not cancel for INT8 (+127 vs -128); make one that does.
        cases.append(("exact cancellation", [pos_max] * N,
                      [1 if i % 2 == 0 else 0xFF for i in range(N)], UNIT, UNIT))
    for label, a, b, sa, sb in cases:
        await h.step(a, b, sa, sb, label=label)
        await h.step(a, b, sa, sb, clear=1, label=f"clear after {label}")
        await h.step(a, b, sa, sb, label=f"{label} from zero")

    # Element pairing: a one-hot element must meet the b element at the same index.
    for i in range(N):
        a = list(zeros)
        a[i] = pos_max
        b = [(j % 3) + 1 for j in range(N)]
        b[i] = neg_max
        await h.step(a, b, label=f"one-hot element {i}", clear=0)
        await h.step(a, b, clear=1, label=f"clear after one-hot {i}")

    # Saturation: both scales 254 overflow FP32 in either sign, and stay saturated.
    big = [neg_max] * N
    _, _, info = h.predict(big, big, 254, 254)
    assert info["saturate"], "saturation corner does not saturate"
    await h.step(big, big, 254, 254, label="saturate positive")
    await h.step(big, big, 254, 254, label="stay saturated")
    await h.step(big, [pos_max] * N, 254, 254, clear=1, label="clear")
    await h.step(big, [pos_max] * N, 254, 254, label="saturate negative")

    # Underflow: both scales 0 are far below FP32's smallest normal.
    await h.step(zeros, zeros, clear=1, label="clear")
    _, _, info = h.predict([pos_max] * N, [pos_max] * N, 0, 0)
    assert info["underflow"], "underflow corner does not underflow"
    await h.step([pos_max] * N, [pos_max] * N, 0, 0, label="underflow to zero")
    one_a, one_b = block_with_sum(h.fmt, 1)
    await h.step(one_a, one_b, *scales_for(h.fmt, 0), label="load 1.0")
    await h.step([pos_max] * N, [pos_max] * N, 0, 0, label="tiny block into 1.0")


@cocotb.test()
async def dp32_mx_rounding(dut):
    """A 24-bit significand of all ones (or ending in 0) built from two blocks, then one small block."""
    h = Harness(dut)
    await h.start()
    fmt = h.fmt
    # Block sums whose bits tile 24 bits: high part at 2^(x+low_bits), low part at 2^x.
    low_bits = 10 if fmt == "mxint8" else 12
    high = (1 << (24 - low_bits)) - 1
    x = -20

    async def build(low, label):
        await h.step([0] * N, [0] * N, clear=1, label=f"clear before {label}")
        for s, e in ((high, x + low_bits), (low, x)):
            a, b = block_with_sum(fmt, s)
            sa, sb = scales_for(fmt, e)
            _, _, info = h.predict(a, b, sa, sb)
            assert not (info["term_sticky"] or info["rest_sticky"] or info["guard"]), f"{label}: setup not exact"
            await h.step(a, b, sa, sb, label=f"{label} setup {s}")

    async def finish(s, e, label, **expect):
        a, b = block_with_sum(fmt, s)
        sa, sb = scales_for(fmt, e)
        _, _, info = h.predict(a, b, sa, sb)
        for key, value in expect.items():
            assert bool(info[key]) == value, f"{label}: reference {key}={info[key]}, corner expects {value}"
        await h.step(a, b, sa, sb, label=label)

    all_ones_low = (1 << low_bits) - 1
    await build(all_ones_low, "odd LSB")
    await finish(1, x - 1, "tie rounds up and overflows the significand",
                 guard=True, round=False, term_sticky=False, rest_sticky=False, increment=True, overflow=True)

    await build(all_ones_low - 1, "even LSB")
    await finish(1, x - 1, "tie keeps the even LSB",
                 guard=True, round=False, term_sticky=False, rest_sticky=False, increment=False)

    await build(all_ones_low - 1, "round bit")
    await finish(3, x - 2, "guard and round round up", guard=True, round=True, increment=True)

    await build(all_ones_low - 1, "sticky")
    await finish(5, x - 3, "guard and sticky round up", guard=True, round=False, increment=True)


@cocotb.test()
async def dp32_mx_nan_and_control(dut):
    """NaN scales (D4) zero the block and set the sticky flag; control priorities."""
    h = Harness(dut)
    await h.start()
    a, b = block_with_sum(h.fmt, 100)
    await h.step(a, b, UNIT, UNIT, label="load")
    await h.step(a, b, NAN_SCALE, UNIT, label="NaN scale_a")
    await h.step(a, b, UNIT, UNIT, label="flag stays set")
    await h.step(a, b, UNIT, UNIT, clear=1, label="clear drops flag")
    await h.step(a, b, UNIT, NAN_SCALE, en=0, label="NaN with en low does not set flag")
    await h.step(a, b, UNIT, NAN_SCALE, label="NaN scale_b")
    await h.step(a, b, NAN_SCALE, NAN_SCALE, label="both scales NaN")
    await h.reset()

    await h.step(a, b, label="load before hold")
    await h.step(b, a, 200, 50, en=0, label="en low holds")
    await h.step(b, a, 200, 50, en=1, clear=1, label="clear wins over en")
    await h.step(a, b, label="reload")
    await h.step(a, b, en=0, clear=1, label="clear with en low")
    await h.step(a, b, NAN_SCALE, UNIT, label="load before reset")
    await h.reset()


def _run(fmt, align_w, n_vec):
    run(SOURCES[fmt], f"dp32_{fmt}", "test_dp32_mx", parameters={"ALIGN_W": align_w},
        extra_env={"FORMATSCOPE_NVEC": str(n_vec), "FORMATSCOPE_ALIGN_W": str(align_w),
                   "FORMATSCOPE_FMT": fmt},
        build_name=f"dp32_{fmt}_a{align_w}")


@pytest.mark.parametrize("align_w", ALIGN_WS)
def test_mxint8(align_w):
    _run("mxint8", align_w, N_VEC_CI)


@pytest.mark.parametrize("align_w", ALIGN_WS)
def test_mxfp4(align_w):
    _run("mxfp4", align_w, N_VEC_CI)


@pytest.mark.slow
@pytest.mark.parametrize("fmt", ["mxint8", "mxfp4"])
@pytest.mark.parametrize("align_w", ALIGN_WS)
def test_dp32_mx_full(fmt, align_w):
    _run(fmt, align_w, N_VEC_FULL)
