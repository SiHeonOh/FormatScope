"""FP8-E4M3 DP32 unit against tb/refs.py at ALIGN_W 24 and 32 (build-plan.md S5.5, S6.4).

Like the INT tests, every cycle is checked and the prior accumulator comes from
the stream itself. The rounding corners (ties to even, round bit, sticky bit,
significand overflow) are built from lanes whose products are exact powers of
two, and each one first asserts in Python that the reference really takes the
intended path, so a corner cannot silently stop testing what its name says.
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
SOURCES = [
    "rtl/common/adder_tree.v", "rtl/common/lzc.v", "rtl/common/fused_stage.v",
    "rtl/common/decode_fp8e4m3.v", "rtl/fp8e4m3/dp32_fp8e4m3.v",
]

N = refs.BLOCK
ZERO = 0x00
ONE = 0x38            # 1.0
MAX = 0x7E            # +448, largest finite
MIN_SUB = 0x01        # +2^-9, smallest subnormal
NAN = 0x7F


def neg(code):
    return code | 0x80


def _pow2_pairs():
    """{k: (a, b)} with positive codes whose product is exactly 2^k.
    Prefers the widest significand product, so the term sits lowest in its field."""
    best = {}
    for ca in range(128):
        for cb in range(128):
            _, ea, ga, na = refs.decode_fp8(ca)
            _, eb, gb, nb = refs.decode_fp8(cb)
            p = ga * gb
            if na or nb or p == 0 or p & (p - 1):
                continue
            k = p.bit_length() - 1 + ea + eb - 18
            if k not in best or p > best[k][0]:
                best[k] = (p, ca, cb)
    return {k: (ca, cb) for k, (_, ca, cb) in best.items()}


POW2 = _pow2_pairs()


def lanes(powers):
    """Vectors whose lane products are 2^k for each k in `powers`, zero elsewhere."""
    a, b = [ZERO] * N, [ZERO] * N
    for lane, k in enumerate(powers):
        a[lane], b[lane] = POW2[k]
    return a, b


class Harness:
    """Drives the FP8 unit and tracks the expected register and flag."""

    def __init__(self, dut):
        self.dut = dut
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
        dut.scale_a.value = 127
        dut.scale_b.value = 127
        for _ in range(2):
            await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        dut.rst_n.value = 1
        self.acc, self.nan = 0, 0
        self.check("after reset")

    def predict(self, a, b):
        """(register, nan, info) the reference expects for accumulating a, b now."""
        info = {}
        acc, nan = refs.dp32_fp8(a, b, self.acc, self.align_w, info)
        return acc, nan, info

    async def step(self, a, b, en=1, clear=0, label=""):
        dut = self.dut
        dut.a_flat.value = refs.pack(a, 8)
        dut.b_flat.value = refs.pack(b, 8)
        dut.en.value = en
        dut.clear.value = clear
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        before = self.acc
        if clear:
            self.acc, self.nan = 0, 0
        elif en:
            self.acc, nan, info = self.predict(a, b)
            self.nan |= nan
            self.coverage.update(k for k in ("term_sticky", "rest_sticky", "increment", "overflow", "zero")
                                 if info[k])
            self.coverage["tie"] += bool(info["guard"] and not info["round"]
                                         and not info["term_sticky"] and not info["rest_sticky"])
        self.check(label, a, b, before, en, clear)

    def check(self, label, a=None, b=None, before=None, en=None, clear=None):
        got = self.dut.acc_out.value.to_unsigned()
        got_nan = int(self.dut.flag_nan.value)
        if got != self.acc or got_nan != self.nan:
            detail = "" if a is None else (
                f"\n  a = {refs.pack(a, 8):064x}\n  b = {refs.pack(b, 8):064x}"
                f"\n  en={en} clear={clear} acc_before=0x{before:08x}"
                f" ({float(refs.fp32_value(before))!r})"
            )
            expected = float(refs.fp32_value(self.acc))
            raise AssertionError(
                f"fp8 a{self.align_w} {label}: expected acc 0x{self.acc:08x} ({expected!r}) "
                f"flag_nan {self.nan}, got 0x{got:08x} flag_nan {got_nan}{detail}"
            )


@cocotb.test()
async def dp32_fp8_random(dut):
    """Seeded random codes: full range, small (subnormal-heavy), and large magnitudes."""
    n = int(os.environ.get("FORMATSCOPE_NVEC", N_VEC_CI))
    h = Harness(dut)
    await h.start()
    rng = np.random.default_rng(0)
    for k in range(n):
        mode = rng.random()
        signs = rng.integers(0, 2, size=(2, N)) << 7
        if mode < 0.6:
            a, b = rng.integers(0, 256, size=(2, N))
        elif mode < 0.8:
            a, b = rng.integers(0, 24, size=(2, N)) | signs          # E <= 2
        else:
            a, b = rng.integers(96, 127, size=(2, N)) | signs        # E >= 12
        a[(a & 0x7F) == NAN] -= 1                                  # NaN only when injected below
        b[(b & 0x7F) == NAN] -= 1
        if rng.random() < 0.01:
            a[rng.integers(N)] = NAN
        en = int(rng.random() >= 0.1)
        clear = int(rng.random() < 0.02)
        await h.step(list(a), list(b), en=en, clear=clear, label=f"vector {k}")
    dut._log.info(f"a{h.align_w} coverage over {n} vectors: {dict(h.coverage)}")


@cocotb.test()
async def dp32_fp8_corners(dut):
    """Extremes, signs, subnormals, cancellation, and very different magnitudes."""
    h = Harness(dut)
    await h.start()
    alt_max = [MAX if i % 2 == 0 else neg(MAX) for i in range(N)]
    cases = [
        ("all zero", [ZERO] * N, [ZERO] * N),
        ("negative zeros", [neg(ZERO)] * N, [neg(ZERO)] * N),
        ("1.0 * 1.0 in lane 0", [ONE] + [ZERO] * 31, [ONE] + [ZERO] * 31),
        ("all max", [MAX] * N, [MAX] * N),
        ("all max negative", [neg(MAX)] * N, [MAX] * N),
        ("all min subnormal", [MIN_SUB] * N, [MIN_SUB] * N),
        ("max * min subnormal", [MAX] * N, [neg(MIN_SUB)] * N),
        ("subnormal mix", [i % 8 for i in range(N)], [(7 - i % 8) | (0x80 * (i % 3 == 0)) for i in range(N)]),
        ("alternating signs cancel", alt_max, [MAX] * N),
    ]
    for label, a, b in cases:
        await h.step(a, b, label=label)
        await h.step(a, b, clear=1, label=f"clear after {label}")
        await h.step(a, b, label=f"{label} from zero")

    # Accumulator much larger than the products, then the reverse.
    await h.step([MAX] * N, [MAX] * N, clear=1, label="clear")
    for k in range(8):
        await h.step([MAX] * N, [MAX] * N, label=f"grow large {k}")
    for k in range(4):
        await h.step([MIN_SUB] * N, [neg(MIN_SUB)] * N, label=f"tiny products into large acc {k}")
    await h.step([MIN_SUB] * N, [MIN_SUB] * N, clear=1, label="clear")
    await h.step([MIN_SUB] * N, [MIN_SUB] * N, label="tiny acc")
    await h.step([neg(MAX)] * N, [MAX] * N, label="huge products into tiny acc")

    # Exact cancellation across two accumulations.
    a = [ONE, MAX, 0x41, 0x0C] + [ZERO] * 28
    b = [ONE, 0x40, 0x39, 0x22] + [ZERO] * 28
    await h.step(a, b, clear=1, label="clear")
    await h.step(a, b, label="cancellation part 1")
    await h.step(a, [neg(c) for c in b], label="cancellation part 2")
    assert h.acc == 0, "cancellation corner did not return to zero in the reference"


@cocotb.test()
async def dp32_fp8_rounding(dut):
    """Round-once decisions, each asserted on the reference before it is run."""
    h = Harness(dut)
    await h.start()

    async def build(powers, label):
        await h.step([ZERO] * N, [ZERO] * N, clear=1, label=f"clear before {label}")
        a, b = lanes(powers)
        _, _, info = h.predict(a, b)
        assert not (info["term_sticky"] or info["rest_sticky"] or info["guard"]), f"{label}: setup not exact"
        await h.step(a, b, label=f"{label} setup")

    async def finish(powers, label, **expect):
        a, b = lanes(powers)
        _, _, info = h.predict(a, b)
        for key, value in expect.items():
            assert bool(info[key]) == value, f"{label}: reference {key}={info[key]}, corner expects {value}"
        await h.step(a, b, label=label)

    # 24 ones from 2^-17 up: adding 2^-18 is an exact tie with an odd LSB -> rounds up, overflows.
    await build(range(-17, 7), "tie, odd LSB")
    await finish([-18], "tie rounds up and overflows the significand",
                 guard=True, round=False, term_sticky=False, rest_sticky=False, increment=True, overflow=True)

    # Same without the 2^-17 lane: even LSB, so the tie rounds down.
    await build(range(-16, 7), "tie, even LSB")
    await finish([-18], "tie keeps the even LSB",
                 guard=True, round=False, term_sticky=False, rest_sticky=False, increment=False)

    # Guard and round bits both set: rounds up regardless of the LSB.
    await build(range(-15, 8), "round bit")
    await finish([-17, -18], "guard and round round up", guard=True, round=True, increment=True)

    # Guard with only sticky below it: rounds up although the LSB is even.
    await build(range(-14, 9), "sticky")
    await finish([-16, -18], "guard and sticky round up", guard=True, round=False, increment=True)


@cocotb.test()
async def dp32_fp8_nan_and_control(dut):
    """NaN lanes contribute zero and set the sticky flag; clear, en, and reset priorities."""
    h = Harness(dut)
    await h.start()
    ones = [ONE] * N
    with_nan = [ONE] * N
    with_nan[3] = NAN
    await h.step(with_nan, ones, label="NaN in a, lane 3")
    await h.step(ones, ones, label="flag stays set")
    await h.step(ones, ones, clear=1, label="clear drops flag")
    nan_b = list(ones)
    nan_b[31] = neg(NAN)
    await h.step(ones, nan_b, en=0, label="NaN with en low does not set flag")
    await h.step(ones, nan_b, label="negative NaN in b, lane 31")
    await h.step([NAN] * N, [NAN] * N, label="all NaN adds nothing")
    await h.reset()

    # A NaN lane decodes to the largest exponent (T = 18 against MAX). It must not
    # set the window, or the tiny products beside it would shift out entirely.
    a = [NAN, ZERO] + [MIN_SUB] * 30
    b = [MAX, MAX] + [MIN_SUB] * 30
    _, _, info = h.predict(a, b)
    assert info["t_max"] == -10 and not info["term_sticky"], "NaN window corner is not testing the window"
    await h.step(a, b, label="NaN and zero lanes do not move the window")
    await h.reset()

    await h.step([MAX] * N, [MAX] * N, label="load before hold")
    await h.step([neg(MAX)] * N, [MAX] * N, en=0, label="en low holds")
    await h.step([neg(MAX)] * N, [MAX] * N, en=1, clear=1, label="clear wins over en")
    await h.step([MAX] * N, [MAX] * N, label="reload")
    await h.step([MAX] * N, [MAX] * N, en=0, clear=1, label="clear with en low")
    await h.step(with_nan, ones, label="load before reset")
    await h.reset()


def _run(align_w, n_vec):
    run(SOURCES, "dp32_fp8e4m3", "test_dp32_fp8", parameters={"ALIGN_W": align_w},
        extra_env={"FORMATSCOPE_NVEC": str(n_vec), "FORMATSCOPE_ALIGN_W": str(align_w)},
        build_name=f"dp32_fp8e4m3_a{align_w}")


@pytest.mark.parametrize("align_w", ALIGN_WS)
def test_dp32_fp8(align_w):
    _run(align_w, N_VEC_CI)


@pytest.mark.slow
@pytest.mark.parametrize("align_w", ALIGN_WS)
def test_dp32_fp8_full(align_w):
    _run(align_w, N_VEC_FULL)
