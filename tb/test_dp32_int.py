"""INT8 and INT4 DP32 units against tb/refs.py (build-plan.md S5.3, S6.4).

Every cycle is checked, so the random prior accumulator comes from the stream
itself: accumulations build it up, with random en gaps and clears mixed in.
No simulation-only preload port is needed.
"""

import os

import numpy as np
import pytest

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

import refs
from conftest import run

N_VEC_CI = 300
N_VEC_FULL = 10_000
SOURCES = {
    "int8": ["rtl/common/adder_tree.v", "rtl/int8/dp32_int8.v"],
    "int4": ["rtl/common/adder_tree.v", "rtl/int4/dp32_int4.v"],
}


class Harness:
    """Drives one DP32 INT unit and tracks the expected register value."""

    def __init__(self, dut):
        self.dut = dut
        self.w = len(dut.a_flat) // refs.BLOCK
        self.acc = 0

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
        self.acc = 0
        self.check("after reset")

    async def start(self):
        Clock(self.dut.clk, 10, unit="ns").start()
        await self.reset()

    async def step(self, a, b, en=1, clear=0, label=""):
        """Drive one cycle on the falling edge, check the register after the next rising edge."""
        dut = self.dut
        dut.a_flat.value = refs.pack(a, self.w)
        dut.b_flat.value = refs.pack(b, self.w)
        dut.en.value = en
        dut.clear.value = clear
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        before = self.acc
        if clear:
            self.acc = 0
        elif en:
            self.acc = refs.dp32_int(a, b, self.acc, self.w)
        self.check(label, a, b, before, en, clear)

    def check(self, label, a=None, b=None, before=None, en=None, clear=None):
        got = self.dut.acc_out.value.to_unsigned()
        if got != self.acc:
            digits = self.w // 4
            detail = "" if a is None else (
                f"\n  a = {refs.pack(a, self.w):0{32 * digits}x}"
                f"\n  b = {refs.pack(b, self.w):0{32 * digits}x}"
                f"\n  en={en} clear={clear} acc_before=0x{before:08x}"
            )
            raise AssertionError(
                f"int{self.w} {label}: expected 0x{self.acc:08x}, got 0x{got:08x}{detail}"
            )
        assert int(self.dut.flag_nan.value) == 0, f"int{self.w} {label}: flag_nan set"


@cocotb.test()
async def dp32_int_random(dut):
    """Seeded random codes over the full two's-complement range, with en gaps and clears."""
    n = int(os.environ.get("FORMATSCOPE_NVEC", N_VEC_CI))
    h = Harness(dut)
    await h.start()
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 1 << h.w, size=(n, 2, refs.BLOCK))
    en = rng.random(n) >= 0.1
    clear = rng.random(n) < 0.02          # independent of en, so clear-with-en occurs too
    for k in range(n):
        await h.step(codes[k, 0], codes[k, 1],
                     en=int(en[k]), clear=int(clear[k]), label=f"vector {k}")


@cocotb.test()
async def dp32_int_corners(dut):
    """Extremes, sign handling, element pairing, and control priority."""
    h = Harness(dut)
    await h.start()
    w, n = h.w, refs.BLOCK
    qmax = (1 << (w - 1)) - 1          # +127 / +7
    qmin = 1 << (w - 1)                # -128 / -8 (code the quantizer never emits, D1)
    neg1 = (1 << w) - 1                # -1
    alt = [1 if i % 2 == 0 else neg1 for i in range(n)]

    cases = [
        ("all zero", [0] * n, [0] * n),
        ("max * max", [qmax] * n, [qmax] * n),
        ("min * min", [qmin] * n, [qmin] * n),
        ("max * min", [qmax] * n, [qmin] * n),
        ("min * max", [qmin] * n, [qmax] * n),
        ("-1 * -1", [neg1] * n, [neg1] * n),
        ("alternating signs", [qmax if i % 2 == 0 else qmin for i in range(n)], [qmax] * n),
        ("exact cancellation", [qmax] * n, alt),
    ]
    for label, a, b in cases:
        await h.step(a, b, label=label)
        await h.step(a, b, clear=1, label=f"clear after {label}")
        await h.step(a, b, label=f"{label} from zero")

    # Element pairing: a one-hot element must meet the b element at the same index.
    b = [(j % qmax) + 1 for j in range(n)]
    for i in range(n):
        a = [0] * n
        a[i] = qmax
        b_i = list(b)
        b_i[i] = qmin
        await h.step(a, b_i, label=f"one-hot element {i}")

    # Control: en low holds, clear wins over en, clear with en low still clears.
    await h.step([qmax] * n, [qmax] * n, label="load before hold")
    await h.step([qmin] * n, [qmax] * n, en=0, label="en low holds")
    await h.step([qmin] * n, [qmax] * n, en=1, clear=1, label="clear wins over en")
    await h.step([qmax] * n, [qmax] * n, label="reload")
    await h.step([qmax] * n, [qmax] * n, en=0, clear=1, label="clear with en low")

    # Synchronous reset mid-stream.
    await h.step([qmax] * n, [qmax] * n, label="load before reset")
    await h.reset()


@cocotb.test()
async def dp32_int_wraparound(dut):
    """INT32 two's-complement wraparound (D13). Reachable in simulation only for INT8:
    2^12 accumulations of 32 * (-128)^2 = 2^19 land exactly on 2^31."""
    h = Harness(dut)
    await h.start()
    if h.w != 8:
        return
    qmin = [1 << 7] * refs.BLOCK
    for k in range((1 << 12) + 2):
        await h.step(qmin, qmin, label=f"wrap accumulation {k}")
    assert refs.to_signed32(h.acc) < 0, "wraparound corner did not cross 2^31"


@pytest.mark.parametrize("fmt", list(SOURCES))
def test_dp32_int(fmt):
    run(SOURCES[fmt], f"dp32_{fmt}", "test_dp32_int",
        extra_env={"FORMATSCOPE_NVEC": str(N_VEC_CI)})


@pytest.mark.slow
@pytest.mark.parametrize("fmt", list(SOURCES))
def test_dp32_int_full(fmt):
    run(SOURCES[fmt], f"dp32_{fmt}", "test_dp32_int",
        extra_env={"FORMATSCOPE_NVEC": str(N_VEC_FULL)})
