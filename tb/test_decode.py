"""Exhaustive decoder tests: every code against quant/tables/<fmt>.json (build-plan.md S6.3)."""

import json
import math

import pytest

import cocotb
from cocotb.triggers import Timer

from conftest import ROOT, run

TABLES = ROOT / "quant" / "tables"


def _table(name):
    with open(TABLES / f"{name}.json") as f:
        return {int(code): value for code, value in json.load(f).items()}


def _same(value, expected):
    return value == expected and math.copysign(1.0, value) == math.copysign(1.0, expected)


async def _settle():
    await Timer(1, unit="ns")


@cocotb.test()
async def decode_fp8e4m3_exhaustive(dut):
    table = _table("fp8e4m3")
    for code in range(256):
        dut.code.value = code
        await _settle()
        sign, exp, sig, is_nan = (int(dut.sign.value), int(dut.exp.value),
                                  int(dut.sig.value), int(dut.is_nan.value))
        if table[code] == "nan":
            assert is_nan, f"code 0x{code:02x}: is_nan not set"
            continue
        value = (-1.0 if sign else 1.0) * sig * 2.0 ** (exp - 9)
        assert not is_nan and _same(value, table[code]), (
            f"code 0x{code:02x}: sign={sign} exp={exp} sig={sig} is_nan={is_nan} "
            f"-> {value}, table says {table[code]}")
    dut._log.info("256/256 FP8 codes match")


@cocotb.test()
async def decode_e2m1_exhaustive(dut):
    table = _table("e2m1")
    for code in range(16):
        dut.code.value = code
        await _settle()
        sign, mag = int(dut.sign.value), int(dut.mag.value)
        value = (-1.0 if sign else 1.0) * mag / 2
        assert _same(value, table[code]), f"code 0x{code:x}: sign={sign} mag={mag}, table says {table[code]}"
    dut._log.info("16/16 E2M1 codes match")


@cocotb.test()
async def decode_e8m0_exhaustive(dut):
    table = _table("e8m0")
    for code in range(256):
        dut.code.value = code
        await _settle()
        exp, is_nan = int(dut.exp.value), int(dut.is_nan.value)
        if table[code] == "nan":
            assert is_nan, f"code 0x{code:02x}: is_nan not set"
        else:
            assert not is_nan and 2.0 ** (exp - 127) == table[code], (
                f"code 0x{code:02x}: exp={exp} is_nan={is_nan}, table says {table[code]}")
    dut._log.info("256/256 E8M0 codes match")


@pytest.mark.parametrize("decoder", ["fp8e4m3", "e2m1", "e8m0"])
def test_decode(decoder):
    run([f"rtl/common/decode_{decoder}.v"], f"decode_{decoder}", "test_decode",
        test_filter=f"decode_{decoder}_exhaustive")
