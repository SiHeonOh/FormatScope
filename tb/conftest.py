"""Shared cocotb runner helper for every tb/test_*.py (build-plan.md S6.1)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TB = ROOT / "tb"

# The simulator's embedded Python gets this process's sys.path as PYTHONPATH,
# so the test modules can import refs (tb/) and quant.formats (repo root).
for path in (str(ROOT), str(TB)):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full 10,000-vector DP32 runs, skipped in CI")


def run(sources, toplevel, test_module, parameters=None, extra_env=None, build_name=None):
    """Build `sources` with `toplevel` and run the cocotb tests in `test_module`.

    Waveforms are written only when FORMATSCOPE_VCD=1, to keep runs fast.
    """
    from cocotb_tools.runner import get_runner

    sim = os.getenv("SIM", "icarus")
    waves = os.getenv("FORMATSCOPE_VCD") == "1"
    build_dir = ROOT / "sim_build" / (build_name or toplevel)
    runner = get_runner(sim)
    runner.build(
        sources=[ROOT / s for s in sources],
        hdl_toplevel=toplevel,
        parameters=parameters or {},
        # The Icarus runner defaults to -g2012; a later -g2005 overrides it and
        # holds the RTL to the Verilog-2005 coding rule (S5.1).
        build_args=["-g2005"] if sim == "icarus" else [],
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        waves=waves,
        always=True,
    )
    runner.test(
        hdl_toplevel=toplevel,
        test_module=test_module,
        build_dir=build_dir,
        test_dir=build_dir,
        extra_env=extra_env or {},
        waves=waves,
    )
