"""Parser tests for sta/run_sta.py (build-plan.md S7.3)."""

import re

import pytest

from sta import run_sta as rsta

REPORT = """\
Startpoint: b_flat[3] (input port clocked by clk)
Endpoint: acc_out[31]$_DFF_PN0_ (rising edge-triggered flip-flop clocked by clk)
Path Group: clk
Path Type: max

Fanout     Cap    Slew   Delay    Time   Description
-----------------------------------------------------------------------------
                          0.000   0.000   clock clk (rise edge)
                          0.000   0.000   clock network delay (ideal)
                          0.000   0.000 v input external delay
     2   0.005   0.000   0.000   0.000 v b_flat[3] (in)
                                           b_flat[3] (net)
                 0.000   0.000   0.000 v _0412_/A (sky130_fd_sc_hd__nand2_1)
     1   0.002   0.061   0.087   3.912 ^ _2291_/X (sky130_fd_sc_hd__a21o_1)
                 0.061   0.000   3.912 ^ acc_out[31]$_DFF_PN0_/D (sky130_fd_sc_hd__dfrtp_1)
                                  3.912   data arrival time

                         10.000  10.000   clock clk (rise edge)
                          0.000  10.000   clock network delay (ideal)
                          0.000  10.000   clock reconvergence pessimism
                                 10.000 ^ acc_out[31]$_DFF_PN0_/CLK (sky130_fd_sc_hd__dfrtp_1)
                         -0.124   9.876   library setup time
                                  9.876   data required time
-----------------------------------------------------------------------------
                                  9.876   data required time
                                 -3.912   data arrival time
-----------------------------------------------------------------------------
                                  5.964   slack (MET)


wns 0.000
tns 0.000
"""


def test_arrival_is_the_first_positive_arrival_line():
    assert rsta.parse_arrival_ns(REPORT) == pytest.approx(3.912)


@pytest.mark.parametrize("line,value", [
    ("wns 0.000", 0.0), ("wns -0.12", -0.12), ("wns max -0.123", -0.123), ("wns max 0.00", 0.0),
])
def test_wns_forms(line, value):
    assert rsta.parse_wns_ns(REPORT.replace("wns 0.000", line)) == pytest.approx(value)


def test_missing_paths_raise():
    with pytest.raises(ValueError, match="data arrival time"):
        rsta.parse_arrival_ns("No paths found.\n")
    with pytest.raises(ValueError, match="wns"):
        rsta.parse_wns_ns("No paths found.\n")


def test_errors_in_report_raise():
    rsta.check_report(REPORT)
    with pytest.raises(RuntimeError, match="STA-0164"):
        rsta.check_report("Error: sta.tcl, 4 STA-0164 module dp32_int8 not found.\n")


def test_netlist_name_parsing():
    meta = rsta.parse_netlist_name("synth/out/sky130hd_fp8e4m3_t2_a32_r3_netlist.v")
    assert meta == {"lib": "sky130hd", "format": "fp8e4m3", "target": "t2",
                    "align_w": 32, "run": 3}
    # A format id with an underscore must not be split at it.
    meta = rsta.parse_netlist_name("synth/out/sky130hd_int4_b32_t1_a24_r0_netlist.v")
    assert meta == {"lib": "sky130hd", "format": "int4_b32", "target": "t1",
                    "align_w": 24, "run": 0}
    for bad in ("smoke_add_netlist.v", "sky130hd_int8_t3_a24_r0_netlist.v",
                "sky130hd_int8_unc_a24_r0_stat.json"):
        with pytest.raises(ValueError):
            rsta.parse_netlist_name(bad)


def test_period_from_targets(tmp_path):
    targets = tmp_path / "targets.toml"
    targets.write_text("[sky130hd]\nt1_ps = 5600\n")
    assert rsta.period_ns("sky130hd", "unc", targets) == rsta.UNC_PERIOD_NS
    assert rsta.period_ns("sky130hd", "t1", targets) == pytest.approx(5.6)
    with pytest.raises(SystemExit, match="t2_ps"):
        rsta.period_ns("sky130hd", "t2", targets)


def test_template_renders_and_keeps_tcl_braces():
    text = (rsta.STA_DIR / "sta.tcl.template").read_text()
    rendered = rsta.render(text, {"LIB": "/pdk/x.lib", "NETLIST": "/repo/n.v",
                                  "TOP": "dp32_int8", "PERIOD_NS": "5.600"})
    assert not re.search(r"\{[A-Z_]+\}", rendered)
    assert "create_clock -name clk -period 5.600 [get_ports clk]" in rendered
    assert "-fields {slew cap input net fanout}" in rendered
    assert rendered.count("{") == rendered.count("}")


def test_sta_prefix_expands_env(monkeypatch):
    monkeypatch.setenv("PDK_ROOT", "/home/rg/.ciel")
    monkeypatch.setenv("FORMATSCOPE_STA", "docker run --rm -i -v $PDK_ROOT:$PDK_ROOT opensta")
    assert rsta.sta_prefix() == ["docker", "run", "--rm", "-i", "-v",
                                 "/home/rg/.ciel:/home/rg/.ciel", "opensta"]
