"""Parser and guard tests for synth/run_synth.py (build-plan.md S7.1, S7.4, S7.5)."""

import csv
import re
import subprocess
from pathlib import Path

import pytest

from synth import run_synth as rs

FLAT_JSON = r"""
{
   "creator": "Yosys 0.57 (git sha1 3aca86049, clang++ 18.1.3 -fPIC -O3)",
   "invocation": "stat -json -liberty sky130_fd_sc_hd__tt_025C_1v80.lib ",
   "modules": {
      "\\dp32_int8": {
         "num_wires":         1873,
         "num_wire_bits":     2412,
         "num_pub_wires":     9,
         "num_pub_wire_bits": 546,
         "num_ports":         9,
         "num_port_bits":     546,
         "num_memories":      0,
         "num_memory_bits":   0,
         "num_processes":     0,
         "num_cells":         2301,
         "area":              21873.512000,
         "num_cells_by_type": {
            "sky130_fd_sc_hd__a21oi_1": 212,
            "sky130_fd_sc_hd__dfrtp_1": 33,
            "sky130_fd_sc_hd__xor2_1": 410
         }
      }
   },
   "design": {
      "num_wires":         1873,
      "num_cells":         2301,
      "area":              21873.512000,
      "num_cells_by_type": {
         "sky130_fd_sc_hd__a21oi_1": 212
      }
   }
}
"""

FLAT_TEXT = r"""
10. Printing statistics.

=== smoke_add ===

   Number of wires:                 17
   Number of wire bits:             26
   Number of public wires:           3
   Number of public wire bits:      13
   Number of cells:                 14
     sky130_fd_sc_hd__a21oi_1        2
     sky130_fd_sc_hd__xor2_1         5

   Chip area for module '\smoke_add': 81.328000
"""

HIER_JSON = r"""
{
   "creator": "Yosys 0.57",
   "modules": {
      "$paramod$9f3c\\adder_tree": {
         "num_cells": 3, "area": 30.0,
         "num_cells_by_type": {"sky130_fd_sc_hd__fa_1": 3}
      },
      "$paramod\\tree_stage\\N=32": {
         "num_cells": 3, "area": 1.0,
         "num_cells_by_type": {"$paramod$9f3c\\adder_tree": 2, "sky130_fd_sc_hd__inv_1": 1}
      },
      "\\mul_stage": {
         "num_cells": 4,
         "num_cells_by_type": {"sky130_fd_sc_hd__nand2_1": 4}
      },
      "\\decode_fp8e4m3": {
         "num_cells": 1,
         "num_cells_by_type": {"sky130_fd_sc_hd__inv_1": 1}
      },
      "\\dp32_fp8e4m3": {
         "num_cells": 37,
         "num_cells_by_type": {
            "\\mul_stage": 1,
            "$paramod\\tree_stage\\N=32": 1,
            "\\decode_fp8e4m3": 32,
            "sky130_fd_sc_hd__dfrtp_1": 2
         }
      }
   },
   "design": {"num_cells": 999, "area": 12345.0}
}
"""

LIBERTY = r"""
library ("sky130_fd_sc_hd__tt_025C_1v80") {
    cell ("sky130_fd_sc_hd__dfrtp_1") {
        leakage_power () { value : 0.001; when : "!CLK&D"; }
        area : 25.02400000;
        pin ("Q") { direction : "output"; }
    }
    cell ("sky130_fd_sc_hd__fa_1") {
        area : 20.01920000;
    }
    cell ("sky130_fd_sc_hd__inv_1") {
        area : 3.75360000;
    }
    cell ("sky130_fd_sc_hd__nand2_1") {
        cell_footprint : "sky130_fd_sc_hd__nand2";
        area : 3.75360000;
    }
}
"""


def test_parse_stat_json_prefers_design_section():
    assert rs.parse_stat_json(FLAT_JSON, "dp32_int8") == (21873.512, 2301)


def test_parse_stat_json_falls_back_to_module_entry_and_skips_log_prefix():
    text = "12. Printing statistics.\n" + FLAT_JSON.replace('"design"', '"unused"')
    assert rs.parse_stat_json(text, "dp32_int8") == (21873.512, 2301)


def test_parse_stat_text_chip_area_and_cells():
    assert rs.parse_stat_text(FLAT_TEXT, "smoke_add") == (81.328, 14)


def test_parse_stat_text_new_table_format():
    text = ("=== smoke_add ===\n\n        +----------Local Count\n"
            "       17 wires\n       14     81.328 cells\n"
            "   Chip area for top module '\\smoke_add': 81.328000\n")
    assert rs.parse_stat_text(text, "smoke_add") == (81.328, 14)


def test_parse_stat_files_falls_back_to_text_then_raises(tmp_path):
    bad_json = tmp_path / "x_stat.json"
    bad_json.write_text("not json")
    text = tmp_path / "x_stat.txt"
    text.write_text(FLAT_TEXT)
    assert rs.parse_stat_files(bad_json, text, "smoke_add") == (81.328, 14)
    text.write_text("nothing useful")
    with pytest.raises(RuntimeError, match="x_stat.json"):
        rs.parse_stat_files(bad_json, text, "smoke_add")


@pytest.mark.parametrize("area", [0.0, -1.0, None])
def test_area_row_refuses_non_positive_area(area):
    with pytest.raises(ValueError, match="optimized away"):
        rs.area_row("sky130hd", "int8", "unc", None, 24, 0, area, 0, "0.57", "2026-09-15")


def test_dtarget_refused_on_unconstrained_run():
    assert rs.dtarget_arg("unc", None) == ""
    assert rs.dtarget_arg("t1", 4321.4) == "-D 4321"
    with pytest.raises(ValueError, match="unc"):
        rs.dtarget_arg("unc", 4000)
    with pytest.raises(ValueError):
        rs.dtarget_arg("t2", None)


def test_liberty_cell_areas_reads_past_leakage_groups():
    areas = rs.liberty_cell_areas(LIBERTY)
    assert areas == {
        "sky130_fd_sc_hd__dfrtp_1": 25.024, "sky130_fd_sc_hd__fa_1": 20.0192,
        "sky130_fd_sc_hd__inv_1": 3.7536, "sky130_fd_sc_hd__nand2_1": 3.7536,
    }


def test_stage_of_handles_paramod_names():
    assert rs.stage_of("$paramod\\tree_stage\\N=32") == "tree"
    assert rs.stage_of("\\normacc_stage") == "normacc"
    assert rs.stage_of("$paramod$abc\\align_stage") == "align"
    assert rs.stage_of("\\decode_e2m1") == "decode"
    assert rs.stage_of("\\dp32_int8") == "other"


def test_breakdown_multiplies_instances_and_inherits_stage():
    areas = rs.liberty_cell_areas(LIBERTY)
    totals = rs.breakdown_from_stat(HIER_JSON, "dp32_fp8e4m3", areas)
    assert totals["mul"] == pytest.approx(4 * 3.7536)
    # tree_stage: one inverter plus two adder_tree instances of three full adders
    assert totals["tree"] == pytest.approx(3.7536 + 2 * 3 * 20.0192)
    assert totals["decode"] == pytest.approx(32 * 3.7536)
    assert totals["other"] == pytest.approx(2 * 25.024)
    assert set(totals) == {"mul", "tree", "decode", "other"}


def test_jobs_perturbation_and_window_rules():
    jobs = list(rs.jobs(["int8", "fp8e4m3"], ["unc", "t1"], [24, 32], True, 1))
    int8 = [j for j in jobs if j[0] == "int8"]
    assert {j[2] for j in int8} == {24}  # INT units skip the window sweep
    t1 = [j for j in jobs if j[0] == "fp8e4m3" and j[1] == "t1" and j[2] == 32]
    assert [j[4] for j in t1] == rs.PERTURB_FACTORS and [j[3] for j in t1] == [0, 1, 2, 3, 4]
    assert all(j[4] is None for j in jobs if j[1] == "unc")


@pytest.mark.parametrize("template", ["synth_flat.ys.template", "synth_hier.ys.template"])
def test_templates_render_every_placeholder(template):
    text = (rs.SYNTH_DIR / template).read_text()
    fields = rs.synth_fields("/pdk/sky130.lib", rs.SOURCES["fp8e4m3"], "dp32_fp8e4m3",
                             "-D 4000", "synth/out/x", 24, constr="/pdk/abc.constr")
    rendered = rs.render(text, fields)
    assert not re.search(r"\{[A-Z_]+\}", rendered)
    assert "chparam -set ALIGN_W 24 dp32_fp8e4m3" in rendered
    # -D is only picoseconds when ABC reads the liberty itself, which -constr triggers.
    assert "abc -liberty /pdk/sky130.lib -constr /pdk/abc.constr -D 4000" in rendered
    with pytest.raises(ValueError, match="constraint file"):
        rs.synth_fields("/pdk/sky130.lib", rs.SOURCES["int8"], "dp32_int8", "-D 4000", "synth/out/x")
    assert rs.abc_constr_path("sky130hd") is not None, "synth/abc_sky130hd.constr must exist"
    assert "stat -json -liberty /pdk/sky130.lib" in rendered
    flatten = "synth -top dp32_fp8e4m3 -flatten -noalumacc" in rendered
    assert flatten == (template == "synth_flat.ys.template")
    # Both flows keep the multipliers and the tree out of a single $macc, so
    # ABC maps them separately (equal or smaller netlists, seconds not minutes).
    assert "-noalumacc" in rendered
    with pytest.raises(ValueError, match="SOURCES"):
        rs.render(text, {k: v for k, v in fields.items() if k != "SOURCES"})


def test_append_rows_writes_header_once_in_appendix_a_order(tmp_path):
    path = tmp_path / "area.csv"
    row = rs.area_row("sky130hd", "int8", "t1", 3920.0, 24, 2, 100.5, 12, "0.57", "2026-09-15")
    rs.append_rows(path, rs.AREA_COLUMNS, [row])
    rs.append_rows(path, rs.AREA_COLUMNS, [row])
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(rs.AREA_COLUMNS)
    assert len(lines) == 3
    assert next(csv.DictReader(lines))["dtarget_ps"] == "3920"


def _use_tmp_repo(monkeypatch, tmp_path, lib_path):
    libs = tmp_path / "libs.toml"
    libs.write_text(f'[testbox]\nsky130hd = "{lib_path}"\n')
    monkeypatch.setenv("FORMATSCOPE_MACHINE", "testbox")
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    monkeypatch.setattr(rs, "AREA_CSV", tmp_path / "results/area.csv")
    monkeypatch.setattr(rs, "load_lib_path",
                        lambda lib_id, must_exist=True: LOAD_LIB_PATH(
                            lib_id, libs_toml=libs, must_exist=must_exist))


LOAD_LIB_PATH = rs.load_lib_path


def test_dry_run_renders_without_yosys(tmp_path, monkeypatch, capsys):
    _use_tmp_repo(monkeypatch, tmp_path, "/nonexistent/sky130.lib")
    rs.main(["--fmt", "int8", "--dry-run"])
    out = capsys.readouterr().out
    assert "sky130hd_int8_unc_a24_r0.ys" in out
    script = (tmp_path / "synth/out/sky130hd_int8_unc_a24_r0.ys").read_text()
    assert "read_verilog -sv rtl/common/adder_tree.v rtl/int8/dp32_int8.v" in script
    abc_line = next(line for line in script.splitlines() if line.startswith("abc "))
    assert abc_line.startswith("abc -liberty /nonexistent/sky130.lib")
    assert " -D " not in abc_line  # no delay target on the unconstrained run


def test_t1_before_targets_are_set_is_a_clear_error(tmp_path):
    targets = tmp_path / "targets.toml"
    targets.write_text("[sky130hd]\n# t1_ps = 0\n")
    with pytest.raises(SystemExit, match="t1_ps is not set"):
        rs.load_target_ps("sky130hd", "t1", targets)
    assert rs.load_target_ps("sky130hd", "unc", targets) is None


def test_demo_prints_area_and_time_and_writes_no_csv(tmp_path, monkeypatch, capsys):
    lib = tmp_path / "sky130.lib"
    lib.write_text(LIBERTY)
    _use_tmp_repo(monkeypatch, tmp_path, lib)
    for src in rs.SOURCES["int8"]:
        (tmp_path / src).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / src).write_text("// stub\n")

    def fake_run(cmd, **kwargs):
        if cmd[-1] == "-V":
            return subprocess.CompletedProcess(cmd, 0, stdout="Yosys 0.57 (git sha1 x)\n")
        stem = cmd[-1][: -len(".ys")]
        Path(f"{stem}_stat.json").write_text(FLAT_JSON)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rs.subprocess, "run", fake_run)
    rs.main(["--lib", "sky130hd", "--fmt", "int8", "--target", "unc", "--demo"])
    out = capsys.readouterr().out
    assert "dp32_int8 on sky130hd (unc): 21,873.5 um^2, 2,301 cells" in out
    assert re.search(r"synthesized in \d+\.\d s with Yosys 0\.57", out)
    assert not (tmp_path / "results/area.csv").exists()


def test_append_rows_with_key_replaces_the_earlier_row(tmp_path):
    path = tmp_path / "area.csv"
    key = ("lib", "format", "unit", "target", "align_w", "run")
    row = dict.fromkeys(rs.AREA_COLUMNS, "")
    row.update(lib="sky130hd", format="int8", unit="dp32", target="t1", align_w=24, run=0,
               area_um2="1.0")
    rs.append_rows(path, rs.AREA_COLUMNS, [row], key=key)
    row["area_um2"] = "2.0"
    rs.append_rows(path, rs.AREA_COLUMNS, [row], key=key)          # same key: replaced
    rs.append_rows(path, rs.AREA_COLUMNS, [dict(row, format="int4", area_um2="3.0")], key=key)
    lines = path.read_text().splitlines()
    assert len(lines) == 3, lines
    assert lines[1].startswith("sky130hd,int8,dp32,t1,,24,0,2.0")
    assert lines[2].startswith("sky130hd,int4,dp32,t1,,24,0,3.0")


def test_abc_margin_tightens_the_mapping_target(tmp_path):
    targets = tmp_path / "targets.toml"
    targets.write_text("[sky130hd]\nt1_ps = 36600\nabc_margin = 0.03\n")
    assert rs.load_abc_margin("sky130hd", targets) == pytest.approx(0.03)
    assert rs.abc_target_ps(36600.0, 0.03) == pytest.approx(36600 / 1.03)
    targets.write_text("[sky130hd]\nt1_ps = 36600\n")
    assert rs.load_abc_margin("sky130hd", targets) == 0.0
