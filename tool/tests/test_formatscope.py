"""Unit tests for the formatscope tool on a tiny synthetic results set (build-plan.md S8.3)."""

import csv

import pytest

from formatscope.cli import main, table_rows
from formatscope.data import load_points
from formatscope.pareto import best_per_area, frontier
from formatscope.recommend import as_percent, best_under_budget, smallest_meeting

ACC_COLS = ["format", "stage", "top1", "top5", "calib_images", "calib_stat",
            "quantize_first_last", "epochs", "seed", "checkpoint", "date"]
AREA_COLS = ["lib", "format", "unit", "target", "dtarget_ps", "align_w", "run",
             "area_um2", "cell_count", "yosys_version", "date"]
TIMING_COLS = ["lib", "format", "unit", "target", "align_w", "run", "period_ns",
               "delay_ps", "wns_ps", "sta_version", "date"]


def _write(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


@pytest.fixture
def rdir(tmp_path):
    acc = [("fp32", "none", 86.34), ("int4", "ptq", 61.65), ("int8", "ptq", 86.39),
           ("fp8e4m3", "ptq", 85.50), ("mxint8", "ptq", 86.39), ("mxfp4", "ptq", 60.71),
           ("int4", "qat", 80.00), ("int4", "ptq", 62.00)]  # repeated key: last row wins
    _write(tmp_path / "accuracy.csv", ACC_COLS,
           [{"format": f, "stage": s, "top1": t} for f, s, t in acc])
    areas = {"int4": [8000, 8200, 8100], "int8": [20000], "fp8e4m3": [30000], "mxfp4": [12000]}
    rows = []
    for fmt, values in areas.items():
        for run, a in enumerate(values):
            rows.append({"lib": "sky130hd", "format": fmt, "unit": "dp32", "target": "t1",
                         "dtarget_ps": 4000, "align_w": 0 if fmt.startswith("int") else 24,
                         "run": run, "area_um2": a})
    rows.append({"lib": "sky130hd", "format": "fp8e4m3", "unit": "dp32", "target": "t1",
                 "dtarget_ps": 4000, "align_w": 32, "run": 0, "area_um2": 35000})
    _write(tmp_path / "area.csv", AREA_COLS, rows)
    _write(tmp_path / "timing.csv", TIMING_COLS,
           [{"lib": "sky130hd", "format": "int8", "unit": "dp32", "target": "t1",
             "align_w": 0, "run": 0, "period_ns": 4, "delay_ps": 3900}])
    return tmp_path


def test_join_median_spread_and_last_row_wins(rdir):
    pts = {p.format: p for p in load_points(rdir, target="t1", align_w=24)}
    assert pts["int4"].area == 8100
    assert pts["int4"].area_spread == (8000, 8200)
    assert pts["int4"].top1_ptq == 62.00
    assert pts["int4"].top1_qat == 80.00
    assert pts["fp8e4m3"].area == 30000  # align_w=32 row excluded
    assert pts["mxint8"].area is None  # accuracy only, no hardware yet
    assert pts["int8"].delay == 3900


def test_align_w_filter_selects_sweep_rows(rdir):
    pts = {p.format: p for p in load_points(rdir, target="t1", align_w=32)}
    assert pts["fp8e4m3"].area == 35000
    assert pts["int8"].area == 20000  # INT units have no window, match either


def test_frontier_drops_dominated_points(rdir):
    pts = load_points(rdir, target="t1")
    front = [p.format for p in frontier(pts)]
    # mxfp4 (12000, 60.71) is dominated by int4 (8100, 62.0); fp8 by int8
    assert front == ["int4", "int8"]
    assert best_per_area(pts).format == "int4"


def test_frontier_empty_and_ties():
    assert frontier([]) == []
    assert best_per_area([]) is None


def test_recommend(rdir):
    pts = load_points(rdir, target="t1")
    assert smallest_meeting(pts, 85.0).format == "int8"
    assert smallest_meeting(pts, 99.0) is None
    assert best_under_budget(pts, 15000).format == "int4"
    assert best_under_budget(pts, 25000).format == "int8"
    assert best_under_budget(pts, 100) is None
    assert as_percent(0.87) == pytest.approx(87.0)
    assert as_percent(87) == 87


def test_recommend_qat_stage(rdir):
    pts = load_points(rdir, target="t1")
    assert smallest_meeting(pts, 70.0, stage="qat").format == "int4"


def test_cli_recommend_default_margin(rdir, capsys):
    assert main(["--results", str(rdir), "recommend", "--target", "t1"]) == 0
    out = capsys.readouterr().out
    assert ">= 85.34%" in out
    assert "int8:" in out


def test_cli_recommend_no_match_exit_code(rdir):
    assert main(["--results", str(rdir), "recommend", "--min-acc", "0.999"]) == 1


def test_table_has_every_configuration(rdir):
    header, rows = table_rows(rdir, "sky130hd", 24)
    assert [r[0] for r in rows] == ["fp32", "int4", "int8", "fp8e4m3", "mxint8", "mxfp4", "int4_b32"]
    assert len(header) == len(rows[0])
    int4 = rows[1]
    assert int4[header.index("PTQ top-1")] == "62.00"
    assert int4[header.index("QAT top-1")] == "80.00"
    assert int4[header.index("area t1 (µm²)")] == "8100"
    # Block-scaled formats carry the 8-bit shared scale amortized over 32 elements.
    bits = {r[0]: r[header.index("bits per number")] for r in rows}
    assert bits == {"fp32": "32", "int4": "4", "int8": "8", "fp8e4m3": "8",
                    "mxint8": "8.25", "mxfp4": "4.25", "int4_b32": "4.25"}


def test_plot_writes_png_and_svg(rdir):
    assert main(["--results", str(rdir), "plot"]) == 0
    figs = rdir / "figures"
    assert (figs / "frontier_sky130hd_t1.png").exists()
    assert (figs / "frontier_sky130hd_t1.svg").exists()


def test_plot_with_no_hardware_rows_fails_cleanly(tmp_path):
    _write(tmp_path / "accuracy.csv", ACC_COLS, [{"format": "int8", "stage": "ptq", "top1": 86}])
    assert main(["--results", str(tmp_path), "plot"]) == 1


def test_cli_run_from_results_on_committed_csvs(capsys):
    assert main(["run", "--from-results"]) == 0
    assert "frontier" in capsys.readouterr().out
