"""Yosys synthesis driver: area.csv and breakdown.csv (build-plan.md S7.1, S7.4, S7.5).

Renders synth/synth_flat.ys.template (or synth_hier for --hier) for every
(library, format, delay target, alignment window, run), runs Yosys from the
repository root, and appends one row per netlist to results/area.csv in the
Appendix A column order.

    python synth/run_synth.py --smoke                     # S3.7a adder, prints area
    python synth/run_synth.py --fmt int8 --target unc     # one unit, one target
    python synth/run_synth.py --target all --perturb      # S7.4 spread runs
    python synth/run_synth.py --hier --target t1          # S7.5 stage breakdown
    python synth/run_synth.py --fmt int8 --dry-run        # render only
    python synth/run_synth.py --fmt int8 --demo           # on-camera run, no CSV (S8.5)

Library paths come from synth/libs.toml, section = $FORMATSCOPE_MACHINE or the
hostname. Delay targets come from synth/targets.toml. Set FORMATSCOPE_YOSYS to
override the yosys binary.
"""

import argparse
import csv
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYNTH_DIR = ROOT / "synth"
OUT_DIR = SYNTH_DIR / "out"
RESULTS_DIR = ROOT / "results"
AREA_CSV = RESULTS_DIR / "area.csv"
BREAKDOWN_CSV = RESULTS_DIR / "breakdown.csv"
LIBS_TOML = SYNTH_DIR / "libs.toml"
TARGETS_TOML = SYNTH_DIR / "targets.toml"

AREA_COLUMNS = [
    "lib", "format", "unit", "target", "dtarget_ps", "align_w", "run",
    "area_um2", "cell_count", "yosys_version", "date",
]
BREAKDOWN_COLUMNS = ["lib", "format", "target", "align_w", "stage", "area_um2", "date"]

UNIT = "dp32"
FORMATS = ["int4", "int8", "fp8e4m3", "mxint8", "mxfp4"]
LIB_IDS = ["sky130hd", "asap7"]
TARGETS = ["unc", "t1", "t2"]
DEFAULT_ALIGN_W = 24
# Only the fused-stage units have an alignment window (S5.4, decision D6).
ALIGNED_FORMATS = {"fp8e4m3", "mxint8", "mxfp4"}
PERTURB_FACTORS = [0.98, 0.99, 1.00, 1.01, 1.02]  # decision D8
STAGES = ["decode", "mul", "align", "tree", "normacc", "other"]

# RTL read for each unit. Edit here when a unit gains or drops a dependency;
# the templates never hard-code sources.
SOURCES = {
    "int4": ["rtl/common/adder_tree.v", "rtl/int4/dp32_int4.v"],
    "int8": ["rtl/common/adder_tree.v", "rtl/int8/dp32_int8.v"],
    "fp8e4m3": [
        "rtl/common/adder_tree.v", "rtl/common/lzc.v", "rtl/common/fused_stage.v",
        "rtl/common/decode_fp8e4m3.v", "rtl/fp8e4m3/dp32_fp8e4m3.v",
    ],
    "mxint8": [
        "rtl/common/adder_tree.v", "rtl/common/lzc.v", "rtl/common/fused_stage.v",
        "rtl/common/decode_e8m0.v", "rtl/mxint8/dp32_mxint8.v",
    ],
    "mxfp4": [
        "rtl/common/adder_tree.v", "rtl/common/lzc.v", "rtl/common/fused_stage.v",
        "rtl/common/decode_e2m1.v", "rtl/common/decode_e8m0.v", "rtl/mxfp4/dp32_mxfp4.v",
    ],
}
SMOKE_DESIGNS = {
    "smoke_add": ["rtl/smoke/smoke_add.v"],
    "smoke_add_reg": ["rtl/smoke/smoke_add_reg.v"],
}

_PLACEHOLDER_RE = re.compile(r"\{[A-Z_]+\}")


# ---------------------------------------------------------------- configuration

def machine_name():
    return os.environ.get("FORMATSCOPE_MACHINE") or socket.gethostname()


def load_lib_path(lib_id, libs_toml=LIBS_TOML, machine=None, must_exist=True):
    """Absolute liberty path for lib_id on this machine, from libs.toml."""
    machine = machine or machine_name()
    with open(libs_toml, "rb") as f:
        libs = tomllib.load(f)
    if machine not in libs:
        raise SystemExit(
            f"{libs_toml}: no [{machine}] section (have: {', '.join(libs) or 'none'}). "
            f"Add one, or set FORMATSCOPE_MACHINE to an existing section."
        )
    if lib_id not in libs[machine]:
        raise SystemExit(f"{libs_toml}: [{machine}] has no '{lib_id}' entry")
    path = os.path.expanduser(os.path.expandvars(libs[machine][lib_id]))
    if must_exist and not os.path.isfile(path):
        raise SystemExit(f"{libs_toml}: [{machine}] {lib_id} = {path} does not exist")
    return path


def load_target_ps(lib_id, target, targets_toml=TARGETS_TOML):
    """Delay target in picoseconds, or None for the unconstrained run."""
    if target == "unc":
        return None
    with open(targets_toml, "rb") as f:
        targets = tomllib.load(f)
    key = f"{target}_ps"
    value = targets.get(lib_id, {}).get(key)
    if not value:
        raise SystemExit(
            f"{targets_toml}: [{lib_id}] {key} is not set. Run every unit at "
            f"--target unc, time it with sta/run_sta.py, and fix T1/T2 per S7.2 first."
        )
    return float(value)


# ---------------------------------------------------------------- rendering

def dtarget_arg(target, dtarget_ps):
    """The ABC -D argument. Refuses a delay target on the unconstrained run (risk 11)."""
    if target == "unc":
        if dtarget_ps is not None:
            raise ValueError("target=unc must not carry an ABC -D delay target")
        return ""
    if dtarget_ps is None or dtarget_ps <= 0:
        raise ValueError(f"target={target} needs a positive delay target, got {dtarget_ps}")
    return f"-D {int(round(dtarget_ps))}"


def render(template_text, fields):
    """Substitute {KEY} fields; raise if any placeholder is left unfilled."""
    text = template_text
    for key, value in fields.items():
        text = text.replace("{" + key + "}", str(value))
    leftover = sorted(set(_PLACEHOLDER_RE.findall(text)))
    if leftover:
        raise ValueError(f"unfilled template placeholders: {', '.join(leftover)}")
    return text


def top_name(fmt):
    return f"{UNIT}_{fmt}"


def out_stem(lib_id, fmt, target, align_w, run):
    return f"{lib_id}_{fmt}_{target}_a{align_w}_r{run}"


def synth_fields(lib_path, sources, top, dtarget, out_prefix, align_w=None):
    chparam = (f"chparam -set ALIGN_W {align_w} {top}" if align_w is not None
               else "# no ALIGN_W parameter on this unit")
    return {
        "LIB": lib_path,
        "SOURCES": " ".join(sources),
        "TOP": top,
        "CHPARAM": chparam,
        "DTARGET": dtarget,
        "OUT": out_prefix,
    }


# ---------------------------------------------------------------- parsing

def _load_stat_json(text):
    """Parse `stat -json` output, tolerating any log text before the JSON object."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in stat output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def _module_key(modules, top):
    """The key for `top` in the stat JSON modules dict ('\\top' or a $paramod name)."""
    for key in (f"\\{top}", top):
        if key in modules:
            return key
    matches = [k for k in modules if re.search(rf"\\{re.escape(top)}(\\|$)", k)]
    if len(matches) == 1:
        return matches[0]
    if len(modules) == 1:
        return next(iter(modules))
    raise ValueError(f"module {top} not found in stat JSON (have {list(modules)})")


def _cells(entry):
    for key in ("num_cells", "cells"):
        if key in entry:
            value = entry[key]
            return int(value["count"] if isinstance(value, dict) else value)
    return None


def _area(entry):
    for key in ("area", "chip_area"):
        if key in entry:
            return float(entry[key])
    return None


def parse_stat_json(text, top):
    """(area_um2, cell_count) for `top` from `stat -json -liberty` output."""
    stat = _load_stat_json(text)
    design = stat.get("design", {})
    area, cells = _area(design), _cells(design)
    if area is None or cells is None:
        entry = stat.get("modules", {})[_module_key(stat.get("modules", {}), top)]
        area = area if area is not None else _area(entry)
        cells = cells if cells is not None else _cells(entry)
    if area is None:
        raise ValueError("stat JSON has no area (was stat run with -liberty?)")
    return area, cells


def parse_stat_text(text, top):
    """(area_um2, cell_count) from the human `stat -liberty` output (fallback)."""
    area = None
    m = re.search(r"Chip area for top module '\\?([^']+)':\s*([-0-9.eE+]+)", text)
    if m:
        area = float(m.group(2))
    else:
        for m in re.finditer(r"Chip area for module '\\?([^']+)':\s*([-0-9.eE+]+)", text):
            if m.group(1) == top or m.group(1).endswith("\\" + top):
                area = float(m.group(2))
    if area is None:
        raise ValueError(f"no 'Chip area for module' line for {top}")
    section = text
    header = re.search(rf"^=== \\?{re.escape(top)} ===$", text, re.M)
    if header:
        section = text[header.end():]
    cells = None
    m = re.search(r"Number of cells:\s+(\d+)", section)
    if m:
        cells = int(m.group(1))
    else:
        m = re.search(r"^\s*(\d+)\s+(?:[-0-9.eE+]+\s+)?cells\s*$", section, re.M)
        if m:
            cells = int(m.group(1))
    return area, cells


def parse_stat_files(json_path, text_path, top):
    """Area and cell count, JSON first, text second; raises naming both files."""
    errors = []
    for path, parser in ((json_path, parse_stat_json), (text_path, parse_stat_text)):
        try:
            return parser(Path(path).read_text(), top)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{path}: {exc}")
    raise RuntimeError("could not parse Yosys stat output:\n  " + "\n  ".join(errors))


def liberty_cell_areas(text):
    """{cell_name: area} from a liberty file's `cell (name) { ... area : X; }` groups."""
    starts = list(re.finditer(r'\bcell\s*\(\s*"?([^")\s]+)"?\s*\)\s*\{', text))
    areas = {}
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        area = re.search(r"\barea\s*:\s*([-0-9.eE+]+)\s*;", text[m.end():end])
        if area:
            areas[m.group(1)] = float(area.group(1))
    return areas


def stage_of(module_name):
    """Map a (possibly $paramod-mangled) module name to a breakdown stage."""
    name = module_name.lower()
    for token, stage in (("mul_stage", "mul"), ("align_stage", "align"),
                         ("tree_stage", "tree"), ("normacc_stage", "normacc"),
                         ("decode_", "decode")):
        if token in name:
            return stage
    return "other"


def breakdown_from_stat(stat_text, top, cell_areas):
    """{stage: area_um2} from hierarchical `stat -json` output.

    Each module's own area is the sum of its liberty cells; submodule instances
    are followed recursively, multiplied by their instance count. A module with
    no stage name of its own inherits its parent's stage, so an adder_tree
    inside tree_stage counts as tree.
    """
    modules = _load_stat_json(stat_text).get("modules", {})
    totals = {}

    def visit(key, count, inherited):
        stage = stage_of(key)
        if stage == "other":
            stage = inherited
        for cell_type, n in modules[key].get("num_cells_by_type", {}).items():
            n = int(n)
            if cell_type in modules:
                visit(cell_type, count * n, stage)
            elif cell_type in cell_areas:
                totals[stage] = totals.get(stage, 0.0) + count * n * cell_areas[cell_type]
            elif not cell_type.startswith("$"):
                raise ValueError(f"cell {cell_type} in {key} is neither a module nor a liberty cell")

    visit(_module_key(modules, top), 1, "other")
    return totals


# ---------------------------------------------------------------- CSV rows

def area_row(lib_id, fmt, target, dtarget_ps, align_w, run, area, cells, yosys_version, date):
    if area is None or area <= 0:
        raise ValueError(
            f"{lib_id} {fmt} {target} a{align_w} r{run}: area {area} <= 0; the design "
            f"was optimized away (outputs registered and used? hierarchy -check clean?)"
        )
    return {
        "lib": lib_id, "format": fmt, "unit": UNIT, "target": target,
        "dtarget_ps": "" if dtarget_ps is None else int(round(dtarget_ps)),
        "align_w": align_w, "run": run, "area_um2": f"{area:.4f}",
        "cell_count": "" if cells is None else cells,
        "yosys_version": yosys_version, "date": date,
    }


def append_rows(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------- running

def yosys_bin():
    return os.environ.get("FORMATSCOPE_YOSYS", "yosys")


def yosys_version():
    out = subprocess.run([yosys_bin(), "-V"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Yosys\s+(\S+)", out)
    return m.group(1) if m else out.strip()


def run_yosys(script_text, out_prefix, dry_run=False):
    script = Path(f"{out_prefix}.ys")
    log = Path(f"{out_prefix}.log")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(script_text)
    cmd = [yosys_bin(), "-q", "-l", str(log), "-s", str(script)]
    if dry_run:
        print(" ".join(cmd))
        return
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"yosys failed ({result.returncode}); see {log}")


def check_sources(sources, dry_run):
    missing = [s for s in sources if not (ROOT / s).is_file()]
    if missing:
        message = "missing RTL: " + ", ".join(missing)
        if not dry_run:
            raise SystemExit(message + " (is the unit merged on this branch?)")
        print("warning: " + message, file=sys.stderr)


def jobs(formats, targets, align_ws, perturb, runs):
    """(fmt, target, align_w, run, factor) tuples to synthesize."""
    for fmt in formats:
        for align_w in align_ws:
            if fmt not in ALIGNED_FORMATS and align_w != DEFAULT_ALIGN_W:
                continue  # INT units have no window; one row at the default is enough
            for target in targets:
                if target == "unc":
                    # No -D here, so repeated runs are the S7.4 determinism check.
                    count = len(PERTURB_FACTORS) if perturb else runs
                    for run in range(count):
                        yield fmt, target, align_w, run, None
                elif perturb:
                    for run, factor in enumerate(PERTURB_FACTORS):
                        yield fmt, target, align_w, run, factor
                else:
                    for run in range(runs):
                        yield fmt, target, align_w, run, 1.0


def demo_lines(lib_id, fmt, target, area, cells, elapsed, version):
    """The two lines `formatscope demo` shows on camera (S8.5)."""
    cells_text = "?" if cells is None else f"{cells:,}"
    return (f"{UNIT}_{fmt} on {lib_id} ({target}): {area:,.1f} um^2, {cells_text} cells\n"
            f"synthesized in {elapsed:.1f} s with Yosys {version}")


def synth_smoke(args):
    lib_path = load_lib_path(args.lib, must_exist=not args.dry_run)
    template = (SYNTH_DIR / "synth_flat.ys.template").read_text()
    for top, sources in SMOKE_DESIGNS.items():
        out_prefix = f"synth/out/{top}"
        fields = synth_fields(lib_path, sources, top, "", out_prefix)
        run_yosys(render(template, fields), ROOT / out_prefix, args.dry_run)
        if args.dry_run:
            continue
        area, cells = parse_stat_files(ROOT / f"{out_prefix}_stat.json",
                                       ROOT / f"{out_prefix}_stat.txt", top)
        status = "PASS" if area > 0 else "FAIL"
        print(f"{status} {top}: area {area:.3f} um^2, {cells} cells, "
              f"netlist {out_prefix}_netlist.v")
        if area <= 0:
            raise SystemExit(1)


def synth_units(args):
    formats = args.fmt or FORMATS
    targets = TARGETS if args.target == "all" else [args.target]
    align_ws = args.align_w or [DEFAULT_ALIGN_W]
    lib_path = load_lib_path(args.lib, must_exist=not args.dry_run)
    template_name = "synth_hier.ys.template" if args.hier else "synth_flat.ys.template"
    template = (SYNTH_DIR / template_name).read_text()
    out_sub = "synth/out/hier" if args.hier else "synth/out"
    version = "dry-run" if args.dry_run else yosys_version()
    date = datetime.date.today().isoformat()
    cell_areas = None

    for fmt, target, align_w, run, factor in jobs(formats, targets, align_ws,
                                                  args.perturb and not (args.hier or args.demo),
                                                  1 if args.demo else args.runs):
        base_ps = load_target_ps(args.lib, target)
        dtarget_ps = None if base_ps is None else base_ps * factor
        sources = SOURCES[fmt]
        check_sources(sources, args.dry_run)
        top = top_name(fmt)
        out_prefix = f"{out_sub}/{out_stem(args.lib, fmt, target, align_w, run)}"
        fields = synth_fields(lib_path, sources, top, dtarget_arg(target, dtarget_ps),
                              out_prefix, align_w if fmt in ALIGNED_FORMATS else None)
        started = time.monotonic()
        run_yosys(render(template, fields), ROOT / out_prefix, args.dry_run)
        elapsed = time.monotonic() - started
        if args.dry_run:
            continue

        if args.demo:
            area, cells = parse_stat_files(ROOT / f"{out_prefix}_stat.json",
                                           ROOT / f"{out_prefix}_stat.txt", top)
            print(demo_lines(args.lib, fmt, target, area, cells, elapsed, version))
        elif args.hier:
            if cell_areas is None:
                cell_areas = liberty_cell_areas(Path(lib_path).read_text(errors="replace"))
            stat_text = (ROOT / f"{out_prefix}_stat.json").read_text()
            totals = breakdown_from_stat(stat_text, top, cell_areas)
            rows = [{"lib": args.lib, "format": fmt, "target": target, "align_w": align_w,
                     "stage": stage, "area_um2": f"{totals[stage]:.4f}", "date": date}
                    for stage in STAGES if totals.get(stage, 0.0) > 0]
            append_rows(BREAKDOWN_CSV, BREAKDOWN_COLUMNS, rows)
            print(f"{fmt} {target} a{align_w}: " +
                  ", ".join(f"{r['stage']} {r['area_um2']}" for r in rows))
        else:
            area, cells = parse_stat_files(ROOT / f"{out_prefix}_stat.json",
                                           ROOT / f"{out_prefix}_stat.txt", top)
            row = area_row(args.lib, fmt, target, dtarget_ps, align_w, run, area, cells,
                           version, date)
            append_rows(AREA_CSV, AREA_COLUMNS, [row])
            print(f"{fmt} {target} a{align_w} r{run}: {row['area_um2']} um^2, {cells} cells")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lib", default="sky130hd", choices=LIB_IDS)
    parser.add_argument("--fmt", action="append", choices=FORMATS,
                        help="repeatable; default all five units")
    parser.add_argument("--target", default="unc", choices=TARGETS + ["all"])
    parser.add_argument("--align-w", type=int, action="append",
                        help="repeatable; default 24 (fused-stage units only)")
    parser.add_argument("--runs", type=int, default=1,
                        help="repeat each job this many times (determinism check)")
    parser.add_argument("--perturb", action="store_true",
                        help="five runs at -D x 0.98..1.02 (S7.4)")
    parser.add_argument("--hier", action="store_true", help="stage breakdown (S7.5)")
    parser.add_argument("--smoke", action="store_true", help="S3.7a smoke adder only")
    parser.add_argument("--demo", action="store_true",
                        help="print area, cells and elapsed time; write no CSV (S8.5)")
    parser.add_argument("--dry-run", action="store_true", help="render scripts, run nothing")
    args = parser.parse_args(argv)
    if args.demo and args.hier:
        parser.error("--demo and --hier are exclusive")
    if args.smoke:
        synth_smoke(args)
    else:
        synth_units(args)


if __name__ == "__main__":
    main()
