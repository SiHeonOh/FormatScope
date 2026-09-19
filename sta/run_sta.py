"""OpenSTA driver: timing.csv (build-plan.md S3.7d, S7.3).

Times every synthesized netlist in synth/out (or the ones given with --netlist)
and appends one row per netlist to results/timing.csv in the Appendix A column
order. The raw OpenSTA report for each netlist is kept in sta/out/.

    python sta/run_sta.py --smoke                  # S3.7d: time the smoke adders
    python sta/run_sta.py                          # every dp32 netlist in synth/out
    python sta/run_sta.py --netlist synth/out/sky130hd_int8_unc_a24_r0_netlist.v

Netlist names carry their metadata: {LIB_ID}_{FMT}_{TARGET}_a{ALIGN_W}_r{RUN}_netlist.v.
delay_ps is the data arrival time of the worst path, which does not depend on
the clock period; the period only sets WNS. Unconstrained netlists are timed
against a 10 ns clock, t1/t2 netlists against their target from synth/targets.toml.

OpenSTA runs as `sta -no_init -no_splash -exit <script>`. To use another
binary or the Docker image built from OpenSTA's Dockerfile.ubuntu24.04, set
FORMATSCOPE_STA to the command prefix. Mount the repository and $PDK_ROOT at
the same absolute paths inside the container so every path in the script
resolves unchanged:

    export FORMATSCOPE_STA='docker run --rm -i -v $PWD:$PWD -v $PDK_ROOT:$PDK_ROOT -w $PWD opensta'

Variables in FORMATSCOPE_STA are expanded when the script runs, so run it from
the repository root (the Makefile does).
"""

import argparse
import datetime
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synth.run_synth import (  # noqa: E402
    TARGETS_TOML, UNIT, append_rows, load_lib_path, render,
)

STA_DIR = ROOT / "sta"
OUT_DIR = STA_DIR / "out"
SYNTH_OUT = ROOT / "synth" / "out"
TIMING_CSV = ROOT / "results" / "timing.csv"
TIMING_COLUMNS = [
    "lib", "format", "unit", "target", "align_w", "run",
    "period_ns", "delay_ps", "wns_ps", "sta_version", "date",
]
UNC_PERIOD_NS = 10.0

NETLIST_RE = re.compile(
    # A format id may contain an underscore (int4_b32); the target anchors the split.
    r"^(?P<lib>sky130hd|asap7)_(?P<format>[a-z0-9]+(?:_[a-z0-9]+)*?)_(?P<target>unc|t1|t2)"
    r"_a(?P<align_w>\d+)_r(?P<run>\d+)_netlist\.v$"
)
SMOKE_NETLISTS = {
    "smoke_add_reg": "synth/out/smoke_add_reg_netlist.v",
    "smoke_add": "synth/out/smoke_add_netlist.v",
}


# ---------------------------------------------------------------- parsing

def parse_netlist_name(name):
    """{lib, format, target, align_w, run} from a synth/out netlist file name."""
    m = NETLIST_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"{name}: not a {{LIB}}_{{FMT}}_{{TARGET}}_a{{W}}_r{{RUN}}_netlist.v name")
    meta = m.groupdict()
    meta["align_w"] = int(meta["align_w"])
    meta["run"] = int(meta["run"])
    return meta


def parse_arrival_ns(report):
    """Data arrival time of the first (worst) path in report_checks output."""
    m = re.search(r"^\s*(-?[0-9.]+)\s+data arrival time\s*$", report, re.M)
    if not m:
        raise ValueError("no 'data arrival time' in the OpenSTA report (no constrained paths?)")
    return float(m.group(1))


def parse_wns_ns(report):
    """WNS from report_wns: accepts both 'wns 0.00' and 'wns max -0.12'."""
    m = re.search(r"^\s*wns(?:\s+max)?\s+(-?[0-9.]+)\s*$", report, re.M)
    if not m:
        raise ValueError("no 'wns' line in the OpenSTA report")
    return float(m.group(1))


def check_report(report):
    """Raise on OpenSTA errors, which do not always set a non-zero exit code."""
    errors = [line for line in report.splitlines() if re.match(r"^\s*Error\b", line)]
    if errors:
        raise RuntimeError("OpenSTA reported errors:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------- running

def period_ns(lib_id, target, targets_toml=TARGETS_TOML):
    if target == "unc":
        return UNC_PERIOD_NS
    with open(targets_toml, "rb") as f:
        value = tomllib.load(f).get(lib_id, {}).get(f"{target}_ps")
    if not value:
        raise SystemExit(f"{targets_toml}: [{lib_id}] {target}_ps is not set (S7.2)")
    return float(value) / 1000.0


def sta_prefix():
    return shlex.split(os.path.expandvars(os.environ.get("FORMATSCOPE_STA", "sta")))


def sta_version():
    result = subprocess.run(sta_prefix() + ["-version"], capture_output=True, text=True)
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"


def run_sta(lib_path, netlist, top, period, stem):
    """Render the Tcl script, run OpenSTA, save and return the report text."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    template = (STA_DIR / "sta.tcl.template").read_text()
    script = OUT_DIR / f"{stem}.tcl"
    script.write_text(render(template, {
        "LIB": lib_path, "NETLIST": Path(netlist).resolve(), "TOP": top,
        "PERIOD_NS": f"{period:.3f}",
    }))
    cmd = sta_prefix() + ["-no_init", "-no_splash", "-exit", str(script)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    report = result.stdout + result.stderr
    report_path = OUT_DIR / f"{stem}_sta.txt"
    report_path.write_text(report)
    if result.returncode != 0:
        raise SystemExit(f"OpenSTA failed ({result.returncode}); see {report_path}")
    check_report(report)
    return report, report_path


def time_smoke(args):
    lib_path = load_lib_path(args.lib)
    ok = True
    for top, netlist in SMOKE_NETLISTS.items():
        if not (ROOT / netlist).is_file():
            raise SystemExit(f"{netlist} missing; run `python synth/run_synth.py --smoke` first")
        report, report_path = run_sta(lib_path, ROOT / netlist, top, UNC_PERIOD_NS, top)
        start = re.search(r"^Startpoint:\s*(.+)$", report, re.M)
        end = re.search(r"^Endpoint:\s*(.+)$", report, re.M)
        slack = re.search(r"^\s*(-?[0-9.]+)\s+slack \((MET|VIOLATED)\)", report, re.M)
        passed = bool(start and end and slack)
        ok = ok and passed
        if passed:
            print(f"PASS {top}: {start.group(1)} -> {end.group(1)}, arrival "
                  f"{parse_arrival_ns(report):.3f} ns, slack {slack.group(1)} ns "
                  f"({slack.group(2)}); report {report_path.relative_to(ROOT)}")
        else:
            print(f"FAIL {top}: no Startpoint/Endpoint/slack block; see {report_path}")
    if not ok:
        raise SystemExit(1)


def time_netlists(args):
    netlists = [Path(n) for n in args.netlist] if args.netlist else sorted(
        p for p in SYNTH_OUT.glob("*_netlist.v") if NETLIST_RE.match(p.name))
    if not netlists:
        raise SystemExit(f"no dp32 netlists in {SYNTH_OUT}; run synth/run_synth.py first")
    version = sta_version()
    date = datetime.date.today().isoformat()
    lib_paths = {}
    for netlist in netlists:
        meta = parse_netlist_name(netlist)
        lib_id = meta["lib"]
        if lib_id not in lib_paths:
            lib_paths[lib_id] = load_lib_path(lib_id)
        period = period_ns(lib_id, meta["target"])
        stem = netlist.name[: -len("_netlist.v")]
        report, _ = run_sta(lib_paths[lib_id], netlist, f"{UNIT}_{meta['format']}", period, stem)
        row = {
            "lib": lib_id, "format": meta["format"], "unit": UNIT, "target": meta["target"],
            "align_w": meta["align_w"], "run": meta["run"], "period_ns": f"{period:.3f}",
            "delay_ps": f"{parse_arrival_ns(report) * 1000:.1f}",
            "wns_ps": f"{parse_wns_ns(report) * 1000:.1f}",
            "sta_version": version, "date": date,
        }
        append_rows(TIMING_CSV, TIMING_COLUMNS, [row],
                    key=("lib", "format", "unit", "target", "align_w", "run"))
        print(f"{stem}: delay {row['delay_ps']} ps, wns {row['wns_ps']} ps @ {period:.3f} ns")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lib", default="sky130hd", help="library for --smoke")
    parser.add_argument("--netlist", action="append", help="repeatable; default all in synth/out")
    parser.add_argument("--smoke", action="store_true", help="S3.7d smoke adders only")
    args = parser.parse_args(argv)
    if args.smoke:
        time_smoke(args)
    else:
        time_netlists(args)


if __name__ == "__main__":
    main()
