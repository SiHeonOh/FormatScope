"""Load and join the committed CSVs (build-plan.md S8.2, Appendix A).

The CSVs are append-only, so when a key repeats the last row wins for
accuracy, and every run is kept for area and timing (the median is the
reported value, min/max is the spread from S7.4).
"""

import csv
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from formatscope import HARDWARE_FORMATS

_REPO_ROOT = Path(__file__).resolve().parents[2]


def results_dir(override=None):
    if override:
        return Path(override)
    return Path(os.environ.get("FORMATSCOPE_RESULTS", _REPO_ROOT / "results"))


def _read(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _float(value):
    return float(value) if value not in (None, "") else None


def load_accuracy(rdir):
    """{(format, stage): top1 percent}, last row per key wins."""
    acc = {}
    for row in _read(rdir / "accuracy.csv"):
        top1 = _float(row.get("top1"))
        if top1 is not None:
            acc[(row["format"], row["stage"])] = top1
    return acc


def fp32_top1(acc):
    return acc.get(("fp32", "none"))


@dataclass
class Point:
    format: str
    lib: str
    target: str
    align_w: int
    top1_ptq: float = None
    top1_qat: float = None
    areas: list = field(default_factory=list)
    delays: list = field(default_factory=list)
    dtarget_ps: float = None

    @property
    def area(self):
        return statistics.median(self.areas) if self.areas else None

    @property
    def area_spread(self):
        return (min(self.areas), max(self.areas)) if self.areas else (None, None)

    @property
    def delay(self):
        return statistics.median(self.delays) if self.delays else None

    def top1(self, stage):
        return self.top1_qat if stage == "qat" else self.top1_ptq


def load_points(rdir, lib="sky130hd", target="t1", align_w=24):
    """One Point per hardware format at (lib, target, align_w).

    INT units have no alignment window, so their rows match any align_w.
    Formats with no area row yet are still returned (area None) so the
    table can show accuracy before synthesis lands.
    """
    acc = load_accuracy(rdir)
    points = {
        fmt: Point(fmt, lib, target, align_w,
                   top1_ptq=acc.get((fmt, "ptq")), top1_qat=acc.get((fmt, "qat")))
        for fmt in HARDWARE_FORMATS
    }

    def matches(row):
        if row.get("lib") != lib or row.get("target") != target:
            return False
        if row.get("format") not in points:
            return False
        if row["format"] in ("int4", "int8"):
            return True
        return int(row.get("align_w") or 0) == align_w

    # The CSVs are append-only, so a repeated `make synth` writes the same run
    # index again. The last row per run index wins; it replaces, not stacks.
    areas, delays = {}, {}
    for row in _read(rdir / "area.csv"):
        if matches(row):
            area = _float(row.get("area_um2"))
            if area and area > 0:
                areas[(row["format"], row.get("run", "0"))] = area
                if row.get("dtarget_ps"):
                    points[row["format"]].dtarget_ps = _float(row["dtarget_ps"])
    for row in _read(rdir / "timing.csv"):
        if matches(row):
            delay = _float(row.get("delay_ps"))
            if delay is not None:
                delays[(row["format"], row.get("run", "0"))] = delay
    for (fmt, _run), area in areas.items():
        points[fmt].areas.append(area)
    for (fmt, _run), delay in delays.items():
        points[fmt].delays.append(delay)
    return [points[fmt] for fmt in HARDWARE_FORMATS]


def load_breakdown(rdir, lib="sky130hd", target="unc", align_w=24):
    """{format: {stage: area_um2}}, last row per (format, stage) wins."""
    out = {}
    for row in _read(rdir / "breakdown.csv"):
        if row.get("lib") != lib or row.get("target") != target:
            continue
        if row["format"] not in ("int4", "int8") and int(row.get("align_w") or 0) != align_w:
            continue
        out.setdefault(row["format"], {})[row["stage"]] = float(row["area_um2"])
    return out
