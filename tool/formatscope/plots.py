"""Frontier, breakdown, and window-sweep figures (build-plan.md S8.4).

matplotlib only. Every figure is written as PNG and SVG at 300 dpi.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from formatscope import TARGETS  # noqa: E402
from formatscope.data import load_breakdown, load_points  # noqa: E402
from formatscope.pareto import frontier, measured  # noqa: E402

COLORS = {
    "int4": "#1b9e77",
    "int8": "#d95f02",
    "fp8e4m3": "#7570b3",
    "mxint8": "#e7298a",
    "mxfp4": "#66a61e",
    "int4_b32": "#e6ab02",
}
LABELS = {"int4": "INT4", "int8": "INT8", "fp8e4m3": "FP8-E4M3", "mxint8": "MXINT8", "mxfp4": "MXFP4",
          "int4_b32": "INT4-b32"}
STAGES = ["decode", "mul", "align", "tree", "normacc", "other"]
TARGET_TITLES = {"unc": "unconstrained", "t1": "T1", "t2": "T2"}

plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _title(points, target):
    title = TARGET_TITLES[target]
    dtarget = next((p.dtarget_ps for p in points if p.dtarget_ps), None)
    return f"{title} ({dtarget:.0f} ps)" if dtarget and target != "unc" else title


def draw_frontier(ax, points, target, logx=False):
    for stage, filled in (("ptq", True), ("qat", False)):
        for p in measured(points, stage):
            color = COLORS[p.format]
            ax.scatter(p.area, p.top1(stage), s=90, zorder=3,
                       color=color if filled else "white", edgecolors=color, linewidths=2)
            if filled:
                ax.annotate(LABELS[p.format], (p.area, p.top1(stage)), xytext=(8, -4),
                            textcoords="offset points", fontsize=12, color=color)
                lo, hi = p.area_spread
                if hi > lo:
                    ax.hlines(p.top1(stage), lo, hi, color=color, linewidth=1, zorder=2)
    ax.scatter([], [], s=90, color="0.3", label="PTQ")
    ax.scatter([], [], s=90, color="white", edgecolors="0.3", linewidths=2, label="QAT (5 epochs)")
    front = frontier(points, "ptq")
    if len(front) > 1:
        ax.step([p.area for p in front], [p.top1("ptq") for p in front],
                where="post", color="0.4", linewidth=1.2, zorder=1)
    if logx:
        ax.set_xscale("log")
    ax.set_title(_title(points, target))
    ax.set_xlabel("DP32 area (µm²)")
    ax.set_ylabel("top-1 (%)")


def frontier_figures(rdir, out_dir, lib="sky130hd", align_w=24, logx=False, targets=TARGETS):
    written = []
    by_target = {t: load_points(rdir, lib, t, align_w) for t in targets}
    for target, points in by_target.items():
        if not measured(points):
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        draw_frontier(ax, points, target, logx)
        ax.legend(frameon=False, fontsize=12, loc="lower right")
        written += _save(fig, out_dir, f"frontier_{lib}_{target}")

    present = [t for t, pts in by_target.items() if measured(pts)]
    if len(present) > 1:
        fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), sharey=True)
        for ax, target in zip(axes, present):
            draw_frontier(ax, by_target[target], target, logx)
        axes[0].legend(frameon=False, fontsize=12, loc="lower right")
        written += _save(fig, out_dir, f"frontier_{lib}_panel")
    return written


def breakdown_figure(rdir, out_dir, lib="sky130hd", target="unc", align_w=24):
    data = load_breakdown(rdir, lib, target, align_w)
    formats = [f for f in COLORS if f in data]
    if not formats:
        return []
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = [0.0] * len(formats)
    cmap = plt.get_cmap("Greys")
    for i, stage in enumerate(STAGES):
        heights = [data[f].get(stage, 0.0) for f in formats]
        if not any(heights):
            continue
        ax.bar([LABELS[f] for f in formats], heights, bottom=bottoms, label=stage,
               color=cmap(0.25 + 0.6 * i / len(STAGES)), edgecolor="white")
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_ylabel("area (µm², hierarchical synthesis)")
    ax.set_title(f"Area by stage, {TARGET_TITLES[target]}")
    ax.legend(frameon=False, fontsize=12, bbox_to_anchor=(1, 1), loc="upper left")
    return _save(fig, out_dir, "breakdown")


def window_sweep_figure(rdir, out_dir, lib="sky130hd", target="unc"):
    formats = ["fp8e4m3", "mxint8", "mxfp4", "int4_b32"]
    by_w = {w: {p.format: p.area for p in load_points(rdir, lib, target, w)} for w in (24, 32)}
    present = [f for f in formats if by_w[24].get(f) and by_w[32].get(f)]
    if not present:
        return []
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(present))
    ax.bar([i - 0.2 for i in x], [by_w[24][f] for f in present], width=0.4, label="ALIGN_W = 24", color="0.55")
    ax.bar([i + 0.2 for i in x], [by_w[32][f] for f in present], width=0.4, label="ALIGN_W = 32", color="0.2")
    ax.set_xticks(list(x), [LABELS[f] for f in present])
    ax.set_ylabel("DP32 area (µm²)")
    ax.set_title(f"Alignment window, {TARGET_TITLES[target]}")
    ax.legend(frameon=False, fontsize=12)
    return _save(fig, out_dir, "window_sweep")


def table_png(rows, header, out_dir):
    fig, ax = plt.subplots(figsize=(1.6 * len(header), 0.5 * (len(rows) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.auto_set_column_width(col=list(range(len(header))))  # else long headers clip
    tbl.scale(1, 1.5)
    return _save(fig, out_dir, "results_table")
