"""The `formatscope` command (build-plan.md S8.1)."""

import argparse
import subprocess
import sys

from formatscope import ALL_FORMATS, BITS_PER_NUMBER, TARGETS
from formatscope.data import _REPO_ROOT, fp32_top1, load_accuracy, load_points, results_dir
from formatscope.pareto import best_per_area, frontier
from formatscope.recommend import DEFAULT_MARGIN, as_percent, best_under_budget, smallest_meeting


def _fmt(value, spec):
    return format(value, spec) if value is not None else "—"


def _describe(p, stage):
    lo, hi = p.area_spread
    spread = f" (runs {lo:.0f}–{hi:.0f})" if hi and hi > lo else ""
    return (f"{p.format}: top-1 {p.top1(stage):.2f}% ({stage}), "
            f"area {p.area:.0f} µm²{spread}, delay {_fmt(p.delay, '.0f')} ps")


def cmd_run(args):
    if not args.from_results:
        subprocess.run(["make", "quant", "synth", "sta"], cwd=_REPO_ROOT, check=True)
    rdir = results_dir(args.results)
    for target in TARGETS:
        points = load_points(rdir, args.lib, target, args.align_w)
        front = frontier(points, args.stage)
        k = best_per_area(points, args.stage)
        print(f"[{args.lib} {target}] frontier: {', '.join(p.format for p in front) or 'no measured points'}"
              + (f"; best accuracy per µm² {k.format}" if k else ""))
    return 0


def cmd_plot(args):
    from formatscope import plots

    rdir = results_dir(args.results)
    out = rdir / "figures"
    targets = [args.target] if args.target else TARGETS
    written = plots.frontier_figures(rdir, out, args.lib, args.align_w, args.logx, targets)
    if not args.target:
        written += plots.breakdown_figure(rdir, out, args.lib, "unc", args.align_w)
        written += plots.window_sweep_figure(rdir, out, args.lib)
    if not written:
        print("nothing to plot: no format has both an accuracy row and an area row yet", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


def cmd_recommend(args):
    rdir = results_dir(args.results)
    points = load_points(rdir, args.lib, args.target, args.align_w)
    if args.area_budget is not None:
        pick = best_under_budget(points, args.area_budget, args.stage)
        ask = f"most accurate unit with area <= {args.area_budget:.0f} µm²"
    else:
        if args.min_acc is not None:
            min_acc = as_percent(args.min_acc)
        else:
            base = fp32_top1(load_accuracy(rdir))
            if base is None:
                print("no fp32 row in accuracy.csv; pass --min-acc", file=sys.stderr)
                return 2
            min_acc = base - DEFAULT_MARGIN
        ask = f"smallest unit with top-1 >= {min_acc:.2f}%"
        pick = smallest_meeting(points, min_acc, args.stage)
    print(f"[{args.lib} {args.target}] {ask}")
    if pick is None:
        print("  no format qualifies")
        return 1
    print("  " + _describe(pick, args.stage))
    return 0


def _bits(fmt):
    return f"{BITS_PER_NUMBER[fmt]:g}"


def table_rows(rdir, lib, align_w):
    acc = load_accuracy(rdir)
    by_target = {t: {p.format: p for p in load_points(rdir, lib, t, align_w)} for t in TARGETS}
    header = ["format", "bits per number", "PTQ top-1", "QAT top-1"]
    header += [f"area {t} (µm²)" for t in TARGETS] + [f"delay {t} (ps)" for t in TARGETS]
    rows = [["fp32", _bits("fp32"), _fmt(acc.get(("fp32", "none")), ".2f"), "—"] + ["—"] * 6]
    for fmt in ALL_FORMATS:
        row = [fmt, _bits(fmt),
               _fmt(acc.get((fmt, "ptq")), ".2f"), _fmt(acc.get((fmt, "qat")), ".2f")]
        for key in ("area", "delay"):
            for t in TARGETS:
                p = by_target[t].get(fmt)
                row.append(_fmt(getattr(p, key) if p else None, ".0f"))
        rows.append(row)
    return header, rows


def cmd_table(args):
    rdir = results_dir(args.results)
    header, rows = table_rows(rdir, args.lib, args.align_w)
    if args.png:
        from formatscope import plots

        for path in plots.table_png(rows, header, rdir / "figures"):
            print(path)
        return 0
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    return 0


def cmd_demo(args):
    print("$ synthesizing dp32_int8 against sky130_fd_sc_hd, unconstrained", flush=True)
    subprocess.run([sys.executable, "synth/run_synth.py", "--lib", args.lib, "--fmt", "int8",
                    "--target", "unc", "--demo"], cwd=_REPO_ROOT, check=True)
    print(flush=True)
    return cmd_recommend(argparse.Namespace(results=args.results, lib=args.lib, target=args.target,
                                            align_w=args.align_w, stage=args.stage,
                                            min_acc=args.min_acc, area_budget=None))


def build_parser():
    parser = argparse.ArgumentParser(prog="formatscope", description=__doc__)
    parser.add_argument("--results", help="results directory (default: repo results/ or $FORMATSCOPE_RESULTS)")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, target_default="t1"):
        p.add_argument("--lib", default="sky130hd", choices=["sky130hd", "asap7"])
        p.add_argument("--target", default=target_default, choices=TARGETS)
        p.add_argument("--align-w", type=int, default=24, choices=[24, 32])
        p.add_argument("--stage", default="ptq", choices=["ptq", "qat"])

    p = sub.add_parser("run", help="run quant + synth + sta, then summarize the frontier")
    common(p)
    p.add_argument("--from-results", action="store_true", help="skip the runs, just reload the CSVs")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("plot", help="regenerate results/figures")
    common(p, target_default=None)
    p.add_argument("--logx", action="store_true")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("recommend", help="smallest unit above an accuracy, or best under an area budget")
    common(p)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--min-acc", type=float, help="top-1 threshold, 0.87 or 87 (default: FP32 minus 1 point)")
    group.add_argument("--area-budget", type=float, help="area cap in µm²")
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("table", help="results table as markdown, or PNG with --png")
    p.add_argument("--lib", default="sky130hd", choices=["sky130hd", "asap7"])
    p.add_argument("--align-w", type=int, default=24, choices=[24, 32])
    p.add_argument("--png", action="store_true")
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("demo", help="one INT8 synthesis on camera, then a recommendation")
    common(p)
    p.add_argument("--min-acc", type=float)
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
