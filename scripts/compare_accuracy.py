"""Compare a reproduction run against the committed accuracy table.

    python scripts/compare_accuracy.py --baseline results/accuracy.csv \
                                       --candidate results/accuracy_desktop-wsl.csv

Exact equality is not the bar and never will be: the committed numbers came
from a different training run on a different backend (MPS), so the FP32
checkpoint differs and every PTQ number shifts with it. What has to hold is
the shape of the table -- each format within --tol of the reference, and the
ordering the whole project rests on (S4.5): INT8 and MXINT8 track FP32, INT4
and MXFP4 fall visibly below, INT4-B32 lands between.
"""

import argparse
import csv
import sys


def read_rows(path):
    """{(format, stage): top1} from an accuracy.csv, last row per key wins."""
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows[(row["format"], row["stage"])] = float(row["top1"])
            except (KeyError, ValueError):
                continue
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="results/accuracy.csv")
    p.add_argument("--candidate", required=True)
    p.add_argument("--tol", type=float, default=2.0,
                   help="max |delta| in top-1 points before a row is flagged")
    args = p.parse_args()

    base = read_rows(args.baseline)
    cand = read_rows(args.candidate)
    if not cand:
        print(f"no rows in {args.candidate}", file=sys.stderr)
        return 1

    failures = []

    print(f"\n{'format':>10s} {'stage':>6s} {'reference':>10s} {'this run':>10s} "
          f"{'delta':>8s}  status")
    print("-" * 60)
    for key in sorted(cand, key=lambda k: (k[1], k[0])):
        fmt, stage = key
        got = cand[key]
        if key not in base:
            print(f"{fmt:>10s} {stage:>6s} {'--':>10s} {got:>10.2f} {'--':>8s}  new")
            continue
        ref = base[key]
        delta = got - ref
        ok = abs(delta) <= args.tol
        if not ok:
            failures.append(f"{fmt}/{stage}: {got:.2f}% vs {ref:.2f}% "
                            f"(delta {delta:+.2f}, tol {args.tol})")
        print(f"{fmt:>10s} {stage:>6s} {ref:>10.2f} {got:>10.2f} {delta:>+8.2f}  "
              f"{'ok' if ok else 'OUT OF TOLERANCE'}")

    # Ordering invariants: these are the actual claims of the project, and they
    # must hold regardless of how far the absolute numbers drifted.
    print()
    ptq = {fmt: top1 for (fmt, stage), top1 in cand.items() if stage == "ptq"}
    fp32 = cand.get(("fp32", "none"))

    def invariant(name, condition, detail):
        if condition is None:
            print(f"  skip  {name} (missing data)")
            return
        print(f"  {'ok  ' if condition else 'FAIL'}  {name}: {detail}")
        if not condition:
            failures.append(f"invariant {name}: {detail}")

    if fp32 is not None and "int8" in ptq:
        invariant("int8 tracks fp32", abs(ptq["int8"] - fp32) <= 1.0,
                  f"int8 {ptq['int8']:.2f} vs fp32 {fp32:.2f}")
    if "int4" in ptq and "int8" in ptq:
        invariant("int4 below int8", ptq["int4"] < ptq["int8"] - 1.0,
                  f"int4 {ptq['int4']:.2f} vs int8 {ptq['int8']:.2f}")
    if "mxfp4" in ptq and "int8" in ptq:
        invariant("mxfp4 below int8", ptq["mxfp4"] < ptq["int8"] - 1.0,
                  f"mxfp4 {ptq['mxfp4']:.2f} vs int8 {ptq['int8']:.2f}")
    if "int4_b32" in ptq and "int4" in ptq and "int8" in ptq:
        invariant("int4_b32 between int4 and int8",
                  ptq["int4"] < ptq["int4_b32"] < ptq["int8"],
                  f"int4 {ptq['int4']:.2f} < int4_b32 {ptq['int4_b32']:.2f} "
                  f"< int8 {ptq['int8']:.2f}")
    for fmt, top1 in sorted(ptq.items()):
        if top1 < 20.0:
            failures.append(f"{fmt} at {top1:.2f}% is near the 10% chance line")

    print()
    if failures:
        print("REPRODUCTION MISMATCH:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("PASS: this run matches the committed table within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
