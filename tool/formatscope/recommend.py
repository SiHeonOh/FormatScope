"""Pick a format under an accuracy margin or an area budget (build-plan.md S8.3)."""

from formatscope.pareto import measured

DEFAULT_MARGIN = 1.0  # decision D17: points of top-1 below FP32


def as_percent(acc):
    """Accept 0.87 or 87 for the same threshold; the CSVs store percent."""
    return acc * 100.0 if acc <= 1.0 else acc


def smallest_meeting(points, min_acc, stage="ptq"):
    ok = [p for p in measured(points, stage) if p.top1(stage) >= min_acc]
    return min(ok, key=lambda p: (p.area, -p.top1(stage))) if ok else None


def best_under_budget(points, area_budget, stage="ptq"):
    ok = [p for p in measured(points, stage) if p.area <= area_budget]
    return max(ok, key=lambda p: (p.top1(stage), -p.area)) if ok else None
