"""Accuracy-vs-area Pareto front and knee (build-plan.md S8.3)."""


def measured(points, stage="ptq"):
    return [p for p in points if p.area is not None and p.top1(stage) is not None]


def frontier(points, stage="ptq"):
    """Points no other point beats on both area (<=) and accuracy (>=).

    Sorted by area ascending, a point stays iff it is strictly more accurate
    than everything smaller. Ties in area keep only the more accurate point.
    """
    ordered = sorted(measured(points, stage), key=lambda p: (p.area, -p.top1(stage)))
    front, best = [], float("-inf")
    for p in ordered:
        if p.top1(stage) > best:
            front.append(p)
            best = p.top1(stage)
    return front


def knee(points, stage="ptq"):
    """Frontier point with the most top-1 accuracy per um^2."""
    front = frontier(points, stage)
    if not front:
        return None
    return max(front, key=lambda p: p.top1(stage) / p.area)
