"""Oracle for rulebook RB-1 (filters) — the subset used in stage 4.2a.

RB-1.1  A range filter keeps a row when  low <= value <= high  (inclusive at both ends).
        Source: RangeFilter.ts 81 and 115, `(col[i] >= low) && (col[i] <= high)`. Status: verified in code.
RB-1.2  A missing value (NaN) is never kept by a range filter, also at the widget's full bounds,
        because NaN fails both comparisons. Status: to verify -> verified by measurement through the scenario
        `selection/rb1_nan_excluded_at_full_bounds` and its report (erratum E-3).
RB-1.3  The selection is the intersection (logical AND) of all active filters.
        Status: intended rule; known defect B1 when two or more filters exclude rows at load — its exact selection is
        `b1_selection` below (used only to recognise the defect, never as the intended rule).
"""
import numpy as np


def range_mask(values, low: float, high: float) -> np.ndarray:
    """RB-1.1, RB-1.2: rows kept by one range filter."""
    v = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (v >= low) & (v <= high)


def intersection(masks, n: int) -> np.ndarray:
    """RB-1.3: rows kept by all filters (all rows if there is none)."""
    out = np.ones(n, dtype=bool)
    for m in masks:
        out &= np.asarray(m, dtype=bool)
    return out


# ------------------------------------------------------------------------------------------- defect selections
# Selections that RootInteractive produces *because of* a known defect. They are used only by expected-failure
# signatures (engine/expect.py, EqualsSelection): a page with the defect must match the defect's selection exactly,
# so that any other deviation stays a normal failure. Each rule states its mechanism and the measurement behind it.

def b1_first_changed(masks_at_load) -> int:
    """Index of the filter whose load report starts B1's chain: the first filter whose packed load vector is not all
    ones. RangeFilter.as_bits packs 32 rows per word and leaves the unused bits of the last word 0, so when the row count
    is not a multiple of 32 even the first filter "changes" at load; otherwise the first filter that excludes a row
    (Claude3, 4.2b v0.2 review, measured; refined rule since stage 4.2c-2a). -1 if no filter changes: then all apply."""
    masks_at_load = [np.asarray(m, dtype=bool) for m in masks_at_load]
    n = len(masks_at_load[0]) if masks_at_load else 0
    if n % 32:
        return 0
    for k, m in enumerate(masks_at_load):
        if not m.all():
            return k
    return -1


def b1_selection(masks_now, masks_at_load) -> np.ndarray:
    """Known defect B1 (LazyIntersectionFilter.ts 191/193/208/210, `_changed_values ||= …` short-circuits the merge of
    later filters at load). Mechanism, measured: at load, the first filter whose packed load vector is not all ones
    (b1_first_changed) is merged and every LATER filter is skipped; filters before it changed nothing at load and are
    applied normally. Each skipped filter then keeps the rows it excluded at load and otherwise follows its widget.
    With RootInteractive's default limits the rows excluded at load are the filter's missing values; with explicit
    widget limits they are also the rows outside those limits (Claude10, 4.2b v0.2 review: the rule held at all five
    steps). Without a changing filter at load (row count a multiple of 32 and no filter excluding a row) B1 is absent.

    Measured 2026-11-07 on RootInteractive 4e8d783: the page equals the histogram of this selection in every cell, and
    the page's selected-row count equals its size, at every step — two filters 18 039 / 12 361 / 8 798 rows, three
    filters 17 989 / 12 256 / 8 751 / 6 279; first filter without missing values: 20 000 rows → no B1, 20 001 → B1."""
    masks_now = [np.asarray(m, dtype=bool) for m in masks_now]
    masks_at_load = [np.asarray(m, dtype=bool) for m in masks_at_load]
    if not masks_now:
        raise ValueError("B1 needs at least one filter")
    k_star = b1_first_changed(masks_at_load)
    out = np.ones_like(masks_now[0])
    for k, (now, load) in enumerate(zip(masks_now, masks_at_load)):
        out &= now if (k_star < 0 or k <= k_star) else (now | ~load)
    return out


SELECTION_RULES = {
    "b1_later_filters_keep_rows_excluded_at_load": b1_selection,
}
