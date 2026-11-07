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

def b1_selection(masks_now, masks_at_load) -> np.ndarray:
    """Known defect B1 (LazyIntersectionFilter.ts 191/193/208/210, `_changed_values ||= …` short-circuits the merge of
    later filters at load). Mechanism, measured: the first filter is applied correctly; every later filter keeps the
    rows it excluded at load and otherwise follows its widget — rows it excluded at load stay selected while it still
    excludes them. With RootInteractive's default limits the rows excluded at load are the filter's missing values.

    Measured 2026-11-07 on RootInteractive 4e8d783 (coder, after the 4.2b review by Claude3): the page equals the
    histogram of this selection in every cell, and the page's selected-row count equals its size, at every step —
    two filters 18 039 / 12 361 / 8 798 rows, three filters 17 989 / 12 256 / 8 751 / 6 279."""
    masks_now = [np.asarray(m, dtype=bool) for m in masks_now]
    masks_at_load = [np.asarray(m, dtype=bool) for m in masks_at_load]
    if not masks_now:
        raise ValueError("B1 needs at least one filter")
    out = masks_now[0].copy()
    for now, load in zip(masks_now[1:], masks_at_load[1:]):
        out &= now | ~load
    return out


SELECTION_RULES = {
    "b1_later_filters_keep_rows_excluded_at_load": b1_selection,
}
