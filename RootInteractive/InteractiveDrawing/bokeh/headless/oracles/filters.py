"""Oracle for rulebook RB-1 (filters) — the subset used in stage 4.2a.

RB-1.1  A range filter keeps a row when  low <= value <= high  (inclusive at both ends).
        Source: RangeFilter.ts 81 and 115, `(col[i] >= low) && (col[i] <= high)`. Status: verified in code.
RB-1.2  A missing value (NaN) is never kept by a range filter, also at the widget's full bounds,
        because NaN fails both comparisons. Status: to verify -> verified by measurement through the scenario
        `selection/rb1_nan_excluded_at_full_bounds` and its report (erratum E-3).
RB-1.3  The selection is the intersection (logical AND) of all active filters.
        Status: intended rule; known defect B1 when two or more filters exclude rows at load.
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
