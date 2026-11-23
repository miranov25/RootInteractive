"""PHASE_5 Group 7: what the invariance dashboards must show, computed with numpy and pandas from the raw table only
(Claude21.RI, 2026-11-22). Nothing here imports RootInteractive.

The rules are the page's documented rules, written again independently:
- a selection keeps a row when every switched-on condition holds: lo <= value <= hi for a range slider (both ends
  included), the row's category in the chosen list, and each switched-on text query (given here with its pandas
  equivalent);
- a histogram's range on each axis is the minimum and maximum of the selected rows' finite values; no selected row
  gives [0, 1]; equal ends give NumPy's [v - 0.5, v + 0.5] (decision L-46); the edges are numpy.linspace(lo, hi, n + 1),
  every bin [left, right) and the last one [left, right] (the bin-edge rule of 16 November, oracles/histograms.py);
- a profile (a projection with `unbinned`) gives, for each cell of the other axes: the entries (the sum of its bins),
  the mean and the standard deviation (divided by the entries) of the rows' values, and each quantile q found the
  page's way: in the first sub-bin where the cumulative count passes q * entries, numpy's linear quantile of that
  sub-bin's values at q' = (q * entries - before) / (count of the sub-bin).
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

from ..oracles import histograms as H

TRANSFORMS = ("diff", "ratio", "logRatio")


def variable(t, name: str, params: Dict[str, float], functions: Dict[str, Callable]) -> np.ndarray:
    """The values of one entry of the template's variable lists, as the page computes them."""
    a = {c: np.asarray(t[c], dtype=np.float64) for c in ("A", "B", "C", "D")}
    known = {
        "A": lambda: a["A"], "B": lambda: a["B"], "C": lambda: a["C"], "D": lambda: a["D"],
        "cat": lambda: np.asarray(t["cat"], dtype=np.float64),
        "A+B": lambda: a["A"] + a["B"], "B+A": lambda: a["B"] + a["A"],
        "A*B": lambda: a["A"] * a["B"], "B*A": lambda: a["B"] * a["A"],
        "(A+B)-B": lambda: (a["A"] + a["B"]) - a["B"], "A+A": lambda: a["A"] + a["A"],
        "A+exponentX*B": lambda: a["A"] + params["exponentX"] * a["B"],
        "(A+B)-(B+A)": lambda: (a["A"] + a["B"]) - (a["B"] + a["A"]), "A*B-B*A": lambda: a["A"] * a["B"] - a["B"] * a["A"],
        "((A+B)-B)-A": lambda: ((a["A"] + a["B"]) - a["B"]) - a["A"], "(A+A)/A": lambda: (a["A"] + a["A"]) / a["A"],
    }
    if name in functions:
        return np.asarray(functions[name](t), dtype=np.float64)
    if name not in known:
        raise KeyError(f"the oracle does not know the variable {name!r}")
    return known[name]()


def transform(y, n, how: str) -> np.ndarray:
    """diffFunc of the template: y - n, y / n, or log(y) - log(n), row by row."""
    y, n = np.asarray(y, np.float64), np.asarray(n, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if how == "diff":
            return y - n
        if how == "ratio":
            return y / n
        if how == "logRatio":
            return np.log(y) - np.log(n)
    raise ValueError(how)


def selection(t, ranges: Dict[str, Sequence[float]], lists: Dict[str, Sequence], queries: Sequence[Callable]) -> np.ndarray:
    """The selected rows: every range lo <= v <= hi, every list, every switched-on query (pandas equivalents)."""
    n = len(t["A"])
    keep = np.ones(n, dtype=bool)
    for c, (lo, hi) in ranges.items():
        v = np.asarray(t[c], dtype=np.float64)
        keep &= (v >= lo) & (v <= hi)
    for c, chosen in lists.items():
        col = np.asarray(t[c])
        # a list shows its items as texts ("0", "1", ...): compared as numbers for a numeric column
        wanted = [float(x) for x in chosen] if np.issubdtype(col.dtype, np.number) else [str(x) for x in chosen]
        keep &= np.isin(col, wanted)
    for q in queries:
        keep &= np.asarray(q(t), dtype=bool)
    return keep


def axis_range(values, mask) -> tuple:
    v = np.asarray(values, np.float64)[np.asarray(mask, bool)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    return (lo - 0.5, hi + 0.5) if lo == hi else (lo, hi)


def histogram(columns: Sequence, nbins: Sequence[int], mask, weights=None, ranges=None) -> dict:
    """Edges and counts (first axis fastest) of the page's histogram of `columns` on the selected rows; with
    `weights`, the sums of the weights (the range does not depend on the weights). `ranges`: the fixed range of each
    axis, as the page declares it (the ToyMC page); without it, the automatic range of the selected rows."""
    ranges = [tuple(r) for r in ranges] if ranges is not None else [axis_range(c, mask) for c in columns]
    edges = [H.canonical_edges(int(nb), lo, hi) for nb, (lo, hi) in zip(nbins, ranges)]
    count = H.counts([np.asarray(c, np.float64) for c in columns], [int(x) for x in nbins], ranges, mask=mask,
                     weights=weights)
    return {"edges": edges, "count": count, "ranges": ranges}


def _bins(columns, nbins, ranges):
    """Per row: the bin of each axis (-1 outside), with the same exact-edge rule as the counts."""
    out = []
    for c, nb, (lo, hi) in zip(columns, nbins, ranges):
        out.append(H.bin_index_1d(np.asarray(c, np.float64), int(nb), lo, hi))
    return out


def profile(columns: Sequence, nbins: Sequence[int], mask, axis: int, quantiles: Sequence[float], weights=None,
            ranges=None) -> dict:
    """The page's unbinned profile along `axis`, one value per cell of the other axes, in the page's order (the axes
    above `axis` outer, those below inner, first axis fastest). With `weights`: the entries are the sums of the
    weights, the mean and the standard deviation are weighted (rows of weight 0 count for nothing), and the quantiles
    are not computed (their weighted rule is left to compilation time, decision DEC-30): NaN. `ranges`: fixed ranges,
    as in `histogram`."""
    nbins = [int(x) for x in nbins]
    ranges = [tuple(r) for r in ranges] if ranges is not None else [axis_range(c, mask) for c in columns]
    idx = _bins(columns, nbins, ranges)
    inside = np.asarray(mask, bool).copy()
    for b in idx:
        inside &= b >= 0
    y = np.asarray(columns[axis], np.float64)
    low_dims = nbins[:axis]
    high_dims = nbins[axis + 1:]
    n_low = int(np.prod(low_dims)) if low_dims else 1
    n_high = int(np.prod(high_dims)) if high_dims else 1
    # the cell of a row: low part (first axis fastest) + n_low * high part
    low = np.zeros(len(y), np.int64)
    stride = 1
    for d in range(axis):
        low += idx[d] * stride
        stride *= nbins[d]
    high = np.zeros(len(y), np.int64)
    stride = 1
    for d in range(axis + 1, len(nbins)):
        high += idx[d] * stride
        stride *= nbins[d]
    cell = high * n_low + low
    ncell = n_low * n_high
    out = {"entries": np.zeros(ncell), "mean": np.full(ncell, np.nan), "std": np.full(ncell, np.nan)}
    for k in range(len(quantiles)):
        out[f"quantile_{k}"] = np.full(ncell, np.nan)
    rows = np.flatnonzero(inside)
    order = np.lexsort((y[rows], idx[axis][rows], cell[rows]))       # by cell, then sub-bin, then value
    rows = rows[order]
    c_sorted = cell[rows]
    starts = np.searchsorted(c_sorted, np.arange(ncell), side="left")
    ends = np.searchsorted(c_sorted, np.arange(ncell), side="right")
    nb = nbins[axis]
    w_all = None if weights is None else np.asarray(weights, np.float64)
    for k in range(ncell):
        r = rows[starts[k]:ends[k]]
        if w_all is not None:
            w = w_all[r]
            e = float(w.sum())
            out["entries"][k] = e
            keep = w != 0
            if e > 0 and keep.any():
                v, w = y[r][keep], w[keep]
                mu = float(np.sum(w * v) / np.sum(w))
                out["mean"][k] = mu
                out["std"][k] = float(np.sqrt(np.sum(w * (v - mu) ** 2) / e))
            continue
        e = len(r)
        out["entries"][k] = e
        if e == 0:
            continue
        v = y[r]
        mu = float(np.mean(v))
        out["mean"][k] = mu
        out["std"][k] = float(np.sqrt(np.mean((v - mu) ** 2)))
        sub = idx[axis][r]
        per_bin = np.bincount(sub, minlength=nb)
        cum = np.cumsum(per_bin)
        for iq, q in enumerate(quantiles):
            if q == 1:
                out[f"quantile_{iq}"][k] = float(v.max())
                continue
            above = np.flatnonzero(cum > q * e)
            if above.size == 0:
                continue
            j = int(above[0])
            before = cum[j] - per_bin[j]
            qs = (q * e - before) / per_bin[j]
            out[f"quantile_{iq}"][k] = float(np.quantile(np.sort(v[sub == j]), qs))
    return out
