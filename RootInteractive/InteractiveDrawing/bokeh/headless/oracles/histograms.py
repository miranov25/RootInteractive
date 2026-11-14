"""Oracle for rulebook RB-2 (histograms) — the subset used in stage 4.2a.

RootInteractive assigns bins with its own arithmetic, which differs from numpy.histogram at bin edges
(e.g. 0.3, 0.6 and 0.7 on [0, 1] with 10 bins land one bin higher than in numpy). The oracle therefore
reproduces RootInteractive's formula, never numpy.histogram.

RB-2.1  1-D (HistogramCDS.ts): scale = nbins/(max-min), origin = -min*scale (143-144);
        bin = (v*scale + origin) | 0 (227). Verified in code.
RB-2.2  N-D (HistoNdCDS.ts): scale = nbins/(max-min), origin = min*scale (167-168);
        per axis bin = (v*scale - origin) | 0, combined with strides, first variable fastest (330-345).
        Verified in code. In IEEE arithmetic a + (-b) equals a - b, so 1-D and N-D give the same bin.
RB-2.3  The last edge is inclusive: v == max goes into the last bin (1-D 225, N-D 341). Verified in code.
RB-2.4  Values outside [min, max] are not counted (1-D: getbin returns -1 / nbins, counting keeps
        0 <= bin < nbins, 174/181/189; N-D: the row gets -1, 337). +-Inf therefore not counted. Verified in code.
RB-2.5  Non-finite values are not counted. N-D: verified in code (337, `!isFinite(val)`).
        1-D: intended rule; known defect B8 (NaN is counted in bin 0), fixed in PHASE_5 Batch 2A.
RB-2.6  (PHASE_5 Batch 2A, B10) A value inside [min, max] whose bin computes as nbins (one rounding step below the
        maximum) is in the last bin. Before the fix it was dropped (1-D) or moved to bin 0 of the next row of bins
        (N-D; dropped in the last row): `b10=True` reproduces that, for the runs with B10 put back by a fault.
"""
import itertools

import numpy as np


def _trunc_int32(x: np.ndarray) -> np.ndarray:
    """JavaScript `x | 0` for the values that reach it: truncation toward zero. Supported domain: finite values with
    |x| < 2**31, which holds for bin positions inside the range (0 <= x <= nbins). JavaScript's ToInt32 wrap-around
    for larger values is not reproduced; histograms with more than 2**31 bins per axis are outside the domain."""
    return np.trunc(x).astype(np.int64)


def bin_index_1d(values, nbins: int, lo: float, hi: float, b10: bool = False) -> np.ndarray:
    """RB-2.1, 2.3 … 2.6: bin per row of a 1-D histogram, -1 where the row is not counted; `b10`: B10 put back."""
    v = np.asarray(values, dtype=np.float64)
    scale = nbins / (hi - lo)
    origin = -lo * scale
    out = np.full(v.shape, -1, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(v) & (v >= lo) & (v <= hi)
    last = inside & (v == hi)
    mid = inside & ~last
    out[last] = nbins - 1
    out[mid] = _trunc_int32(v[mid] * scale + origin)
    if not b10:
        out[mid] = np.minimum(out[mid], nbins - 1)        # RB-2.6
    out[(out < 0) | (out >= nbins)] = -1                  # counting keeps 0 <= bin < nbins
    return out


def bin_index_nd(columns, nbins, ranges, b10: bool = False) -> np.ndarray:
    """RB-2.2 … 2.6: flat bin per row of an N-D histogram (first variable fastest), -1 where not counted; `b10`: B10
    put back."""
    n = len(columns[0])
    flat = np.zeros(n, dtype=np.int64)
    dropped = np.zeros(n, dtype=bool)
    stride = 1
    for col, nb, (lo, hi) in zip(columns, nbins, ranges):
        v = np.asarray(col, dtype=np.float64)
        scale = nb / (hi - lo)
        origin = lo * scale
        with np.errstate(invalid="ignore"):
            bad = (v < lo) | (v > hi) | ~np.isfinite(v)
        dropped |= bad
        last = ~bad & (v == hi)
        mid = ~bad & ~last
        b = np.zeros(n, dtype=np.int64)
        b[last] = nb - 1
        b[mid] = _trunc_int32(v[mid] * scale - origin)
        if not b10:
            b[mid] = np.minimum(b[mid], nb - 1)          # RB-2.6
        flat += b * stride
        stride *= nb
    flat[dropped] = -1
    flat[(flat < 0) | (flat >= stride)] = -1             # counting keeps 0 <= bin < length
    return flat


def counts(columns, nbins, ranges, mask=None, weights=None, b10: bool = False) -> np.ndarray:
    """Expected `bin_count` of a histogram: 1-D if one variable, N-D otherwise; rows outside `mask`
    are not counted; `weights` sums the weight instead of 1; `b10`: B10 put back (RB-2.6)."""
    if len(columns) == 1:
        idx = bin_index_1d(columns[0], nbins[0], *ranges[0], b10=b10)
        length = nbins[0]
    else:
        idx = bin_index_nd(columns, nbins, ranges, b10=b10)
        length = int(np.prod(nbins))
    keep = idx >= 0
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)[keep]
    return np.bincount(idx[keep], weights=w, minlength=length).astype(np.float64)


# ----------------------------------------------------------------------------- the rule as a statement (4.2c-2a v0.2)
def rule_bounds(columns, nbins, ranges, weights=None, ulps: int = 4):
    """RB-2 as a rule, independent of RootInteractive's arithmetic (4.2c-2a review, Claude10's per-cell rule): a
    finite row inside [lo, hi] on every axis is counted once, in a cell whose bin on each axis contains it — the top
    edge belongs to the last bin; a value within `ulps` rounding steps (of max(|lo|, |hi|)) of an INTERIOR edge may go to either
    neighbouring bin (which one is arithmetic, not rule). Returns (lower, upper, total): per cell the weight that
    must be there and the weight that may be there, and the total weight that must be counted."""
    n = len(columns[0])
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    inside = np.ones(n, dtype=bool)
    kmin, kmax = [], []
    for col, nb, (lo, hi) in zip(columns, nbins, ranges):
        v = np.asarray(col, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            inside &= np.isfinite(v) & (v >= lo) & (v <= hi)
        edges = lo + (hi - lo) * np.arange(nb + 1) / nb
        edges[-1] = hi
        interior = edges[1:-1]
        # the tolerance follows the arithmetic's resolution on this axis (v * scale - lo * scale rounds on the scale of
        # max(|lo|, |hi|)), not the edge's own value: an interior edge at 0 would otherwise get none (4.2c-2a, Claude10)
        tol = np.full(interior.shape, ulps * np.spacing(max(abs(lo), abs(hi))))
        vv = np.where(np.isfinite(v), v, lo)[:, None]
        kmin.append((vv > interior + tol).sum(axis=1))
        kmax.append((vv >= interior - tol).sum(axis=1))
    size = int(np.prod(nbins))
    lower, upper = np.zeros(size), np.zeros(size)
    strides = np.cumprod([1] + list(nbins[:-1]))
    certain = inside.copy()
    for a, b in zip(kmin, kmax):
        certain &= a == b
    flat_min = sum(k * st for k, st in zip(kmin, strides))
    np.add.at(lower, flat_min[certain], w[certain])
    # a row may be in every cell kmin + c with c in {0, 1} per axis and c <= kmax - kmin: distinct cells by construction
    for choice in itertools.product((0, 1), repeat=len(nbins)):
        ok = inside.copy()
        for a, b, c in zip(kmin, kmax, choice):
            ok &= (a + c) <= b
        flat = sum((a + c) * st for a, c, st in zip(kmin, choice, strides))
        np.add.at(upper, flat[ok], w[ok])
    return lower, upper, float(w[inside].sum())


def rule_violations(page, lower, upper, total, rel: float = 1e-12, abs_: float = 1e-9) -> list:
    """Cells where the page is outside [lower, upper], and the total if it differs (tolerances for weights)."""
    page = np.asarray(page, dtype=np.float64)
    tol = abs_ + rel * np.maximum(np.abs(upper), 1.0)
    bad = [f"cell {i}: {page[i]:g} not in [{lower[i]:g}, {upper[i]:g}]"
           for i in np.nonzero((page < lower - tol) | (page > upper + tol))[0]]
    if abs(page.sum() - total) > abs_ + rel * max(abs(total), 1.0):
        bad.append(f"total {page.sum():g} != {total:g} values in range")
    return bad

