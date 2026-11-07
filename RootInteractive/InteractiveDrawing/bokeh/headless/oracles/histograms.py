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
        1-D: intended rule; known defect B8 (NaN is counted in bin 0).
"""
import numpy as np


def _trunc_int32(x: np.ndarray) -> np.ndarray:
    """JavaScript `x | 0` for the values that reach it (finite, inside the range): truncation toward zero."""
    return np.trunc(x).astype(np.int64)


def bin_index_1d(values, nbins: int, lo: float, hi: float) -> np.ndarray:
    """RB-2.1, 2.3, 2.4, 2.5: bin per row of a 1-D histogram, -1 where the row is not counted."""
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
    out[(out < 0) | (out >= nbins)] = -1                  # counting keeps 0 <= bin < nbins
    return out


def bin_index_nd(columns, nbins, ranges) -> np.ndarray:
    """RB-2.2 … 2.5: flat bin per row of an N-D histogram (first variable fastest), -1 where not counted."""
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
        flat += b * stride
        stride *= nb
    flat[dropped] = -1
    flat[(flat < 0) | (flat >= stride)] = -1             # counting keeps 0 <= bin < length
    return flat


def counts(columns, nbins, ranges, mask=None, weights=None) -> np.ndarray:
    """Expected `bin_count` of a histogram: 1-D if one variable, N-D otherwise; rows outside `mask`
    are not counted; `weights` sums the weight instead of 1."""
    if len(columns) == 1:
        idx = bin_index_1d(columns[0], nbins[0], *ranges[0])
        length = nbins[0]
    else:
        idx = bin_index_nd(columns, nbins, ranges)
        length = int(np.prod(nbins))
    keep = idx >= 0
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)[keep]
    return np.bincount(idx[keep], weights=w, minlength=length).astype(np.float64)
