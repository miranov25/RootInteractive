"""RB-2: independent explicit-binary64-edge histogram oracle.

Canonical edges are np.linspace(lo, hi, nbins + 1).  Each bin is [left,right)
except the final bin [left,right].  No epsilon or adjacent-bin tolerance.

The optional ``b10=True`` mode is a *historical B10 defect simulator* used
ONLY in negative-control tests.  It changes a bin only when the old arithmetic
incorrectly computes an in-range coordinate >= nbins near the upper edge.
It must never be used as the healthy scientific reference.
"""
import numpy as np


def canonical_edges(nbins: int, lo: float, hi: float) -> np.ndarray:
    """Versioned binary64 reference; reject degenerate/nonrepresentable ranges."""
    if not isinstance(nbins, (int, np.integer)) or not (0 < nbins < 2**31):
        raise ValueError("histogram nbins must be an integer in [1, 2**31)")
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi and np.isfinite(hi-lo)):
        raise ValueError("histogram range must be finite, increasing, with finite width")
    edges = np.linspace(float(lo), float(hi), int(nbins)+1, dtype=np.float64)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("histogram binary64 edges must be strictly increasing")
    return edges


def _inside(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])


def _b10_legacy_raw(values: np.ndarray, nbins: int, lo: float, hi: float) -> np.ndarray:
    """Historical B10 arithmetic ONLY, not a reference bin assignment."""
    scale = nbins / (hi - lo)
    return values * scale - lo * scale


def bin_index_1d(values, nbins: int, lo: float, hi: float, b10: bool = False) -> np.ndarray:
    """Exact represented-edge index, -1 for excluded values."""
    v = np.asarray(values, dtype=np.float64)
    edges = canonical_edges(nbins, lo, hi)
    inside = _inside(v, edges)
    out = np.full(v.shape, -1, dtype=np.int64)
    out[inside] = np.minimum(np.searchsorted(edges, v[inside], side="right") - 1, nbins-1)
    if b10:
        # Reintroduce only B10, NOT all the old interior-edge rounding defects.
        affected = inside & (v < hi) & (_b10_legacy_raw(v, nbins, lo, hi) >= nbins)
        out[affected] = -1
    return out


def bin_index_nd(columns, nbins, ranges, b10: bool = False) -> np.ndarray:
    """Exact represented-edge flat index, first variable fastest."""
    if not columns or len(columns) != len(nbins) or len(columns) != len(ranges):
        raise ValueError("one bin geometry required for each nonempty data column")
    n = len(columns[0])
    flat = np.zeros(n, dtype=np.int64)
    inside_all = np.ones(n, dtype=bool)
    stride = 1
    for col, nb, (lo, hi) in zip(columns, nbins, ranges):
        v = np.asarray(col, dtype=np.float64)
        if len(v) != n:
            raise ValueError("histogram columns must have equal lengths")
        edges = canonical_edges(nb, lo, hi)
        inside = _inside(v, edges)
        axis_bin = np.zeros(n, dtype=np.int64)
        axis_bin[inside] = np.minimum(np.searchsorted(edges, v[inside], side="right") - 1, nb-1)
        if b10:
            affected = inside & (v < hi) & (_b10_legacy_raw(v, nb, lo, hi) >= nb)
            axis_bin[affected] = nb  # Old B10: spill into the next axis, or off the last axis.
        flat += axis_bin * stride
        inside_all &= inside
        stride *= nb
    flat[~inside_all | (flat >= stride)] = -1
    return flat


def counts(columns, nbins, ranges, mask=None, weights=None, b10: bool = False) -> np.ndarray:
    """Independent expected cell counts or weight sums (float64)."""
    if len(columns) == 1:
        idx = bin_index_1d(columns[0], nbins[0], *ranges[0], b10=b10)
        length = nbins[0]
    else:
        idx = bin_index_nd(columns, nbins, ranges, b10=b10)
        length = int(np.prod(nbins))
    keep = idx >= 0
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    weight_vec = None if weights is None else np.asarray(weights, dtype=np.float64)[keep]
    return np.bincount(idx[keep], weights=weight_vec, minlength=length).astype(np.float64)


def rule_bounds(columns, nbins, ranges, weights=None, ulps: int = 0):
    """Strict independent rule by edge comparisons, NOT searchsorted or RI arithmetic.

    Kept as (lower,upper,total) for existing family-test integration.  Under the
    new exact-edge contract every cell has ONE permissible value: lower==upper.
    A nonzero ulps cannot weaken this contract.
    """
    if ulps != 0:
        raise ValueError("RB-2 uses exact represented edges; ulps tolerance is forbidden")
    if not columns:
        raise ValueError("no histogram columns")
    n = len(columns[0])
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    inside = np.ones(n, dtype=bool)
    flat = np.zeros(n, dtype=np.int64)
    stride = 1
    for col, nb, (lo, hi) in zip(columns, nbins, ranges):
        v = np.asarray(col, dtype=np.float64)
        edges = canonical_edges(nb, lo, hi)
        inside &= _inside(v, edges)
        # Second independent classification: count edge-comparison predicates,
        # rather than using the searchsorted method in the primary oracle.
        valid = np.isfinite(v)
        axis = np.zeros(n, dtype=np.int64)
        axis[valid] = np.count_nonzero(v[valid, None] >= edges[None, 1:-1], axis=1)
        flat += axis * stride
        stride *= nb
    expect = np.bincount(flat[inside], weights=w[inside], minlength=stride).astype(np.float64)
    return expect.copy(), expect.copy(), float(w[inside].sum())


def rule_violations(page, lower, upper, total, rel: float = 1e-12, abs_: float = 1e-9) -> list:
    """Every bin must equal its strict edge-rule expectation (weighted float tolerance only)."""
    page = np.asarray(page, dtype=np.float64)
    tol = abs_ + rel * np.maximum(np.abs(upper), 1.0)
    bad = [f"cell {i}: {page[i]:g} not equal to [{lower[i]:g}, {upper[i]:g}]"
           for i in np.nonzero((page < lower - tol) | (page > upper + tol))[0]]
    if abs(page.sum() - total) > abs_ + rel * max(abs(total), 1.0):
        bad.append(f"total {page.sum():g} != {total:g} values in range")
    return bad
