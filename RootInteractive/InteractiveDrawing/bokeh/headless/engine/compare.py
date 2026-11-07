"""Comparisons of page results with oracle values (proposal v0.2 §7.3). Unit-tested themselves.

Rules: counts are compared exactly; floats with |page - oracle| <= abs_tol + rel_tol*|oracle|; NaN equals NaN
only when `nan_equal` is set; +Inf differs from -Inf; -0 equals +0; a shape mismatch always fails. The page/oracle
ratio per cell is reported as a diagnostic (judged against 1), never used as the acceptance test.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Comparison:
    ok: bool
    kind: str
    n: int
    n_diff: int
    message: str
    worst: Optional[Tuple[int, float, float]] = None      # (cell, page, oracle)
    ratio_range: Optional[Tuple[float, float]] = None
    diff_cells: List[int] = field(default_factory=list)


def _cell_equal(p: np.ndarray, o: np.ndarray, abs_tol: float, rel_tol: float, nan_equal: bool) -> np.ndarray:
    pn, on = np.isnan(p), np.isnan(o)
    eq = np.zeros(p.shape, dtype=bool)
    eq |= pn & on & nan_equal
    both = ~pn & ~on
    pinf, oinf = np.isinf(p), np.isinf(o)
    inf_case = both & (pinf | oinf)
    eq |= inf_case & (p == o)
    fin = both & ~pinf & ~oinf
    with np.errstate(invalid="ignore", over="ignore"):
        eq |= fin & (np.abs(p - o) <= abs_tol + rel_tol * np.abs(o))
    return eq


def compare(page, oracle, kind: str = "count", abs_tol: float = 0.0, rel_tol: float = 0.0,
            nan_equal: bool = False) -> Comparison:
    p = np.asarray(page, dtype=np.float64).ravel()
    o = np.asarray(oracle, dtype=np.float64).ravel()
    if p.shape != o.shape:
        return Comparison(False, kind, int(o.size), int(max(p.size, o.size)),
                          f"shape mismatch: page {p.size} cells, oracle {o.size}")
    if kind == "count":
        eq = _cell_equal(p, o, 0.0, 0.0, False)
    elif kind == "float":
        eq = _cell_equal(p, o, abs_tol, rel_tol, nan_equal)
    else:
        raise ValueError(f"unknown comparison kind {kind!r}")
    bad = np.flatnonzero(~eq)
    ratio = None
    nz = (o != 0) & np.isfinite(o) & np.isfinite(p)
    if nz.any():
        r = p[nz] / o[nz]
        ratio = (float(r.min()), float(r.max()))
    if bad.size == 0:
        return Comparison(True, kind, int(o.size), 0, f"{o.size} cells equal", ratio_range=ratio)
    with np.errstate(invalid="ignore"):
        dev = np.where(np.isfinite(p[bad]) & np.isfinite(o[bad]), np.abs(p[bad] - o[bad]), np.inf)
    k = int(bad[int(np.argmax(dev))])
    return Comparison(False, kind, int(o.size), int(bad.size),
                      f"{bad.size} of {o.size} cells differ; worst cell {k}: page {p[k]!r}, oracle {o[k]!r}",
                      worst=(k, float(p[k]), float(o[k])), ratio_range=ratio, diff_cells=bad[:20].tolist())
