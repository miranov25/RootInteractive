"""Expected responses of the original WeightsRef dashboard, computed with numpy from the raw table only (PHASE_4_7,
Claude20.RI, 2026-11-12). Nothing here imports RootInteractive.

The dashboard (`test_interactiveTemplateWeightsRef`, template `getDefaultVarsRefWeights`) compares two weightings of
the same rows. The tab "histoXYWeight" shows, for X = varX (A) and Y = varY (B), 100 x 120 bins:
- the ANALYSIS side, weighted by the parameter `weights` (default A): the 2-D histogram `bin_count` and its profile
  `histoXYData_1` (entries, mean, std, the 35 % and 50 % quantiles of Y in each X bin), computed from the rows' values
  (unbinned);
- the REFERENCE side, weighted by `weightsRef` (default B/(1+C)): the 2-D histogram `histoXYRef` and its projection
  `projXYRef_1`, declared without `unbinned`, so today computed from the bin centres (binned);
- the COMPARISON: `diffFuncWeights(analysis, reference)`, the difference (default), ratio or log ratio of the two, for the
  counts of every 2-D bin and for the mean, median and std of every X bin.

The oracle answers two questions for every state of the page:
- DESIRED: what the page should show by the Architect's decisions (decisions document v0.3):
  - DEC-11 the oracle rebuilds the page's relative16 rounding;
  - DEC-21 a range slider at its end does not cut;
  - DEC-22 / DEC-10 weighted statistics from the rows' values;
  - R2 "proper statistical formula": the reference side is computed from the values too (unbinned), like the analysis
    side, so that equal weights give equal curves; Q-T-02 is read this way, see the CRR;
  - DEC-24 the error bar of a mean is s / sqrt(N_eff), N_eff = (sum w)^2 / sum w^2. The proper error bars of the
    median and of the std wait for the Architect's confirmation ("R2-errors"): they are reported, not judged.
- AS SHOWN: the same with the known defects switched on, each predicted exactly from the data:
  - WT-D6  the A range filter compares the rounded values with the slider's RAW end values, so at the slider's end
           the rows whose rounded A lies outside are lost (row 44894 on the default table);
  - WR-D1  the reference side is binned: mean, std and quantiles from the bin centres (linear inside the bin);
  - WT-D14 every error bar is drawn as std / sqrt(entries), entries = sum of the weights.
A state is PASS when the page equals DESIRED; KNOWN_ISSUE_RED when it differs from DESIRED but equals AS SHOWN exactly,
and the defects that make the difference are named; FAIL otherwise.

Quantiles: a weighted quantile has no single convention for real-valued weights (DEC-10, weighted case: the rule
inside a bin is left to PHASE_5). Inside the Y bin where the cumulative weight passes q * entries, the value must lie
between the inverse of the weighted CDF and the next value (both included); zero-weight rows never count.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

WR_FUNCTION = "test_interactiveTemplateWeightsRef"
QUANTILES = (0.35, 0.5)
DEFECTS = ("WT-D6", "WR-D1", "WT-D14")
DERIVED = {                                  # computed in Python from the raw table, then transferred (rounded)
    "A*A": lambda t: t["A"] * t["A"],
    "A*A+B": lambda t: t["A"] * t["A"] + t["B"],
    "B/(1+C)": lambda t: t["B"] / (1 + t["C"]),
}
TRANSFORMS = ("diff", "ratio", "logRatio")


def round16(x) -> np.ndarray:
    """relative16 as the page receives it: the mantissa rounded to 16 bits, stored as float32 (compressArray.py)."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    m = np.rint(m * 2.0 ** 16) / 2.0 ** 16
    return np.ldexp(m.astype(np.float32), e).astype(np.float32).astype(np.float64)


def page_columns(t: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    cols = {c: round16(t[c]) for c in "ABCD"}
    cols.update({name: round16(f(t)) for name, f in DERIVED.items()})
    return cols


def selection(cols: Dict[str, np.ndarray], widgets: Dict[str, dict], defects: Sequence[str] = ()) -> np.ndarray:
    """The selected rows. DESIRED: a handle at the slider's start or end does not cut (DEC-21); otherwise
    start <= value <= end on the rounded value (DEC-11). WT-D6: the A range is applied with its raw end values."""
    sel = np.ones(len(cols["A"]), dtype=bool)
    for c in "ABCD":
        w = widgets[c]
        lo, hi = w["value"]
        v = cols[c]
        if c == "A" and "WT-D6" in defects:
            sel &= (v >= lo) & (v <= hi)
            continue
        if lo > w["start"]:
            sel &= v >= lo
        if hi < w["end"]:
            sel &= v <= hi
    return sel


def axis(values: np.ndarray, sel: np.ndarray, nbins: int) -> dict:
    """HistoNdCDS auto range over the selected rows, its bin index (truncation, the maximum included) and edges."""
    v = values[sel & np.isfinite(values)]
    lo, hi = float(v.min()), float(v.max())
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    scale = nbins / (hi - lo)
    origin = lo * scale
    inv = 1 / scale
    with np.errstate(invalid="ignore"):
        idx = np.trunc(values * scale - origin)
    idx = np.where(np.isfinite(idx), idx, -1).astype(np.int64)
    idx[values == hi] = nbins - 1
    idx[~((values >= lo) & (values <= hi))] = -1
    k = np.arange(nbins)
    return {"index": idx, "lo": lo, "hi": hi, "bottom": lo + k * inv, "center": lo + (k + .5) * inv,
            "top": lo + (k + 1) * inv}


def _quantile_bracket(y, w, yb, counts_y, entries, q):
    """(lower, upper) bracket of the weighted quantile q, DEC-10 weighted case (see the module docstring)."""
    if not entries > 0:
        return np.nan, np.nan
    cum = np.cumsum(counts_y)
    above = np.flatnonzero(cum > q * entries)
    if above.size == 0:
        return np.nan, np.nan
    j = int(above[0])
    low = cum[j] - counts_y[j]
    k = q * entries - low
    inb = (yb == j) & (w > 0)
    v, ww = y[inb], w[inb]
    order = np.argsort(v, kind="stable")
    v, c = v[order], np.cumsum(ww[order])
    i_lo = int(np.argmax(c >= k - 1e-12 * max(1.0, abs(k))))
    i_hi = int(np.argmax(c > k + 1e-12 * max(1.0, abs(k)))) if (c > k).any() else len(v) - 1
    return float(v[i_lo]), float(v[max(i_hi, i_lo)])


def profile_unbinned(y, w, bx, by, sel, nx, ny, counts) -> dict:
    """Per X bin, from the selected rows' values: entries (sum of the binned weighted counts, in Y order), weighted mean,
    weighted population std, quantile brackets, and N_eff for the proper error of the mean."""
    out = {k: np.full(nx, np.nan) for k in ("entries", "mean", "std", "n_eff")}
    for q in QUANTILES:
        out[f"q{q}_lo"], out[f"q{q}_hi"] = np.full(nx, np.nan), np.full(nx, np.nan)
    ok = sel & (bx >= 0) & (by >= 0)
    for i in range(nx):
        m = ok & (bx == i)
        yy, ww, yb = y[m], w[m], by[m]
        cy = counts[:, i]
        entries = float(np.cumsum(cy)[-1]) if ny else 0.0
        out["entries"][i] = entries
        sw = ww.sum()
        if entries > 0 and sw > 0:
            mu = float(np.sum(ww * yy) / sw)
            out["mean"][i] = mu
            out["std"][i] = float(np.sqrt(np.sum(ww * (yy - mu) ** 2) / sw))
            out["n_eff"][i] = float(sw ** 2 / np.sum(ww ** 2))
            for q in QUANTILES:
                out[f"q{q}_lo"][i], out[f"q{q}_hi"][i] = _quantile_bracket(yy, ww, yb, cy, entries, q)
    return out


def profile_binned(counts: np.ndarray, ax_y: dict) -> dict:
    """WR-D1 (as shown): the projection from the bin centres; quantiles linear inside the bin, as HistoNdProfile does
    without `unbinned`."""
    ny, nx = counts.shape
    out = {k: np.full(nx, np.nan) for k in ("entries", "mean", "std")}
    for q in QUANTILES:
        out[f"q{q}_lo"] = np.full(nx, np.nan)
    yc, yl, yr = ax_y["center"], ax_y["bottom"], ax_y["top"]
    for i in range(nx):
        c = counts[:, i]
        cum = np.cumsum(c)
        e = float(cum[-1])
        out["entries"][i] = e
        if e > 0:
            mean = 0.0
            for j in range(ny):
                mean += c[j] * yc[j]
            mean /= e
            std = 0.0
            for j in range(ny):
                std += (yc[j] - mean) * (yc[j] - mean) * c[j]
            out["mean"][i], out["std"][i] = mean, float(np.sqrt(std / e))
            for q in QUANTILES:
                above = np.flatnonzero(cum > q * e)
                if above.size:
                    j = int(above[0])
                    low = cum[j] - c[j]
                    mm = (q * e - low) / (cum[j] - low)
                    out[f"q{q}_lo"][i] = yl[j] * (1 - mm) + yr[j] * mm
    for q in QUANTILES:
        out[f"q{q}_hi"] = out[f"q{q}_lo"].copy()
    return out


def transform(a, b, how: str) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if how == "diff":
            return a - b
        if how == "ratio":
            return a / b
        if how == "logRatio":
            return np.log(a) - np.log(b)
    raise ValueError(how)


def expected(t: Dict[str, np.ndarray], state: dict, defects: Sequence[str] = ()) -> dict:
    """The page's response in one state. `state`: widgets (the A-D sliders as in the manifest, with "value"),
    weights, weightsRef, transform, nbinsX, nbinsY. `defects`: the known defects switched on (AS SHOWN)."""
    cols = page_columns(t)
    sel = selection(cols, state["widgets"], defects)
    nx, ny = int(state["nbinsX"]), int(state["nbinsY"])
    ax, ay = axis(cols["A"], sel, nx), axis(cols["B"], sel, ny)
    ok = sel & (ax["index"] >= 0) & (ay["index"] >= 0)
    flat = ax["index"] + ay["index"] * nx
    w_an, w_ref = cols[state["weights"]], cols[state["weightsRef"]]
    h_an = np.bincount(flat[ok], weights=w_an[ok], minlength=nx * ny).reshape(ny, nx)
    h_ref = np.bincount(flat[ok], weights=w_ref[ok], minlength=nx * ny).reshape(ny, nx)
    y, bx, by = cols["B"], ax["index"], ay["index"]
    an = profile_unbinned(y, w_an, bx, by, sel, nx, ny, h_an)
    ref = profile_binned(h_ref, ay) if "WR-D1" in defects else profile_unbinned(y, w_ref, bx, by, sel, nx, ny, h_ref)
    how = state["transform"]
    with np.errstate(divide="ignore", invalid="ignore"):
        err = {}
        for side, p in (("analysis", an), ("reference", ref)):
            drawn = p["std"] / np.sqrt(p["entries"])                       # WT-D14, as the page draws it
            proper = p["std"] / np.sqrt(p["n_eff"]) if "n_eff" in p else drawn
            err[side] = {"mean": drawn if "WT-D14" in defects else proper,
                         "median_reported": drawn, "std_reported": drawn}
    xc2, yc2 = np.tile(ax["center"], ny), np.repeat(ay["center"], nx)
    return {
        "state": {k: state[k] for k in ("weights", "weightsRef", "transform", "nbinsX", "nbinsY")},
        "A": list(state["widgets"]["A"]["value"]),
        "defects": sorted(defects),
        "selected_count": int(sel.sum()),
        "selected_rows": sel,
        "x_range": [ax["lo"], ax["hi"]], "y_range": [ay["lo"], ay["hi"]],
        "x_center": ax["center"], "xy_center_0": xc2, "xy_center_1": yc2,
        "bin_count": h_an.ravel(), "histoXYRef": h_ref.ravel(),
        "analysis": an, "reference": ref,
        # the median comparison is checked against the page's own medians, each checked against its bracket
        "comparison": {"mean": transform(an["mean"], ref["mean"], how),
                       "std": transform(an["std"], ref["std"], how),
                       "counts": transform(h_an.ravel(), h_ref.ravel(), how)},
        "errors": err,
    }
