"""The fail-closed comparator of the oracle test dashboards (PHASE_4_7, Claude20.RI).

`compare(expected, actual)` returns the list of differences; an empty list is the only PASS. The browser test and the
falsifiers (deliberately wrong actual responses, on a snapshot and live in the browser) use this same function. Every
field the expected response names must be present in the actual one: a missing field is a difference, never a pass.

What a drawn plot must satisfy (revision of 2026-11-12, after GPT24's consolidated Cycle A review, items G1 and G2):
- every expected plot, visible or hidden, is present exactly once, and no other plot of the checked sources is there;
- a visible plot: data from the right histogram, shown (its canvas visible and the renderer itself visible), the drawn
  x field and x values (bin centres), the drawn y values, the axis labels and the title;
- painted geometry: the screen coordinates Bokeh computed when it painted are finite, one per point, and an increasing
  straight-line image of the current x and y values (x to the right, y upwards). A plot whose values changed but
  whose painting did not follow fails here;
- a hidden plot: present and not shown (its tab is not active). Its values are checked when its tab is opened.
"""
from __future__ import annotations

from typing import List

import numpy as np

CDF_RTOL = 1e-12             # cdf is cumulative / total in float64 on both sides
EDGE_RTOL = 1e-12             # bin edges and bin centres
WIDGET_ATOL_FRACTION = 1e-12  # of the slider's full range
PAINT_ATOL_PX = 0.01          # screen coordinates are float32; a straight line through them must fit within 0.01 px

RESPONSE_KEYS = ("widgets", "tab", "selected_count", "selected_sha256", "edges", "bin_count", "cumulative", "cdf",
                 "renderers")


def _first_diff(got, want, rtol: float = 0.0):
    g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if g.shape != w.shape:
        return f"shape {g.shape} != {w.shape}"
    bad = ~np.isclose(g, w, rtol=rtol, atol=0.0) if rtol else g != w
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        return f"bin {i}: got {g[i]!r}, expected {w[i]!r} ({int(bad.sum())} bins differ)"
    return None


def _renderers(actual: dict, source: str, yfield: str) -> list:
    return [r for r in actual.get("renderers", []) if r.get("source") == source and r.get("yfield") == yfield]


def _straight_image(values, screen, increasing: bool):
    """None if `screen` is a straight-line image of `values` (increasing or decreasing as asked), else why not."""
    v, s = np.asarray(values, dtype=float), np.asarray(screen, dtype=float)
    if v.shape != s.shape or v.size == 0:
        return f"{s.size} screen coordinates for {v.size} values"
    if not np.isfinite(s).all():
        return "screen coordinates not finite"
    if np.ptp(v) == 0:
        return None if np.ptp(s) <= PAINT_ATOL_PX else "equal values painted at different places"
    slope, offset = np.polyfit(v, s, 1)
    if (slope > 0) != increasing:
        return f"painted in the wrong direction (slope {slope:.3g})"
    worst = float(np.max(np.abs(s - (slope * v + offset))))
    return None if worst <= PAINT_ATOL_PX else f"painted positions do not follow the values (off by up to {worst:.3g} px)"


def compare(expected: dict, actual: dict, widget_span: float = 1.0) -> List[str]:
    out: List[str] = [f"{key}: missing from the actual response" for key in RESPONSE_KEYS if key not in actual]
    if out:
        return out
    tol = WIDGET_ATOL_FRACTION * widget_span
    for name, want in expected["widgets"].items():
        got = actual["widgets"].get(name)
        if got is None or len(got) != len(want) or any(abs(g - w) > tol for g, w in zip(got, want)):
            out.append(f"widget {name}: read back {got}, expected {want}")
    if actual["tab"] != expected["tab"]:
        out.append(f"tab: active {actual['tab']!r}, expected {expected['tab']!r}")
    if actual["selected_count"] != expected["selected_count"]:
        out.append(f"selection: {actual['selected_count']} rows, expected {expected['selected_count']}")
    if actual["selected_sha256"] != expected["selected_sha256"]:
        out.append("selection: the selected rows differ from the expected rows")
    for key, rtol in (("edges", EDGE_RTOL), ("bin_count", 0.0), ("cumulative", 0.0), ("cdf", CDF_RTOL)):
        d = _first_diff(actual[key], expected[key], rtol)
        if d:
            out.append(f"histoA.{key}: {d}")

    column = {"bin_count": expected["bin_count"], "cumulative": expected["cumulative"], "cdf": expected["cdf"]}
    plots = expected.get("plots", {})
    wanted = {tuple(p) for p in expected["visible_renderers"]} | {tuple(p) for p in expected["hidden_renderers"]}
    for r in actual["renderers"]:
        if (r.get("source"), r.get("yfield")) not in wanted:
            out.append(f"renderer {r.get('source')}.{r.get('yfield')}: on the page, but not expected")
    for source, yfield in expected["visible_renderers"]:
        name = f"renderer {source}.{yfield}"
        found = _renderers(actual, source, yfield)
        if len(found) != 1:
            out.append(f"{name}: found {len(found)} glyph renderers, expected exactly one")
            continue
        r = found[0]
        if ["HistogramCDS", "histoA"] not in r.get("chain", []):
            out.append(f"{name}: its data does not come from HistogramCDS histoA (chain {r.get('chain')})")
        if r.get("visible") is not True:
            out.append(f"{name}: not visible (canvas shown: {r.get('canvas_visible')}, renderer visible: "
                       f"{r.get('model_visible')})")
        if r.get("n") != len(column[yfield]):
            out.append(f"{name}: draws {r.get('n')} points, expected {len(column[yfield])}")
        d = _first_diff(r.get("y") or [], column[yfield], CDF_RTOL if yfield == "cdf" else 0.0)
        if d:
            out.append(f"{name} (drawn values): {d}")
        spec = plots.get(f"{source}.{yfield}")
        if spec is None:
            out.append(f"{name}: the expected response has no plot description (x field, x values, labels)")
            continue
        if r.get("xfield") != spec["xfield"]:
            out.append(f"{name}: draws x from {r.get('xfield')!r}, expected {spec['xfield']!r}")
        d = _first_diff(r.get("x") or [], spec["x"], EDGE_RTOL)
        if d:
            out.append(f"{name} (drawn x): {d}")
        for key in ("xlabel", "ylabel", "title"):
            got = r.get(key)
            if not isinstance(got, str) or got.strip() != spec[key]:
                out.append(f"{name}: {key} {got!r}, expected {spec[key]!r}")
        for axis, values, increasing in (("x", spec["x"], True), ("y", column[yfield], False)):
            why = _straight_image(values, r.get("s" + axis) or [], increasing)
            if why:
                out.append(f"{name} (painted {axis}): {why}")
    for source, yfield in expected["hidden_renderers"]:
        found = _renderers(actual, source, yfield)
        if len(found) != 1:
            out.append(f"renderer {source}.{yfield}: found {len(found)} glyph renderers on the hidden tab, "
                       f"expected exactly one")
        elif found[0].get("visible") is not False:
            out.append(f"renderer {source}.{yfield}: visible, but its tab is not active")
    return out


def differs(baseline: dict, changed: dict) -> List[str]:
    """Non-vacuity: what makes a changed state really different from the baseline (empty = the change did nothing)."""
    why = []
    if baseline["selected_count"] != changed["selected_count"]:
        why.append(f"selection {baseline['selected_count']} -> {changed['selected_count']} rows")
    n = int(np.sum(np.asarray(baseline["bin_count"]) != np.asarray(changed["bin_count"])))
    if n:
        why.append(f"{n} of {len(baseline['bin_count'])} bins changed")
    return why
