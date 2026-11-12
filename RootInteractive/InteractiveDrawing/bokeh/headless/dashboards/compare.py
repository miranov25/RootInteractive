"""The fail-closed comparator of the oracle test dashboards (PHASE_4_7, Claude20.RI).

`compare(expected, actual)` returns the list of differences; an empty list is the only PASS. The browser test and the
falsifiers (deliberately wrong actual responses, on a snapshot and live in the browser) use this same function. Every
field the expected response names must be present in the actual one: a missing field is a difference, never a pass.

What a drawn plot must satisfy (revision of 2026-11-12, after GPT24's consolidated Cycle A review, items G1 and G2):
- every expected plot, visible or hidden, is present exactly once, and no other plot of the checked sources is there;
- a visible plot: data from the right histogram, shown (its canvas visible and the renderer itself visible), the drawn
  x field and x values (bin centres), the drawn y values, the axis labels and the title;
- screen coordinates, anchored (revision of 2026-11-12, GPT25's summary item H-03): the coordinates Bokeh computed for
  each point (the glyph view's sx, sy) must equal the position the plot's own frame and axis ranges give for the
  EXPECTED x and y values, within 0.01 px, on linear axes; and every point must lie inside the frame. A plot whose
  values changed but whose drawing did not follow, a plot shifted or scaled, or an axis range that cuts points off,
  fails here. What this is NOT: a check of the pixels on the canvas (no raster claim);
- a hidden plot: present and not shown (its tab is not active). Its values are checked when its tab is opened.
"""
from __future__ import annotations

from typing import List

import numpy as np

CDF_RTOL = 1e-12             # cdf is cumulative / total in float64 on both sides
EDGE_RTOL = 1e-12             # bin edges and bin centres
WIDGET_ATOL_FRACTION = 1e-12  # of the slider's full range
ANCHOR_ATOL_PX = 0.01         # screen coordinates (float32; 3e-5 px seen) against the frame-and-range position

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


def _anchored(r: dict, axis: str, values) -> list:
    """Differences between the screen coordinates Bokeh computed and the position the plot's frame and range give for
    the expected values: x runs from the frame's left to its right edge, y from its bottom to its top edge."""
    frame, rng, scale = r.get("frame"), r.get(f"{axis}_range"), r.get(f"{axis}_scale")
    if not isinstance(frame, dict) or not rng or len(rng) != 2:
        return ["the plot's frame or axis range is missing from the actual response"]
    if scale != "LinearScale":
        return [f"{axis} axis scale {scale!r}; only LinearScale is supported by this check"]
    start, end = float(rng[0]), float(rng[1])
    if not np.isfinite([start, end]).all() or end == start:
        return [f"{axis} range {rng} is not usable"]
    v = np.asarray(values, dtype=float)
    s = np.asarray(r.get("s" + axis) or [], dtype=float)
    if s.shape != v.shape or v.size == 0:
        return [f"{s.size} screen coordinates for {v.size} values"]
    if axis == "x":
        lo, hi = frame["left"], frame["right"]
        want = lo + (v - start) / (end - start) * (hi - lo)
    else:
        lo, hi = frame["top"], frame["bottom"]
        want = hi - (v - start) / (end - start) * (hi - lo)
    out = []
    bad = ~(np.abs(s - want) <= ANCHOR_ATOL_PX)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        out.append(f"point {i} at {s[i]:.2f} px, the frame and range put it at {want[i]:.2f} px "
                   f"({int(bad.sum())} points differ)")
    outside = (want < lo - ANCHOR_ATOL_PX) | (want > hi + ANCHOR_ATOL_PX)
    if outside.any():
        out.append(f"{int(outside.sum())} points lie outside the plot frame (axis range {rng})")
    return out


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
        for axis, values in (("x", spec["x"]), ("y", column[yfield])):
            for why in _anchored(r, axis, values):
                out.append(f"{name} (screen {axis}): {why}")
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
