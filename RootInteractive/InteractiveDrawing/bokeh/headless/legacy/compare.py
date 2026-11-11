"""The fail-closed comparator of the dashboard tests (PHASE_4_7, Claude20.RI).

`compare(expected, actual)` returns the list of differences; an empty list is the only PASS. The browser test and the
falsifiers (deliberately wrong actual responses) use this same function. Every field the expected response names must
be present in the actual one: a missing field is a difference, never a pass.
"""
from __future__ import annotations

from typing import List

import numpy as np

CDF_RTOL = 1e-12             # cdf is cumulative / total in float64 on both sides
EDGE_RTOL = 1e-12
WIDGET_ATOL_FRACTION = 1e-12  # of the slider's full range


def _first_diff(got, want, rtol: float = 0.0):
    g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if g.shape != w.shape:
        return f"shape {g.shape} != {w.shape}"
    bad = ~np.isclose(g, w, rtol=rtol, atol=0.0) if rtol else g != w
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        return f"bin {i}: got {g[i]!r}, expected {w[i]!r} ({int(bad.sum())} bins differ)"
    return None


def _renderer(actual: dict, source: str, yfield: str):
    found = [r for r in actual.get("renderers", []) if r.get("source") == source and r.get("yfield") == yfield]
    return found[0] if len(found) == 1 else len(found)


def compare(expected: dict, actual: dict, widget_span: float = 1.0) -> List[str]:
    out: List[str] = []
    for key in ("widgets", "tab", "selected_count", "selected_sha256", "edges", "bin_count", "cumulative", "cdf",
                "renderers"):
        if key not in actual:
            out.append(f"{key}: missing from the actual response")
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
    for source, yfield in expected["visible_renderers"]:
        r = _renderer(actual, source, yfield)
        if not isinstance(r, dict):
            out.append(f"renderer {source}.{yfield}: found {r} glyph renderers, expected exactly one")
            continue
        if ["HistogramCDS", "histoA"] not in r.get("chain", []):
            out.append(f"renderer {source}.{yfield}: its data does not come from HistogramCDS histoA "
                       f"(chain {r.get('chain')})")
        if r.get("visible") is not True:
            out.append(f"renderer {source}.{yfield}: not visible")
        if r.get("n") != len(column[yfield]):
            out.append(f"renderer {source}.{yfield}: draws {r.get('n')} points, expected {len(column[yfield])}")
        d = _first_diff(r.get("y", []), column[yfield], CDF_RTOL if yfield == "cdf" else 0.0)
        if d:
            out.append(f"renderer {source}.{yfield} (drawn values): {d}")
    for source, yfield in expected["hidden_renderers"]:
        r = _renderer(actual, source, yfield)
        if isinstance(r, dict) and r.get("visible") is not False:
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
