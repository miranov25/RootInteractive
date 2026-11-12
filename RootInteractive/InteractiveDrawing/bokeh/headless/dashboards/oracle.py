"""Expected responses of the original dashboards, computed with numpy from the raw table only (PHASE_4_7, Claude20.RI).

Nothing here imports RootInteractive. The raw table is regenerated from (rows, seed) exactly as the opt-in in
test_bokehClientHistogram.py builds it (numpy.random.default_rng(seed).random((rows, 4)), columns A, B, C, D) and
its hash must equal the one the captured page recorded, so the oracle and the page start from the same events.

H01 contract (testBokehClientHistogramOnlyHisto; histogram `histoA`: variable A, 20 bins, range histoRangeA = [0, 1]):
- a row is selected when A > 0 (the page's initial query "A>0") and every range widget holds start <= value <= end
  (both ends included) and its DDC category is among the selected ones; a range end that is itself a data value
  (the slider's full range is min..max of the column) selects that row on both sides by exact comparison;
- bin i counts the selected rows with edge[i] <= A < edge[i+1], edges = linspace(0, 1, 21); the oracle checks that no
  selected value lies within 1e-9 of an edge, so this agrees with numpy.histogram;
- cumulative = running sum of the counts, left to right;
- cdf = cumulative / (sum of the counts in the histogram range), last bin 1 (decision DEC-06, for non-empty
  histograms; an empty histogram is a separate, open policy and is never produced by these tests);
- the drawn plots (H01_PLOTS): x is the bin centre (edge[i] + edge[i+1]) / 2; the axis labels and titles follow the
  declaration in testBokehClientHistogramOnlyHisto and the table's metadata (test_bokehClientHistogram.py line 32,
  'A.AxisTitle': "A (cm)"): a plot of histoA is labelled with A's title, a plot of the stacked source histo1Ds (a
  multi-select of histograms, so no single variable) with the plain field name "bin_center"; the declared y title of
  histo1Ds, "{transformY}(N)", reads "(N)" while transformY is None (no transform). Labels are compared without
  leading or trailing spaces.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Sequence

import numpy as np

H01_FUNCTION = "testBokehClientHistogramOnlyHisto"
H01_NBINS = 20
H01_RANGE = (0.0, 1.0)
EDGE_MARGIN = 1e-9
DDC_NAMES = {0: "A0", 1: "A1", 2: "A2", 3: "A3", 4: "A4"}
H01_PLOTS = {   # (source, drawn y field) -> drawn x field and labels; x values are the bin centres
    ("histo1Ds", "bin_count"): {"xfield": "bin_center", "xlabel": "bin_center", "ylabel": "(N)",
                                "title": "(N) vs bin_center"},
    ("histoA", "cumulative"): {"xfield": "bin_center", "xlabel": "A (cm)", "ylabel": "cumulative",
                               "title": "cumulative vs A (cm)"},
    ("histoA", "cdf"): {"xfield": "bin_center", "xlabel": "A (cm)", "ylabel": "cdf", "title": "cdf vs A (cm)"},
}


def regenerate_table(rows: int, seed: int) -> Dict[str, np.ndarray]:
    raw = np.random.default_rng(int(seed)).random(size=(int(rows), 4))
    return {c: raw[:, i].copy() for i, c in enumerate("ABCD")}


def table_sha256(t: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for col in "ABCD":
        h.update(np.ascontiguousarray(t[col], dtype="<f8").tobytes())
    return h.hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=bool)).tobytes()).hexdigest()


def selection(t: Dict[str, np.ndarray], widgets: Dict[str, dict]) -> np.ndarray:
    """Rows the page should select: the initial query A>0 and every range and category widget at its value."""
    mask = t["A"] > 0
    for title, w in widgets.items():
        if w["type"] == "RangeSlider" and title in t:
            lo, hi = w["value"]
            mask &= (t[title] >= lo) & (t[title] <= hi)
    ddc = widgets.get("DDC")
    if ddc is not None and set(ddc["value"]) != set(ddc["options"]):
        names = np.vectorize(DDC_NAMES.get)(np.round(t["D"] * 4).astype(int))
        mask &= np.isin(names, list(ddc["value"]))
    return mask


def boundary_margin(values: np.ndarray, cuts: Sequence[float]) -> float:
    """Smallest distance between any value and any cut (bin edge or range end)."""
    v = np.sort(values)
    out = np.inf
    for c in cuts:
        i = np.searchsorted(v, c)
        for j in (i - 1, i):
            if 0 <= j < len(v):
                out = min(out, abs(v[j] - c))
    return float(out)


def h01_expected(t: Dict[str, np.ndarray], widgets: Dict[str, dict], tab: str, checkpoint: str,
                 check_margin: bool = True) -> dict:
    """The full expected response of H01 for one widget state and the active tab. check_margin=False only for
    describing a page whose slider value is known to sit next to a data value (the H01-D1 signature)."""
    mask = selection(t, widgets)
    a = t["A"][mask]
    edges = np.linspace(H01_RANGE[0], H01_RANGE[1], H01_NBINS + 1)
    # A range end that IS a data value (the full range: min and max of A) is compared exactly on both sides; only
    # computed cuts (bin edges, keyboard steps) need a margin against rounding.
    cuts = list(edges) + [v for v in widgets["A"]["value"] if not np.any(t["A"] == v)]
    margin = boundary_margin(t["A"], cuts)
    if check_margin and margin <= EDGE_MARGIN:
        raise AssertionError(f"H01 fixture: a value of A lies {margin:g} from a bin edge or range end; "
                             f"choose another seed or step (membership would depend on rounding)")
    counts = np.histogram(a, bins=edges)[0].astype(np.int64)
    if (a == edges[-1]).any():
        raise AssertionError("H01 fixture: a selected A equals the histogram's upper edge")
    cumulative = np.cumsum(counts)
    total = cumulative[-1]
    if total <= 0:
        raise AssertionError("H01: empty histogram; its CDF is an open policy and is not tested")
    visible = ([["histo1Ds", "bin_count"]] if tab == "Histograms"
               else [["histoA", "cumulative"], ["histoA", "cdf"]])
    hidden = ([["histoA", "cumulative"], ["histoA", "cdf"]] if tab == "Histograms" else [["histo1Ds", "bin_count"]])
    centres = ((edges[:-1] + edges[1:]) / 2).tolist()
    plots = {f"{s}.{y}": dict(H01_PLOTS[(s, y)], x=centres) for s, y in visible}
    return {
        "scenario": "H01", "function": H01_FUNCTION, "checkpoint": checkpoint,
        "widgets": {"A": list(widgets["A"]["value"])}, "tab": tab,
        "selected_count": int(mask.sum()), "selected_sha256": mask_sha256(mask),
        "edges": edges.tolist(), "bin_count": counts.tolist(), "cumulative": cumulative.tolist(),
        "cdf": (cumulative / total).tolist(),
        "visible_renderers": visible, "hidden_renderers": hidden, "plots": plots,
        "boundary_margin": margin,
    }


def with_a_range(widgets: Dict[str, dict], lo: float, hi: float) -> Dict[str, dict]:
    out = {k: dict(v) for k, v in widgets.items()}
    out["A"] = dict(out["A"], value=[float(lo), float(hi)])
    return out


def keyboard_range(widget: dict, k_low: int, k_high: int):
    """The legal values k_low steps right of start (lower handle) and k_high steps left of end (upper handle)."""
    start, end, step = widget["start"], widget["end"], widget["step"]
    n = int(round((end - start) / step))
    return start + k_low * step, start + (n - k_high) * step
