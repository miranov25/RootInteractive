"""PHASE_5_2 stage 6: quantile lines per curve (FR-RI-3 part a item 3, the Architect's item (c)), and in a facet grid.

In plain words.
- A 1-D histogram of several curves (a list of weights chosen by a multiSelect, or a list of variables) is drawn by
  RootInteractive as a stack of histograms, one per option. The feature request asks for the quantiles of each curve
  from its own rows, never of the pooled rows. Stage 4 refused several curves by name.
- Now each curve has its own lines: its quantiles from its own selected rows and weights, its own legend entries and
  texts named after the curve ("wa, median", "wa, 10%: 0.2899"), and only while the curve is selected. The colour of
  a curve's lines: the "color" of quantile_lines if given; else the figure's colour mapper on the curves'
  "_source_index" (factor_cmap("_source_index", palette, curves)); else the histogram's colour.
- In a facet grid (stage 5) the grid legend keeps the entries of differently coloured curves apart and draws each
  entry with the dash of its level (the median solid, the other levels dashed), as the cells do. Found while writing
  these tests: stage 5's grid legend took string values ("bevel", "square") for columns and left out the column dash
  that a quantile line reads, so Bokeh looked for it on every draw and the page never became idle.
- Before the fix (commit 1 of stage 6, the tests only): several curves are refused ("several curves") and the grid
  legend's stand-in of a quantile line has no dash column. Strict expected failures (KnownDefect); since the
  fix, regressions that name it.
Author: Claude10.RI, 2026-11-22.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_phase5_2_band import ref_quantile
from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_phase5_2_quantile_lines import (
    SET, _build_error, _eval, _from_array, _node, _open)


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect or missing feature."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


def regression(condition: bool, message: str) -> None:
    """After the fix, the exact signature of a fixed defect is a regression: an ordinary failure that names it."""
    if condition:
        raise AssertionError("REGRESSION, a fixed defect is back: " + message)


STRICT = dict(strict=True, raises=KnownDefect)
UNIT = pytest.mark.headless_unit
LEVELS = [0.1, 0.5, 0.9]
CURVES = ["wa", "wb"]
OWN = ["#d62728", "#2ca02c"]                          # the figure's own colours of wa, wb


def _two_groups(n=4000) -> pd.DataFrame:
    """Group a: A ~ Gaussian(0.35, 0.15); group b: A ~ Gaussian(0.65, 0.15); wa, wb their 0/1 weights; wr a weight from
    0 to 3 (zeros included) on all rows."""
    from RootInteractive.Tools.pandaTools import initMetadata
    rng = np.random.default_rng(5)
    g = rng.integers(0, 2, n)
    d = pd.DataFrame({"A": np.where(g == 0, rng.normal(0.35, 0.15, n), rng.normal(0.65, 0.15, n))})
    d["wa"] = (g == 0).astype(float)
    d["wb"] = (g == 1).astype(float)
    d["wr"] = np.round(rng.random(n) * 3, 1)
    initMetadata(d)
    return d


PARAMS = [{"name": "weights", "value": list(CURVES), "options": list(CURVES)}]
WIDGETS = (("range", ["A"]), ("multiSelect", ["weights"]))


def _hist():
    return {"name": "hA", "variables": ["A"], "nbins": 10, "range": [0, 1], "weights": "weights"}


def _figure(**extra):
    q = {"levels": list(LEVELS), "text": "both"}
    q.update(extra.pop("quantile_lines", {}))
    return [["A"], ["hA"], {"quantile_lines": q, **extra}]


def _several_refused(df, figures, **kw) -> bool:
    """Stage 5 HEAD's signature: the page is refused because the histogram has several curves."""
    error = _build_error(df, figures, [_hist()], params=PARAMS, widgets=WIDGETS, **kw)
    return error is not None and error[0] == "ValueError" and "several curves" in error[1] \
        and error[2] == "bokehTools.py:addQuantileLines"


def _page(tmp_path, tag, df, figures, **kw):
    from bokeh.document import Document
    regression(_several_refused(df, figures, **kw), "FR-RI-3 (c): several curves are refused")
    doc = Document()
    doc.add_root(_from_array(df, figures, [_hist()], params=PARAMS, widgets=WIDGETS, **kw).pAll)
    out = tmp_path / f"qcurves_{tag}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


STATE = r"""() => {
  const all = window.__riH.models();
  const out = {};
  for (const q of all.filter(m => m.type === 'HistogramQuantiles')) {
    const col = (c) => Array.from(q.get_column(c) ?? []);
    const x = col('x').map(Number);                   // reading a column computes the values first
    out[q.prefix] = {values: [...q.values.entries()], x, level: col('level').map(Number), label: col('label'),
                     dash: col('dash'), text: col('text'), active: q.active};
  }
  return out;
}"""

RANGE = r"""(a) => {
  for (const q of window.__riH.models().filter(m => m.type === 'HistogramQuantiles')) { q.x_range.start = a[0]; q.x_range.end = a[1]; }
  return true;
}"""


def _want(df, lo, hi, curve):
    sel = df[(df.A >= lo) & (df.A <= hi)]
    return {q: ref_quantile(sel.A.to_numpy(), sel[curve].to_numpy(), q, "inverted_cdf") for q in LEVELS}


# ------------------------------------------------------------------ Node: each curve its own numbers
@_node
@pytest.mark.feature("DISPLAY.quantile_lines")
def test_each_curve_has_the_quantiles_of_its_own_rows_not_of_the_pooled_rows(node_session, tmp_path):
    """Two curves (weights wa, wb). Each curve's levels equal dfdraw's weighted inverted_cdf of its own selected rows,
    at two slider positions; each median differs from the median of the pooled selected rows by more than 0.05. The
    legend labels and texts
    name the curve: "wa, median", "wa, 10%: 0.2899"."""
    df = _two_groups()
    _open(node_session, _page(tmp_path, "own", df, [_figure()]))
    _eval(node_session, RANGE, [0.0, 1.0])
    for lo, hi in ((float(df.A.min()), float(df.A.max())), (0.3, 0.9)):
        _eval(node_session, SET, {"name": "A", "value": [lo, hi]})
        s = _eval(node_session, STATE)
        assert sorted(s) == CURVES, s
        pooled = float(np.quantile(df.A[(df.A >= lo) & (df.A <= hi)], 0.5))
        for curve in CURVES:
            want = _want(df, lo, hi, curve)
            got = {float(q): v for q, v in s[curve]["values"]}
            assert got == pytest.approx(want, rel=1e-12), (curve, got, want)
            assert abs(want[0.5] - pooled) > 0.05, (curve, want, pooled)
            shown = [q for q in LEVELS if 0 <= want[q] <= 1]
            assert s[curve]["level"] == shown, s[curve]
            names = {0.1: "q=10%", 0.5: "median", 0.9: "q=90%"}
            assert s[curve]["label"] == [f"{curve}, {names[q]}" for q in shown], s[curve]
            texts = {0.1: "10%", 0.5: "median", 0.9: "90%"}
            assert s[curve]["text"] == [f"{curve}, {texts[q]}: {float(f'{want[q]:.4g}'):g}" for q in shown], s[curve]


@_node
@pytest.mark.feature("DISPLAY.quantile_lines")
def test_a_curve_deselected_draws_no_line_and_comes_back(node_session, tmp_path):
    """The multiSelect of the curves: with only wa selected, wb has no line (no row, so no text and no legend entry)
    and wa is unchanged; with both selected again, wb's lines are back with its own values."""
    df = _two_groups()
    _open(node_session, _page(tmp_path, "deselect", df, [_figure()]))
    _eval(node_session, RANGE, [0.0, 1.0])
    both = _eval(node_session, STATE)
    _eval(node_session, SET, {"name": "weights", "value": ["wa"]})
    one = _eval(node_session, STATE)
    assert one["wb"]["x"] == [] and one["wb"]["label"] == [] and one["wb"]["active"] is False, one["wb"]
    assert one["wa"]["x"] == both["wa"]["x"] and one["wa"]["active"] is True, (one["wa"], both["wa"])
    _eval(node_session, SET, {"name": "weights", "value": ["wa", "wb"]})
    again = _eval(node_session, STATE)
    assert again["wb"]["x"] == both["wb"]["x"] and len(again["wb"]["x"]) >= 2, (again["wb"], both["wb"])


# ------------------------------------------------------------------ Python: colours and legend entries
def _spans(page):
    from bokeh.models import GlyphRenderer
    return [m for m in page.pAll.references() if isinstance(m, GlyphRenderer) and type(m.glyph).__name__ == "VSpan"]


@UNIT
@pytest.mark.feature("DISPLAY.quantile_lines")
def test_the_colour_of_each_curves_lines_and_one_legend_entry_per_curve():
    """With the figure's colour mapper on _source_index, each curve's lines take its colour; with the option's own
    color, every line takes it; with neither, the histogram's colour. One legend entry per curve (on the column label
    of its own rows), so that each curve's levels are listed in its colour."""
    from bokeh.models import Legend
    from bokeh.transform import factor_cmap
    df = _two_groups(500)
    regression(_several_refused(df, [_figure()]), "FR-RI-3 (c): several curves are refused")
    cases = [(_figure(color=factor_cmap("_source_index", OWN, CURVES)), OWN),
             (_figure(quantile_lines={"color": "black"}), ["black", "black"]),
             (_figure(color="#9467bd"), ["#9467bd", "#9467bd"])]
    for figure, colours in cases:
        page = _from_array(df, [figure], [_hist()], params=PARAMS, widgets=WIDGETS)
        spans = sorted(_spans(page), key=lambda r: r.data_source.prefix)     # a page's references are a set
        assert [s.data_source.prefix for s in spans] == CURVES, spans
        assert [s.glyph.line_color for s in spans] == colours, (figure[2], [s.glyph.line_color for s in spans])
        items = [it for lg in page.pAll.references() if isinstance(lg, Legend) for it in lg.items
                 if any(r in spans for r in it.renderers)]
        assert sorted([r.data_source.prefix for r in it.renderers] for it in items) == [["wa"], ["wb"]], items
        assert all(it.label["field"] == "label" for it in items), items


# ------------------------------------------------------------------ Chromium: in a facet grid
GRID_LEGEND = r"""() => {
  const all = window.__riH.models();
  const legend = all.find(m => m.type === 'Legend' && m.name === 'facet_grid_legend' && m.visible);
  const out = [];
  for (const item of legend.items) {
    if (!item.visible) continue;
    for (const r of item.renderers) {
      if (r.glyph.type !== 'VSpan') continue;
      const rows = r.data_source;
      out.push({colour: r.glyph.line_color.value, labels: Array.from(rows.get_column('label') ?? []).map(String),
                dashes: Array.from(rows.get_column('dash') ?? []).map(String)});
    }
  }
  return out;
}"""


def _grid_page(df, curves):
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    from bokeh.transform import factor_cmap
    before = sa.show
    sa.show = lambda *a, **k: None
    params = [{"name": "weights", "value": list(CURVES), "options": list(CURVES)}] if curves else []
    hist = {**_hist(), "weights": "weights"} if curves else {k: v for k, v in _hist().items() if k != "weights"}
    extra = {"color": factor_cmap("_source_index", OWN, CURVES)} if curves else {"color": OWN[0]}
    figures = [_figure(**extra) for _ in range(2)]
    try:
        return sa.bokehDrawSA.fromArray(df, None, figures, [list(w) for w in (WIDGETS if curves else WIDGETS[:1])],
                                        histogramArray=[hist], parameterArray=params,
                                        widgetLayout=[[0, 1] if curves else [0]],
                                        layout={"facetGrid": {"cells": [[0, 1]], "xTitle": "A",
                                                              "columnHeadings": ["left", "right"]}})
    finally:
        sa.show = before


def _dash_missing(page) -> bool:
    """Stage 5 HEAD's signature: the grid legend's stand-in of a quantile line reads its dash from the column "dash",
    which its rows (LegendRows) do not have."""
    from bokeh.models import GlyphRenderer
    rows = [m.data_source for m in page.pAll.references() if isinstance(m, GlyphRenderer)
            and type(m.glyph).__name__ == "VSpan" and type(m.data_source).__name__ == "LegendRows"]
    return bool(rows) and any("dash" not in r.column_names for r in rows)


@pytest.mark.browser_display
@pytest.mark.feature("DISPLAY.quantile_lines")
@pytest.mark.feature("DISPLAY.facet_grid")
@pytest.mark.parametrize("curves", [False, True], ids=["one_curve", "two_curves"])
def test_browser_quantile_lines_in_a_facet_grid_legend(headless_session, tmp_path, curves):
    """Chromium, two cells with quantile lines. The grid legend lists each curve's levels once (both cells' entries
    of a curve are one entry), in the curve's colour, with the dash of the level (the median solid, the others
    dashed); two curves stay two entries. No console error."""
    from bokeh.io import save
    from bokeh.resources import INLINE
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_display import _open_browser_page
    df = _two_groups(2000)
    if curves:
        regression(_several_refused(df, [_figure()]), "FR-RI-3 (c): several curves are refused")
    page_model = _grid_page(df, curves)
    regression(_dash_missing(page_model), "stage 5: the grid legend's entry of a quantile line has no dash column")
    p = tmp_path / "qcurves_grid.html"
    save(page_model.pAll, filename=str(p), resources=INLINE, title="quantile lines in a facet grid")
    ctx, page, errors = _open_browser_page(headless_session, p)
    try:
        page.wait_for_timeout(300)
        entries = page.evaluate(GRID_LEGEND)
        prefix = (lambda c: f"{c}, ") if curves else (lambda c: "")
        want = [{"colour": OWN[k], "labels": [f"{prefix(c)}q=10%", f"{prefix(c)}median", f"{prefix(c)}q=90%"],
                 "dashes": ["dashed", "solid", "dashed"]} for k, c in enumerate(CURVES if curves else CURVES[:1])]
        assert entries == want, entries
        (tmp_path / "qcurves_grid.png").write_bytes(page.screenshot(full_page=True))
        assert not errors, errors
    finally:
        ctx.close()
