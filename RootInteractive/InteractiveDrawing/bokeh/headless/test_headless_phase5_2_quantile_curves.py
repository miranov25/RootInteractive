"""PHASE_5_2 stage 7: quantile curves on a 2-D heatmap with boxed labels (the Architect's option (y)).

In plain words.
- dfdraw's QA pages (TaylorFits_V3_MAD, 2026-11-14) draw a 2-D histogram as a heatmap with, for each level (5 %, 10 %,
  50 %, 90 %, 95 %), a curve: the level's quantile of the second variable in each bin of the first, over the selected
  rows. Each curve has its colour (Matplotlib's tab10 from orange) and dash (the cycle dashed, dashdot, dotted), a
  legend entry ("q=5%", "median") and a label in a white box with the level in the curve's colour ("5%", "50%"), all
  labels at one X position.
- RootInteractive now draws this from the heatmap figure's option "quantile_curves": {"levels": [...], "method",
  "colors", "dashes", "labels" ("boxed" or "none"), "label_x" (where the labels stand, a fraction of the shown X range,
  0.4 by default), "legend", "width"}. The numbers are a profile of the histogram (HistoNdProfile) with these levels:
  with "unbinned_projections" and quantile_method "linear" they are NumPy's linear quantile of the cell's rows.
- Refused by name: a wrong key, levels, method, labels, label_x, colors or dashes; a method that reads rows without
  unbinned_projections; the option on a 1-D histogram or on another drawing than a heatmap; a histogram of several
  curves.
- Before the fix (commit 1 of stage 7, the tests only) the option was silently ignored: the page had the heatmap and
  no curve. Strict expected failures (KnownDefect); since the fix, regressions that name it.
Author: Claude10.RI, 2026-11-22.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

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
LEVELS = [0.05, 0.1, 0.5, 0.9, 0.95]
TEXTS = ["5%", "10%", "50%", "90%", "95%"]
TAB10_FROM_ORANGE = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
DASHES = ["dashed", "dashdot", "dotted", "dashed", "dashdot"]


def _table(n=6000) -> pd.DataFrame:
    """A uniform in [0, 6); B = 0.5 A + (0.3 + 0.1 A) U(-2, 2), always inside [-2, 6] (the histogram's Y range), its
    spread growing with A; C uniform in [0, 1), a second selection."""
    from RootInteractive.Tools.pandaTools import initMetadata
    rng = np.random.default_rng(11)
    d = pd.DataFrame({"A": rng.random(n) * 6, "C": rng.random(n)})
    d["B"] = 0.5 * d.A + (0.3 + 0.1 * d.A) * (rng.random(n) * 4 - 2)
    initMetadata(d)
    return d


def _hist(**extra):
    return {"name": "h2", "variables": ["A", "B"], "nbins": [6, 20], "range": [[0, 6], [-2, 6]],
            "unbinned_projections": True, "quantile_method": "linear", **extra}


def _figure(**curves):
    q = {"levels": list(LEVELS)}
    q.update(curves)
    return [["A"], ["h2"], {"quantile_curves": q}]


WIDGETS = (("range", ["C"]),)


def _ignored(df, figures, histograms, **kw) -> bool:
    """The snapshot's signature: the page is built and holds no line (the heatmap's Quad only)."""
    from bokeh.models import GlyphRenderer
    if _build_error(df, figures, histograms, widgets=WIDGETS, **kw) is not None:
        return False
    page = _from_array(df, figures, histograms, widgets=WIDGETS, **kw)
    return not any(isinstance(m, GlyphRenderer) and type(m.glyph).__name__ == "Line" for m in page.pAll.references())


def _page(tmp_path, tag, df, figures, histograms, **kw):
    from bokeh.document import Document
    regression(_ignored(df, figures, histograms, **kw), "option (y): quantile_curves is ignored")
    doc = Document()
    doc.add_root(_from_array(df, figures, histograms, widgets=WIDGETS, **kw).pAll)
    out = tmp_path / f"qcurves2d_{tag}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


STATE = r"""() => {
  const all = window.__riH.models();
  const lines = all.filter(m => m.type === 'GlyphRenderer' && m.glyph.type === 'Line');
  const profile = lines[0].data_source;
  const curves = lines.map(r => Array.from(profile.get_column(r.glyph.y.field)).map(Number));
  const labels = all.find(m => m.type === 'QuantileCurveLabels');
  const col = (c) => labels == null ? null : Array.from(labels.get_column(c) ?? []);
  return {x: Array.from(profile.get_column('bin_center_0')).map(Number), curves,
          labels: labels == null ? null : {x: col('x').map(Number), y: col('y').map(Number), text: col('text'),
                                           color: col('color')}};
}"""

RANGE = r"""(a) => {
  const labels = window.__riH.models().find(m => m.type === 'QuantileCurveLabels');
  labels.x_range.start = a[0]; labels.x_range.end = a[1];
  return true;
}"""


def _oracle(df, lo, hi):
    """The levels of B in each A bin [k, k + 1) over the rows with lo <= C <= hi: NumPy's linear quantile."""
    sel = df[(df.C >= lo) & (df.C <= hi)]
    return [[float(np.quantile(sel.B[(sel.A >= k) & (sel.A < k + 1)], q)) for k in range(6)] for q in LEVELS]


# ------------------------------------------------------------------ Node: the numbers and the labels
@_node
@pytest.mark.feature("DISPLAY.quantile_curves")
def test_each_curve_is_the_levels_quantile_of_b_in_each_a_bin_of_the_selected_rows(node_session, tmp_path):
    """Five curves, one per level; each curve's value in an A bin equals NumPy's linear quantile of B over the
    selected rows of that bin, for all rows and after the slider on C selects [0.3, 0.8]."""
    df = _table()
    _open(node_session, _page(tmp_path, "numbers", df, [_figure()], [_hist()]))
    for lo, hi in ((float(df.C.min()), float(df.C.max())), (0.3, 0.8)):
        _eval(node_session, SET, {"name": "C", "value": [lo, hi]})
        s = _eval(node_session, STATE)
        assert s["x"] == [k + 0.5 for k in range(6)], s["x"]
        want = _oracle(df, lo, hi)
        for k, q in enumerate(LEVELS):
            assert s["curves"][k] == pytest.approx(want[k], rel=1e-12), (q, s["curves"][k], want[k])


@_node
@pytest.mark.feature("DISPLAY.quantile_curves")
def test_the_boxed_labels_stand_at_one_x_on_their_curves(node_session, tmp_path):
    """With the shown X range [0, 6], all labels stand at the bin nearest 0.4 of it (2.4: the bin centre 2.5), each at
    its curve's value there, with its level and colour; with the range [3, 6] at 4.5 (4.2); after the slider on C
    selects [0.3, 0.8], on the new values of the curves; with label_x 0.1 and [0, 6] at 0.5. Nothing while the range
    is not known."""
    df = _table()
    _open(node_session, _page(tmp_path, "labels", df, [_figure()], [_hist()]))
    assert _eval(node_session, STATE)["labels"]["x"] == [], "no label before the shown range is known"
    for range_, at in (([0, 6], 2), ([3, 6], 4)):
        _eval(node_session, RANGE, range_)
        s = _eval(node_session, STATE)
        assert s["labels"]["x"] == [at + 0.5] * 5, s["labels"]
        assert s["labels"]["y"] == [s["curves"][k][at] for k in range(5)], s["labels"]
        assert s["labels"]["text"] == TEXTS and s["labels"]["color"] == TAB10_FROM_ORANGE, s["labels"]
    before = _eval(node_session, STATE)["labels"]["y"]
    _eval(node_session, SET, {"name": "C", "value": [0.3, 0.8]})
    s = _eval(node_session, STATE)
    assert s["labels"]["y"] == [s["curves"][k][4] for k in range(5)] and s["labels"]["y"] != before, (s, before)
    _open(node_session, _page(tmp_path, "labels_x", df, [_figure(label_x=0.1)], [_hist()]))
    _eval(node_session, RANGE, [0, 6])
    assert _eval(node_session, STATE)["labels"]["x"] == [0.5] * 5


@_node
@pytest.mark.feature("DISPLAY.quantile_curves")
def test_the_curves_follow_the_histograms_weights_chosen_by_a_select(node_session, tmp_path):
    """The histogram's weights chosen by a select (one value): w1 = 1 on all rows, wh = 1 where C > 0.5 and 0
    elsewhere. With wh the curves are the quantiles of the rows with C > 0.5 (a zero weight carries no mass)."""
    df = _table()
    df["w1"] = 1.0
    df["wh"] = (df.C > 0.5).astype(float)
    params = [{"name": "weights", "value": "w1", "options": ["w1", "wh"]}]
    widgets = WIDGETS + (("select", ["weights"]),)
    figure = _figure()
    regression(_ignored(df, [figure], [_hist(weights="weights")], params=params),
                 "option (y): quantile_curves is ignored")
    from bokeh.document import Document
    doc = Document()
    doc.add_root(_from_array(df, [figure], [_hist(weights="weights")], params=params, widgets=widgets).pAll)
    page = tmp_path / "qcurves2d_weights.json"
    page.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    _open(node_session, page)
    s = _eval(node_session, STATE)
    for k in range(5):
        assert s["curves"][k] == pytest.approx(_oracle(df, 0, 1)[k], rel=1e-12), k
    _eval(node_session, SET, {"name": "weights", "value": "wh"})
    s = _eval(node_session, STATE)
    want = _oracle(df, 0.5 + 1e-12, 1)
    for k in range(5):
        assert s["curves"][k] == pytest.approx(want[k], rel=1e-12), (k, s["curves"][k], want[k])


# ------------------------------------------------------------------ Python: the look, the options, the refusals
def _lines(page):
    from bokeh.models import GlyphRenderer
    found = [m for m in page.pAll.references() if isinstance(m, GlyphRenderer) and type(m.glyph).__name__ == "Line"]
    return sorted(found, key=lambda r: r.glyph.y)                  # quantile_0, quantile_1, ...: a page's references are a set


@UNIT
@pytest.mark.feature("DISPLAY.quantile_curves")
def test_the_look_dfdraws_colours_dashes_legend_and_the_options():
    """By default dfdraw's look: tab10 from orange, the dash cycle dashed, dashdot, dotted, legend "q=5%" ... "median",
    a LabelSet in white boxes; the options colors, dashes, width, legend False and labels "none" are followed."""
    from bokeh.models import LabelSet, Legend
    df = _table(1000)
    regression(_ignored(df, [_figure()], [_hist()]), "option (y): quantile_curves is ignored")
    page = _from_array(df, [_figure()], [_hist()], widgets=WIDGETS)
    lines = _lines(page)
    assert [r.glyph.line_color for r in lines] == TAB10_FROM_ORANGE and [r.glyph.line_dash for r in lines] == \
        [[6], [6, 4, 2, 4], [2, 4], [6], [6, 4, 2, 4]], [(r.glyph.line_color, r.glyph.line_dash) for r in lines]
    labels = [it.label["value"] for lg in page.pAll.references() if isinstance(lg, Legend) for it in lg.items]
    assert labels == ["q=5%", "q=10%", "median", "q=90%", "q=95%"], labels
    boxes = [m for m in page.pAll.references() if isinstance(m, LabelSet)]
    assert len(boxes) == 1 and boxes[0].background_fill_color == "white", boxes
    own = _figure(colors=["red", "blue", "black", "blue", "red"], dashes=["solid"] * 5, width=3, legend=False,
                  labels="none")
    page = _from_array(df, [own], [_hist()], widgets=WIDGETS)
    lines = _lines(page)
    assert [r.glyph.line_color for r in lines] == ["red", "blue", "black", "blue", "red"]
    assert all(r.glyph.line_dash == [] and r.glyph.line_width == 3 for r in lines)
    assert not any(isinstance(m, LabelSet) for m in page.pAll.references())
    assert not any(it for lg in page.pAll.references() if isinstance(lg, Legend) for it in lg.items)


@UNIT
@pytest.mark.feature("DISPLAY.quantile_curves")
@pytest.mark.parametrize("figure,histogram,named", [
    (_figure(level=[0.5]), _hist(), ["'level'", "'levels'"]),
    (_figure(levels=[0.5, 1.5]), _hist(), ["levels", "0 to 1"]),
    (_figure(method="median"), _hist(), ["'median'", "'linear'"]),
    (_figure(method="linear"), _hist(unbinned_projections=False, quantile_method="subbin"), ["unbinned_projections"]),
    (_figure(colors=["red"]), _hist(), ["colors", "5"]),
    (_figure(labels="box"), _hist(), ["'box'", "'boxed'"]),
    (_figure(label_x=2), _hist(), ["label_x", "0 to 1"]),
    ([["A"], ["h2"], {"quantile_curves": {"levels": [0.5]}, "visualization_type": "colZ"}], _hist(),
     ["heatmap", "'colZ'"]),
    ([["A"], ["h1"], {"quantile_curves": {"levels": [0.5]}}], {"name": "h1", "variables": ["A"], "nbins": 6,
                                                               "range": [0, 6]}, ["1-D", "quantile_lines"])],
    ids=["key", "levels", "method", "needs_unbinned", "colors", "labels", "label_x", "colZ", "1d"])
def test_a_wrong_option_is_refused_by_name(figure, histogram, named):
    df = _table(500)
    error = _build_error(df, [figure], [histogram], widgets=WIDGETS)
    regression(error is None, "option (y): quantile_curves is ignored")
    assert error is not None and error[0] == "ValueError" and all(n in error[1] for n in named), error


@UNIT
@pytest.mark.feature("DISPLAY.quantile_curves")
def test_a_histogram_of_several_curves_is_refused_by_name():
    df = _table(500)
    df["w1"], df["w2"] = 1.0, 2.0
    params = [{"name": "weights", "value": ["w1", "w2"], "options": ["w1", "w2"]}]
    error = _build_error(df, [_figure()], [_hist(weights="weights")], params=params,
                         widgets=(("multiSelect", ["weights"]),))
    regression(error is None, "option (y): quantile_curves is ignored")
    assert error is not None and error[0] == "ValueError" and "several curves" in error[1], error


# ------------------------------------------------------------------ in a facet grid: the shared range
def _grid_of_quantile_lines(df):
    """Two cells of 1-D histograms with stage 4's quantile lines, in a facet grid sharing X (commonX)."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    before = sa.show
    sa.show = lambda *a, **k: None
    try:
        return sa.bokehDrawSA.fromArray(df, None, [[["A"], ["h1"], {"quantile_lines": {"levels": [0.5]}}]] * 2,
                                        [list(w) for w in WIDGETS], widgetLayout=[[0]],
                                        histogramArray=[{"name": "h1", "variables": ["A"], "nbins": 6, "range": [0, 6]}],
                                        layout={"facetGrid": {"cells": [[0, 1]]}})
    finally:
        sa.show = before


@UNIT
@pytest.mark.feature("DISPLAY.facet_grid")
@pytest.mark.feature("DISPLAY.quantile_curves")
def test_in_a_facet_grid_every_cell_reads_the_shared_range():
    """A facet grid shares one X range among its cells. Stage 4's quantile lines and stage 7's curve labels draw only
    inside the shown range, so each cell's must read the shared range, not the cell's own one, which no plot computes
    any more (stage 5 left them on it: only the first cell drew its lines)."""
    from bokeh.models import GlyphRenderer, GridBox, Plot
    from RootInteractive.InteractiveDrawing.bokeh.HistogramQuantiles import HistogramQuantiles
    df = _table(1000)
    page = _grid_of_quantile_lines(df)
    grid = [m for m in page.pAll.references() if isinstance(m, GridBox) and m.name == "facet_grid"][0]
    shared = [c[0] for c in grid.children if isinstance(c[0], Plot) and c[0].name != "facet_grid_legend"][0].x_range
    sources = [m for m in page.pAll.references() if isinstance(m, HistogramQuantiles)]
    regression(len(sources) == 2 and sum(s.x_range is shared for s in sources) == 1,
                 "stage 5: a cell's quantile lines read the cell's own range, not the grid's shared one")
    assert len(sources) == 2 and all(s.x_range is shared for s in sources), [s.x_range for s in sources]
    curves = _grid_page(df)
    grid = [m for m in curves.pAll.references() if isinstance(m, GridBox) and m.name == "facet_grid"][0]
    shared = [c[0] for c in grid.children if isinstance(c[0], Plot) and c[0].name != "facet_grid_legend"][0].x_range
    labels = [m for m in curves.pAll.references() if type(m).__name__ == "QuantileCurveLabels"]
    assert len(labels) == 2 and all(s.x_range is shared for s in labels), [s.x_range for s in labels]
    cells = [r for r in curves.pAll.references() if isinstance(r, Plot) and r.name != "facet_grid_legend" and
             any(isinstance(g, GlyphRenderer) and type(g.glyph).__name__ == "Line" for g in r.renderers)]
    assert len(cells) == 2 and all(r.x_range is shared for r in cells), cells


# ------------------------------------------------------------------ Chromium: drawn, alone and in a facet grid
LOOK = r"""() => {
  const all = window.__riH.models();
  const sets = all.filter(m => m.type === 'LabelSet');
  const legend = all.find(m => m.type === 'Legend' && m.visible && m.items.some(i => i.visible));
  const entries = legend.items.filter(i => i.visible).map(i => ({label: String(i.label.value),
    colour: i.renderers[0].glyph.line_color.value, dash: i.renderers[0].glyph.line_dash.value}));
  return {rows: sets.map(s => s.source.get_length()), legend: legend.name, entries};
}"""


def _grid_page(df):
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    before = sa.show
    sa.show = lambda *a, **k: None
    try:
        return sa.bokehDrawSA.fromArray(df, None, [_figure(), _figure()], [list(w) for w in WIDGETS],
                                        histogramArray=[_hist()], widgetLayout=[[0]],
                                        layout={"facetGrid": {"cells": [[0, 1]], "xTitle": "A", "yTitle": "B"}})
    finally:
        sa.show = before


@pytest.mark.browser_display
@pytest.mark.feature("DISPLAY.quantile_curves")
@pytest.mark.parametrize("grid", [False, True], ids=["alone", "facet_grid"])
def test_browser_curves_labels_and_legend_are_drawn(headless_session, tmp_path, grid):
    """Chromium: the heatmap with its five curves; each cell's LabelSet has five rows (the shown range is known once
    drawn); the legend (the cell's, or in a facet grid the grid's) lists the five levels once, each in its curve's
    colour and dash; no console error."""
    from bokeh.io import save
    from bokeh.resources import INLINE
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_display import _open_browser_page
    df = _table(3000)
    regression(_ignored(df, [_figure()], [_hist()]), "option (y): quantile_curves is ignored")
    page_model = _grid_page(df) if grid else _from_array(df, [_figure()], [_hist()], widgets=WIDGETS)
    p = tmp_path / "qcurves2d.html"
    save(page_model.pAll, filename=str(p), resources=INLINE, title="quantile curves")
    ctx, page, errors = _open_browser_page(headless_session, p)
    try:
        page.wait_for_timeout(300)
        look = page.evaluate(LOOK)
        assert look["rows"] == [5, 5] if grid else look["rows"] == [5], look
        assert look["legend"] == ("facet_grid_legend" if grid else None), look
        dash = {"dashed": [6], "dashdot": [6, 4, 2, 4], "dotted": [2, 4]}
        want = [{"label": label, "colour": c, "dash": dash[d]}
                for label, c, d in zip(["q=5%", "q=10%", "median", "q=90%", "q=95%"], TAB10_FROM_ORANGE, DASHES)]
        assert look["entries"] == want, look["entries"]
        (tmp_path / "qcurves2d.png").write_bytes(page.screenshot(full_page=True))
        assert not errors, errors
    finally:
        ctx.close()
