"""PHASE_5_2 stage 5: a facet grid with its titles, one legend and a footer (FR-RI-5, with the option of a footer).

In plain words.
- dfdraw draws a faceted call as one grid: a title over the whole grid, one legend for the grid, column headings over
  the top row and row headings left of each row (or a title per cell), one X title under the grid and one Y title
  left of it (vertical), the same ranges in every cell, tick labels only on the outer cells. The Architect's examples
  (dEdxAI QA pages, 2026-11-22) also have a footer with the meaning of each variable.
- RootInteractive now builds this from a layout element {"facetGrid": {...}} with the keys cells, title, xTitle,
  yTitle, columnHeadings, rowHeadings, cellTitles, legend ("top" or "none"), commonX, commonY, footer, plot_height,
  min_border_left, min_border_bottom. Every key and count is checked; a wrong one is refused by name.
- The grid legend: one strip above the cells. Each entry is drawn by a stand-in that shares the cell's glyph (so it
  keeps the cell's colours) and draws nothing itself; a legend on a column lists the values in the cell's rows and
  follows them; an entry hidden in the cell is hidden in the strip. The cells' own legends are hidden.
- Before the fix (commit 1 of stage 5, the tests only) the layout dict was read as tabs and the build failed with a
  KeyError in processBokehLayoutArrayRenderers: strict expected failures (KnownDefect). Since the fix (commit 2), the
  same signature is a regression that names it.
Author: Claude10.RI, 2026-11-22.
"""
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
STAGE_5_TODAY = pytest.mark.xfail(**STRICT, reason="FR-RI-5: the layout element facetGrid is not known")
UNIT = pytest.mark.headless_unit


def _table(n=3000) -> pd.DataFrame:
    """A uniform in [0, 4), F the facet 0..5, cat a group a/b/c (only a in the first cell, F == 0: the grid legend lists
    the groups of all cells, not those of the first one), B = 1 + 0.2 A (1 + F) + noise."""
    from RootInteractive.Tools.pandaTools import initMetadata
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"A": rng.random(n) * 4, "F": rng.integers(0, 6, n).astype(float),
                      "cat": np.array(["a", "b", "c"])[rng.integers(0, 3, n)]})
    d["B"] = 1 + 0.2 * d.A * (1 + d.F) + rng.normal(0, 0.3, n)
    d.loc[d.F == 0, "cat"] = "a"
    initMetadata(d)
    return d


FIGURES = [[["A"], ["B"], {"filter": f"F == {k}", "legend_field": "cat"}] for k in range(6)]
GRID = {"cells": [[0, 1, 2], [3, 4, 5]], "title": "B vs A  [faceted by F]\nselection: all", "xTitle": "A",
        "yTitle": "B, the second variable", "columnHeadings": ["H=0", "H=1", "H=2"], "rowHeadings": ["Q=low", "Q=high"],
        "footer": ["A: the first variable", "B: 1 + 0.2 A (1 + F) + noise"]}
WIDGETS = [["range", ["A"]], ["multiSelect", ["cat"]]]


def _from_array(grid=None, figures=FIGURES):
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    before = sa.show
    sa.show = lambda *a, **k: None
    try:
        return sa.bokehDrawSA.fromArray(_table(), None, list(figures), [list(w) for w in WIDGETS],
                                        layout={"facetGrid": dict(GRID if grid is None else grid)},
                                        widgetLayout=[list(range(len(WIDGETS)))])
    finally:
        sa.show = before


def _build_error(grid=None, figures=FIGURES):
    try:
        _from_array(grid, figures)
        return None
    except Exception as exc:                                    # noqa: BLE001 - the signature names the exception
        last = traceback.extract_tb(exc.__traceback__)[-1]
        return type(exc).__name__, str(exc), f"{Path(last.filename).name}:{last.name}"


def _grid_of(page):
    from bokeh.models import GridBox
    grids = [m for m in page.pAll.references() if isinstance(m, GridBox) and m.name == "facet_grid"]
    return grids[0] if grids else None


def _no_grid(grid=None) -> bool:
    """The snapshot's signature (stage 4 HEAD 2fc9158): the layout dict is read as tabs, so the build fails with a
    KeyError on a key of the dict inside processBokehLayoutArrayRenderers; or a page is built without a facet grid."""
    error = _build_error(grid)
    if error is not None:
        return error[0] == "KeyError" and error[2] == "bokehTools.py:processBokehLayoutArrayRenderers"
    return _grid_of(_from_array(grid)) is None


# ------------------------------------------------------------------ Python: the grid as built
@UNIT
@pytest.mark.feature("DISPLAY.facet_grid")
@STAGE_5_TODAY
def test_the_grid_is_built_with_its_titles_shared_ranges_and_one_legend():
    known_defect(_no_grid(), "FR-RI-5: the layout element facetGrid is read as tabs")
    page = _from_array()
    grid = _grid_of(page)
    from bokeh.models import Div, Legend, Plot
    texts = [c[0].text for c in grid.children if isinstance(c[0], Div)]
    assert texts[0] == "B vs A  [faceted by F]<br>selection: all", texts
    assert {"B, the second variable", "Q=low", "Q=high", "A", "A: the first variable<br>B: 1 + 0.2 A (1 + F) + noise"} <= set(texts), texts
    cells = [c for c in grid.children if isinstance(c[0], Plot) and c[0].name != "facet_grid_legend"]
    assert [(r, c) for _, r, c, _, _ in cells] == [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)], cells
    figures = [c[0] for c in cells]
    assert [f.title.text for f in figures] == ["H=0", "H=1", "H=2", " ", " ", " "]
    assert len({id(f.x_range) for f in figures}) == 1 and len({id(f.y_range) for f in figures}) == 1
    assert all(a.axis_label is None for f in figures for a in f.xaxis + f.yaxis)
    hidden = [[(f.xaxis[0].major_label_text_font_size == "0px", f.yaxis[0].major_label_text_font_size == "0px")
               for f in figures[3 * i:3 * i + 3]] for i in range(2)]
    assert hidden == [[(True, False), (True, True), (True, True)], [(False, False), (False, True), (False, True)]], hidden
    legends = [m for m in page.pAll.references() if isinstance(m, Legend)]
    shown = [lg for lg in legends if lg.visible]
    assert len(shown) == 1 and shown[0].name == "facet_grid_legend", [(lg.name, lg.visible) for lg in legends]
    from bokeh.core.validation.check import check_integrity
    assert [(e.code, e.name) for e in check_integrity(page.pAll.references()).error] == []


@UNIT
@pytest.mark.feature("DISPLAY.facet_grid")
@pytest.mark.parametrize("change,named", [
    ({"titel": "x"}, ["'titel'", "'title'"]),
    ({"rowHeadings": ["only one"]}, ["rowHeadings", "2"]),
    ({"columnHeadings": ["a", "b"]}, ["columnHeadings", "3"]),
    ({"cells": [[0, 1, 2], [3, 4, 9]]}, ["9", "not a figure"]),
    ({"cellTitles": [["a"], ["b"]]}, ["cellTitles", "shape"]),
    ({"legend": "right"}, ["'right'", "'top'", "'none'"])])
@STAGE_5_TODAY
def test_a_wrong_grid_is_refused_by_name(change, named):
    grid = {**GRID, **change}
    known_defect(_no_grid(grid), "FR-RI-5: the layout element facetGrid is read as tabs")
    error = _build_error(grid)
    assert error is not None and error[0] == "ValueError" and all(n in error[1] for n in named), error


# ------------------------------------------------------------------ Chromium: the geometry, the legend, the ranges
GEOMETRY = r"""() => {
  const views = []; const walk = v => { views.push(v); for (const c of (v.child_views ?? [])) walk(c); };
  for (const v of Object.values(Bokeh.index)) walk(v);
  const box = (v) => { const r = v.el.getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height}; };
  const out = {divs: [], cells: [], strip: null};
  for (const v of views) {
    if (v.model.type === 'Div') {       // the text as drawn: the union of its line boxes (the Div's own box is its grid area)
      const range = document.createRange(); range.selectNodeContents(v.el.shadowRoot ?? v.el);
      const rs = Array.from(range.getClientRects()).filter(q => q.width > 0 && q.height > 0);
      const x0 = Math.min(...rs.map(q => q.x)), y0 = Math.min(...rs.map(q => q.y));
      const ink = {x: x0, y: y0, w: Math.max(...rs.map(q => q.x + q.width)) - x0, h: Math.max(...rs.map(q => q.y + q.height)) - y0};
      out.divs.push({text: v.model.text, ...box(v), ink});
    }
    if (v.model.type === 'Figure' || v.model.type === 'Plot') {
      const b = box(v), f = v.frame.bbox;
      const entry = {name: v.model.name, title: v.model.title ? v.model.title.text : null, ...b,
                     frame: {x: b.x + f.left, y: b.y + f.top, w: f.width, h: f.height},
                     x_range: [v.model.x_range.start, v.model.x_range.end], y_range: [v.model.y_range.start, v.model.y_range.end]};
      if (v.model.name === 'facet_grid_legend') out.strip = entry; else out.cells.push(entry);
    }
  }
  const legends = window.__riH.models().filter(m => m.type === 'Legend');
  out.legends = legends.map(l => ({name: l.name, visible: l.visible,
    labels: l.items.filter(it => it.visible).flatMap(it => Array.from(it.get_labels_list_from_label_prop()).map(String))}));
  return out;
}"""

SET = r"""async (a) => {
  const w = window.__riH.models().filter(m => 'value' in m && m.js_property_callbacks && (m.name === a.name || (m.name == null && m.title === a.name)));
  if (w.length !== 1) throw Error('widget identity: ' + a.name + ': ' + w.length);
  w[0].value = a.value;
  await new Promise(resolve => setTimeout(resolve, 0));
  return true;
}"""


def _div(g, text):
    found = [d for d in g["divs"] if d["text"] == text]
    assert len(found) == 1, (text, [d["text"] for d in g["divs"]])
    return found[0]


@pytest.mark.browser_display
@pytest.mark.feature("DISPLAY.facet_grid")
@STAGE_5_TODAY
def test_browser_the_grid_geometry_one_legend_and_shared_ranges(headless_session, tmp_path):
    """Chromium: the title above all cells and the strip legend between them; the X title under the cells and centred
    on them, the footer under it; the Y title left of the cells and vertical (its drawn text more than three times
    taller than wide; the Div's own box is its grid area and says nothing); each row heading left of its row, centred
    on it and vertical; six frames of equal size, in two rows and three columns; one visible legend, listing the groups of the
    cells' rows, and following the group selection; the same ranges in every cell, also after the slider moves; no
    console error."""
    from bokeh.io import save
    from bokeh.resources import INLINE
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_display import _open_browser_page
    known_defect(_no_grid(), "FR-RI-5: the layout element facetGrid is read as tabs")
    p = tmp_path / "facet_grid.html"
    save(_from_array().pAll, filename=str(p), resources=INLINE, title="facet grid")
    ctx, page, errors = _open_browser_page(headless_session, p)
    try:
        g = page.evaluate(GEOMETRY)
        cells, strip = g["cells"], g["strip"]
        assert len(cells) == 6 and strip is not None, g
        top = min(c["y"] for c in cells)
        bottom = max(c["y"] + c["h"] for c in cells)
        left = min(c["x"] for c in cells)
        title = _div(g, "B vs A  [faceted by F]<br>selection: all")
        assert title["y"] + title["h"] <= strip["y"] + 1 and strip["y"] + strip["h"] <= top + 1, (title, strip, top)
        x_title, footer = _div(g, "A"), _div(g, "A: the first variable<br>B: 1 + 0.2 A (1 + F) + noise")
        assert x_title["y"] >= bottom - 1 and footer["y"] >= x_title["y"] + x_title["h"] - 1, (x_title, footer, bottom)
        right = max(c["x"] + c["w"] for c in cells)
        assert abs(x_title["ink"]["x"] + x_title["ink"]["w"] / 2 - (left + right) / 2) < 0.1 * (right - left), x_title
        y_title = _div(g, "B, the second variable")
        ink = y_title["ink"]
        assert y_title["x"] + y_title["w"] <= left + 1 and ink["h"] > 3 * ink["w"], y_title
        for i, heading in enumerate(["Q=low", "Q=high"]):
            h = _div(g, heading)
            row = sorted(cells, key=lambda c: c["y"])[3 * i]
            ink = h["ink"]
            assert h["x"] + h["w"] <= left + 1 and ink["h"] > 1.5 * ink["w"], h
            assert row["y"] - 1 <= ink["y"] + ink["h"] / 2 <= row["y"] + row["h"] + 1, (h, row)
        frames = [c["frame"] for c in cells]
        assert max(f["w"] for f in frames) - min(f["w"] for f in frames) <= 1, frames
        assert max(f["h"] for f in frames) - min(f["h"] for f in frames) <= 1, frames
        assert len({round(f["y"]) for f in frames}) == 2 and len({round(f["x"]) for f in frames}) == 3, frames
        assert [c["title"] for c in sorted(cells, key=lambda c: (c["y"], c["x"]))][:3] == ["H=0", "H=1", "H=2"]
        shown = [lg for lg in g["legends"] if lg["visible"]]
        assert len(shown) == 1 and shown[0]["name"] == "facet_grid_legend", g["legends"]
        assert sorted(shown[0]["labels"]) == ["a", "b", "c"], shown
        assert len({tuple(c["x_range"]) for c in cells}) == 1 and len({tuple(c["y_range"]) for c in cells}) == 1
        page.evaluate(SET, {"name": "cat", "value": ["a", "c"]})
        page.evaluate(SET, {"name": "A", "value": [1.0, 3.0]})
        page.wait_for_function("() => Bokeh.documents[0].is_idle")
        page.wait_for_timeout(300)
        g = page.evaluate(GEOMETRY)
        shown = [lg for lg in g["legends"] if lg["visible"]]
        assert sorted(shown[0]["labels"]) == ["a", "c"], shown
        assert len({tuple(c["x_range"]) for c in g["cells"]}) == 1 and len({tuple(c["y_range"]) for c in g["cells"]}) == 1
        (tmp_path / "facet_grid.png").write_bytes(page.screenshot(full_page=True))
        assert not errors, errors
    finally:
        ctx.close()


def _weighted_page(kind):
    """The compiler's case: profiles of groups given as weights (stage 1b), one series per weight, in two cells. With
    markers ("scatter+line") the legend is one entry on the column weights; lines alone ("line") have one entry per
    weight, hidden with its series."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_phase5_2_lines import _hist, _table as _wtable
    before = sa.show
    sa.show = lambda *a, **k: None
    try:
        return sa.bokehDrawSA.fromArray(
            _wtable(), None, [[["bin_center_0"], [y], {"source": "hp_1", "visualization_type": kind}]
                              for y in ("mean", "std")], [["range", ["A"]], ["multiSelect", ["weights"]]],
            parameterArray=[{"name": "weights", "value": ["w0", "w1"], "options": ["w0", "w1", "w2"]}],
            histogramArray=[_hist("weights")], widgetLayout=[[0, 1]],
            layout={"facetGrid": {"cells": [[0, 1]], "columnHeadings": ["mean", "std"], "xTitle": "A"}})
    finally:
        sa.show = before


@pytest.mark.browser_display
@pytest.mark.feature("DISPLAY.facet_grid")
@pytest.mark.parametrize("kind", ["scatter+line", "line"])
@STAGE_5_TODAY
def test_browser_the_grid_legend_follows_the_weights_shown_in_the_cells(headless_session, tmp_path, kind):
    """Chromium: series given as weights, with markers and as lines alone. The strip lists the weights shown in the cells (w0, w1), once each although
    two cells draw them; after the weights selection changes to w1, w2 it lists w1, w2: an entry hidden in the cells
    is hidden in the strip. No console error."""
    from bokeh.io import save
    from bokeh.resources import INLINE
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_display import _open_browser_page
    try:
        page_model = _weighted_page(kind)
    except KeyError as exc:
        known_defect(True, f"FR-RI-5: the layout element facetGrid is read as tabs ({exc!r})")
    known_defect(_grid_of(page_model) is None, "FR-RI-5: the layout element facetGrid is read as tabs")
    p = tmp_path / "facet_grid_weights.html"
    save(page_model.pAll, filename=str(p), resources=INLINE, title="facet grid, weights")
    ctx, page, errors = _open_browser_page(headless_session, p)
    try:
        def labels():
            g = page.evaluate(GEOMETRY)
            shown = [lg for lg in g["legends"] if lg["visible"]]
            assert len(shown) == 1 and shown[0]["name"] == "facet_grid_legend", g["legends"]
            return shown[0]["labels"]
        assert sorted(labels()) == ["w0", "w1"], labels()
        page.evaluate(SET, {"name": "weights", "value": ["w1", "w2"]})
        page.wait_for_function("() => Bokeh.documents[0].is_idle")
        page.wait_for_timeout(300)
        assert sorted(labels()) == ["w1", "w2"], labels()
        (tmp_path / "facet_grid_weights.png").write_bytes(page.screenshot(full_page=True))
        assert not errors, errors
    finally:
        ctx.close()
