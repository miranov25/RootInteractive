"""PHASE_5_2 stage 1: an unknown histogram type is refused by name (BR-RI-1), and a line through the points of a
profile (FR-RI-1).

In plain words.
- BR-RI-1. A histogram figure accepts the types "points" and "bars" (1-D) and "heatmap" and "colZ" (2-D). Any other
  type is refused today with an error that does not name it (a HoverTool ValueError, or an UnboundLocalError for
  histoGlyphRenderer), or, with the tooltips switched off, the page is built and the histogram is not drawn at all.
  After the fix the page builder refuses it with a ValueError that names the type and the accepted ones.
- FR-RI-1. A figure entry with "visualization_type": "line" draws each Y column as a line through its points, in row
  order; "scatter+line" draws the markers and the line. The option "line_style" gives the dash, width, colour and
  alpha. Lines are drawn on histogram projections (profiles), whose rows are the bins in order and are never
  downsampled; a line on any other source is refused by name. A profile of several weights gets one line per weight
  series, each with only its own points, never joining two series (Bokeh 3.3.2 refuses a filtered view on a line,
  validation error 1024). The lines follow the weight selection: shown, coloured as their markers, listed in the
  legend when the figure has no markers.
- Before the fix, every line test fails with the exact signature of today: the build raises NotImplementedError
  "Visualization type not supported: line" (or "scatter+line") in bokehTools.py bokehDrawArray. That signature is a
  strict expected failure (KnownDefect) here; after the fix it is a regression that names it.
Controls (pass before and after the fix): the markers of the same profiles equal the NumPy oracle, and the accepted
histogram types still draw their glyphs.
Author: Claude10.RI, 2026-11-22.
"""
from __future__ import annotations

import json
import re
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
BR_RI_1_TODAY = pytest.mark.xfail(**STRICT, reason="BR-RI-1: an unknown histogram type is not refused by name")
FR_RI_1_TODAY = pytest.mark.xfail(**STRICT, reason="FR-RI-1: visualization_type line and scatter+line are not supported")
UNIT = pytest.mark.headless_unit
NBX, MIN_ENTRIES = 6, 3
SPARSE = {(0, 2): 1, (0, 3): 0, (1, 4): 2}            # (group, A bin): rows; every other pair has 8 rows


def _node(f):
    """The target, backend and layer marks of a Node test. The feature marks are written out on each test, because
    scripts/generate_capability_matrix.py reads only the literal form @pytest.mark.feature("ID")."""
    for m in reversed([pytest.mark.node_display, pytest.mark.backend("node"), pytest.mark.layer("integration")]):
        f = m(f)
    return f


# ------------------------------------------------------------------ data and pages (the public builder)
def _table(seed=7) -> pd.DataFrame:
    """A in [0, 6), B in (0.05, 3.95); groups 0, 1, 2 with w0, w1, w2 = 1 for the rows of a group, else 0.
    Every (group, A bin) has 8 rows, except the pairs of SPARSE: group 0 has 1 row in bin 2 and none in bin 3, group 1
    has 2 rows in bin 4. So with "entries >= 3" the unweighted profile keeps every bin, and the weighted one drops
    bins 2 and 3 of w0 (inner bins: the line joins bins 1 and 4) and bin 4 of w1."""
    from RootInteractive.Tools.pandaTools import initMetadata
    rng = np.random.default_rng(seed)
    a, g = [], []
    for k in range(3):
        for b in range(NBX):
            n = SPARSE.get((k, b), 8)
            a += list(b + 0.02 + 0.96 * rng.random(n))
            g += [k] * n
    d = pd.DataFrame({"A": np.array(a), "g": np.array(g)})
    d["B"] = 0.05 + 3.9 * rng.random(len(d))
    for k in range(3):
        d[f"w{k}"] = (d["g"] == k).astype(float)
    initMetadata(d)
    return d


def _hist(weights=None):
    h = {"name": "hp", "variables": ["A", "B"], "nbins": [NBX, 4], "range": [[0, NBX], [0, 4]],
         "unbinned_projections": True}
    if weights is not None:
        h["weights"] = weights
    return h


def _from_array(df, figures, *, weighted=False, widgets=(("range", ["A"]),), histograms=None, **kw):
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    params = [{"name": "weights", "value": ["w0", "w1"], "options": ["w0", "w1", "w2"]}] if weighted else []
    widgets = [list(w) for w in widgets] + ([["multiSelect", ["weights"]]] if weighted else [])
    if histograms is None:
        histograms = [_hist("weights" if weighted else None)]
    before = sa.show
    sa.show = lambda *a, **k: None
    try:
        return sa.bokehDrawSA.fromArray(df, None, list(figures), widgets, parameterArray=params,
                                        histogramArray=list(histograms), layout=[list(range(len(figures)))],
                                        widgetLayout=[list(range(len(widgets)))], **kw)
    finally:
        sa.show = before


def _build_error(df, figures, **kw):
    """None if the page builds, else (exception type, message, file:function where it was raised)."""
    try:
        _from_array(df, figures, **kw)
        return None
    except Exception as exc:                                    # noqa: BLE001 - the signature names the exception
        last = traceback.extract_tb(exc.__traceback__)[-1]
        return type(exc).__name__, str(exc), f"{Path(last.filename).name}:{last.name}"


def _page(tmp_path, tag, df, figures, **kw):
    from bokeh.document import Document
    doc = Document()
    doc.add_root(_from_array(df, figures, **kw).pAll)
    out = tmp_path / f"lines_{tag}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


def _validation_errors(page):
    from bokeh.core.validation.check import check_integrity
    found = check_integrity(page.pAll.references())
    return [(e.code, e.name) for e in found.error]


def _glyphs(page):
    """The glyph type names of the page's renderers, sorted."""
    from bokeh.models import GlyphRenderer
    return sorted(type(m.glyph).__name__ for m in page.pAll.references() if isinstance(m, GlyphRenderer))


# ------------------------------------------------------------------ figures used below
LINE = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "line"}]
LINE_FILTERED = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "line", "filter": "entries >= 3"}]
BOTH_FILTERED = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "scatter+line",
                                              "filter": "entries >= 3"}]
MARKERS_FILTERED = [["bin_center_0"], ["mean"], {"source": "hp_1", "filter": "entries >= 3"}]


# ------------------------------------------------------------------ signatures (what production gives today)
def _is_missing_line(error, kind="line") -> bool:
    """FR-RI-1 today, exact: NotImplementedError 'Visualization type not supported: <kind>' in bokehDrawArray."""
    return error == ("NotImplementedError", f"Visualization type not supported: {kind}", "bokehTools.py:bokehDrawArray")


def _is_br_ri_1(error) -> bool:
    """BR-RI-1 today, exact: the unknown type is not named. Either the HoverTool refuses the missing renderer
    (ValueError from Bokeh's property check), or histoGlyphRenderer is unbound in the 2-D helper."""
    if error is None:
        return False
    name, message, where = error
    hover = name == "ValueError" and re.search(r"HoverTool\(id='[^']*', \.\.\.\)\.renderers: .* got \[None\]$", message) is not None
    unbound = name == "UnboundLocalError" and "'histoGlyphRenderer'" in message and where == "bokehTools.py:addHisto2dGlyph"
    return hover or unbound


def _line_or_defect(error, kind="line"):
    """At the snapshot the line cannot be built: the strict expected failure. After the fix it builds."""
    known_defect(_is_missing_line(error, kind), f"FR-RI-1: NotImplementedError 'Visualization type not supported: {kind}'")
    assert error is None, error


# ------------------------------------------------------------------ Python only: the signatures are not vacuous
@UNIT
def test_line_signatures_are_exact():
    today = ("NotImplementedError", "Visualization type not supported: line", "bokehTools.py:bokehDrawArray")
    assert _is_missing_line(today) and not _is_missing_line(None)
    assert not _is_missing_line(("NotImplementedError", "Visualization type not supported: band", today[2]))
    assert _is_missing_line(("NotImplementedError", "Visualization type not supported: scatter+line", today[2]), "scatter+line")
    hover = ("ValueError", "failed to validate HoverTool(id='p1059', ...).renderers: expected an element of either Auto "
             "or List(Instance(DataRenderer)), got [None]", "bases.py:prepare_value")
    unbound = ("UnboundLocalError", "local variable 'histoGlyphRenderer' referenced before assignment",
               "bokehTools.py:addHisto2dGlyph")
    assert _is_br_ri_1(hover) and _is_br_ri_1(unbound) and not _is_br_ri_1(None)
    assert not _is_br_ri_1(("ValueError", "Histogram figure: visualization_type 'lines' is not supported", "x"))


@UNIT
@pytest.mark.feature("DISPLAY.line")
def test_control_the_table_has_the_sparse_bins_it_claims():
    d = _table()
    counts = {(k, b): int(((d.g == k) & (np.floor(d.A) == b)).sum()) for k in range(3) for b in range(NBX)}
    assert counts == {(k, b): SPARSE.get((k, b), 8) for k in range(3) for b in range(NBX)}


# ------------------------------------------------------------------ Python only: BR-RI-1
HISTO_1D = {"name": "h1", "variables": ["A"], "nbins": NBX, "range": [0, NBX]}
HISTO_2D = {"name": "h2", "variables": ["A", "B"], "nbins": [NBX, 4], "range": [[0, NBX], [0, 4]]}


def _histo_figure(name, kind):
    return [["A"], [name], {"visualization_type": kind}]


@UNIT
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.parametrize("histogram,kind,accepted", [
    (HISTO_1D, "lines", ("points", "bars")), (HISTO_1D, "line", ("points", "bars")),
    (HISTO_2D, "points", ("heatmap", "colZ")), (HISTO_2D, "contour", ("heatmap", "colZ"))])
@pytest.mark.parametrize("tooltips", ["default", "none"])
@BR_RI_1_TODAY
def test_br_ri_1_an_unknown_histogram_type_is_refused_by_name(histogram, kind, accepted, tooltips):
    if tooltips == "none":
        histogram = dict(histogram, tooltips=None)            # the histogram's own tooltips switched off
    figure = _histo_figure(histogram["name"], kind)
    error = _build_error(_table(), [figure], histograms=[histogram], widgets=())
    if tooltips == "none":
        # today the page is built and the histogram is silently not drawn; that is the defect's signature here
        known_defect(error is None, f"BR-RI-1: the unknown type '{kind}' builds a page without the histogram")
    else:
        known_defect(_is_br_ri_1(error), f"BR-RI-1: the unknown type '{kind}' is refused without its name: {error}")
    assert error is not None and error[0] == "ValueError", error
    assert f"'{kind}'" in error[1] and all(f"'{a}'" in error[1] for a in accepted), error


@UNIT
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.parametrize("histogram,kind,glyph", [
    (HISTO_1D, "points", "Scatter"), (HISTO_1D, "bars", "Quad"),
    (HISTO_2D, "heatmap", "Quad"), (HISTO_2D, "colZ", "Scatter")])
def test_control_br_ri_1_the_accepted_histogram_types_draw_their_glyph(histogram, kind, glyph):
    page = _from_array(_table(), [_histo_figure(histogram["name"], kind)], histograms=[histogram], widgets=())
    assert glyph in _glyphs(page), _glyphs(page)
    assert _validation_errors(page) == []


# ------------------------------------------------------------------ Python only: FR-RI-1 when the page is built
@UNIT
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.parametrize("weighted", [False, True])
@FR_RI_1_TODAY
def test_a_line_is_built_with_line_renderers_and_is_valid(weighted):
    """One Line per Y column; one per weight option for a profile of several weights; no marker; Bokeh's validation
    (error 1024 included) finds nothing."""
    error = _build_error(_table(), [LINE_FILTERED], weighted=weighted)
    _line_or_defect(error)
    page = _from_array(_table(), [LINE_FILTERED], weighted=weighted)
    assert _glyphs(page) == ["Line"] * (3 if weighted else 1), _glyphs(page)
    assert _validation_errors(page) == []


@UNIT
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.parametrize("weighted", [False, True])
@FR_RI_1_TODAY
def test_markers_and_line_are_built_together(weighted):
    error = _build_error(_table(), [BOTH_FILTERED], weighted=weighted)
    _line_or_defect(error, "scatter+line")
    page = _from_array(_table(), [BOTH_FILTERED], weighted=weighted)
    assert _glyphs(page) == sorted(["Line"] * (3 if weighted else 1) + ["Scatter"]), _glyphs(page)
    assert _validation_errors(page) == []


@UNIT
@pytest.mark.feature("DISPLAY.line")
@FR_RI_1_TODAY
def test_the_line_style_is_given_to_the_line():
    from bokeh.models import GlyphRenderer, Line
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "line",
                                           "line_style": {"dash": "dashed", "width": 3, "color": "red", "alpha": 0.5}}]
    _line_or_defect(_build_error(_table(), [figure]))
    lines = [m.glyph for m in _from_array(_table(), [figure]).pAll.references()
             if isinstance(m, GlyphRenderer) and isinstance(m.glyph, Line)]
    assert len(lines) == 1
    dashed = Line(line_dash="dashed").line_dash                       # Bokeh stores a named dash as its pattern
    assert (lines[0].line_dash, lines[0].line_width, lines[0].line_color, lines[0].line_alpha) == (dashed, 3, "red", 0.5)


@UNIT
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.parametrize("style,named", [
    ({"dash": "dash-dot"}, ["line_style", "'dash-dot'", "'solid'", "'dashed'", "'dotted'", "'dotdash'", "'dashdot'"]),
    ({"dahs": "dashed"}, ["'dahs'", "'dash'", "'width'", "'color'", "'alpha'"]),
    ({"width": -1}, ["line_style", "width", "-1"]),
    ({"alpha": 2}, ["line_style", "alpha", "2"])])
@FR_RI_1_TODAY
def test_a_wrong_line_style_is_refused_by_name(style, named):
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "line", "line_style": style}]
    error = _build_error(_table(), [figure])
    known_defect(_is_missing_line(error), "FR-RI-1: NotImplementedError 'Visualization type not supported: line'")
    assert error is not None and error[0] == "ValueError", error
    assert all(n in error[1] for n in named), error


@UNIT
@pytest.mark.feature("DISPLAY.line")
@FR_RI_1_TODAY
def test_a_line_on_a_table_is_refused_by_name():
    """The rows of a table are downsampled when drawn (1000 by default), so a line through them would silently skip
    rows: refused, naming the source and saying why."""
    error = _build_error(_table(), [[["A"], ["B"], {"visualization_type": "line"}]], widgets=())
    known_defect(_is_missing_line(error), "FR-RI-1: NotImplementedError 'Visualization type not supported: line'")
    assert error is not None and error[0] == "ValueError", error
    assert "line" in error[1] and "projection" in error[1], error


@UNIT
@pytest.mark.feature("DISPLAY.line")
def test_control_an_unknown_figure_type_is_refused_by_name():
    error = _build_error(_table(), [[["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "lien"}]])
    assert error is not None and error[0] in ("NotImplementedError", "ValueError") and "lien" in error[1], error


# ------------------------------------------------------------------ Node: read the page
STATE = r"""() => {
  const all = window.__riH.models();
  const scalar = (c) => (c != null && typeof c === 'object' && 'value' in c) ? c.value : c;
  const nums = (ds, f) => f == null ? null : Array.from(ds.get_column(f) ?? []).map(Number);
  const strs = (ds, f) => { try { const c = ds.get_column(f); return c == null ? null : Array.from(c).map(String); } catch (e) { return null; } };
  const renderers = all.filter(m => m.type === 'GlyphRenderer').map(r => {
    const g = r.glyph, ds = r.data_source;
    const out = {glyph: g.type, visible: r.visible, source: ds.name, x: nums(ds, g.x && g.x.field), y: nums(ds, g.y && g.y.field),
                 weights: strs(ds, 'weights'), id: r.id};
    if (g.type === 'Line') Object.assign(out, {color: String(scalar(g.line_color)), dash: Array.from(scalar(g.line_dash) ?? []),
                                               width: Number(scalar(g.line_width))});
    if (g.type === 'Scatter') {
      const fc = g.fill_color;
      out.palette = (fc && fc.transform && fc.transform.factors) ? Object.fromEntries(Array.from(fc.transform.factors).map((f, k) =>
        [String(f), k < fc.transform.palette.length ? String(fc.transform.palette[k]) : String(fc.transform.nan_color)])) : null;
    }
    return out;
  });
  const profile = all.find(m => m.type === 'HistoNdProfile' && m.name === 'hp_1');
  const columns = {};
  for (const c of ['bin_center_0', 'mean', 'entries']) columns[c] = nums(profile, c);
  columns.weights = strs(profile, 'weights');
  const items = all.filter(m => m.type === 'LegendItem').map(it => ({
    label: it.label == null ? null : (it.label.value != null ? String(it.label.value) : (it.label.field != null ? '@' + it.label.field : String(it.label))),
    visible: it.visible, renderers: it.renderers.map(r => r.id)}));
  return {renderers, columns, items};
}"""

SET_WIDGET = r"""async (a) => {
  const all = window.__riH.models();
  const possible = all.filter(w => 'value' in w && w.js_property_callbacks && (w.name === a.name || (w.name == null && w.title === a.name)));
  if (possible.length !== 1) throw Error('widget identity: ' + a.name + ': ' + possible.length);
  const w = possible[0], cbs = (w.js_property_callbacks || {})['change:value'] || [];
  const pending = [], methods = [];
  for (const cb of cbs) {
    const original = cb.execute;
    methods.push([cb, original]);
    cb.execute = function (...args) { const p = Promise.resolve(original.apply(this, args)); pending.push(p); return p; };
  }
  try {
    w.value = a.value;
    const done = await Promise.allSettled(pending);
    if (done.some(x => x.status === 'rejected')) throw Error('callback rejected');
    await new Promise(resolve => setTimeout(resolve, 0));
    return Array.from(w.value);
  } finally { for (const [cb, original] of methods) cb.execute = original; }
}"""

WIDGET_VALUE = r"""(a) => { const w = window.__riH.models().filter(m => m.name === a.name && 'value' in m); return Array.from(w[0].value); }"""


def _open(session, path):
    r = session.request("new_doc", path=str(path))
    assert r.get("ok") and not r.get("page_errors"), r
    assert not r.get("console_errors"), r.get("console_errors")
    return r


def _eval(session, code, arg=None):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = session.request("eval", code=code, arg=arg)
    assert r.get("ok") and not r.get("page_errors"), r
    assert not r.get("console_errors"), r.get("console_errors")
    return decode(r["result"])


def _state(session):
    return _eval(session, STATE)


def _set(session, name, value):
    _eval(session, SET_WIDGET, {"name": name, "value": list(value)})


def _page_or_defect(tmp_path, tag, df, figure, kind="line", **kw):
    """Build the page; at the snapshot the line's NotImplementedError is the strict expected failure."""
    _line_or_defect(_build_error(df, [figure], **kw), kind)
    return _page(tmp_path, tag, df, [figure], **kw)


# ------------------------------------------------------------------ the NumPy oracle
def _oracle(df, lo, hi, weight=None):
    """Per A bin of the rows with lo <= A <= hi (and weight 1): (bin centre, number of rows, mean of B)."""
    rows = df[(df.A >= lo) & (df.A <= hi)]
    if weight is not None:
        rows = rows[rows[weight] == 1]
    b = np.floor(rows.A.to_numpy()).astype(int)
    out = []
    for k in range(NBX):
        sel = rows.B.to_numpy()[b == k]
        out.append((k + 0.5, len(sel), float(sel.mean()) if len(sel) else float("nan")))
    return out


def _kept(oracle):
    """The oracle's points that pass "entries >= 3", in bin order."""
    return [(x, m) for x, n, m in oracle if n >= MIN_ENTRIES]


def _points(r):
    return list(zip(r["x"], r["y"]))


def _close(got, expected):
    assert len(got) == len(expected), (got, expected)
    for (gx, gy), (ex, ey) in zip(got, expected):
        assert gx == pytest.approx(ex, abs=1e-9) and gy == pytest.approx(ey, rel=1e-9, abs=1e-12), (got, expected)


def _profile_rows(state, weight=None):
    """The projection's own rows that pass "entries >= 3" (of one weight series), in row order."""
    c = state["columns"]
    keep = [i for i, n in enumerate(c["entries"]) if n >= MIN_ENTRIES and (weight is None or c["weights"][i] == weight)]
    return [(c["bin_center_0"][i], c["mean"][i]) for i in keep]


def _of(state, glyph):
    return [r for r in state["renderers"] if r["glyph"] == glyph]


# ------------------------------------------------------------------ Node: controls (markers, before and after the fix)
@_node
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.feature("HIST.profile")
def test_control_the_markers_of_the_profile_equal_the_numpy_oracle(node_session, tmp_path):
    """Healthy control: the oracle of this file is the profile RootInteractive draws today, before and after a slider
    cuts A."""
    df = _table()
    _open(node_session, _page(tmp_path, "markers", df, [MARKERS_FILTERED]))
    lo, hi = _eval(node_session, WIDGET_VALUE, {"name": "A"})
    s = _state(node_session)
    (markers,) = _of(s, "Scatter")
    _close(_points(markers), _kept(_oracle(df, lo, hi)))
    _set(node_session, "A", [1.5, 4.5])
    s = _state(node_session)
    _close(_points(_of(s, "Scatter")[0]), _kept(_oracle(df, 1.5, 4.5)))


@_node
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.feature("HIST.profile")
def test_control_the_markers_of_each_weight_equal_the_numpy_oracle(node_session, tmp_path):
    df = _table()
    _open(node_session, _page(tmp_path, "markers_w", df, [MARKERS_FILTERED], weighted=True))
    lo, hi = _eval(node_session, WIDGET_VALUE, {"name": "A"})
    (markers,) = _of(_state(node_session), "Scatter")
    for w in ("w0", "w1"):
        got = [p for p, ww in zip(_points(markers), markers["weights"]) if ww == w]
        _close(got, _kept(_oracle(df, lo, hi, w)))


# ------------------------------------------------------------------ Node: FR-RI-1
@_node
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.feature("HIST.profile")
@FR_RI_1_TODAY
def test_the_line_points_equal_the_projection_and_the_numpy_oracle(node_session, tmp_path):
    """The line goes through the bins that pass the filter, in bin order, equal to the projection's own rows and to
    the NumPy oracle, before and after the slider cuts A."""
    df = _table()
    _open(node_session, _page_or_defect(tmp_path, "line", df, LINE_FILTERED))
    lo, hi = _eval(node_session, WIDGET_VALUE, {"name": "A"})
    s = _state(node_session)
    (line,) = _of(s, "Line")
    assert _of(s, "Scatter") == []
    _close(_points(line), _profile_rows(s))
    _close(_points(line), _kept(_oracle(df, lo, hi)))
    _set(node_session, "A", [1.5, 4.5])
    s = _state(node_session)
    (line,) = _of(s, "Line")
    _close(_points(line), _profile_rows(s))
    _close(_points(line), _kept(_oracle(df, 1.5, 4.5)))
    assert [x for x, _ in _points(line)] == [1.5, 2.5, 3.5, 4.5]


@_node
@pytest.mark.feature("DISPLAY.line")
@FR_RI_1_TODAY
def test_a_line_without_a_filter_keeps_the_nan_of_an_empty_bin(node_session, tmp_path):
    """Without a filter the line has every bin, in order; the empty bin 3 of w0 is NaN, where Bokeh breaks the line
    into two segments (the drawing is checked in Chromium)."""
    df = _table()
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "line"}]
    _open(node_session, _page_or_defect(tmp_path, "line_nan", df, figure, weighted=True))
    s = _state(node_session)
    lines = {r["weights"][0]: r for r in _of(s, "Line") if r["visible"]}
    assert sorted(lines) == ["w0", "w1"], lines
    y = lines["w0"]["y"]
    assert len(y) == NBX and np.isnan(y[3]) and not np.isnan(y[2]) and not any(np.isnan(v) for v in y[:3] + y[4:]), y


@_node
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.feature("HIST.profile")
@FR_RI_1_TODAY
def test_one_line_per_weight_series_with_only_its_own_points(node_session, tmp_path):
    """Two weights selected: two shown lines, each through only its own series' bins (w0 joins bins 1 and 4, w1
    skips bin 4), equal to the projection and to the oracle; w2's line is hidden. The weight selection is followed."""
    df = _table()
    _open(node_session, _page_or_defect(tmp_path, "line_w", df, LINE_FILTERED, weighted=True))
    lo, hi = _eval(node_session, WIDGET_VALUE, {"name": "A"})
    s = _state(node_session)
    lines = _of(s, "Line")
    assert len(lines) == 3
    shown = {r["weights"][0]: r for r in lines if r["visible"]}
    assert sorted(shown) == ["w0", "w1"], shown
    for w, r in shown.items():
        assert set(r["weights"]) == {w}, r["weights"]                 # never two series in one line
        _close(_points(r), _profile_rows(s, w))
        _close(_points(r), _kept(_oracle(df, lo, hi, w)))
    assert [x for x, _ in _points(shown["w0"])] == [0.5, 1.5, 4.5, 5.5]
    assert [x for x, _ in _points(shown["w1"])] == [0.5, 1.5, 2.5, 3.5, 5.5]
    _set(node_session, "weights", ["w2", "w0"])
    _set(node_session, "A", [0.0, 3.5])
    s = _state(node_session)
    shown = {r["weights"][0]: r for r in _of(s, "Line") if r["visible"] and r["weights"]}
    assert sorted(shown) == ["w0", "w2"], shown
    for w, r in shown.items():
        assert set(r["weights"]) == {w}
        _close(_points(r), _kept(_oracle(df, 0.0, 3.5, w)))


@_node
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.feature("HIST.profile")
@FR_RI_1_TODAY
def test_each_line_has_the_colour_of_its_markers_and_follows_the_selection(node_session, tmp_path):
    df = _table()
    _open(node_session, _page_or_defect(tmp_path, "both_w", df, BOTH_FILTERED, "scatter+line", weighted=True))
    for selection in (["w0", "w1"], ["w2", "w0"], ["w1"]):
        if selection != ["w0", "w1"]:
            _set(node_session, "weights", selection)
        s = _state(node_session)
        (markers,) = _of(s, "Scatter")
        shown = {r["weights"][0]: r for r in _of(s, "Line") if r["visible"]}
        assert sorted(shown) == sorted(selection), (selection, shown)
        for w, r in shown.items():
            assert r["color"] == markers["palette"][w], (selection, w, r["color"], markers["palette"])
        assert len({r["color"] for r in shown.values()}) == len(selection)
        # with markers, the markers' legend stands: no legend item of a line
        line_ids = {r["id"] for r in _of(s, "Line")}
        assert not any(set(it["renderers"]) & line_ids for it in s["items"]), s["items"]


@_node
@pytest.mark.feature("DISPLAY.line")
@pytest.mark.feature("DISPLAY.legend")
@FR_RI_1_TODAY
def test_a_line_only_figure_lists_the_shown_series_in_its_legend(node_session, tmp_path):
    df = _table()
    _open(node_session, _page_or_defect(tmp_path, "legend_w", df, LINE_FILTERED, weighted=True))
    for selection in (["w0", "w1"], ["w2"]):
        if selection != ["w0", "w1"]:
            _set(node_session, "weights", selection)
        s = _state(node_session)
        by_id = {r["id"]: r for r in _of(s, "Line")}
        shown = sorted(it["label"] for it in s["items"] if it["visible"] and any(by_id[i]["visible"] for i in it["renderers"]))
        hidden_shown = [it for it in s["items"] if it["visible"] and not any(by_id[i]["visible"] for i in it["renderers"])]
        assert shown == sorted(selection), (selection, s["items"])
        assert hidden_shown == [], hidden_shown


@_node
@pytest.mark.feature("DISPLAY.line")
@FR_RI_1_TODAY
def test_markers_line_and_error_bars_draw_the_same_points(node_session, tmp_path):
    """Markers, a styled line and error bars on one figure: the line goes through the markers, the error bars are
    drawn on the same rows, and the style is the one asked for (Bokeh's "dotted" is the dash pattern [2, 4])."""
    df = _table()
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "scatter+line", "filter": "entries >= 3",
                                           "errY": "std", "line_style": {"dash": "dotted", "width": 2, "color": "black"}}]
    _open(node_session, _page_or_defect(tmp_path, "styled", df, figure, "scatter+line"))
    s = _state(node_session)
    (line,), (markers,) = _of(s, "Line"), _of(s, "Scatter")
    bars = [r for r in s["renderers"] if r["glyph"] in ("HBar", "Quad", "VBar")]
    assert _points(line) == _points(markers) and len(_points(line)) == NBX
    assert len(bars) == 1 and bars[0]["source"] == markers["source"], bars
    assert (line["color"], line["width"]) == ("black", 2) and line["dash"] == [2, 4], line


# ------------------------------------------------------------------ Chromium: the line as drawn
def _browser(headless_session, tmp_path, tag, df, figure, kind="line", **kw):
    """Save the page as a self-contained HTML file and open it in Chromium (the route of test_headless_display.py)."""
    from bokeh.io import save
    from bokeh.resources import INLINE
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_display import _open_browser_page
    _line_or_defect(_build_error(df, [figure], **kw), kind)
    p = tmp_path / f"lines_{tag}.html"
    save(_from_array(df, [figure], **kw).pAll, filename=str(p), resources=INLINE, title=tag)
    return _open_browser_page(headless_session, p)


def _red_runs(png: bytes):
    """The runs of neighbouring pixel columns of the screenshot that hold a strongly red pixel (the line)."""
    import io
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    red = (a[..., 0] > 200) & (a[..., 1] < 90) & (a[..., 2] < 90)
    columns = np.flatnonzero(red.any(axis=0))
    if len(columns) == 0:
        return [], 0
    breaks = np.flatnonzero(np.diff(columns) > 1)
    starts = np.concatenate([[columns[0]], columns[breaks + 1]])
    ends = np.concatenate([columns[breaks], [columns[-1]]])
    return list(zip(starts.tolist(), ends.tolist())), int(red.sum())


@pytest.mark.browser_display
@pytest.mark.feature("DISPLAY.line")
@FR_RI_1_TODAY
def test_browser_a_nan_breaks_the_drawn_line_into_two_segments(headless_session, tmp_path):
    """Only w0 selected, no filter: its bins 0 to 5 with bin 3 empty (NaN). Chromium draws the solid red line in two
    pieces, bins 0 to 2 and bins 4 to 5, and nothing else red (the legend, whose sample is red, is hidden for the
    count); with the filter "entries >= 3" (bins 2 and 3 dropped) the line joins bins 1 and 4: one piece. The
    screenshots are kept in the test's folder."""
    df = _table()
    style = {"dash": "solid", "width": 3, "color": "red"}
    for tag, extra, pieces in (("nan", {}, 2), ("joined", {"filter": "entries >= 3"}, 1)):
        figure = [["bin_center_0"], ["mean"], {"source": "hp_1", "visualization_type": "line", "line_style": style, **extra}]
        ctx, page, errors = _browser(headless_session, tmp_path, tag, df, figure, weighted=True)
        try:
            page.evaluate("""() => { const w = window.__riH.models().find(m => 'value' in m && m.js_property_callbacks &&
                                       (m.name === 'weights' || (m.name == null && m.title === 'weights')));
                                     w.value = ['w0'];
                                     for (const m of window.__riH.models()) if (m.type === 'Legend') m.visible = false; }""")
            page.wait_for_function("() => Bokeh.documents[0].is_idle")
            page.wait_for_timeout(200)
            canvas = page.locator("canvas").first
            png = canvas.screenshot()
            (tmp_path / f"lines_{tag}.png").write_bytes(png)
            runs, n_red = _red_runs(png)
            assert len(runs) == pieces and n_red > 100, (tag, runs, n_red)
            assert not errors, errors
        finally:
            ctx.close()
