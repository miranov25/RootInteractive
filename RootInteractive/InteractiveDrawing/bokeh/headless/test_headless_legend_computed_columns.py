"""PHASE_5 legend group: a legend on a column that the page computes, and the display text of a weight.

In plain words. RootInteractive draws a legend on a column for three kinds of figures:
- the profile of several weights (column "weights": one series per weight column);
- the stacked histogram (column "_source_index": one part per selected weight);
- any figure with the option legend_field.
The first two columns are computed by the page when it draws; they are in no table sent with the page. These tests
build pages with RootInteractive's public builder (bokehDrawSA.fromArray), check them with Bokeh's own validation in
Python, load them in Node with the real TypeScript models, and read the legend the way Bokeh's legend computes it
(LegendItem.get_labels_list_from_label_prop).

The two defects and the missing feature:
    LG-D1  (the register's observation O-1) a legend on a computed column makes Bokeh report error E-1001
           (BAD_COLUMN_NAME) when the page is built, and log "[bokeh] Bad column name on label: [object Object]" when it
           loads. The legend itself works. The cause: Bokeh checks the column against the downsampler that the glyph
           draws from (DownsamplerCDS.column_names in Python, its data when the page loads), and both are empty until
           the page draws. The test harness allows the message today (engine/runner.py CONSOLE_ALLOWED).
    LG-D2  a figure with the option legend_field cannot be built: UnboundLocalError for x_label in bokehTools.py
           bokehDrawArray (that branch never sets the axis titles).
    LG-F1  the legend of a profile of several weights shows the weight columns' names; a weight column's display text
           in its table's metadata (<column>.AxisTitle, the same text that axis titles use) is not used.
Each has the exact signature of the result before the fix (now a regression check that names it) and a healthy
control. Fixed in the legend group's fix commit; LG-F1 is a new, optional feature.
Author: Claude20.RI, 2026-11-20 (the review of GPT36.RI's legend request found the cause).
"""
from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def regression(condition: bool, message: str) -> None:
    """After the fix, the exact signature of a fixed defect is a regression: an ordinary failure that names it."""
    if condition:
        raise AssertionError("REGRESSION, a fixed defect is back: " + message)

UNIT = pytest.mark.headless_unit
NODE = [pytest.mark.node_display, pytest.mark.backend("node"), pytest.mark.layer("integration"),
        pytest.mark.feature("DISPLAY.legend")]
LEGEND_MESSAGE = "[bokeh] Bad column name on label: [object Object]"
TITLES = {"w0": "π±", "w1": "K±", "w2": "0.5 ≤ pT < 1.0"}          # a label need not be a column name


def _node(*marks):
    def wrap(f):
        for m in reversed(NODE + list(marks)):
            f = m(f)
        return f
    return wrap


# ------------------------------------------------------------------ data and pages (the public builder)
def _table(titles=None, n=240, seed=3) -> pd.DataFrame:
    """x, y in [0, 4); three groups; w0, w1, w2 = 1 for the rows of a group, else 0; cat = the group's letter."""
    from RootInteractive.Tools.pandaTools import initMetadata
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"x": rng.random(n) * 4, "y": rng.random(n) * 4})
    g = rng.integers(0, 3, n)
    for k in range(3):
        d[f"w{k}"] = (g == k).astype(float)
    d["cat"] = np.array(["a", "b", "c"])[g]
    initMetadata(d)
    for column, text in (titles or {}).items():
        d.meta.metaData[f"{column}.AxisTitle"] = text
    return d


def _build(kind, df):
    """The page object of one of the three legend kinds."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    if kind == "profile":
        params = [{"name": "weights", "value": ["w0", "w1"], "options": ["w0", "w1", "w2"]}]
        hists = [{"name": "hp", "variables": ["x", "y"], "nbins": [4, 4], "range": [[0, 4], [0, 4]],
                  "unbinned_projections": True, "weights": "weights"}]
        figures, widgets = [[["bin_center_0"], ["mean"], {"source": "hp_1"}]], [["multiSelect", ["weights"]]]
    elif kind == "stack":
        params = [{"name": "weightChoice", "value": ["w0", "w1"], "options": ["w0", "w1", "w2"]}]
        hists = [{"name": "hs", "variables": ["x"], "nbins": 4, "range": [0, 4], "weights": "weightChoice"}]
        figures, widgets = [[["bin_center"], ["bin_count"], {"source": "hs"}]], [["multiSelect", ["weightChoice"]]]
    elif kind == "option":
        params, hists, widgets = [], [], []
        figures = [[["x"], ["y"], {"legend_field": "cat"}]]
    elif kind == "fixed_text":
        params, hists, widgets = [], [], []
        figures = [[["x"], ["y"]]]
    else:
        raise ValueError(kind)
    before = sa.show
    sa.show = lambda *a, **kw: None
    try:
        return sa.bokehDrawSA.fromArray(df, None, figures, widgets, parameterArray=params, histogramArray=hists,
                                        layout=[[0]], widgetLayout=[list(range(len(widgets)))])
    finally:
        sa.show = before


def _page(tmp_path, kind, df, suffix=""):
    from bokeh.document import Document
    doc = Document()
    doc.add_root(_build(kind, df).pAll)
    out = tmp_path / f"legend_{kind}{suffix}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


def _build_error(kind, df):
    """None if the page builds, else (exception type, message, file:function where it was raised)."""
    try:
        _build(kind, df)
        return None
    except Exception as exc:                                    # noqa: BLE001 - the signature names the exception
        last = traceback.extract_tb(exc.__traceback__)[-1]
        return type(exc).__name__, str(exc), f"{Path(last.filename).name}:{last.name}"


def _validation_errors(kind, df):
    """Bokeh's validation errors of the page: (code, name, the type of the model it names, that model's label field)."""
    from bokeh.core.validation.check import check_integrity
    refs = _build(kind, df).pAll.references()
    by_id = {m.id: m for m in refs}
    out = []
    for e in check_integrity(refs).error:
        found = re.search(r"id='([^']+)'", e.extra or "")
        model = by_id.get(found.group(1)) if found else None
        out.append((e.code, e.name, type(model).__name__ if model is not None else None,
                    getattr(getattr(model, "label", None), "field", None)))
    return out


# ------------------------------------------------------------------ signatures (what production gives today)
def _is_lg_d1_python(errors, field) -> bool:
    """LG-D1 at build time, exact: one E-1001 error, raised by the LegendItem whose label is the field."""
    return errors == [(1001, "BAD_COLUMN_NAME", "LegendItem", field)]


def _is_lg_d1_console(console_errors, n_legends=1) -> bool:
    """LG-D1 at load time, exact: the only console errors are Bokeh's legend message, once per field legend."""
    return len(console_errors) == n_legends and all(m == LEGEND_MESSAGE for m in console_errors)


def _is_lg_d2(error) -> bool:
    """LG-D2, exact: the build raises UnboundLocalError for x_label inside bokehTools.py bokehDrawArray."""
    return (error is not None and error[0] == "UnboundLocalError" and "'x_label'" in error[1]
            and error[2] == "bokehTools.py:bokehDrawArray")


def _is_lg_f1(entries, names) -> bool:
    """LG-F1, exact: although the weight columns have display texts, the legend shows exactly their names."""
    return [e["label"] for e in entries] == list(names) and all(e["label"] == e["weight"] for e in entries)


# ------------------------------------------------------------------ Node: open a page, read its legend
LEGEND = r"""() => {
  const all = window.__riH.models();
  const items = all.filter(m => m.type === 'LegendItem' && m.label != null && m.label.field != null);
  return items.map(it => {
    const r = it.renderers[0], ds = r.data_source, field = it.label.field;
    const labels = Array.from(it.get_labels_list_from_label_prop()).map(String);
    const column = Array.from(ds.get_column(field) ?? []).map(String);
    const weights = (field === 'weights' || field === 'weights_label')
      ? Array.from(ds.get_column('weights') ?? []).map(String) : null;
    const fc = r.glyph.fill_color;
    const colour = (i) => {
      if (fc == null || fc.transform == null || fc.transform.factors == null) return fc && fc.value != null ? String(fc.value) : null;
      const v = String(ds.get_column(fc.field)[i]);
      const k = Array.from(fc.transform.factors).map(String).indexOf(v);
      return k >= 0 && k < fc.transform.palette.length ? String(fc.transform.palette[k]) : String(fc.transform.nan_color);
    };
    const entries = labels.map(l => { const i = column.indexOf(l); return {label: l, weight: weights ? weights[i] : null, colour: colour(i)}; });
    return {field, source: ds.type, labels, column, weights, entries};
  });
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


def _open_raw(session, path):
    """The harness reply as it is: no console message is allowed or hidden here."""
    r = session.request("new_doc", path=str(path))
    assert r.get("ok") and not r.get("page_errors"), r
    return r


def _eval(session, code, arg=None):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = session.request("eval", code=code, arg=arg)
    assert r.get("ok") and not r.get("page_errors"), r
    return decode(r["result"]), list(r.get("console_errors") or [])


def _legend(session):
    got, console = _eval(session, LEGEND)
    assert len(got) == 1, f"expected one legend on a column, found {len(got)}"
    return got[0], console


def _select(session, name, value):
    _eval(session, SET_WIDGET, {"name": name, "value": list(value)})


def _pairs_exactly(legend, titles):
    """Every row's label is its own weight's display text (or the weight's name without one)."""
    expected = [titles.get(w, w) for w in legend["weights"]]
    return legend["column"] == expected


# ------------------------------------------------------------------ Python only: the signatures are not vacuous
@UNIT
def test_lg_signatures_are_exact():
    item = (1001, "BAD_COLUMN_NAME", "LegendItem", "weights")
    assert _is_lg_d1_python([item], "weights")
    assert not _is_lg_d1_python([], "weights")                                    # the right answer is not the defect
    assert not _is_lg_d1_python([item] * 2, "weights")
    assert not _is_lg_d1_python([(1001, "BAD_COLUMN_NAME", "GlyphRenderer", None)], "weights")
    assert not _is_lg_d1_python([item], "_source_index")
    assert _is_lg_d1_console([LEGEND_MESSAGE])
    assert not _is_lg_d1_console([]) and not _is_lg_d1_console([LEGEND_MESSAGE, "other"])
    assert not _is_lg_d1_console([LEGEND_MESSAGE, LEGEND_MESSAGE])
    assert _is_lg_d2(("UnboundLocalError", "local variable 'x_label' referenced before assignment",
                      "bokehTools.py:bokehDrawArray"))
    assert not _is_lg_d2(None) and not _is_lg_d2(("KeyError", "'x_label'", "bokehTools.py:bokehDrawArray"))
    names = [{"label": "w0", "weight": "w0"}, {"label": "w1", "weight": "w1"}]
    assert _is_lg_f1(names, ["w0", "w1"])
    assert not _is_lg_f1([{"label": "π±", "weight": "w0"}, {"label": "w1", "weight": "w1"}], ["w0", "w1"])
    assert not _is_lg_f1(names, ["w1", "w0"])


@UNIT
def test_lg_titles_are_kept_by_the_table():
    """The display texts reach the page builder: a copied table keeps its metadata (bokehDrawArray copies it)."""
    d = _table(TITLES)
    assert d.copy().meta.metaData["w2.AxisTitle"] == "0.5 ≤ pT < 1.0"


# ------------------------------------------------------------------ Python only: LG-D1 and LG-D2 when the page is built
@UNIT
def test_lg_d1_control_a_legend_with_a_fixed_text_is_valid():
    """Healthy control: a scatter whose legend is a fixed text passes Bokeh's validation, before and after the fix."""
    assert _validation_errors("fixed_text", _table()) == []


@UNIT
@pytest.mark.parametrize("kind,field", [("profile", "weights"), ("stack", "_source_index")])
def test_lg_d1_the_legend_column_is_known_when_the_page_is_built(kind, field):
    errors = _validation_errors(kind, _table())
    regression(_is_lg_d1_python(errors, field), f"LG-D1: Bokeh reports E-1001 for the legend on '{field}'")
    assert errors == [], errors


@UNIT
def test_lg_d2_a_figure_with_the_legend_option_is_built_and_valid():
    error = _build_error("option", _table())
    regression(_is_lg_d2(error), "LG-D2: UnboundLocalError for x_label in bokehTools.py bokehDrawArray")
    assert error is None, error
    assert _validation_errors("option", _table()) == []


@UNIT
def test_lg_d1_the_harness_allows_no_legend_message():
    """After the fix no console message of the legend is allowed, so every test that opens a weighted or stacked page
    fails by name if the message comes back."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import CONSOLE_ALLOWED
    allowed = [p for p, _ in CONSOLE_ALLOWED if re.search(p, LEGEND_MESSAGE)]
    regression(len(allowed) == 1, "LG-D1: engine/runner.py CONSOLE_ALLOWED allows the legend message")
    assert allowed == [], allowed


# ------------------------------------------------------------------ Node: LG-D1 when the page loads
@_node()
@pytest.mark.parametrize("kind", ["profile", "stack"])
def test_lg_d1_no_console_error_when_the_page_loads(node_session, tmp_path, kind):
    r = _open_raw(node_session, _page(tmp_path, kind, _table(), "_load"))
    console = list(r.get("console_errors") or [])
    regression(_is_lg_d1_console(console), f"LG-D1: '{LEGEND_MESSAGE}' when the {kind} page loads")
    assert console == [], console


# ------------------------------------------------------------------ Node: the legend follows the selection (controls)
@_node()
def test_lg_control_the_profile_legend_lists_the_selected_weights(node_session, tmp_path):
    """Healthy control, before and after the fix: the legend lists the selected weights in their order, each with the
    colour its series is drawn with, and follows the weight selection."""
    _open_raw(node_session, _page(tmp_path, "profile", _table(), "_follow"))
    legend, _ = _legend(node_session)
    assert legend["labels"] == ["w0", "w1"], legend["labels"]
    assert all(e["colour"] not in (None, "None") for e in legend["entries"]), legend["entries"]
    _select(node_session, "weights", ["w2", "w0"])
    legend, _ = _legend(node_session)
    assert legend["labels"] == ["w2", "w0"], legend["labels"]
    assert len({e["colour"] for e in legend["entries"]}) == 2, legend["entries"]


@_node()
def test_lg_control_the_stack_legend_lists_the_selected_parts(node_session, tmp_path):
    """Healthy control, before and after the fix: the stacked histogram's legend lists exactly the selected parts."""
    _open_raw(node_session, _page(tmp_path, "stack", _table(), "_follow"))
    legend, _ = _legend(node_session)
    assert sorted(legend["labels"]) == ["w0", "w1"], legend["labels"]
    _select(node_session, "weightChoice", ["w1", "w2"])
    legend, _ = _legend(node_session)
    assert sorted(legend["labels"]) == ["w1", "w2"], legend["labels"]


# ------------------------------------------------------------------ Node: LG-D2, the legend option on a page
@_node()
def test_lg_d2_the_legend_option_lists_the_column_values(node_session, tmp_path):
    error = _build_error("option", _table())
    regression(_is_lg_d2(error), "LG-D2: UnboundLocalError for x_label in bokehTools.py bokehDrawArray")
    r = _open_raw(node_session, _page(tmp_path, "option", _table(), "_load"))
    assert list(r.get("console_errors") or []) == []
    legend, _ = _legend(node_session)
    assert sorted(legend["labels"]) == ["a", "b", "c"], legend["labels"]


# ------------------------------------------------------------------ Node: LG-F1, the display text of a weight
@_node()
def test_lg_f1_control_without_titles_the_legend_shows_the_column_names(node_session, tmp_path):
    """Healthy control, before and after the fix: without display texts the legend shows the weight columns' names."""
    _open_raw(node_session, _page(tmp_path, "profile", _table(), "_names"))
    legend, _ = _legend(node_session)
    assert legend["labels"] == ["w0", "w1"] and _pairs_exactly(legend, {}), legend


@_node()
@pytest.mark.parametrize("titles", [TITLES, {"w1": "K±"}], ids=["all", "only_w1"])
def test_lg_f1_the_legend_shows_the_weights_display_texts(node_session, tmp_path, titles):
    """The legend shows each weight's display text, follows the weight selection, pairs each text with its own series,
    and keeps the colours of the page without texts."""
    _open_raw(node_session, _page(tmp_path, "profile", _table(), "_plain"))
    plain, _ = _legend(node_session)
    _open_raw(node_session, _page(tmp_path, "profile", _table(titles), f"_titles_{len(titles)}"))
    legend, _ = _legend(node_session)
    regression(_is_lg_f1(legend["entries"], ["w0", "w1"]), "LG-F1: the legend shows the column names w0, w1")
    assert legend["labels"] == [titles.get(w, w) for w in ("w0", "w1")], legend["labels"]
    assert _pairs_exactly(legend, titles), legend
    assert [e["colour"] for e in legend["entries"]] == [e["colour"] for e in plain["entries"]]
    _select(node_session, "weights", ["w1", "w2"])
    legend, _ = _legend(node_session)
    assert legend["labels"] == [titles.get(w, w) for w in ("w1", "w2")], legend["labels"]
    assert _pairs_exactly(legend, titles), legend
