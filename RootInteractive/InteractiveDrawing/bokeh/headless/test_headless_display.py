"""PHASE_4_6 C2 — widgets and display: independent, test-only review candidate.

For GPT27.RI's widget team and Claude20.RI's *later* integration on alma2.
No RootInteractive production classes are modified. Source reference: RI 2026-11-09
LLMBUNDLE plus A3 integration diff at code 175687e (not an updated 11-Nov HEAD).

Node exercises live deserialized production Bokeh models and numerical oracles.
Chromium is reserved for physical text input, a tab click, and display updates.
The prescribed pinned Bokeh 3.3.2/Node/jsdom/Chromium runtime is REQUIRED before
claiming any integration PASS. Do not mark an unverified failure as XFAIL.

Integration (Claude20.RI, on 96860e5; GPT27.RI's candidate, revised v0.1a by GPT26.RI, approved for integration).
Run on the pinned toolchain: 13 of 20 cases passed as delivered; every failure traced to a product defect, none to the
tests. Defects, each with an exact signature, a strict expected-failure partner and a healthy control:
  TQ-D1  a text query does not get the page's parameters: its function receives the parameter list of the LAST ALIAS
         built (bokehTools.py passes the variable `customJsArgList` left over from the alias loop), with the values
         at build time, never updated. Without such an alias the query throws "ReferenceError: cut is not defined"
         (logged) and every row stays selected; with it, the query computes once and ignores later changes;
         Fixed in PHASE_5 Group 4: the query gets every parameter of the page and follows its changes;
  TQ-D2  when a query's text becomes empty (cleared, or a toggleable query enabled while its text area is empty), its
         function is dropped (CustomJSNAryFunction.update_func), the previous selection stays, and the next query
         logs "TypeError: Function.prototype.apply was called on null" once (the text area's callback sets `func`,
         which recomputes the column before update_func builds the new function); the result is then right.
         Fixed in PHASE_5 Group 4: an empty text is no condition ("return true"), every row is selected.
         For both, the signature tests are removed, the strict tests are ordinary tests, the signatures REGRESSIONS;
  CS-D1  an axis label's model value (ConcatenatedString.value) is undefined until its parameter first changes
         (#is_dirty starts false); the axis text itself is right, set by Python at build.
         Fixed in PHASE_5 Group 1: the strict mark is off, the signature test is removed, the exact signature is a
         regression (an ordinary failure), and the value is checked at load, in Node and in Chromium;
  DS-D1  after the selection shrinks, the downsampler's columns keep the old entries at the end (DownsamplerCDS
         get_column grows its arrays but never shrinks them): get_column(c).length > get_length(); the first
         get_length() entries are right and Bokeh paints get_length() rows (A4 observation O-A4-2, now a test);
  WG-D1  (A3, same signature) also for a toggleable TEXT QUERY alone on its table - the form the standard dashboard
         template (bokehInteractiveTemplate.py) uses three times.
  DS-D1 and WG-D1 are fixed in PHASE_5 Batch 1: their strict marks are off and their signature tests removed; their
  exact signatures are now regressions (ordinary failures).
Changes: the defect tests and helpers below; GPT's tests that hit a defect are its strict partners; the Chromium
axis-label tests check the axis text at load and the full state after the real input (the model value at load is
CS-D1, tested in Node). Targets node_display (Node) and browser_display (Chromium).
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_widgets import WG_D1, _load_failure


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect (integration, Claude20.RI)."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


STRICT = dict(strict=True, raises=KnownDefect)


def regression(condition: bool, message: str) -> None:
    """PHASE_5 Batch 1: the exact signature of a fixed defect is a regression - an ordinary failure."""
    if condition:
        raise AssertionError("REGRESSION, a fixed defect is back: " + message)


TQ_D1_ERROR = "ReferenceError: cut is not defined"
# The end of V8's message depends on the Node version ("which is null and not a function" in Node 22, other wording
# in Node 18 on alma2), so the signature uses the part every version shares, plus the function that raises it.
TQ_D2_ERROR = "TypeError: Function.prototype.apply was called on null"


def _one_tq_d2_message(messages) -> bool:
    return len(messages) == 1 and messages[0].startswith(TQ_D2_ERROR)

N = 12
X = np.array([0.25, .75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, .5, 1.5, 2.5, 3.5])
Y = np.array([.25, 1.25, 2.25, .75, 1.75, 2.5, 1.5, .5, 2.25, .25, 1.25, 2.75])
GATE = np.array([-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., -1.25, .75, 1.25])
A = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2], dtype=np.int32)
B = np.array([0, 2, 1, 3, 0, 1, 2, 0, 1, 3, 2, 1], dtype=np.int32)
W = np.array([1., 2., 1.5, .5, 3., 2.5, 4., 1.25, 2.25, 3.5, .75, 1.75])
BINS = (4, 3)
RANGES = ((0., 4.), (0., 3.))
HIST_PROBES = [["h", "bin_count", ["HistoNdCDS"]]]
# PHASE_5 Group 5 (G4-P1-A): parameter names that cannot be variables of strict-mode JavaScript
ODD_NAMES = ["default", "static", "eval", "class", "n-bins"]
# PHASE_5 Group 5d (review of group 5c, P1-G5c-01): column names that cannot be variables of strict-mode JavaScript.
# The page builder gives a query only the columns with a plain name, so on a page they are words of JavaScript.
ODD_COLUMNS = ["default", "static", "let", "arguments"]
NON_ASCII = "\u03c3cut"   # sigma-cut: not plain ASCII (Group 5a, F1)
CLASH_CUT = np.array([0., 5.] * (N // 2))   # the column named like the parameter `cut` (G4-POL-1, decision N1)


def frame():
    return pd.DataFrame(dict(x=X.copy(), y=Y.copy(), gate=GATE.copy(),
                             A=A.copy(), B=B.copy(), w=W.copy()))


def query_oracle(expression, cut=None):
    """The intended query semantics, independent of RI and its JS compiler."""
    if expression == "return true":
        return np.ones(N, dtype=bool)
    if expression == "return A > 1":
        return A > 1
    if expression == "return A > 1 && B < 2":
        return (A > 1) & (B < 2)
    if expression == "return A > cut":
        assert cut is not None
        return A > cut
    raise ValueError("unsupported oracle expression " + expression)


def gate_oracle(lo=-2., hi=2.):
    return np.isfinite(GATE) & (GATE >= lo) & (GATE <= hi)


def weighted_bins(df, mask, bins=BINS, ranges=RANGES):
    mask = np.asarray(mask, dtype=bool)
    assert mask.shape == (len(df),)
    raw, _ = np.histogramdd(df.loc[mask, ["x", "y"]].to_numpy(),
                            bins=list(bins), range=[list(r) for r in ranges],
                            weights=df.loc[mask, "w"].to_numpy())
    return raw.ravel(order="F")  # RootInteractive x-fastest order, not C order


def assert_histogram(df, state, mask):
    expected = np.asarray(mask, dtype=bool)
    actual = np.asarray(state["mask"], dtype=bool)
    np.testing.assert_array_equal(actual, expected, err_msg="textQuery row-mask mismatch")
    np.testing.assert_allclose(np.asarray(state["bins"], dtype=float),
                               weighted_bins(df, expected), atol=1e-12, rtol=1e-12,
                               err_msg="textQuery weighted histogram mismatch")


def assert_sample(df, state, selected, limit):
    """NO pseudo-random hash reimplementation. All properties independently checkable."""
    idx = np.asarray(state["index"], dtype=np.int64)
    mask = np.asarray(selected, dtype=bool)
    assert idx.ndim == 1
    assert len(idx) == min(int(limit), int(mask.sum())), (idx.tolist(), limit, int(mask.sum()))
    assert len(idx) == len(np.unique(idx)), "downsampler repeated a source row"
    assert np.all(np.diff(idx) > 0), "downsampler indices must be strictly increasing"
    assert np.all((0 <= idx) & (idx < len(df)))
    assert np.all(mask[idx]), "downsampler included a filtered-out row"
    for key in ("x", "y"):
        np.testing.assert_array_equal(np.asarray(state[key]), df[key].to_numpy()[idx],
                                      err_msg=f"downsampler {key} lost source correspondence")
    if limit >= int(mask.sum()):
        np.testing.assert_array_equal(idx, np.flatnonzero(mask),
                                      err_msg="nPointRender >= eligible must draw every eligible row")


def assert_tab_bins(observed, expected):
    np.testing.assert_allclose(np.asarray(observed, dtype=float),
                               np.asarray(expected, dtype=float), atol=1e-12, rtol=1e-12,
                               err_msg="LazyTabs stale or wrong visible histogram")


def assert_axis_state(observed, expected):
    assert observed["modelValue"] == expected, ("ConcatenatedString.value", observed, expected)
    assert observed["axisLabel"] == expected, ("Bokeh Axis.axis_label", observed, expected)


def axis_oracle(value, *, option=False):
    if not option:
        return f"{value} (N)"
    return {"A": "Amplitude (N)", "B": "Background (N)", "None": " (N)"}[str(value)]


# Lazy imports: the two Python-only falsification controls can execute without
# the RootInteractive package, even when the pinned JS toolchain is absent.
def _from_array(kind, df, **settings):
    from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA
    from RootInteractive.Tools.pandaTools import initMetadata
    initMetadata(df)
    df.meta.metaData.update({"A.AxisTitle": "Amplitude", "B.AxisTitle": "Background"})
    compression = [(".*", [("code", 0), ("zip", 0), ("base64", 0)])]
    params = []
    extra = {}
    aliases = []
    if kind in {"query", "query_toggle", "query_parameter", "query_parameter_alias", "query_toggle_shown",
                "query_alone", "query_odd_names", "query_clash", "query_non_ascii", "query_odd_column"}:
        widgets = [["textQuery", {"source": None, "name": "text_query", "value": "return true",
                                  **({"toggleable": True} if kind in ("query_toggle", "query_toggle_shown",
                                                                       "query_alone") else {})}]]
        if kind in ("query_toggle_shown", "query_alone"):
            # integration: the dashboard template's form - the query text is shown in the text area (options.value)
            widgets[0][1]["options"] = {"value": "return true"}
        if kind == "query_alone":
            widgets = widgets[:1]                         # WG-D1: the toggleable query is the table's only filter
        if kind in ("query_toggle", "query_toggle_shown"):
            # WG-D1's healthy page requires a second active filter.
            widgets.append(["range", ["gate", -2., 2.]])
        if kind in ("query_parameter", "query_parameter_alias", "query_odd_names", "query_clash", "query_odd_column"):
            widgets.append(["spinner", ["cut"]])
            params = [{"name": "cut", "value": 1., "range": [0., 4.], "step": 1.}]
        if kind == "query_odd_names":
            # PHASE_5 Group 5 (G4-P1-A): parameters the query does not use, whose names cannot be JavaScript variables
            params += [{"name": name, "value": 2.} for name in ODD_NAMES]
        if kind == "query_non_ascii":
            # PHASE_5 Group 5a (review finding F1): a parameter name that is not plain ASCII
            params = [{"name": NON_ASCII, "value": 2.}]
        if kind == "query_parameter_alias":
            aliases = [("acut", "A > cut")]                # TQ-D1 control: an alias that uses the parameter
        plots = [
            [["bin_center_0"], ["bin_count"], {"source": "h"}],
            # Request both query-only columns through the real source graph;
            # a query typed later cannot itself inform initial column loading.
            [["A"], ["B"] + (["cut"] if kind == "query_clash" else [])
             # PHASE_5 Group 5d: the page shows the column with a name of ODD_COLUMNS, so the query is given it.
             # Group 6 (F1): or the column NON_ASCII, whose name is not plain ASCII.
             + ([c for c in df.columns if c in ODD_COLUMNS or c == NON_ASCII] if kind == "query_odd_column"
                else []),
             {"visualization_type": "scatter"}],
        ]
        extra["layout"] = [[0, 1]]
        extra["sourceArray"] = [{"name": "h", "variables": ["x", "y"],
                                 "nbins": [4, 3], "range": [[0, 4], [0, 3]], "weights": "w"}]
    elif kind == "alias_text":
        # PHASE_5 Group 6c: an alias whose function is the text of a parameter, edited in a text widget (the form of
        # the dashboard template's funCustomForm0)
        params = [{"name": "funForm", "value": "return A * 2"}]
        aliases = [{"name": "funA", "variables": ["A"], "func": "funForm"}]
        widgets = [["text", ["funForm"], {"name": "funForm"}]]
        plots = [[["A"], ["funA"], {"visualization_type": "scatter"}]]
    elif kind == "sample":
        plots = [[["x"], ["y"], {"visualization_type": "scatter"}]]
        widgets = [["range", ["gate", -2., 2.]], ["spinner", ["nRender"]]]
        params = [{"name": "nRender", "value": 5, "range": [1, 20], "step": 1}]
        extra["nPointRender"] = "nRender"
        extra["tooltips"] = [("source row", "@index")]
    elif kind == "tabs":
        # The FIGURE layout must be tabbed for LazyTabs.renderers to include the
        # two plotted DownsamplerCDS objects; widgetLayout tabs alone do not.
        plots = [
            [["bin_center"], ["bin_count"], {"source": "ha"}],
            [["bin_center"], ["bin_count"], {"source": "hb"}],
        ]
        widgets = [["range", ["gate", -2., 2.]]]
        extra["layout"] = {"X histogram": [0], "Y histogram": [1]}
        extra["sourceArray"] = [
            {"name": "ha", "variables": ["x"], "nbins": 4, "range": [0, 4], "weights": "w"},
            {"name": "hb", "variables": ["y"], "nbins": 3, "range": [0, 3], "weights": "w"},
        ]
    elif kind in {"axis_plain", "axis_options"}:
        if kind == "axis_plain":
            params = [{"name": "scale", "value": 2, "range": [1, 9], "step": 1}]
            widgets = [["spinner", ["scale"]]]
            label = "{scale} (N)"
        else:
            params = [{"name": "varY", "value": "A", "options": ["A", "B", "None"]}]
            widgets = [["select", ["varY"]]]
            label = "{varY} (N)"
        plots = [[["x"], ["y"], {"visualization_type": "scatter", "yAxisTitle": label}]]
    else:
        raise ValueError(kind)
    extra.setdefault("layout", [[0]])
    extra.update(settings)
    return bokehDrawSA.fromArray(
        df, None, plots, widgets, aliasArray=aliases, parameterArray=params,
        widgetLayout=[list(range(len(widgets)))],
        arrayCompression=compression, **extra,
    )


def _build_doc(tmp_path, kind, df, *, mute_text=False):
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    from bokeh.document import Document
    saved = sa.show
    sa.show = lambda *args, **kwargs: None
    try:
        root = _from_array(kind, df).pAll
    finally:
        sa.show = saved
    doc = Document()
    doc.add_root(root)
    if mute_text:
        # Fault injection is confined to a DISPOSABLE document, never production.
        from bokeh.models import TextAreaInput
        inputs = list(doc.select({"type": TextAreaInput}))
        assert len(inputs) == 1
        assert inputs[0].js_property_callbacks, "there is no live callback to mute"
        inputs[0].js_property_callbacks = {}
    p = tmp_path / ("display_" + kind + ("_muted" if mute_text else "") + ".json")
    p.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf8")
    return p


def _node_page(node_session, tmp_path, kind, df, *, mute_text=False):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
    result = ScenarioResult(id=f"display/{kind}/{'muted' if mute_text else 'healthy'} [node]", outdir=str(tmp_path))
    page = node_session.new_page(result)
    page.goto(_build_doc(tmp_path, kind, df, mute_text=mute_text))
    page.wait_for_function(page.load_condition)
    _settle(page, HIST_PROBES if kind.startswith("query") and kind != "query_alone" else [])
    _assert_no_errors(result)
    return page, result


def _assert_no_errors(report):
    assert not report.page_errors and not report.console_errors, (report.page_errors, report.console_errors)


def _settle(page, probes=()):
    ret = page.evaluate("p => window.__riH.settle(p, 50)", list(probes))
    assert ret and ret.get("settled"), ret


def _node_set(page, model_type, prop, value, *, title=None, label=None):
    """Set a Bokeh WIDGET, not a filter or underlying numerical result."""
    out = page.evaluate("""async a => {
      const ms = window.__riH.models().filter(m => m.type === a.model &&
        (a.title == null || m.title === a.title) &&
        (a.label == null || m.label === a.label));
      if (ms.length !== 1) throw Error('ambiguous or missing widget '+a.model+' / '+ms.length);
      ms[0][a.prop] = a.value;
      await Promise.resolve();
      return {actual: ms[0][a.prop]};
    }""", {"model": model_type, "prop": prop, "value": value,
          "title": title, "label": label})
    assert out["actual"] == value, out
    _settle(page, HIST_PROBES if model_type == "TextAreaInput" else [])


def _hist_state(page, n=N):
    return page.evaluate("""n => {
      const h = window.__riH.model('h', ['HistoNdCDS']);
      if (!h || !h.filter) throw Error('production histogram/filter missing');
      const raw = h.filter.v_compute();
      if (!raw || raw.length < n) throw Error('short/missing row mask');
      return {mask: Array.from(raw, Boolean).slice(0, n),
              bins: Array.from(h.get_column('bin_count'))};
    }""", n)


def _sample_state(page):
    return page.evaluate("""() => {
      const ds = window.__riH.models().filter(m => m.type === 'DownsamplerCDS' &&
            m.name === 'default cds');
      if (ds.length !== 1) throw Error('unique source DownsamplerCDS absent: '+ds.length);
      const d = ds[0];
      const index = Array.from(d.get_column('index'));
      return {index, x:Array.from(d.get_column('x')), y:Array.from(d.get_column('y')),
              nPoints:d.nPoints, length:d.get_length(), watched:d.watched,
              sourceSelection:Array.from(d.source.selected.indices),
              displayedSelection:Array.from(d.selected.indices)};
    }""")


def _tabs_state(page):
    return page.evaluate("""() => {
      const all = window.__riH.models().filter(m => m.type === 'LazyTabs');
      if (all.length !== 1) throw Error('unique LazyTabs absent: '+all.length);
      const tab = all[0];
      return {active:tab.active, watched:tab.renderers.map(rs => rs.map(r => r.watched)),
              counts:tab.renderers.map(rs=>rs.length)};
    }""")


def _tab_bins(page, index):
    return page.evaluate("""i => {
      const tab = window.__riH.models().find(m => m.type === 'LazyTabs');
      if (!tab || tab.active !== i || !tab.renderers[i].length)
        throw Error('requested tab not active or has no renderers');
      const ds = tab.renderers[i][0];
      if (!ds.watched) throw Error('active tab renderer is not watched');
      return Array.from(ds.get_column('bin_count'));
    }""", index)


def _expected_tab_bins(df, mask, index):
    field, count, region = (("x", 4, (0, 4)), ("y", 3, (0, 3)))[index]
    return np.histogram(df.loc[mask, field], bins=count, range=region,
                        weights=df.loc[mask, "w"])[0]


def _axis_state(page):
    return page.evaluate("""() => {
      const models = window.__riH.models();
      const labels = models.filter(m => m.type === 'ConcatenatedString' &&
            Array.isArray(m.components) && m.components.includes(' (N)'));
      if (labels.length < 1) throw Error('parametric concatenated axis label missing');
      const axes = models.filter(m => /Axis$/.test(m.type) && typeof m.axis_label === 'string' &&
            m.axis_label.endsWith(' (N)'));
      if (axes.length !== 1) throw Error('unique parameterized y-axis missing: '+axes.length);
      // The source can produce multiple ConcatenatedStrings (plot title etc),
      // so require at least one whose current text agrees with the y axis.
      const match = labels.find(l => l.components.join('') === axes[0].axis_label);
      if (!match) throw Error('axis not linked to a matching ConcatenatedString');
      return {modelValue:match.value, components:Array.from(match.components),
              axisLabel:axes[0].axis_label};
    }""")


def _browser_page(headless_session, tmp_path, kind, df):
    """Uses A3's pinned offline script route, not live internet assets."""
    from bokeh.plotting import output_file
    p = tmp_path / (kind + "_browser.html")
    output_file(str(p), mode="inline")
    _from_array(kind, df)
    return _open_browser_page(headless_session, p)


def _open_browser_page(headless_session, p):
    """Open the saved page `p` in Chromium (PHASE_5 Group 6c: split from _browser_page, for the template page)."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine import vendor
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import HELPER_JS
    routes = {pat.replace("**/", "/"): path for pat, path in vendor.routes().items()}
    ctx = headless_session.browser.new_context()
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda exc: errors.append("pageerror: " + str(exc)))
    page.on("console", lambda msg: errors.append("console.error: " + msg.text) if msg.type == "error" else None)

    def route(req):
        uri = req.request.url
        if uri.startswith("file:"):
            return req.continue_()
        for suffix, local in routes.items():
            if uri.split("?")[0].endswith(suffix):
                return req.fulfill(path=str(local), content_type="text/javascript")
        errors.append("blocked request: " + uri)
        return req.abort()
    page.route("**/*", route)
    page.add_init_script(HELPER_JS)
    page.goto(p.resolve().as_uri())
    page.wait_for_function("window.Bokeh && Bokeh.documents.length > 0 && Bokeh.documents[0].is_idle")
    return ctx, page, errors


def _assert_axis(page, expected):
    assert_axis_state(_axis_state(page), expected)


# --------------------------- pure-Python, independently runnable controls
@pytest.mark.headless_unit
def test_display_oracle_falsifiers():
    df = frame()
    good = query_oracle("return A > 1 && B < 2")
    assert_histogram(df, {"mask":good.tolist(), "bins":weighted_bins(df,good).tolist()}, good)
    corrupt = good.copy(); corrupt[0] = not corrupt[0]
    with pytest.raises(AssertionError, match="textQuery row-mask"):
        assert_histogram(df, {"mask":corrupt, "bins":weighted_bins(df,good)}, good)
    corrupt_bin = weighted_bins(df,good); corrupt_bin[0] += .25
    with pytest.raises(AssertionError, match="textQuery weighted histogram"):
        assert_histogram(df, {"mask":good, "bins":corrupt_bin}, good)
    assert not np.array_equal(query_oracle("return true"), good)
    assert not np.array_equal(query_oracle("return A > 1"), good)


@pytest.mark.headless_unit
def test_display_downsample_tabs_axis_falsifiers():
    df = frame()
    selected = gate_oracle()
    good = {"index":[0, 2, 4], "x":X[[0,2,4]].tolist(), "y":Y[[0,2,4]].tolist()}
    assert_sample(df, good, selected, 3)
    for bad in ({**good,"index":[2,0,4]}, {**good,"index":[0,2,2]},
                {**good,"x":[999, *good["x"][1:]]}):
        with pytest.raises((AssertionError, ValueError)):
            assert_sample(df, bad, selected, 3)
    assert not np.array_equal(_expected_tab_bins(df, selected, 0),
                              _expected_tab_bins(df, gate_oracle(-1,1), 0))
    assert axis_oracle(2) != axis_oracle(3)
    assert axis_oracle("A",option=True) != axis_oracle("B",option=True)
    assert axis_oracle("None",option=True) == " (N)"
    # Falsify the same helpers used by the live tab and axis tests.
    expected_tab = _expected_tab_bins(df, gate_oracle(-1, 1), 0)
    assert_tab_bins(expected_tab, expected_tab)
    wrong_tab = expected_tab.copy(); wrong_tab[0] += 1
    with pytest.raises(AssertionError, match="LazyTabs stale"):
        assert_tab_bins(wrong_tab, expected_tab)
    good_axis = {"modelValue":"Amplitude (N)","axisLabel":"Amplitude (N)"}
    assert_axis_state(good_axis, "Amplitude (N)")
    with pytest.raises(AssertionError, match="ConcatenatedString"):
        assert_axis_state({**good_axis, "modelValue":"wrong"}, "Amplitude (N)")
    with pytest.raises(AssertionError, match="Bokeh Axis"):
        assert_axis_state({**good_axis, "axisLabel":"wrong"}, "Amplitude (N)")


# --------------------------- textQuery / ColumnFilter (Node)
@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("expression", ["return true", "return A > 1", "return A > 1 && B < 2"])
def test_node_textquery_exact_masks_and_weighted_bins(node_session, tmp_path, expression):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query", df)
    _node_set(page, "TextAreaInput", "value", expression)
    assert_histogram(df, _hist_state(page), query_oracle(expression))
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_node_textquery_parameter_changed_after_input(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    regression(_is_tq_d1_unknown(page, report), "TQ-D1: the query does not see the parameter")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _node_set(page, "Spinner", "value", 3., title="cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=3.))
    _node_set(page, "Spinner", "value", 1., title="cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_node_textquery_toggle_with_second_filter(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_toggle", df)
    # Query is disabled at load, but the independent range on gate remains live.
    assert_histogram(df, _hist_state(page), gate_oracle())
    _node_set(page, "Toggle", "active", False, label="disable")
    _node_set(page, "TextAreaInput", "value", "return A > 1 && B < 2")
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1 && B < 2") & gate_oracle())
    _node_set(page, "Toggle", "active", True, label="disable")
    assert_histogram(df, _hist_state(page), gate_oracle())
    regression(_one_tq_d2_message(report.console_errors) and not report.page_errors, "TQ-D2: the logged TypeError")
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_node_textquery_disconnected_callback_mutant_is_detected(node_session, tmp_path):
    df = frame()
    healthy, hr = _node_page(node_session, tmp_path, "query", df)
    _node_set(healthy,"TextAreaInput","value","return A > 1")
    assert_histogram(df, _hist_state(healthy), query_oracle("return A > 1"))
    _assert_no_errors(hr)
    mutant, mr = _node_page(node_session, tmp_path, "query", df, mute_text=True)
    assert_histogram(df, _hist_state(mutant), query_oracle("return true"))
    _node_set(mutant,"TextAreaInput","value","return A > 1")
    with pytest.raises(AssertionError, match="textQuery row-mask"):
        assert_histogram(df, _hist_state(mutant), query_oracle("return A > 1"))
    _assert_no_errors(mr)


@pytest.mark.node_display
def test_node_textquery_invalid_js_characterization(node_session, tmp_path, record_property):
    """Record—not adjudicate—the unresolved invalid-JS policy; valid oracle stays enforced."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query", df)
    valid = "return A > 1"
    _node_set(page,"TextAreaInput","value",valid)
    assert_histogram(df, _hist_state(page), query_oracle(valid))
    errors = []
    try:
        _node_set(page,"TextAreaInput","value","return ((bad syntax !!!")
        state = _hist_state(page)
        status = "kept_previous_selection" if state["mask"] == query_oracle(valid).tolist() else "changed_selection"
        # Even without a policy on invalid JS, selected rows and histogram
        # must not disagree numerically. Do not assert which rows are intended.
        np.testing.assert_allclose(state["bins"], weighted_bins(df, state["mask"]), atol=1e-12, rtol=1e-12)
    except Exception as exc:
        detail = type(exc).__name__ + ": " + str(exc)[:300]
        # A missing widget, model or harness is NOT an invalid-JS outcome.
        if any(token in detail.lower() for token in ("ambiguous or missing", "no node", "model missing", "bundle", "not started")):
            raise
        status = "exception"
        errors.append(detail)
    observation = {"status":status, "error":errors, "page_errors":report.page_errors,
                   "console_errors":report.console_errors}
    (tmp_path/"invalid_js_observation.json").write_text(json.dumps(observation, indent=2),encoding="utf8")
    record_property("invalid_js_observation",observation)
    # PASS means only that an explicit observation was recorded and the *valid*
    # part was checked. It does NOT certify a normative invalid-JS policy.
    assert status in {"kept_previous_selection", "changed_selection", "exception"}


# --------------------------- DownsamplerCDS (Node)
@pytest.mark.node_display
@pytest.mark.feature("DISPLAY.downsampler")
def test_node_downsampler_exact_invariants_after_changes(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "sample", df)
    for low,high,limit in ((-2.,2.,5),(-1.5,1.5,5),(-1.5,1.5,8),(-1.5,1.5,20),(-2.,2.,20)):
        _node_set(page,"RangeSlider","value",[low,high],title="gate")
        _node_set(page,"Spinner","value",limit,title="nRender")
        state = _sample_state(page)
        assert state["nPoints"] == limit and state["length"] == len(state["index"]), state
        assert_sample(df,state,gate_oracle(low,high),limit)
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("DISPLAY.downsampler")
def test_node_downsampler_seeded_page_reproducibility(node_session, tmp_path):
    df = frame()
    p1, r1 = _node_page(node_session, tmp_path, "sample", df)
    p2, r2 = _node_page(node_session, tmp_path, "sample", df)
    a,b = _sample_state(p1),_sample_state(p2)
    assert_sample(df,a,gate_oracle(),5)
    assert_sample(df,b,gate_oracle(),5)
    np.testing.assert_array_equal(a["index"],b["index"],err_msg="same named table must have deterministic sampled rows")
    _assert_no_errors(r1); _assert_no_errors(r2)


@pytest.mark.node_display
@pytest.mark.feature("DISPLAY.downsampler")
def test_node_downsampler_selection_maps_back_without_losing_hidden_rows(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "sample", df)
    sample = _sample_state(page)
    assert_sample(df,sample,gate_oracle(),5)
    drawn = sample["index"][0]
    undrawn = next(i for i in range(N) if i not in sample["index"])
    out = page.evaluate("""async a => {
      const ds = window.__riH.models().find(m => m.type === 'DownsamplerCDS' && m.name === 'default cds');
      ds.source.selected.indices = [a.undrawn];
      ds.get_column('index');
      ds.selected.indices = [a.drawnPosition];
      await Promise.resolve();
      const done = await window.__riH.settle([],50);
      if (!done.settled) throw Error('selection did not settle');
      return {source:Array.from(ds.source.selected.indices), draw:Array.from(ds.selected.indices)};
    }""",{"undrawn":undrawn,"drawnPosition":0})
    assert sorted(out["source"]) == sorted([undrawn,drawn]), out
    _assert_no_errors(report)


# --------------------------- LazyTabs (Node)
@pytest.mark.node_display
@pytest.mark.feature("DISPLAY.lazy_tabs")
def test_node_lazy_tabs_watched_and_catchup_after_hidden_filter_moves(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "tabs", df)
    initial = _tabs_state(page)
    assert initial["active"] == 0 and initial["counts"] == [1,1],initial
    assert initial["watched"] == [[True],[False]],initial
    for lo,hi,open_tab in ((-.5,1.5,1),(-1.,.5,0),(-2.,2.,1),(-2.,2.,0)):
        _node_set(page,"RangeSlider","value",[lo,hi],title="gate")
        page.evaluate("i => {const t=window.__riH.models().find(m=>m.type==='LazyTabs');t.active=i}",open_tab)
        _settle(page)
        state = _tabs_state(page)
        assert state["active"] == open_tab,state
        assert state["watched"] == [[open_tab==0],[open_tab==1]],state
        expected = _expected_tab_bins(df,gate_oracle(lo,hi),open_tab)
        assert_tab_bins(_tab_bins(page,open_tab), expected)
    _assert_no_errors(report)


# --------------------------- ConcatenatedString / axis labels (Node)
@pytest.mark.node_display
@pytest.mark.feature("DISPLAY.axis_label")
@pytest.mark.parametrize("kind,model,states", [
    pytest.param("axis_plain","Spinner",[2,5,1,2],id="plain"),
    pytest.param("axis_options","Select",["A","B","None","A"],id="option-metadata-none"),
])
def test_node_axis_label_parameter_value_and_target(node_session, tmp_path, kind, model, states):
    df = frame()
    page, report = _node_page(node_session,tmp_path,kind,df)
    regression(_is_cs_d1(_axis_state(page), axis_oracle(states[0], option=(kind=="axis_options"))),
               "CS-D1: the label's value is undefined at load")
    _assert_axis(page,axis_oracle(states[0],option=(kind=="axis_options")))      # at load, before any change
    for value in states:
        _node_set(page,model,"value",value,title="scale" if kind=="axis_plain" else "varY")
        _assert_axis(page,axis_oracle(value,option=(kind=="axis_options")))
    _assert_no_errors(report)


# --------------------------- physical browser interactions (minimal)
@pytest.mark.browser_display
@pytest.mark.feature("FILTER.text_query")
def test_browser_textquery_real_typed_input_updates_weighted_histogram(headless_session,tmp_path):
    df=frame()
    ctx,page,errors=_browser_page(headless_session,tmp_path,"query",df)
    try:
        textarea=page.locator("textarea.bk-input")
        assert textarea.count()==1, "inspect pinned Bokeh 3.3.2 TextAreaInput DOM"
        textarea.fill("return A > 1 && B < 2")
        textarea.press("Tab")
        page.wait_for_function("() => window.__riH.models().some(m=>m.type==='TextAreaInput' && m.value==='return A > 1 && B < 2')")
        _settle(page,HIST_PROBES)
        assert_histogram(df,_hist_state(page),query_oracle("return A > 1 && B < 2"))
        assert not errors,errors
    finally:
        ctx.close()


@pytest.mark.browser_display
@pytest.mark.feature("DISPLAY.lazy_tabs")
def test_browser_lazy_tabs_real_tab_click_refreshes_hidden_plot(headless_session,tmp_path):
    df=frame()
    ctx,page,errors=_browser_page(headless_session,tmp_path,"tabs",df)
    try:
        _node_set(page,"RangeSlider","value",[-.5,1.5],title="gate")
        headers=page.locator(".bk-tab")
        assert headers.count()==2, "inspect pinned Bokeh 3.3.2 Tabs header DOM"
        headers.nth(1).click()
        page.wait_for_function("() => window.__riH.models().some(m=>m.type==='LazyTabs' && m.active===1)")
        _settle(page)
        assert_tab_bins(_tab_bins(page,1), _expected_tab_bins(df,gate_oracle(-.5,1.5),1))
        assert not errors,errors
    finally:
        ctx.close()


@pytest.mark.browser_display
@pytest.mark.feature("DISPLAY.axis_label")
@pytest.mark.parametrize("kind,value",[("axis_plain","5"),("axis_options","B"),("axis_options","None")])
def test_browser_real_widget_change_updates_axis_label_and_redraws(headless_session,tmp_path,kind,value):
    df=frame()
    ctx,page,errors=_browser_page(headless_session,tmp_path,kind,df)
    try:
        # at load the axis text and the label's model value are checked (the value was undefined at load before the
        # fix of CS-D1 in PHASE_5 Group 1); after the real input below the full state is checked again
        at_load = axis_oracle(2 if kind=="axis_plain" else "A",option=kind=="axis_options")
        assert _axis_state(page)["axisLabel"] == at_load
        assert _axis_state(page)["modelValue"] == at_load
        # Capturing only the plot's canvas avoids mistaking a changed input for
        # graphical axis redraw. Model value is asserted separately below.
        canvases=page.locator("canvas")
        assert canvases.count()>0, "no actual plot canvas"
        before=hashlib.sha256(canvases.first.screenshot()).hexdigest()
        if kind=="axis_plain":
            inp=page.locator('input.bk-input')
            assert inp.count()>=1
            inp.first.fill(value)
            inp.first.press("Tab")
        else:
            select=page.locator('select:not([multiple])')
            assert select.count()==1
            select.select_option(value)
        expected=axis_oracle(value,option=kind=="axis_options")
        page.wait_for_function("e => window.__riH.models().some(m=>/Axis$/.test(m.type) && m.axis_label===e)",arg=expected)
        _settle(page)
        _assert_axis(page,expected)
        after=hashlib.sha256(canvases.first.screenshot()).hexdigest()
        assert before!=after, "plot canvas did not change after axis label update"
        assert not errors,errors
    finally:
        ctx.close()


# ===================================================================================== integration: defect tests
def _is_tq_d1_unknown(page, report) -> bool:
    """TQ-D1 on a page without an alias that uses the parameter: the query function has no parameters, the
    evaluation error is logged, and every row stays selected."""
    keys = page.evaluate("""() => window.__riH.models().filter(m => m.type === 'CustomJSNAryFunction' &&
                              m.func === 'return A > cut').map(m => Object.keys(m.parameters))""")
    state = _hist_state(page)
    return (keys == [[]] and report.console_errors == [TQ_D1_ERROR] and not report.page_errors
            and all(state["mask"]))


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_control_tq_d1_with_the_parameter_in_scope_the_query_computes_it(node_session, tmp_path):
    """TQ-D1's control: the same query with `cut` in the function's scope (the alias page) gives exactly A > 1 and
    its weighted histogram, without any error - the query machinery itself works."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter_alias", df)
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_strict_a_query_follows_its_parameter(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter_alias", df)
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    _node_set(page, "Spinner", "value", 3., title="cut")
    state = _hist_state(page)
    regression(state["mask"] == query_oracle("return A > cut", cut=1.).tolist(), "TQ-D1: build-time value kept")
    assert_histogram(df, state, query_oracle("return A > cut", cut=3.))


RECORD_CONSOLE_JS = """() => { console.__c2_sink = [];
  if (console.__c2_wrapped) return;     // pages get new windows but share one console: wrap console.error only once
  console.__c2_wrapped = true;
  const orig = console.error;
  console.error = function (...a) {
    console.__c2_sink.push(a.map(x => (x && x.stack) ? String(x.stack).split("\\n").slice(0, 2).join(" @ ") : String(x)).join(" | "));
    return orig.apply(this, a); }; }"""


def _clear_and_retype(page):
    """Query A > 1, clear the text, type the next query; returns (mask after the clear, recorded errors)."""
    _node_set(page, "TextAreaInput", "value", "return A > 1")
    _node_set(page, "TextAreaInput", "value", "")
    after_clear = _hist_state(page)["mask"]
    page.evaluate(RECORD_CONSOLE_JS)
    _node_set(page, "TextAreaInput", "value", "return A > 1 && B < 2")
    return after_clear, page.evaluate("() => console.__c2_sink")


def _is_tq_d2(after_clear, recorded, report) -> bool:
    return (after_clear == query_oracle("return A > 1").tolist() and len(recorded) == 1
            and recorded[0].startswith(TQ_D2_ERROR) and "at CustomJSNAryFunction.compute" in recorded[0]
            and _one_tq_d2_message(report.console_errors) and not report.page_errors)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_control_tq_d2_a_toggle_with_the_query_shown_logs_nothing(node_session, tmp_path):
    """TQ-D2's control: the toggleable query in the template's form (its text shown in the text area): enabling it
    and typing the next query gives the right rows and histograms, and nothing is logged."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_toggle_shown", df)
    assert_histogram(df, _hist_state(page), gate_oracle())
    _node_set(page, "Toggle", "active", False, label="disable")
    assert_histogram(df, _hist_state(page), gate_oracle())                         # "return true" keeps every row
    _node_set(page, "TextAreaInput", "value", "return A > 1 && B < 2")
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1 && B < 2") & gate_oracle())
    _node_set(page, "Toggle", "active", True, label="disable")
    assert_histogram(df, _hist_state(page), gate_oracle())
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_strict_clearing_and_retyping_a_query_logs_nothing(node_session, tmp_path):
    """PHASE_5 Group 4: an empty query is no condition, so after clearing the text every row is selected."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query", df)
    after_clear, recorded = _clear_and_retype(page)
    regression(_is_tq_d2(after_clear, recorded, report), "TQ-D2: the logged TypeError")
    assert all(after_clear) and len(after_clear) == len(df), after_clear
    assert recorded == [], recorded
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1 && B < 2"))
    _assert_no_errors(report)


# PHASE_5 Group 5 (review of Groups 3 and 4): G4-P1-A, unusual parameter names; G4-POL-1, a name that is both a column
# and a parameter, which the Architect's decision N1 (2026-11-18) forbids.
ODD_NAMES_FAILURE = ("SyntaxError", "Unexpected token 'default'")
INJECT = """a => {
  const q = window.__riH.models().filter(m => m.type === 'CustomJSNAryFunction' && m.func === 'return true');
  if (q.length !== 1) throw Error('query function: ' + q.length);
  q[0].parameters = {...q[0].parameters, [a.name]: 100.};
}"""
REFUSED_A = 'the name "A" is both a column and a parameter'


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_parameters_whose_names_cannot_be_variables_do_not_break_a_query(node_session, tmp_path):
    """Since Group 4 a text query's function gets every parameter of the page. Parameters named default, static,
    eval, class and n-bins, which the query does not use and which cannot be variables of strict-mode JavaScript, must
    not stop the page: it loads, `return A > cut` selects by the parameter cut and follows it, and nothing is logged.
    After Group 4 the page did not load (SyntaxError when the query's function was built)."""
    df = frame()
    failure = _load_failure(node_session, _build_doc(tmp_path, "query_odd_names", df))
    regression(failure is not None and failure[:2] == ODD_NAMES_FAILURE,
               f"G4-P1-A: the page does not load: {failure}")
    assert failure is None, failure
    page, report = _node_page(node_session, tmp_path, "query_odd_names", df)
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _node_set(page, "Spinner", "value", 3., title="cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=3.))
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_query_using_a_name_that_is_both_a_column_and_a_parameter_is_refused(node_session, tmp_path):
    """The Architect's decision N1: such a name is forbidden. The query `return A > 1`, whose function is given a
    parameter A by hand next to the column A, is refused by name: it selects no row, and one error names A. After
    Group 4 the column won without a message (every row with A > 1 was selected)."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    page.evaluate(INJECT, {"name": "A"})
    _node_set(page, "TextAreaInput", "value", "return A > 1")
    state = _hist_state(page)
    regression(state["mask"] == query_oracle("return A > 1").tolist() and not report.console_errors
               and not report.page_errors, "G4-POL-1: the column A won without a message")
    assert state["mask"] == [False] * N, state["mask"]
    assert not report.page_errors, report.page_errors
    assert len(report.console_errors) == 1 and REFUSED_A in report.console_errors[0], report.console_errors


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_control_a_parameter_named_like_a_column_the_query_does_not_use_changes_nothing(node_session, tmp_path):
    """N1's control, right before and after the fix: a parameter B next to the column B, and the query `return A > 1`
    that does not use B. The query selects by A, and nothing is logged."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    page.evaluate(INJECT, {"name": "B"})
    _node_set(page, "TextAreaInput", "value", "return A > 1")
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1"))
    _assert_no_errors(report)


# PHASE_5 Group 5a (the review of group 5): F1, a parameter name that is not plain ASCII (the Architect's rule of
# 2026-11-18: parameter names are plain ASCII); F2, a name only mentioned in a comment, a string or after a dot is not
# a use of it (decision N1 is about names the function uses); F3, a change of the parameters after the function is
# built is checked again.
MENTIONS = {"comment": "return B > 1 // A", "string": 'return B > 1 && "A" === "A"',
            "property": "return B > 1 && [0].A === undefined", "escaped_quote": "return B > 1 && 'it\\'s A' !== ''",
            "template_text": "return B > 1 && `A` === 'A'", "longer_name": "const \u03c3A = 0; return B > 1 || \u03c3A > 0"}
USES = {"template_expression": "return B > 1 && `${A}` !== ''", "spread": "return Math.max(...[A]) > 1",
        "after_a_string": "return 'x' + A !== 'x1'", "block_comment_before": "/* A */ return A > 1"}
REFUSED_NON_ASCII = 'the parameter "' + NON_ASCII + '" cannot be used'


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_parameter_name_that_is_not_plain_ascii_is_refused_when_the_page_is_built(tmp_path):
    """The Architect's rule: parameter names are plain ASCII. The page builder refuses such a name by name. After
    Group 5 the page was built, and a query that used the name failed in the browser without naming the cause."""
    error = None
    try:
        _build_doc(tmp_path, "query_non_ascii", frame())
    except ValueError as e:
        error = str(e)
    regression(error is None, "F1: the page is built with a parameter name that is not plain ASCII")
    assert NON_ASCII in error and "ASCII" in error, error


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_function_that_uses_a_parameter_that_cannot_be_passed_is_refused(node_session, tmp_path):
    """A parameter that cannot be passed to the function (here a name that is not plain ASCII, given by hand, as the
    page builder now refuses it) and that the function uses: the function is refused by name, selects no row and
    logs one error naming it. After Group 5 the name was left out and the query failed with a ReferenceError."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    page.evaluate(INJECT, {"name": NON_ASCII})
    _node_set(page, "TextAreaInput", "value", "return A > " + NON_ASCII)
    state = _hist_state(page)
    regression(len(report.console_errors) == 1 and "ReferenceError" in report.console_errors[0]
               and NON_ASCII + " is not defined" in report.console_errors[0],
               f"F1: the query fails with {report.console_errors}")
    assert state["mask"] == [False] * N, state["mask"]
    assert not report.page_errors, report.page_errors
    assert len(report.console_errors) == 1 and REFUSED_NON_ASCII in report.console_errors[0], report.console_errors


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("case", list(MENTIONS))
def test_a_name_only_mentioned_is_not_used_and_is_not_refused(node_session, tmp_path, case):
    """Decision N1 refuses a name that is both a column and a parameter when the function USES it. A parameter A next
    to the column A, mentioned only in a comment, a string or after a dot, is not used: the query `return B > 1 ...`
    selects by B, and nothing is logged. After Group 5 the text was searched as it is, and the query was refused."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    page.evaluate(INJECT, {"name": "A"})
    _node_set(page, "TextAreaInput", "value", MENTIONS[case])
    state = _hist_state(page)
    regression(state["mask"] == [False] * N and len(report.console_errors) == 1
               and REFUSED_A in report.console_errors[0], f"F2: the {case} mention is refused")
    assert_histogram(df, state, B > 1)
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("case", list(USES))
def test_control_a_name_used_in_other_forms_is_refused(node_session, tmp_path, case):
    """F2's control, right before and after the fix: a parameter A next to the column A, USED by the query in other
    forms (inside ${...} of a template literal, after a spread, after a string, after a comment). The query is refused
    by name: no row, one error naming A."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    page.evaluate(INJECT, {"name": "A"})
    _node_set(page, "TextAreaInput", "value", USES[case])
    assert _hist_state(page)["mask"] == [False] * N
    assert len(report.console_errors) == 1 and REFUSED_A in report.console_errors[0], report.console_errors


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_change_of_the_parameters_after_the_function_is_built_is_checked_again(node_session, tmp_path):
    """The query `return A > 1` is built first, with the column A. Then a parameter A is added and the arguments are
    updated: the function now uses a name that is both a column and a parameter, and is refused by name (no row,
    one error naming A). After Group 5 an update of the arguments did not check again, and the column was used."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > 1")
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1"))
    page.evaluate("""() => {
      const q = window.__riH.models().filter(m => m.type === 'CustomJSNAryFunction' && m.func === 'return A > 1');
      if (q.length !== 1) throw Error('query function: ' + q.length);
      q[0].parameters = {...q[0].parameters, A: 100.};
      q[0].update_args();
    }""")
    _settle(page, HIST_PROBES)
    state = _hist_state(page)
    regression(state["mask"] == query_oracle("return A > 1").tolist() and not report.console_errors,
               "F3: the column A is used without a check")
    assert state["mask"] == [False] * N, state["mask"]
    assert len(report.console_errors) == 1 and REFUSED_A in report.console_errors[0], report.console_errors



# PHASE_5 Group 5b (the review of group 5a): the names that a function uses must be read as JavaScript reads them.
# G5a-F2-A, a name written with escapes (\u0041 and \u{41} are A); G5a-F2-B, a regular expression whose text
# looks like the start of a comment or the end of ${...}; and more forms of the same kind found by the coder: a
# regular expression with // or a quote or a backtick or an escaped or bracketed "/", a line comment that ends at a
# line separator (U+2028), and the ")" of if or the "}" of a block before a regular expression. Each hides a real
# use of A from group 5a's reading, and the function is computed with A without a message. "parameter": the
# parameter A is used (the column A is not passed, as the code does not write its name); "column": the column A is
# used. G5a-F3-P2: the scalar and the vector function of one model must agree on the fields.
HIDDEN_USES = {
    "escape_four_digits": ("return \\u0041 > 1", "return B.map((b, i) => \\u0041[i] > 1)", "parameter"),
    "escape_braces": ("return \\u{41} > 1", "return B.map((b, i) => \\u{41}[i] > 1)", "parameter"),
    "regex_with_comment_start": ('return /[/*]/.test("/") && A > 1',
                                 'return B.map((b, i) => /[/*]/.test("/") && A[i] > 1)', "column"),
    "regex_with_line_comment": ('return /[//]/.test("/") && A > 1',
                                'return B.map((b, i) => /[//]/.test("/") && A[i] > 1)', "column"),
    "regex_with_quote": ('return /"/.test("x") ? false : A > 1',
                         'return B.map((b, i) => /"/.test("x") ? false : A[i] > 1)', "column"),
    "regex_with_backtick": ('return /`/.test("x") ? false : A > 1',
                            'return B.map((b, i) => /`/.test("x") ? false : A[i] > 1)', "column"),
    "regex_with_slash_in_class": ('return /[/"]/.test("x") ? false : A > 1 // "',
                                  'return B.map((b, i) => /[/"]/.test("x") ? false : A[i] > 1) // "', "column"),
    "regex_with_escaped_slash": ('return /\\/"/.test("x") ? false : A > 1 // "',
                                 'return B.map((b, i) => /\\/"/.test("x") ? false : A[i] > 1) // "', "column"),
    "regex_with_brace_in_template": ('return `${ /}/.test("}") ? A : 0 }` > 1',
                                     'return B.map((b, i) => `${ /}/.test("}") ? A[i] : 0 }` > 1)', "column"),
    "line_separator_ends_comment": ("return false // x  || A > 1",
                                    "return B.map((b, i) => false // x  || A[i] > 1)", "column"),
    "regex_after_if": ('if (B > 99) /[/*]/.test(" */ "); return A > 1 // "',
                       'if (B[0] > 99) /[/*]/.test(" */ "); return B.map((b, i) => A[i] > 1) // "', "column"),
    "regex_after_block": ('{} /[/*]/.test(""); return A > 1',
                          '{} /[/*]/.test(""); return B.map((b, i) => A[i] > 1)', "column"),
}
VECTOR_A, VECTOR_B = [3., .5], [2., 2.]
BUILD_VECTOR = """a => {
  const q = window.__riH.models().filter(m => m.type === 'CustomJSNAryFunction' && m.func === 'return true');
  if (q.length !== 1) throw Error('query function: ' + q.length);
  const m = new q[0].constructor({parameters: {A: 100.}, fields: ["A", "B"], v_func: a.code});
  const out = m.v_compute([a.A, a.B], null);
  return Array.isArray(out) ? out.map(v => Number.isNaN(v) ? "NaN" : v) : String(out);
}"""
BOTH_BODIES = """() => {
  const q = window.__riH.models().filter(m => m.type === 'CustomJSNAryFunction' && m.func === 'return true');
  if (q.length !== 1) throw Error('query function: ' + q.length);
  const m = new q[0].constructor({parameters: {}, fields: ["A", "B", "x"], auto_fields: true,
                                  func: "return A + 1", v_func: "return B.map(b => 2 * b)"});
  const fields = m.get_fields();
  const row = {A: 3., B: 5., x: 7.};
  return {fields, scalar: m.compute(fields.map(f => row[f])),
          vector: Array.from(m.v_compute(fields.map(f => [row[f]]), null))};
}"""


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("case", list(HIDDEN_USES))
def test_a_use_that_a_simple_reading_misses_is_found_and_the_query_is_refused(node_session, tmp_path, case):
    """A parameter A next to the column A, and a query that really uses A, in a form that group 5a's reading of the
    code missed (see HIDDEN_USES). Decision N1: the query is refused by name, selects no row, and one error names A.
    After Group 5a the query was computed without a message, with the parameter A (escapes) or the column A."""
    code, _, used = HIDDEN_USES[case]
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    page.evaluate(INJECT, {"name": "A"})
    _node_set(page, "TextAreaInput", "value", code)
    state = _hist_state(page)
    computed = [True] * N if used == "parameter" else query_oracle("return A > 1").tolist()
    regression(state["mask"] == computed and not report.console_errors and not report.page_errors,
               f"G5a-F2: the {case} query is computed with the {used} A without a message")
    assert state["mask"] == [False] * N, state["mask"]
    assert not report.page_errors, report.page_errors
    assert len(report.console_errors) == 1 and REFUSED_A in report.console_errors[0], report.console_errors


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("case", list(HIDDEN_USES))
def test_a_vector_function_that_hides_its_use_of_a_name_is_refused(node_session, tmp_path, case):
    """The same forms in a vector function (one call for a whole column), built directly with the fields A and B and a
    parameter A: refused by name, NaN for every row, one error naming A. After Group 5a the function was computed
    with the column A ([true, false] for A = [3, 0.5]) without a message."""
    _, code, _ = HIDDEN_USES[case]
    page, report = _node_page(node_session, tmp_path, "query_parameter", frame())
    out = page.evaluate(BUILD_VECTOR, {"code": code, "A": VECTOR_A, "B": VECTOR_B})
    regression(out == [True, False] and not report.console_errors,
               f"G5a-F2: the {case} vector function gives {out} without a message")
    assert out == ["NaN", "NaN"], out
    assert not report.page_errors, report.page_errors
    assert len(report.console_errors) == 1 and REFUSED_A in report.console_errors[0], report.console_errors


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_name_only_inside_a_regular_expression_is_not_used_and_is_not_refused(node_session, tmp_path):
    """The text of a regular expression is not a use, as the text of a string is not. A parameter A next to the column
    A, and the query `return B > 1 && /A/.test("x") === false`: it selects by B, and nothing is logged. After Group 5a
    the A inside /A/ counted as a use, and the query was refused (on the safe side)."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    page.evaluate(INJECT, {"name": "A"})
    _node_set(page, "TextAreaInput", "value", 'return B > 1 && /A/.test("x") === false')
    state = _hist_state(page)
    regression(state["mask"] == [False] * N and len(report.console_errors) == 1
               and REFUSED_A in report.console_errors[0], "G5a-F2: the A inside /A/ is counted as a use")
    assert_histogram(df, state, B > 1)
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_control_divisions_are_not_read_as_regular_expressions(node_session, tmp_path):
    """The control of group 5b's reading, right before and after the fix: a "/" after a name or a number divides.
    The query `return B / 2 / 1 > 0.5 // A`, with a parameter A next to the column A, selects by B (B > 1), and the A
    of its comment is not a use: nothing is logged."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    page.evaluate(INJECT, {"name": "A"})
    _node_set(page, "TextAreaInput", "value", "return B / 2 / 1 > 0.5 // A")
    assert_histogram(df, _hist_state(page), B > 1)
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_model_with_a_scalar_and_a_vector_function_uses_the_same_fields_for_both(node_session, tmp_path):
    """One model with both a scalar function (`return A + 1`) and a vector function (`return B.map(b => 2 * b)`) and
    automatic fields A, B and x. The fields it asks for must serve both functions: A and B; with A = 3 and B = 5 the
    scalar function gives 4 and the vector function [10]. After Group 5a the function built last chose the fields
    (only B), and the scalar function was given B's value as A (6). The page builder never makes such a model; the
    test is of the model alone."""
    page, report = _node_page(node_session, tmp_path, "query_parameter", frame())
    got = page.evaluate(BOTH_BODIES)
    regression(got["fields"] == ["B"] and got["scalar"] == 6, f"G5a-F3-P2: the model gives {got}")
    assert got == {"fields": ["A", "B"], "scalar": 4, "vector": [10]}, got
    _assert_no_errors(report)

# PHASE_5 Group 5c (the review of group 5b): more forms that hide a real use of A from group 5b's reading, each valid
# strict-mode JavaScript that the browser accepts. Claude10.RI F1: HTML-like comments, <!-- anywhere and --> at the
# start of a line, which the browser reads as line comments (group 5b read a /* after them as a block comment and
# skipped the code up to its */). Claude10.RI F2 and F3: eval, which reads the function's own variables from a string
# (the name may even be built at run time), and arguments, which reads them by position. GPT32.RI and GPT33.RI
# (G5b-F1): a regular expression on the line after debugger, break or continue (group 5b read its "/" as a division,
# and the // inside the expression as a comment). Each was computed without a message. "Before" = group 5b.
HIDDEN_USES_5C = {
    "html_open_comment": ("const t = 1 <!-- /*\nreturn A > 1 // */",
                          "const t = 1 <!-- /*\nreturn B.map((b, i) => A[i] > 1) // */"),
    "html_close_comment": ("const t = 1\n--> /*\nreturn A > 1 // */",
                           "const t = 1\n--> /*\nreturn B.map((b, i) => A[i] > 1) // */"),
    "eval": ('return eval("A > 1")', 'return B.map((b, i) => eval("A[i] > 1"))'),
    "eval_name_built_at_run_time": ('return eval(String.fromCharCode(65) + " > 1")',
                                    'return B.map((b, i) => eval(String.fromCharCode(65) + "[i] > 1"))'),
    "eval_written_with_escapes": ('return \\u0065val("A > 1")', 'return B.map((b, i) => \\u0065val("A[i] > 1"))'),
    "arguments": ("return arguments[0] > 1", "return B.map((b, i) => arguments[0][i] > 1)"),
    "regex_after_debugger": ('debugger\n/[//]/.test("/"); const r = A > 1\nreturn r',
                             'debugger\n/[//]/.test("/"); const r = B.map((b, i) => A[i] > 1)\nreturn r'),
    "regex_after_break": ('let r = false\nfor (let k = 0; k < 2; k++) { if (k === 1) break\n/[//]/.test("/"); r = A > 1 }\nreturn r',
                          'let r = null\nfor (let k = 0; k < 2; k++) { if (k === 1) break\n/[//]/.test("/"); r = B.map((b, i) => A[i] > 1) }\nreturn r'),
    "regex_after_continue": ('let r = false\nfor (let k = 0; k < 2; k++) { if (k === 1) continue\n/[//]/.test("/"); r = A > 1 }\nreturn r',
                             'let r = null\nfor (let k = 0; k < 2; k++) { if (k === 1) continue\n/[//]/.test("/"); r = B.map((b, i) => A[i] > 1) }\nreturn r'),
    "regex_after_break_with_label": ('let r = false\nouter: for (let k = 0; k < 2; k++) { if (k === 1) break outer\n/[//]/.test("/"); r = A > 1 }\nreturn r',
                                     'let r = null\nouter: for (let k = 0; k < 2; k++) { if (k === 1) break outer\n/[//]/.test("/"); r = B.map((b, i) => A[i] > 1) }\nreturn r'),
}
# The same forms without a name that is both a column and a parameter (the query uses B, the page has the parameter
# cut): computed as before, nothing logged. eval and arguments refuse only where some name is both.
# PHASE_5 Group 5d (GPT32.RI, P2-G5c-03): the selection that group 5b computed for each form, without a message,
# measured with group 5b's reader: with the column A (A > 1); for the name built at run time with the parameter A
# (100 > 1, every row); for arguments[0] with the first parameter, cut (1 > 1, no row).
GROUP_5B_SELECTION = {case: query_oracle("return A > 1").tolist() for case in HIDDEN_USES_5C}
GROUP_5B_SELECTION["eval_name_built_at_run_time"] = [True] * N
GROUP_5B_SELECTION["arguments"] = [False] * N
COMPUTED_5C = {
    "html_open_comment": "const t = 1 <!-- /*\nreturn B > 1 // */",
    "eval": 'return eval("B > 1")',
    "arguments": "return arguments[0] > 0",
    "regex_after_debugger": 'debugger\n/[//]/.test("/"); const r = B > 1\nreturn r',
    "regex_after_break": 'let r = false\nfor (let k = 0; k < 2; k++) { if (k === 1) break\n/[//]/.test("/"); r = B > 1 }\nreturn r',
}


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("case", list(HIDDEN_USES_5C))
def test_a_use_hidden_from_group_5b_reading_is_refused(node_session, tmp_path, case):
    """A parameter A next to the column A, and a query that really uses A in a form of HIDDEN_USES_5C. Decision N1:
    the query is refused by name, selects no row, and one error names A. Group 5b computed it without a message."""
    code = HIDDEN_USES_5C[case][0]
    page, report = _node_page(node_session, tmp_path, "query_parameter", frame())
    page.evaluate(INJECT, {"name": "A"})
    _node_set(page, "TextAreaInput", "value", code)
    state = _hist_state(page)
    regression(state["mask"] == GROUP_5B_SELECTION[case] and not report.console_errors and not report.page_errors,
                 f"G5b review: the {case} query selects as group 5b did, without a message")
    assert state["mask"] == [False] * N, state["mask"]
    assert not report.page_errors, report.page_errors
    assert len(report.console_errors) == 1 and REFUSED_A in report.console_errors[0], report.console_errors


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("case", list(HIDDEN_USES_5C))
def test_a_vector_function_hiding_a_use_from_group_5b_reading_is_refused(node_session, tmp_path, case):
    """The same forms in a vector function, built with the fields A and B and a parameter A: refused by name, NaN for
    every row, one error naming A. Group 5b computed it with the column A ([true, false]) without a message."""
    code = HIDDEN_USES_5C[case][1]
    page, report = _node_page(node_session, tmp_path, "query_parameter", frame())
    out = page.evaluate(BUILD_VECTOR, {"code": code, "A": VECTOR_A, "B": VECTOR_B})
    regression(out == [True, False] and not report.console_errors,
                 f"G5b review: the {case} vector function gives {out} without a message")
    assert out == ["NaN", "NaN"], out
    assert not report.page_errors, report.page_errors
    assert len(report.console_errors) == 1 and REFUSED_A in report.console_errors[0], report.console_errors


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("case", list(COMPUTED_5C))
def test_control_the_same_forms_without_a_shared_name_are_computed(node_session, tmp_path, case):
    """The control, right before and after the fix: the forms of HIDDEN_USES_5C with B and no name that is both a
    column and a parameter (eval's string reads B; arguments[0] is the parameter cut = 1). The query is computed as
    JavaScript computes it, and nothing is logged."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", COMPUTED_5C[case])
    assert_histogram(df, _hist_state(page), np.ones(N, bool) if case == "arguments" else B > 1)
    _assert_no_errors(report)


ESCAPED_A_FAILURE = "A is not defined"


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_column_written_with_escapes_is_given_to_the_query(node_session, tmp_path):
    """The query `return \u0041 > 1` (\u0041 is A) and no parameter A: the column A must be given to the function, and
    the query selects the rows with A > 1. Group 5b chose the columns by a plain search of the text, did not give A,
    and the query failed with a ReferenceError for A."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return \\u0041 > 1")
    state = _hist_state(page)
    regression(len(report.console_errors) == 1 and "ReferenceError" in report.console_errors[0]
                 and ESCAPED_A_FAILURE in report.console_errors[0],
                 f"G5b review: the column A is not given: {report.console_errors}")
    assert_histogram(df, state, query_oracle("return A > 1"))
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_query_that_writes_its_parameter_with_escapes_follows_it(node_session, tmp_path):
    """The query `return A > \u0063ut` (\u0063ut is cut) uses the parameter cut. When the spinner sets cut from 1 to
    3, the query must follow: the rows with A > 3. The page builder decided by a plain search of the text whether a
    parameter change concerns the query, did not find cut, and the old selection stayed without a message."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > \\u0063ut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _node_set(page, "Spinner", "value", 3., title="cut")
    state = _hist_state(page)
    regression(state["mask"] == query_oracle("return A > cut", cut=1.).tolist() and not report.console_errors,
                 "G5b review: the query keeps the selection of cut = 1 without a message")
    assert_histogram(df, state, query_oracle("return A > cut", cut=3.))
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_eval_reads_a_column_whose_name_is_built_at_run_time(node_session, tmp_path):
    """The query `return eval(String.fromCharCode(66) + " > 1")` reads the column B through a name built at run time
    (66 is B), with no name that is both a column and a parameter: the column B must be given to the function, and the
    query selects the rows with B > 1. Group 5b gave only the columns whose name is in the text, not B, and the query
    failed with a ReferenceError for B. Group 5c gives every column to a function that uses eval or arguments."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", 'return eval(String.fromCharCode(66) + " > 1")')
    state = _hist_state(page)
    regression(len(report.console_errors) == 1 and "ReferenceError" in report.console_errors[0]
                 and "B is not defined" in report.console_errors[0],
                 f"G5c: the column B is not given: {report.console_errors}")
    assert_histogram(df, state, B > 1)
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_control_a_query_reading_its_parameter_through_eval_follows_it(node_session, tmp_path):
    """The control, right before and after the fix: the query `return eval("A > cut")` reads the parameter cut through
    eval. When the spinner sets cut from 1 to 3, the query follows (the rows with A > 3). Group 5c decides by
    reading the names (uses_parameter); a function that uses eval may read any name, so every change concerns it."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", 'return eval("A > cut")')
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _node_set(page, "Spinner", "value", 3., title="cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=3.))
    _assert_no_errors(report)


# PHASE_5 Group 5d (the review of group 5c, finding P1-G5c-01 of GPT32.RI, reproduced by GPT33.RI): a column whose
# name cannot be a variable of strict-mode JavaScript, such as default, static, let, arguments or n-bins. Group 5c
# gives every column to a function that uses eval or arguments, so it built the function with such a column as a
# variable, and JavaScript refused to build it with a SyntaxError. On a page the query then kept its previous
# selection; the only sign was an unhandled rejection of the error in the page (its text depends on the name and on
# the JavaScript engine). Before group 5c the same happened to any function whose text merely contains such a name,
# for example in a comment, because the columns were chosen by a plain search of the text. "Before" = group 5c.
ODD_COLUMN_FAILURE = "SyntaxError"
UNHANDLED = "unhandled rejection"
BUILD_WITH_ODD_COLUMN = """a => {
  const q = window.__riH.models().filter(m => m.type === 'CustomJSNAryFunction' && m.func === 'return true');
  if (q.length !== 1) throw Error('query function: ' + q.length);
  try {
    const m = new q[0].constructor({parameters: {}, fields: ["A", "B", a.name], auto_fields: true,
                                    func: 'return eval("A > 1")', v_func: 'return B.map((b, i) => eval("A[i] > 1"))'});
    const fields = m.get_fields();
    const column = {A: [3., .5], B: [2., 2.], [a.name]: [7., 7.]};
    return {fields, scalar: m.compute(fields.map(f => column[f][0])),
            vector: Array.from(m.v_compute(fields.map(f => column[f]), null))};
  } catch (error) {
    return {error: String(error)};
  }
}"""


def _odd_column_frame(name):
    df = frame()
    df[name] = B * 1.
    return df


ENGINE_ERROR = """a => {
  try {
    new Function(...a.names, '"use strict";\\n' + a.body);
    return null;
  } catch (error) {
    return {name: error.name, message: error.message};
  }
}"""


def _engine_error(page, names, body):
    """PHASE_5 Group 6: the error that this JavaScript engine gives when it builds a strict-mode function with the
    arguments `names` and the code `body`, as CustomJSNAryFunction builds it; None if it builds. Its text depends on
    the engine and its version, so the tests ask the engine instead of writing the text down."""
    return page.evaluate(ENGINE_ERROR, {"names": list(names), "body": body})


def _unhandled(error) -> str:
    """The page error that the Node harness records for an unhandled rejection of `error` (engine/node/runner.mjs)."""
    return f"{UNHANDLED}: {error['message'][:280]}"


def _query_not_built(report, state, error) -> bool:
    """The signature on a page: the previous selection ("return true": every row) stays, no console error, and one
    page error, the unhandled rejection of exactly `error`, the SyntaxError of building the function (_engine_error).
    PHASE_5 Group 6 (the review of group 5d, F3): before, any unhandled rejection matched. Since group 6 one console
    error says that a function cannot be built, with exactly `error`, so the same fault shows in that form: the
    previous selection stays and the message says that the previous function is kept (F4), or, with no function in
    force before, no row is selected and the message says that the function is refused."""
    if (state["mask"] == [True] * N and not report.console_errors
            and report.page_errors == [_unhandled(error)]):
        return True
    return ((state["mask"] == [True] * N and _said_not_built(report, error, KEPT))
            or (state["mask"] == [False] * N and _refused_not_built(report, error)))


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("name", ODD_COLUMNS)
def test_eval_works_on_a_page_with_a_column_named_like_a_word_of_javascript(node_session, tmp_path, name):
    """A page that shows a column named default, static, let or arguments, and the query `return eval("A > 1")`, with
    no name that is both a column and a parameter. The query must select the rows with A > 1, and nothing is logged.
    Group 5c gave the function every column of the page, also that one, and the function could not be built: the
    previous selection (every row) stayed, with no console error and one page error, the unhandled SyntaxError."""
    df = _odd_column_frame(name)
    page, report = _node_page(node_session, tmp_path, "query_odd_column", df)
    error = _engine_error(page, [name], 'return eval("A > 1")')
    assert error is not None and error["name"] == ODD_COLUMN_FAILURE, error
    _node_set(page, "TextAreaInput", "value", 'return eval("A > 1")')
    state = _hist_state(page)
    regression(_query_not_built(report, state, error),
               f"P1-G5c-01: the column {name} breaks the query: {report.page_errors}")
    assert_histogram(df, state, query_oracle("return A > 1"))
    _assert_no_errors(report)


TEXT_WITH_ODD_COLUMN = {
    "default_in_a_comment": ("default", "return A > 1 // the default cut"),
    "let_inside_a_word": ("let", "return A > 1 // a complete selection"),
}


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("case", list(TEXT_WITH_ODD_COLUMN))
def test_a_query_whose_text_contains_the_name_of_such_a_column_works(node_session, tmp_path, case):
    """A page that shows a column named default (or let), and the query `return A > 1 // the default cut` (or
    `// a complete selection`, which contains "let"). The query does not use the column; it must select the rows with
    A > 1, and nothing is logged. Since the columns were chosen by a plain search of the text (before group 5c, and in
    group 5c too), the function was given that column, and could not be built: the previous selection (every row)
    stayed, with no console error and one page error, the unhandled SyntaxError."""
    name, text = TEXT_WITH_ODD_COLUMN[case]
    df = _odd_column_frame(name)
    page, report = _node_page(node_session, tmp_path, "query_odd_column", df)
    error = _engine_error(page, [name], text)
    assert error is not None and error["name"] == ODD_COLUMN_FAILURE, error
    _node_set(page, "TextAreaInput", "value", text)
    state = _hist_state(page)
    regression(_query_not_built(report, state, error),
               f"the column {name} breaks the query {text!r}: {report.page_errors}")
    assert_histogram(df, state, query_oracle("return A > 1"))
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("name", ODD_COLUMNS + ["n-bins"])
def test_a_function_using_eval_built_with_such_a_column_works(node_session, tmp_path, name):
    """The model alone, as an alias written in JavaScript may build it: automatic fields A, B and the column `name`,
    the scalar function `return eval("A > 1")` and the vector function `return B.map((b, i) => eval("A[i] > 1"))`. It
    must be built and get the fields A and B; with A = 3 the scalar function gives true, and with A = [3, 0.5] the
    vector function gives [true, false]. Group 5c gave both functions the column `name` too, and building the model
    failed with a SyntaxError. A name such as n-bins reaches the model only this way; the page builder leaves it out."""
    page, report = _node_page(node_session, tmp_path, "query_parameter", frame())
    got = page.evaluate(BUILD_WITH_ODD_COLUMN, {"name": name})
    # Group 6: one console error says that a function cannot be built, so the same fault now shows in that form
    regression(got.get("error", "").startswith(ODD_COLUMN_FAILURE)
               or any(NOT_BUILT in e and ODD_COLUMN_FAILURE in e for e in report.console_errors),
               f"P1-G5c-01: the model with {name}: {got} {report.console_errors}")
    assert got == {"fields": ["A", "B"], "scalar": True, "vector": [True, False]}, got
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_control_a_shared_name_is_refused_on_a_page_with_such_a_column(node_session, tmp_path):
    """The control, right before and after the fix: a page that shows a column named default, a parameter A next to
    the column A, and the query `return eval("A > 1")`. Decision N1: the query is refused by name, selects no row, and
    one error names A. The refusal looks at every column, also those the function is not given."""
    page, report = _node_page(node_session, tmp_path, "query_odd_column", _odd_column_frame("default"))
    page.evaluate(INJECT, {"name": "A"})
    _node_set(page, "TextAreaInput", "value", 'return eval("A > 1")')
    state = _hist_state(page)
    assert state["mask"] == [False] * N, state["mask"]
    assert not report.page_errors, report.page_errors
    assert len(report.console_errors) == 1 and REFUSED_A in report.console_errors[0], report.console_errors


# PHASE_5 Group 6 (the review of group 5d: Claude10.RI F1; GPT33.RI's summary F2, F4 and F5): a function that
# JavaScript cannot build. The forms: a text query with a typing error; a vector function given a column or a
# parameter named like one of its own arguments (data_source, $output); fields listed by hand whose names cannot be
# variables. Before, the error left the model: on a page a query kept its previous selection, and the only sign was an
# unhandled rejection; a model built directly threw. Now one console error says that the function cannot be built and
# why. A text query with a typing error keeps its previous selection (F4, the Architect, 2026-11-21: "Keep the previous
# selections."; a red frame on the widget comes later, by a bug report). A function with no previous function, such
# as a model built directly, is refused by name, as rule N1 refuses a shared name: NaN for every row.
# Also (F1, the Architect, 2026-11-21: "Column names has to be plain ascii. We should force it."): a function that uses
# a column whose name is not plain ASCII, such as sigma-cut, is refused by name.
NOT_BUILT = "cannot be built"
KEPT = "the previous function is kept"
NOT_BUILT_REFUSED = "it is refused and gives NaN"
NOT_ASCII_COLUMN = f'the column "{NON_ASCII}" cannot be used in the function'
NOT_ASCII_RULE = "a column used in a function needs a plain ASCII name"
TYPING_ERRORS = {"unfinished": "return A >", "unbalanced": "return ((A > 1"}
BUILD_MODEL = """a => {
  const q = window.__riH.models().filter(m => m.type === 'CustomJSNAryFunction' && m.func === 'return true');
  if (q.length !== 1) throw Error('query function: ' + q.length);
  try {
    const m = new q[0].constructor(a.attrs);
    const fields = m.get_fields();
    const got = {fields};
    if (a.attrs.func) got.scalar = String(m.compute(fields.map(f => a.columns[f][0])));
    if (a.attrs.v_func) got.vector = Array.from(m.v_compute(fields.map(f => a.columns[f]), null), String);
    return got;
  } catch (error) {
    return {error: String(error)};
  }
}"""


def _said_not_built(report, error, fate) -> bool:
    """No page error, and one console error that says that the function cannot be built, names the cause `error` (a
    text, or the {name, message} of _engine_error), and says what follows, `fate`: KEPT or NOT_BUILT_REFUSED."""
    if isinstance(error, dict):
        error = f"{error['name']}: {error['message']}"
    return (not report.page_errors and len(report.console_errors) == 1 and NOT_BUILT in report.console_errors[0]
            and error in report.console_errors[0] and fate in report.console_errors[0])


def _refused_not_built(report, error) -> bool:
    """Refused because the function cannot be built, and there is no previous function to keep."""
    return _said_not_built(report, error, NOT_BUILT_REFUSED)


def _refused_not_ascii(report) -> bool:
    """F1: refused because it uses the column NON_ASCII: no page error, one console error that names the column and
    the rule."""
    return (not report.page_errors and len(report.console_errors) == 1
            and NOT_ASCII_COLUMN in report.console_errors[0] and NOT_ASCII_RULE in report.console_errors[0])


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("case", list(TYPING_ERRORS))
def test_a_query_with_a_typing_error_keeps_the_previous_selection_and_says_why(node_session, tmp_path, case):
    """The user types `return A > 1`, then a query with a typing error (`return A >`, or `return ((A > 1`). The
    previous selection (A > 1) must stay (F4, the Architect: "Keep the previous selections."), there is no page error,
    and one console error says that the function cannot be built, with JavaScript's SyntaxError, and that the previous
    function is kept. Before, the previous selection stayed too, but with no console error and one page error, the
    unhandled rejection of the SyntaxError: nothing told the user that the new text was not used."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query", df)
    _node_set(page, "TextAreaInput", "value", "return A > 1")
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1"))
    text = TYPING_ERRORS[case]
    error = _engine_error(page, [], text)
    assert error is not None and error["name"] == "SyntaxError", error
    _node_set(page, "TextAreaInput", "value", text)
    state = _hist_state(page)
    regression(state["mask"] == query_oracle("return A > 1").tolist() and not report.console_errors
               and report.page_errors == [_unhandled(error)],
               f"F4: the query {text!r} keeps the previous selection silently: {report.page_errors}")
    assert_histogram(df, state, query_oracle("return A > 1"))
    assert _said_not_built(report, error, KEPT), (report.page_errors, report.console_errors)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_control_after_a_typing_error_the_kept_query_follows_its_parameter(node_session, tmp_path):
    """The control, right before and after the fix: the page with the parameter cut (a spinner, 1 at load) and the
    query `return A > cut`, then the query with a typing error `return A > cut +`. The kept query must still follow
    its parameter: with cut = 3 it selects the rows with A > 3. Then the corrected query `return A > cut` selects
    again. Only the console error (after the fix) or the page error (before) of the typing error is recorded."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _node_set(page, "TextAreaInput", "value", "return A > cut +")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _node_set(page, "Spinner", "value", 3., title="cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=3.))
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=3.))
    assert len(report.page_errors) + len(report.console_errors) == 1, (report.page_errors, report.console_errors)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_control_a_valid_query_after_a_typing_error_selects_again(node_session, tmp_path):
    """The control, right before and after the fix: after a query with a typing error, the corrected query
    `return A > 1` selects the rows with A > 1. Only the console error (after the fix) or the page error (before)
    of the typing error is recorded."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query", df)
    _node_set(page, "TextAreaInput", "value", "return A >")
    _node_set(page, "TextAreaInput", "value", "return A > 1")
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1"))
    assert len(report.page_errors) + len(report.console_errors) == 1, (report.page_errors, report.console_errors)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_after_a_typing_error_that_removes_the_parameter_the_kept_query_follows_it(node_session, tmp_path):
    """The page with the parameter cut (a spinner, 1 at load) and the query `return A > cut`; then the user deletes
    the parameter's name, and the text `return A >` cannot be built. The kept query must follow its parameter: with
    cut = 3 it selects the rows with A > 3. Before, the page asked the new text whether it uses cut, so the kept query
    no longer followed its parameter: the selection stayed A > 1, with no console error and one page error, the
    unhandled SyntaxError."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    _node_set(page, "TextAreaInput", "value", "return A >")
    error = _engine_error(page, [], "return A >")
    _node_set(page, "Spinner", "value", 3., title="cut")
    state = _hist_state(page)
    regression(state["mask"] == query_oracle("return A > cut", cut=1.).tolist() and not report.console_errors
               and report.page_errors == [_unhandled(error)],
               f"F4: the kept query does not follow its parameter: {report.page_errors}")
    assert_histogram(df, state, query_oracle("return A > cut", cut=3.))
    assert _said_not_built(report, error, KEPT), (report.page_errors, report.console_errors)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("name", ["data_source", "$output"])
def test_a_column_named_like_an_argument_of_the_vector_function_is_refused_by_name(node_session, tmp_path, name):
    """The model alone: automatic columns A and `name` (data_source or $output, the names of the vector function's
    own arguments), the scalar function `return eval("A > 1")` and the vector function
    `return A.map((a) => eval("a > 1"))`, which may read any name and so get every column. The scalar function must
    compute (true for A = 3); the vector function must be refused by name: NaN for each row, and one console error
    that names `name`. Before, building the model failed with JavaScript's SyntaxError for a name given twice."""
    page, report = _node_page(node_session, tmp_path, "query_parameter", frame())
    attrs = {"parameters": {}, "fields": ["A", name], "auto_fields": True,
             "func": 'return eval("A > 1")', "v_func": 'return A.map((a) => eval("a > 1"))'}
    got = page.evaluate(BUILD_MODEL, {"attrs": attrs, "columns": {"A": [3., .5], name: [7., 7.]}})
    regression(got.get("error", "").startswith("SyntaxError") and not report.console_errors,
               f"F2: building the model with the column {name} fails: {got}")
    assert got == {"fields": ["A", name], "scalar": "true", "vector": ["NaN", "NaN"]}, got
    assert _refused_not_built(report, f'the name "{name}" is given to it twice'), report.console_errors


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("name", ["default", "n-bins"])
def test_columns_listed_by_hand_whose_names_cannot_be_variables_are_refused_by_name(node_session, tmp_path, name):
    """The model alone, with its columns listed by hand (no automatic columns): A and `name`, and the function
    `return A > 1`. The columns are passed by position, so none can be left out. The function must be refused by
    name: NaN, and one console error that says that it cannot be built, with JavaScript's SyntaxError. Before,
    building the model failed with that SyntaxError."""
    page, report = _node_page(node_session, tmp_path, "query_parameter", frame())
    error = _engine_error(page, ["A", name], "return A > 1")
    assert error is not None and error["name"] == "SyntaxError", error
    attrs = {"parameters": {}, "fields": ["A", name], "auto_fields": False, "func": "return A > 1"}
    got = page.evaluate(BUILD_MODEL, {"attrs": attrs, "columns": {"A": [3., .5], name: [7., 7.]}})
    regression(got == {"error": f"{error['name']}: {error['message']}"} and not report.console_errors,
               f"F5: building the model with the column {name} listed by hand fails: {got}")
    assert got == {"fields": ["A", name], "scalar": "NaN"}, got
    assert _refused_not_built(report, error), report.console_errors


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.parametrize("auto_fields", [True, False], ids=["automatic", "by_hand"])
def test_a_function_that_uses_a_column_whose_name_is_not_plain_ascii_is_refused_by_name(node_session, tmp_path,
                                                                                         auto_fields):
    """The model alone: the columns A and sigma-cut (a JavaScript name, not plain ASCII), automatic or listed by hand,
    and the function `return A > 0 && sigma-cut > 1`. The Architect: column names are plain ASCII, and this is
    enforced. The function must be refused by name: NaN, and one console error that names the column and the rule.
    Before, with automatic columns the column was left out and computing failed with a ReferenceError; listed by hand
    it was given, and the function computed (true)."""
    page, report = _node_page(node_session, tmp_path, "query_parameter", frame())
    attrs = {"parameters": {}, "fields": ["A", NON_ASCII], "auto_fields": auto_fields,
             "func": f"return A > 0 && {NON_ASCII} > 1"}
    got = page.evaluate(BUILD_MODEL, {"attrs": attrs, "columns": {"A": [3., .5], NON_ASCII: [2., .5]}})
    before = ({"error": f"ReferenceError: {NON_ASCII} is not defined"} if auto_fields
              else {"fields": ["A", NON_ASCII], "scalar": "true"})
    regression(got == before and not report.console_errors,
               f"F1: the column {NON_ASCII} is used without a refusal: {got}")
    assert got == {"fields": ["A"] if auto_fields else ["A", NON_ASCII], "scalar": "NaN"}, got
    assert _refused_not_ascii(report), report.console_errors


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_query_that_uses_a_column_whose_name_is_not_plain_ascii_is_refused_on_a_page(node_session, tmp_path):
    """A page that shows the column sigma-cut, and the query `return sigma-cut > 1`. The query must be refused by
    name: it selects no row, there is no page error, and one console error names the column and the rule. Before, the
    page builder did not give the query the column, and computing the query failed with a ReferenceError: the previous
    selection (every row) stayed, and the only console error was "ReferenceError: sigma-cut is not defined"."""
    df = _odd_column_frame(NON_ASCII)
    page, report = _node_page(node_session, tmp_path, "query_odd_column", df)
    _node_set(page, "TextAreaInput", "value", f"return {NON_ASCII} > 1")
    state = _hist_state(page)
    regression(state["mask"] == [True] * N and not report.page_errors
               and report.console_errors == [f"ReferenceError: {NON_ASCII} is not defined"],
               f"F1: the query uses the column {NON_ASCII} without a refusal: {report.console_errors}")
    assert_histogram(df, state, [False] * N)
    assert _refused_not_ascii(report), (report.page_errors, report.console_errors)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_control_a_query_that_does_not_use_the_column_whose_name_is_not_plain_ascii_works(node_session, tmp_path):
    """The control, right before and after the fix: the same page, and the query `return A > 1`, which does not use
    the column sigma-cut. It must select the rows with A > 1, and nothing is logged."""
    df = _odd_column_frame(NON_ASCII)
    page, report = _node_page(node_session, tmp_path, "query_odd_column", df)
    _node_set(page, "TextAreaInput", "value", "return A > 1")
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1"))
    _assert_no_errors(report)


def _alone_doc(tmp_path):
    return _build_doc(tmp_path, "query_alone", frame())


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_strict_a_toggleable_text_query_alone_loads_and_selects_every_row(node_session, tmp_path):
    """WG-D1 (fixed in PHASE_5 Batch 1): the dashboard template's toggleable text query alone on its table, disabled
    at load, broke the page. Its control is A3's (a second active filter) and
    test_control_tq_d2_a_toggle_with_the_query_shown_logs_nothing."""
    failure = _load_failure(node_session, _alone_doc(tmp_path))
    regression(failure == WG_D1, "WG-D1: the page raises at load")
    assert failure is None, failure
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_alone", df)
    assert_histogram(df, _hist_state(page), np.ones(N, dtype=bool))


def _is_cs_d1(state, expected) -> bool:
    return (state["modelValue"] is None and "".join(map(str, state["components"])) == expected
            and state["axisLabel"] == expected)


@pytest.mark.node_display
@pytest.mark.feature("DISPLAY.axis_label")
@pytest.mark.parametrize("kind,model,states", [
    pytest.param("axis_plain", "Spinner", [5, 1, 2], id="plain"),
    pytest.param("axis_options", "Select", ["B", "None", "A"], id="option-metadata-none"),
])
def test_control_cs_d1_after_each_change_the_value_and_the_axis_follow(node_session, tmp_path, kind, model, states):
    """CS-D1's control: from the first change on, the model value and the axis text equal the oracle at every state."""
    page, report = _node_page(node_session, tmp_path, kind, frame())
    for value in states:
        _node_set(page, model, "value", value, title="scale" if kind == "axis_plain" else "varY")
        _assert_axis(page, axis_oracle(value, option=(kind == "axis_options")))
    _assert_no_errors(report)


DS_COLUMNS_JS = """() => {
  const ds = window.__riH.models().filter(m => m.type === 'DownsamplerCDS' && m.name === 'default cds');
  if (ds.length !== 1) throw Error('unique source DownsamplerCDS absent: ' + ds.length);
  const d = ds[0];
  return {length: d.get_length(), index: Array.from(d.get_column('index')), x: Array.from(d.get_column('x'))};
}"""


def _shrink(page, low, high):
    """nPointRender 20 (every selected row drawn): the full gate (12 rows), then a narrower one."""
    _node_set(page, "Spinner", "value", 20, title="nRender")
    _node_set(page, "RangeSlider", "value", [-2., 2.], title="gate")
    before = page.evaluate(DS_COLUMNS_JS)
    _node_set(page, "RangeSlider", "value", [low, high], title="gate")
    return before, page.evaluate(DS_COLUMNS_JS)


def _is_ds_d1(before, after, low, high) -> bool:
    n = int(gate_oracle(low, high).sum())
    return (after["length"] == n < before["length"] == len(before["index"]) == len(after["index"])
            and after["index"][:n] == np.flatnonzero(gate_oracle(low, high)).tolist()
            and after["index"][n:] == before["index"][n:])


@pytest.mark.node_display
@pytest.mark.feature("DISPLAY.downsampler")
def test_control_ds_d1_when_the_selection_grows_the_columns_match_the_length(node_session, tmp_path):
    """DS-D1's control: 6 rows, then 12: every column has exactly get_length() entries, the selected rows."""
    page, report = _node_page(node_session, tmp_path, "sample", frame())
    _node_set(page, "Spinner", "value", 20, title="nRender")
    for low, high in ((-1., 1.), (-2., 2.)):
        _node_set(page, "RangeSlider", "value", [low, high], title="gate")
        state = page.evaluate(DS_COLUMNS_JS)
        assert state["length"] == len(state["index"]) == len(state["x"]), state
        assert state["index"] == np.flatnonzero(gate_oracle(low, high)).tolist(), state
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("DISPLAY.downsampler")
def test_strict_after_the_selection_shrinks_every_column_has_get_length_entries(node_session, tmp_path):
    """DS-D1 (fixed in PHASE_5 Batch 1): 12 rows drawn, then 6. On master get_length() was 6 while the columns kept 12
    entries, the last 6 the old entries 6..11. Now every column has exactly the 6 selected rows."""
    page, report = _node_page(node_session, tmp_path, "sample", frame())
    before, after = _shrink(page, -1., 1.)
    regression(_is_ds_d1(before, after, -1., 1.), "DS-D1: old entries kept")
    assert after["length"] == len(after["index"]) == len(after["x"]), after
    assert after["index"] == np.flatnonzero(gate_oracle(-1., 1.)).tolist(), after
    _assert_no_errors(report)


# ======================================================= PHASE_5 Group 6c: the state of a text field (TQ-D3 and TQ-D4)
# TQ-D3 (the Architect, 2026-11-21: "red frame or something similar"): a text that becomes a function (a text query,
# or the text widget of an alias) shows its state on its field. Normal: unchanged. Orange: typed, not applied yet.
# Red, with the reason in the title: the text cannot be built (the previous function is kept), the function is
# refused, or computing it failed (the previous result is kept). Enter applies the text, Shift+Enter adds a new line;
# leaving the field applies it, as before. A query switched off by its "disable" toggle shows no colour.
# TQ-D4: a function that throws while computing, on row 3, left rows 0 to 2 with the new result and the rest with the
# old one. The previous result is now kept whole.
# Before the fix the field showed nothing: no class, the title unchanged, and the model had no property "problem".
FIELD_STATE = r"""(name) => {
  const ms = window.__riH.models();
  const area = ms.find(m => m.type === 'TextAreaInput' && (name == null || m.name === name));
  const functions = ms.filter(m => m.type === 'CustomJSNAryFunction');
  return {classes: [...area.css_classes], title: area.title, problems: functions.map(f => f.problem ?? null)};
}"""
KEPT_MASK = [False, True, True, True, True, False, True, True, True, True, True, True]          # return A > 0.5
MIXED_MASK = [False, False, True, True, True, False, True, True, True, True, True, True]          # TQ-D4, before


def _field(page, name=None):
    state = page.evaluate(FIELD_STATE, name)
    state["problem"] = next((p for p in state.pop("problems") if p), state.get("problem"))
    return state


def _shows_nothing(state) -> bool:
    """Before the fix: no class, no title, and no property "problem" on the model (None)."""
    return state["classes"] == [] and state["title"] == "" and state["problem"] is None


def _is_red(state, start, end) -> bool:
    return (state["classes"] == ["ri-problem"] and state["problem"].startswith(start)
            and state["problem"].endswith(end) and state["title"] == state["problem"])


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_query_that_cannot_be_built_is_red_and_says_why(node_session, tmp_path):
    """After `return A > 0.5`, the text `return A >`: the previous selection stays (group 6), and the field turns red
    with the reason in its title: "not used: SyntaxError: ...; the previous one is kept"."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > 0.5")
    _node_set(page, "TextAreaInput", "value", "return A >")
    state = _field(page)
    regression(_shows_nothing(state), f"TQ-D3: the field shows nothing: {state}")
    assert _hist_state(page)["mask"] == KEPT_MASK
    assert _is_red(state, "not used: SyntaxError", "; the previous one is kept"), state


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_refused_query_is_red_and_says_why(node_session, tmp_path):
    """A query that uses the column sigma-cut, whose name is not plain ASCII (group 6, F1): no row is selected, and
    the field turns red: "refused: the column "sigma-cut" has no plain ASCII name; no result (NaN)"."""
    df = _odd_column_frame(NON_ASCII)
    page, report = _node_page(node_session, tmp_path, "query_odd_column", df)
    _node_set(page, "TextAreaInput", "value", f"return {NON_ASCII} > 1")
    state = _field(page)
    regression(_shows_nothing(state), f"TQ-D3: the field shows nothing: {state}")
    assert _hist_state(page)["mask"] == [False] * N
    assert state["problem"] == f'refused: the column "{NON_ASCII}" has no plain ASCII name; no result (NaN)', state
    assert state["classes"] == ["ri-problem"] and state["title"] == state["problem"], state


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_query_that_fails_while_computing_is_red_and_keeps_the_previous_selection(node_session, tmp_path):
    """After `return A > 0.5`, the query `return Ax > 1`, with a column name that does not exist: it is built, and
    computing it fails on the first row. The previous selection stays, and the field turns red: "error while
    computing: ReferenceError: Ax is not defined; the previous result is kept"."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > 0.5")
    _node_set(page, "TextAreaInput", "value", "return Ax > 1")
    state = _field(page)
    regression(_shows_nothing(state), f"TQ-D3: the field shows nothing: {state}")
    assert _hist_state(page)["mask"] == KEPT_MASK
    assert state["problem"] == ("error while computing: ReferenceError: Ax is not defined; "
                                "the previous result is kept"), state
    assert state["classes"] == ["ri-problem"], state


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_query_that_fails_on_row_3_keeps_the_previous_selection_whole(node_session, tmp_path):
    """TQ-D4: after `return A > 0.5`, a query that throws on row 3. The previous selection must stay whole. Before,
    rows 0 to 2 had the new result and rows 3 to 11 the old one: a selection that is neither."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > 0.5")
    _node_set(page, "TextAreaInput", "value", "if (A > 2) throw new Error('row ' + A); return A > 1")
    mask = _hist_state(page)["mask"]
    regression(mask == MIXED_MASK, f"TQ-D4: a mixed selection: {mask}")
    assert mask == KEPT_MASK, mask
    state = _field(page)
    assert state["problem"] == "error while computing: Error: row 3; the previous result is kept", state


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_a_corrected_query_clears_the_red_field(node_session, tmp_path):
    """After a text that cannot be built, a valid query: it selects, and the field is normal again (no class, the
    title as before). Before the fix the field was never marked, so this checks only that it is not marked."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A >")
    _node_set(page, "TextAreaInput", "value", "return A > 1")
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1"))
    state = _field(page)
    assert state["classes"] == [] and state["title"] == "" and state["problem"] in ("", None), state


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_control_a_valid_query_shows_no_state(node_session, tmp_path):
    """The control, right before and after the fix: a valid query selects, and the field is not marked."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > 1")
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1"))
    state = _field(page)
    assert state["classes"] == [] and state["title"] == "" and state["problem"] in ("", None), state
    _assert_no_errors(report)


ALIAS_VALUES = r"""() => Array.from(window.__riH.models()
  .filter(m => m.type === 'CDSAlias' && m.mapping && m.mapping.funA)[0].get_column('funA'), Number)"""


def _set_text(page, value):
    """Set the page's one text widget (the alias_text page has no histogram for _node_set to wait on)."""
    page.evaluate("""async v => { window.__riH.models().find(m => m.type === 'TextAreaInput').value = v;
                                  await Promise.resolve(); }""", value)
    _settle(page)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_the_text_widget_of_an_alias_is_red_when_its_text_cannot_be_built(node_session, tmp_path):
    """The alias funA computes the text of the parameter funForm, edited in a text widget (the form of the dashboard
    template's funCustomForm0). The text `return A *` cannot be built: the alias keeps its values (A * 2), and the text
    widget turns red with the reason in its title, after the widget's own title."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "alias_text", df)
    assert page.evaluate(ALIAS_VALUES) == [2 * a for a in df["A"]]
    _set_text(page, "return A *")
    state = _field(page, "funForm")
    regression(state["classes"] == [] and state["title"] == "funForm" and state["problem"] is None,
               f"TQ-D3: the text widget shows nothing: {state}")
    assert page.evaluate(ALIAS_VALUES) == [2 * a for a in df["A"]]
    assert state["classes"] == ["ri-problem"], state
    assert state["problem"].startswith("not used: SyntaxError") and state["title"] == "funForm: " + state["problem"]
    _set_text(page, "return A + 1")
    assert page.evaluate(ALIAS_VALUES) == [a + 1 for a in df["A"]]
    assert _field(page, "funForm")["classes"] == []


# --------------------------------------------- the browser: what the user sees (colours, pixels, keys, the template)
RED_FRAME, RED_FILL, RED_TEXT = (211, 47, 47), (255, 235, 238), "rgb(211, 47, 47)"
ORANGE_FRAME, ORANGE_FILL = (245, 124, 0), (255, 243, 224)
AREA_OF = r"""(a) => {
  const ms = window.__riH.models().filter(m => m.type === 'TextAreaInput' && (a.name == null || m.name === a.name));
  const m = ms[a.index ?? 0];
  return [...Bokeh.index.query((v) => v.model === m)][0].input_el;
}"""
PIXELS = r"""async (a) => {
  const bytes = Uint8Array.from(atob(a.png), (c) => c.charCodeAt(0));
  const bmp = await createImageBitmap(new Blob([bytes], {type: 'image/png'}));
  const canvas = new OffscreenCanvas(bmp.width, bmp.height), g = canvas.getContext('2d');
  g.drawImage(bmp, 0, 0);
  const d = g.getImageData(0, 0, bmp.width, bmp.height).data, w = bmp.width, h = bmp.height;
  const near = (x, y, rgb, tol) => { const i = 4 * (y * w + x);
    return Math.abs(d[i] - rgb[0]) <= tol && Math.abs(d[i + 1] - rgb[1]) <= tol && Math.abs(d[i + 2] - rgb[2]) <= tol; };
  // the frame: along each side, the share of positions with the frame colour among the 4 outermost pixels (the
  // element may sit half a pixel off the screenshot's grid); the third best of the four sides, because a layout can
  // hide one edge of a field (the dashboard template cuts the last 2 pixels of its text fields)
  const side = (n, at) => { let hit = 0; for (let k = 6; k < n - 6; k++) if ([0, 1, 2, 3].some((j) => at(k, j))) hit++;
    return hit / Math.max(1, n - 12); };
  const sides = [side(w, (k, j) => near(k, j, a.frame, 16)), side(w, (k, j) => near(k, h - 1 - j, a.frame, 16)),
                 side(h, (k, j) => near(j, k, a.frame, 16)), side(h, (k, j) => near(w - 1 - j, k, a.frame, 16))];
  // the inside, 6 pixels in from the edge: the share with the light background (the text covers the rest)
  let inner = 0, hit = 0;
  for (let y = 6; y < h - 6; y++) for (let x = 6; x < w - 6; x++) { inner++; if (near(x, y, a.fill, 6)) hit++; }
  return {frame: sides.sort((x, y) => y - x)[2], fill: hit / Math.max(1, inner)};
}"""


def _states_dir() -> Path:
    """The screenshots and pages of these tests, for people to look at (in the test output folder)."""
    d = Path(os.environ.get("RI_HEADLESS_OUT", Path(__file__).resolve().parent / "_out")) / "text_field_states"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _area(page, name=None, index=0):
    return page.evaluate_handle(AREA_OF, {"name": name, "index": index}).as_element()


def _seen(page, area, shot=None) -> dict:
    """What the browser computes for the field, and what it draws: the share of the frame's pixels in red and in
    orange, and of the inside in their light backgrounds. `shot` saves the screenshot under that name."""
    import base64
    png = area.screenshot()
    if shot:
        (_states_dir() / f"{shot}.png").write_bytes(png)
    b64 = base64.b64encode(png).decode()
    red = page.evaluate(PIXELS, {"png": b64, "frame": RED_FRAME, "fill": RED_FILL})
    orange = page.evaluate(PIXELS, {"png": b64, "frame": ORANGE_FRAME, "fill": ORANGE_FILL})
    style = area.evaluate("e => { const s = getComputedStyle(e); return [s.borderTopColor, s.backgroundColor] }")
    return {"red": red, "orange": orange, "border": style[0], "background": style[1]}


def _drawn(seen, colour) -> bool:
    """The frame is drawn in `colour` on at least three of its four sides (at 3 of 4 positions of a side at least), and
    the inside in its light background (at least half of the pixels; the text covers the rest)."""
    return seen[colour]["frame"] >= 0.75 and seen[colour]["fill"] >= 0.5


def _plain(seen) -> bool:
    """Neither red nor orange is drawn."""
    return max(seen["red"]["frame"], seen["orange"]["frame"], seen["red"]["fill"], seen["orange"]["fill"]) < 0.1


def _title_colour(page, area) -> str:
    return area.evaluate("e => getComputedStyle(e.closest('.bk-input-group')?.querySelector('label') ?? e).color")


@pytest.mark.browser_display
@pytest.mark.feature("FILTER.text_query")
def test_browser_a_text_query_shows_typing_and_failure_in_colour_and_applies_with_enter(headless_session, tmp_path):
    """In Chromium, with real keys: the field is normal at load; orange while the user types; Enter applies the text
    (normal again, and the query selects); Shift+Enter adds a new line without applying; a text that cannot be built
    turns the field red after Enter, with the reason in red in the title, and the previous selection stays; typing
    again turns it orange; leaving the field applies the corrected text, and the field is normal. Each state is
    checked as the browser computes it and as it draws it (pixels), and saved as a screenshot."""
    df = frame()
    ctx, page, errors = _browser_page(headless_session, tmp_path, "query", df)
    try:
        area = _area(page)
        seen = _seen(page, area, "query_1_normal_at_load")
        assert _plain(seen), seen
        area.click()
        area.press("Control+A")
        area.type("return A > 1")
        seen = _seen(page, area, "query_2_orange_typed_not_applied")
        regression(_plain(seen), f"TQ-D3: typing shows nothing: {seen}")
        assert _drawn(seen, "orange"), seen
        area.press("Enter")
        page.wait_for_function("() => window.__riH.models().some(m => m.type === 'TextAreaInput' && m.value === 'return A > 1')")
        _settle(page, HIST_PROBES)
        assert_histogram(df, _hist_state(page), query_oracle("return A > 1"))
        assert _plain(_seen(page, area, "query_3_normal_after_enter")), "after Enter the field must be normal"
        area.press("End")
        area.press("Shift+Enter")
        area.type("&& B < 2")
        assert page.evaluate("() => window.__riH.models().find(m => m.type === 'TextAreaInput').value") == "return A > 1"
        area.press("Enter")
        page.wait_for_function("() => window.__riH.models().some(m => m.type === 'TextAreaInput' && m.value === 'return A > 1\\n&& B < 2')")
        _settle(page, HIST_PROBES)
        assert_histogram(df, _hist_state(page), query_oracle("return A > 1 && B < 2"))
        area.press("Control+A")
        area.type("return A >")
        area.press("Enter")
        page.wait_for_function("() => window.__riH.models().some(m => m.type === 'TextAreaInput' && m.value === 'return A >')")
        _settle(page, HIST_PROBES)
        assert_histogram(df, _hist_state(page), query_oracle("return A > 1 && B < 2"))
        seen = _seen(page, area, "query_4_red_not_used")
        assert _drawn(seen, "red"), seen
        assert seen["border"] == "rgb(211, 47, 47)" and seen["background"] == "rgb(255, 235, 238)", seen
        assert _title_colour(page, area) == RED_TEXT
        title = page.evaluate("() => window.__riH.models().find(m => m.type === 'TextAreaInput').title")
        assert title.startswith("not used: SyntaxError") and title.endswith("the previous one is kept"), title
        area.type(" 1")
        seen = _seen(page, area, "query_5_orange_typing_after_a_failure")
        assert _drawn(seen, "orange"), seen
        page.locator("body").click(position={"x": 2, "y": 2})          # leaving the field applies the text
        page.wait_for_function("() => window.__riH.models().some(m => m.type === 'TextAreaInput' && m.value === 'return A > 1')")
        _settle(page, HIST_PROBES)
        assert_histogram(df, _hist_state(page), query_oracle("return A > 1"))
        assert _plain(_seen(page, area, "query_6_normal_after_leaving_the_field")), "after a correction it must be normal"
        assert page.evaluate("() => window.__riH.models().find(m => m.type === 'TextAreaInput').title") == ""
        assert [e for e in errors if "cannot be built" not in e] == [], errors
    finally:
        ctx.close()


def _template_page(headless_session, tmp_path):
    """The default dashboard template (getDefaultVars) on 200 rows of 4 columns: its 3 text queries, switched off by
    their "disable" toggles at load, and its 3 text fields of custom functions (funCustomForm0 to 2)."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import getDefaultVars
    from bokeh.plotting import output_file
    aliases, variables, parameters, widgets, widget_layout, histograms, figures, layout = getDefaultVars(
        normalization=None, variables=["A", "B", "C", "D"])
    rng = np.random.default_rng(20261121)
    df = pd.DataFrame(rng.random(size=(200, 4)), columns=list("ABCD"))
    p = tmp_path / "template_browser.html"
    output_file(str(p), mode="inline")
    sa.bokehDrawSA.fromArray(df, None, figures, widgets, layout=layout, parameterArray=parameters,
                             widgetLayout=widget_layout, histogramArray=histograms, aliasArray=aliases)
    shutil.copyfile(p, _states_dir() / "template_page.html")
    return _open_browser_page(headless_session, p)


@pytest.mark.browser_display
@pytest.mark.feature("FILTER.text_query")
def test_browser_the_dashboard_template_shows_the_state_of_its_six_text_fields(headless_session, tmp_path):
    """The default dashboard template, in Chromium, with real clicks and keys. In its tab "Custom" all 6 text fields
    are normal at load.
    The first query is switched on with its "disable" toggle, a text that cannot be built is typed and applied with
    Enter: only that field turns red. The text field funCustomForm0 gets a text that cannot be built: it turns red
    too, and the other fields stay normal. The query switched off again shows no colour. The template page and a
    screenshot of each state are saved."""
    ctx, page, errors = _template_page(headless_session, tmp_path)
    try:
        page.locator(".bk-tab", has_text="Custom").click()           # the template's tab of the six text fields
        n = page.evaluate("() => window.__riH.models().filter(m => m.type === 'TextAreaInput').length")
        assert n == 6, n
        areas = [_area(page, index=i) for i in range(n)]
        assert all(_plain(_seen(page, a)) for a in areas)
        names = page.evaluate("() => window.__riH.models().filter(m => m.type === 'TextAreaInput').map(m => m.name)")
        queries = [i for i, name in enumerate(names) if not (name or "").startswith("funCustomForm")]
        forms = [i for i, name in enumerate(names) if (name or "").startswith("funCustomForm")]
        assert len(queries) == 3 and len(forms) == 3, names
        page.locator("button:has-text('disable')").first.click()
        page.wait_for_function("() => window.__riH.models().filter(m => m.type === 'TextAreaInput' && !m.disabled).length === 4")
        on = page.evaluate("() => window.__riH.models().filter(m => m.type === 'TextAreaInput').findIndex(m => !m.disabled && !(m.name || '').startsWith('funCustomForm'))")
        query = areas[on]
        query.click()
        query.press("Control+A")
        query.type("return A >")
        seen = _seen(page, query, "template_1_query_orange_typed")
        regression(_plain(seen), f"TQ-D3: typing in the template's query shows nothing: {seen}")
        assert _drawn(seen, "orange"), seen
        query.press("Enter")
        page.wait_for_function("(i) => window.__riH.models().filter(m => m.type === 'TextAreaInput')[i].value === 'return A >'", arg=on)
        _settle(page)
        assert _drawn(_seen(page, query, "template_2_query_red"), "red")
        form_index = names.index("funCustomForm0")
        form = areas[form_index]
        form.click()
        form.press("Control+A")
        form.type("return A *")
        form.press("Enter")
        page.wait_for_function("(i) => window.__riH.models().filter(m => m.type === 'TextAreaInput')[i].value === 'return A *'", arg=form_index)
        _settle(page)
        assert _drawn(_seen(page, form, "template_3_function_field_red"), "red")
        others = [i for i in range(n) if i not in (on, form_index)]
        assert all(_plain(_seen(page, areas[i])) for i in others), "only the two fields with a problem are red"
        page.locator("button:has-text('disable')").first.click()
        page.wait_for_function("(i) => window.__riH.models().filter(m => m.type === 'TextAreaInput')[i].disabled", arg=on)
        assert _plain(_seen(page, query, "template_4_query_switched_off_no_colour")), "a query switched off shows no colour"
        page.screenshot(path=str(_states_dir() / "template_5_whole_page.png"), full_page=True)
        assert [e for e in errors if "cannot be built" not in e] == [], errors
    finally:
        ctx.close()
