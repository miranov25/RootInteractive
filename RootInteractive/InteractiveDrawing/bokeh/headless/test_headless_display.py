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
  TQ-D2  when a query's text becomes empty (cleared, or a toggleable query enabled while its text area is empty), its
         function is dropped (CustomJSNAryFunction.update_func), the previous selection stays, and the next query
         logs "TypeError: Function.prototype.apply was called on null" once (the text area's callback sets `func`,
         which recomputes the column before update_func builds the new function); the result is then right;
  CS-D1  an axis label's model value (ConcatenatedString.value) is undefined until its parameter first changes
         (#is_dirty starts false); the axis text itself is right, set by Python at build;
  DS-D1  after the selection shrinks, the downsampler's columns keep the old entries at the end (DownsamplerCDS
         get_column grows its arrays but never shrinks them): get_column(c).length > get_length(); the first
         get_length() entries are right and Bokeh paints get_length() rows (A4 observation O-A4-2, now a test);
  WG-D1  (A3, same signature) also for a toggleable TEXT QUERY alone on its table - the form the standard dashboard
         template (bokehInteractiveTemplate.py) uses three times.
Changes: the defect tests and helpers below; GPT's tests that hit a defect are its strict partners; the Chromium
axis-label tests check the axis text at load and the full state after the real input (the model value at load is
CS-D1, tested in Node). Targets node_display (Node) and browser_display (Chromium).
"""

import hashlib
import json

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
                "query_alone"}:
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
        if kind in ("query_parameter", "query_parameter_alias"):
            widgets.append(["spinner", ["cut"]])
            params = [{"name": "cut", "value": 1., "range": [0., 4.], "step": 1.}]
        if kind == "query_parameter_alias":
            aliases = [("acut", "A > cut")]                # TQ-D1 control: an alias that uses the parameter
        plots = [
            [["bin_center_0"], ["bin_count"], {"source": "h"}],
            # Request both query-only columns through the real source graph;
            # a query typed later cannot itself inform initial column loading.
            [["A"], ["B"], {"visualization_type": "scatter"}],
        ]
        extra["layout"] = [[0, 1]]
        extra["sourceArray"] = [{"name": "h", "variables": ["x", "y"],
                                 "nbins": [4, 3], "range": [[0, 4], [0, 3]], "weights": "w"}]
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
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine import vendor
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import HELPER_JS
    p = tmp_path / (kind + "_browser.html")
    output_file(str(p), mode="inline")
    _from_array(kind, df)
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
@pytest.mark.xfail(**STRICT, reason="TQ-D1: a text query does not get the page's parameters")
def test_node_textquery_parameter_changed_after_input(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    known_defect(_is_tq_d1_unknown(page, report), "TQ-D1: the query does not see the parameter")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _node_set(page, "Spinner", "value", 3., title="cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=3.))
    _node_set(page, "Spinner", "value", 1., title="cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _assert_no_errors(report)


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.xfail(**STRICT, reason="TQ-D2: enabling the query with an empty text area drops its function; "
                                    "the next query logs a TypeError")
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
    known_defect(_one_tq_d2_message(report.console_errors) and not report.page_errors, "TQ-D2: the logged TypeError")
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
@pytest.mark.xfail(**STRICT, reason="CS-D1: an axis label's model value is undefined until its parameter changes")
def test_node_axis_label_parameter_value_and_target(node_session, tmp_path, kind, model, states):
    df = frame()
    page, report = _node_page(node_session,tmp_path,kind,df)
    known_defect(_is_cs_d1(_axis_state(page), axis_oracle(states[0], option=(kind=="axis_options"))),
                 "CS-D1: the label's value is undefined at load")
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
        # integration: at load only the axis text is checked - the label's model value is undefined at load (CS-D1,
        # its own Node tests); after the real input below the full state (value and text) is checked
        assert _axis_state(page)["axisLabel"] == axis_oracle(2 if kind=="axis_plain" else "A",option=kind=="axis_options")
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
def test_signature_tq_d1_a_query_without_a_matching_alias_cannot_see_the_parameter(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter", df)
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    assert _is_tq_d1_unknown(page, report), (report.console_errors, _hist_state(page)["mask"])


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_signature_tq_d1_the_query_keeps_the_parameter_value_from_build_time(node_session, tmp_path):
    """With an alias that uses `cut` built last, the query sees cut = 1 (its build-time value) and keeps it: after the
    spinner moves to 3 the selection is still A > 1."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter_alias", df)
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))
    _node_set(page, "Spinner", "value", 3., title="cut")
    assert_histogram(df, _hist_state(page), query_oracle("return A > cut", cut=1.))      # not cut = 3
    assert not np.array_equal(query_oracle("return A > cut", cut=1.), query_oracle("return A > cut", cut=3.))
    _assert_no_errors(report)


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
@pytest.mark.xfail(**STRICT, reason="TQ-D1: a text query keeps the parameter's build-time value")
def test_strict_a_query_follows_its_parameter(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_parameter_alias", df)
    _node_set(page, "TextAreaInput", "value", "return A > cut")
    _node_set(page, "Spinner", "value", 3., title="cut")
    state = _hist_state(page)
    known_defect(state["mask"] == query_oracle("return A > cut", cut=1.).tolist(), "TQ-D1: build-time value kept")
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
def test_signature_tq_d2_after_a_cleared_query_the_next_query_logs_a_type_error(node_session, tmp_path):
    """TQ-D2: the empty text drops the function and the previous selection (A > 1) stays; the next query logs exactly
    one TypeError, raised in CustomJSNAryFunction.compute; the final selection is right."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query", df)
    after_clear, recorded = _clear_and_retype(page)
    assert _is_tq_d2(after_clear, recorded, report), (after_clear, recorded, report.console_errors)
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1 && B < 2"))


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
@pytest.mark.xfail(**STRICT, reason="TQ-D2: a cleared query drops its function; the next query logs a TypeError")
def test_strict_clearing_and_retyping_a_query_logs_nothing(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query", df)
    after_clear, recorded = _clear_and_retype(page)
    known_defect(_is_tq_d2(after_clear, recorded, report), "TQ-D2: the logged TypeError")
    assert_histogram(df, _hist_state(page), query_oracle("return A > 1 && B < 2"))
    _assert_no_errors(report)


def _alone_doc(tmp_path):
    return _build_doc(tmp_path, "query_alone", frame())


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
def test_signature_wg_d1_a_toggleable_text_query_alone_breaks_the_page(node_session, tmp_path):
    """WG-D1 (A3), the same exact signature, for the dashboard template's toggleable text query alone on its table."""
    assert _load_failure(node_session, _alone_doc(tmp_path)) == WG_D1


@pytest.mark.node_display
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.xfail(**STRICT, reason="WG-D1: a toggleable filter alone on its table breaks the page")
def test_strict_a_toggleable_text_query_alone_loads_and_selects_every_row(node_session, tmp_path):
    """WG-D1's control is A3's (a second active filter) and test_control_tq_d2_a_toggle_with_the_query_shown_logs_nothing."""
    failure = _load_failure(node_session, _alone_doc(tmp_path))
    known_defect(failure == WG_D1, "WG-D1")
    assert failure is None, failure
    df = frame()
    page, report = _node_page(node_session, tmp_path, "query_alone", df)
    assert_histogram(df, _hist_state(page), np.ones(N, dtype=bool))


def _is_cs_d1(state, expected) -> bool:
    return (state["modelValue"] is None and "".join(map(str, state["components"])) == expected
            and state["axisLabel"] == expected)


@pytest.mark.node_display
@pytest.mark.feature("DISPLAY.axis_label")
@pytest.mark.parametrize("kind,first", [("axis_plain", 2), ("axis_options", "A")], ids=["plain", "option-metadata"])
def test_signature_cs_d1_the_label_value_is_undefined_at_load(node_session, tmp_path, kind, first):
    """CS-D1: at load the label's model value is undefined, while its components and the axis text are right."""
    page, report = _node_page(node_session, tmp_path, kind, frame())
    assert _is_cs_d1(_axis_state(page), axis_oracle(first, option=(kind == "axis_options"))), _axis_state(page)
    _assert_no_errors(report)


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
def test_signature_ds_d1_after_the_selection_shrinks_the_columns_keep_old_entries(node_session, tmp_path):
    """DS-D1: 12 rows drawn, then 6: get_length() is 6, the columns still have 12 entries; the first 6 are the new rows,
    the last 6 are the old entries 6..11."""
    page, report = _node_page(node_session, tmp_path, "sample", frame())
    before, after = _shrink(page, -1., 1.)
    assert _is_ds_d1(before, after, -1., 1.), (before, after)
    _assert_no_errors(report)


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
@pytest.mark.xfail(**STRICT, reason="DS-D1: the downsampler's columns keep old entries after the selection shrinks")
def test_strict_after_the_selection_shrinks_every_column_has_get_length_entries(node_session, tmp_path):
    page, report = _node_page(node_session, tmp_path, "sample", frame())
    before, after = _shrink(page, -1., 1.)
    known_defect(_is_ds_d1(before, after, -1., 1.), "DS-D1: old entries kept")
    assert after["length"] == len(after["index"]) == len(after["x"]), after
