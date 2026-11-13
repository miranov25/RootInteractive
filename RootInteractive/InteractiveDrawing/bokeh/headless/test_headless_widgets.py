"""Interactive widgets in Node and Chromium — PHASE_4_5 stage A3 (basic). Oracle and invariant tests only.

Plain words: a WIDGET (slider, range, select, multiSelect, multiSelectBitmask, toggle, spinner, spinnerRange) either
selects rows through its filter, or changes a PARAMETER (here: the number of bins, the legend's visibility). After
every widget change the selected rows and the weighted 2-D histogram must equal numpy on the same data; a profile
must equal the binned formula. Pages are built by RootInteractive's own builder (`bokehDrawSA.fromArray`); Node runs
the production models from the page document, Chromium gets real keyboard, click and select input.

Source: GPT27.RI's reviewed candidate `test_headless_widgets_GPT27_RI_CANDIDATE_v0_2a.py` (proposal v0.2a, approved
for test-only integration by GPT26.RI and GPT29.RI), integrated by Claude20.RI. Its 26 cases are kept with their
names, data and oracles. Changes on integration (each needed to run on the pinned toolchain; measured):
  * the filter vector is longer than the table (32 entries for 12 rows): the row mask is its first n entries, as
    engine/runner.py `__riH.selected` reads it (all selection tests failed on a shape check before);
  * Chromium pages get their external scripts from the pinned local copies, every other request blocked - the
    engine's own rule (engine/runner.py `_browser_run`); without it the page needs the internet;
  * Bokeh 3.3.2 renders a Spinner as `<input class="bk-input" type="text">`, not type="number";
  * targets: `node_widgets` (Node) and `browser_widgets` (Chromium); the two Python controls stay `headless_unit`;
  * two defects found, each with an exact signature, a strict partner and a control (end of the file):
    WG-D1  a toggleable range widget alone on its table (disabled at load, its default) breaks the page;
    WG-D2  with a second filter the page loads, but the enabled and moved toggleable range still selects every row.
  The original candidate test of the toggleable range is WG-D1's strict partner.
  PHASE_5 Batch 1 fixed both: the strict marks are off, the signature tests removed (the Chromium one now checks
  that the page loads and shows the widget), and each exact signature is a regression (an ordinary failure).

Original candidate notes (GPT27.RI):
- A missing Node bundle, jsdom or Chromium is a skipped/environmental case under the existing fixtures, NOT an
  acceptance PASS. Set RI_HEADLESS_REQUIRED=1.
- Most pages isolate one control; B1 multi-filter semantics remain covered by existing strict signature/oracle tests
  in the headless scenario/filter families.
- All numeric fixture points are inside bins, not at the known B10 edge defect.
- @pytest.mark.feature uses existing taxonomy IDs for histogram/range/profile correctness; the taxonomy has no
  per-widget IDs.
- v0.2a C0: selection pages require a filter; parameter-only pages require filter == null and their weighted
  histogram equals the all-row oracle.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import HELPER_JS


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect (integration, Claude20)."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


STRICT = dict(strict=True, raises=KnownDefect)


def regression(condition: bool, message: str) -> None:
    """PHASE_5 Batch 1: the exact signature of a fixed defect is a regression - an ordinary failure."""
    if condition:
        raise AssertionError("REGRESSION, a fixed defect is back: " + message)


# Numerically unambiguous synthetic data; no boundary positions or NaN needed
# for testing interactive callbacks. Each row has a distinctive weight so a
# wrong selection or stale histogram is very unlikely to match accidentally.
X = np.asarray([0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75,
                0.5, 1.5, 2.5, 3.5], dtype=np.float64)
Y = np.asarray([0.25, 1.25, 2.25, 0.75, 1.75, 2.5, 1.5, 0.5,
                2.25, 0.25, 1.25, 2.75], dtype=np.float64)
GATE = np.asarray([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
                   2.0, -1.25, 0.75, 1.25], dtype=np.float64)
BITS = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 4, 7], dtype=np.int32)
WEIGHTS = np.asarray([1.0, 2.0, 1.5, 0.5, 3.0, 2.5, 4.0, 1.25,
                      2.25, 3.5, 0.75, 1.75], dtype=np.float64)
MAPPING = {"A": 1, "B": 2, "C": 4}
BINS = (4, 3)
RANGES = ((0.0, 4.0), (0.0, 3.0))
HIST_MODEL = ("h", ["HistoNdCDS"])
PROBES = [["h", "bin_count", ["HistoNdCDS"]]]


def frame() -> pd.DataFrame:
    return pd.DataFrame({"x": X.copy(), "y": Y.copy(), "gate": GATE.copy(),
                         "bits": BITS.copy(), "category": np.asarray(["A", "B", "A", "C", "B", "C", "A", "B", "C", "A", "B", "C"]),
                         "w": WEIGHTS.copy()})


def oracle_range(df, low, high):
    x = df["gate"].to_numpy()
    return np.isfinite(x) & (x >= low) & (x <= high)


def oracle_multiselect(df, selected, how):
    """Explicit reference for the supported bitmask policies; NOT RI code."""
    codes = df["bits"].to_numpy(dtype=np.int64)
    selection = 0
    for name in selected:
        selection |= MAPPING[name]
    if how == "any":
        return (codes & selection) != 0
    if how == "all":
        return (codes & selection) == selection
    raise ValueError("unexpected policy " + str(how))


def oracle_categories(df, selected):
    """Single-column categorical selection, independent of factor encodings."""
    return df["category"].isin(list(selected)).to_numpy(dtype=bool)


def oracle_bins(df, mask):
    """2-D binned sum of weights. x is fastest-changing axis in RI's output."""
    positions = df.loc[np.asarray(mask, dtype=bool), ["x", "y"]].to_numpy()
    weights = df.loc[np.asarray(mask, dtype=bool), "w"].to_numpy(dtype=np.float64)
    bins, _ = np.histogramdd(positions, bins=BINS, range=RANGES, weights=weights)
    return bins.ravel(order="F")       # independent NumPy operation; NOT RI histogram code


def assert_state(df, state, mask):
    """Failure must identify *both* selected-row and histogram differences."""
    assert isinstance(state, dict) and "mask" in state and "bins" in state, state
    expected_mask = np.asarray(mask, dtype=bool)
    actual_mask = np.asarray(state["mask"], dtype=bool)
    assert actual_mask.shape == expected_mask.shape, (actual_mask.shape, expected_mask.shape)
    np.testing.assert_array_equal(actual_mask, expected_mask, err_msg="widget -> selection mismatch")
    observed = np.asarray(state["bins"], dtype=np.float64)
    expected = oracle_bins(df, expected_mask)
    assert observed.shape == expected.shape, (observed.shape, expected.shape)
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12,
                               err_msg="selection -> HistoNdCDS weighted-bin mismatch")


def _widgets(kind, how="any"):
    """Production widget declarations; each path is used by bokehDrawSA.fromArray."""
    basic = {
        "range": [["range", ["gate", -2.0, 2.0]]],
        "multiselect": [["multiSelectBitmask", ["bits"],
                          {"mapping": MAPPING, "how": how, "title": "bits"}]],
        "categorical": [["multiSelect", ["category"]]],
        "slider_selection": [["slider", ["gate", -2.0, 2.0], {"bins": 8}]],
        "select_selection": [["select", ["category"]]],
        "spinner_range": [["spinnerRange", ["gate"], {"name": "gate_spinner"}]],
        "toggleable_range": [["range", ["gate", -2.0, 2.0], {"toggleable": True}]],
        # integration (WG-D1, WG-D2): the gate range, toggleable or plain, with a range on x that keeps every row,
        # in both orders of the two filters
        "toggleable_plus_range": [["range", ["gate", -2.0, 2.0], {"toggleable": True}], ["range", ["x", 0.0, 4.0]]],
        "range_plus_toggleable": [["range", ["x", 0.0, 4.0]], ["range", ["gate", -2.0, 2.0], {"toggleable": True}]],
        "plain_plus_range": [["range", ["gate", -2.0, 2.0]], ["range", ["x", 0.0, 4.0]]],
        "range_plus_plain": [["range", ["x", 0.0, 4.0]], ["range", ["gate", -2.0, 2.0]]],
        "parameter_slider": [["slider", ["nbinsX"]]],
        "parameter_select": [["select", ["nbinsX"]]],
        "parameter_spinner": [["spinner", ["nbinsX"]]],
        "parameter_toggle": [["toggle", ["legendVisible"], {"label": "show legend"}]],
    }
    if kind not in basic:
        raise ValueError(kind)
    return basic[kind]


def _parameters(kind):
    if kind == "parameter_select":
        return [{"name": "nbinsX", "value": 4, "options": [2, 3, 4, 6]}]
    if kind in {"parameter_slider", "parameter_spinner"}:
        return [{"name": "nbinsX", "value": 4, "range": [2, 8], "step": 1}]
    if kind == "parameter_toggle":
        return [{"name": "legendVisible", "value": True}]
    return []


def _from_array(df, kind, how="any"):
    """Actual production builder; models are not mocked or manually constructed."""
    from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import bokehDrawSA

    parameter_hist = kind in {"parameter_slider", "parameter_select", "parameter_spinner"}
    opts = {"source": "h"}
    if kind == "parameter_toggle":
        opts["legend_options"] = {"visible": "legendVisible"}
    return bokehDrawSA.fromArray(
        df, None,
        [[["bin_center_0"], ["bin_count"], opts]],
        _widgets(kind, how),
        sourceArray=[{"name": "h", "variables": ["x", "y"],
                      "nbins": ["nbinsX", 3] if parameter_hist else list(BINS),
                      "range": [list(p) for p in RANGES], "weights": "w"}],
        aliasArray=[], parameterArray=_parameters(kind),
        layout=[[0]], widgetLayout=[list(range(len(_widgets(kind, how))))],
        arrayCompression=[(".*", [("code", 0), ("zip", 0), ("base64", 0)])],
    )


def build_document(tmp_path: Path, df: pd.DataFrame, kind, how="any", disconnect=False):
    """Same build + show suppression + Bokeh Document path as engine/page.py."""
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa

    show = sa.show
    sa.show = lambda *args, **kwargs: None
    try:
        root = _from_array(df, kind, how).pAll
    finally:
        sa.show = show
    doc = Document()
    doc.add_root(root)
    if disconnect:
        # Disposable test document ONLY: demonstrate that a broken live
        # RangeSlider->RangeFilter callback is detected by the numerical oracle.
        from bokeh.models import RangeSlider
        sliders = list(doc.select({"type": RangeSlider}))
        assert len(sliders) == 1, "control fixture needs exactly one RangeSlider"
        links = sliders[0].js_property_callbacks
        assert links and any(links.values()), "no link exists to break; test invalid"
        sliders[0].js_property_callbacks = {}
    out = tmp_path / ("ri_widgets_document_muted.json" if disconnect else "ri_widgets_document.json")
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


def _node_page(node_session, tmp_path, df, kind, how="any"):
    """NodePage handles error collection, document isolation and decoding."""
    report = ScenarioResult(id=f"widgets/{kind}/{how} [node]", outdir=str(tmp_path))
    page = node_session.new_page(report)
    page.goto(build_document(tmp_path, df, kind, how))
    page.wait_for_function(page.load_condition)
    settled = page.evaluate("p => window.__riH.settle(p, 50)", PROBES)
    assert settled.get("settled"), settled
    assert not report.page_errors and not report.console_errors, (
        "Node document emitted unexpected errors", report.page_errors, report.console_errors)
    return page, report


STATE_JS = """(n) => {
      const h = window.__riH.model('h', ['HistoNdCDS']);
      if (!h || !h.filter) throw Error('h/filter missing; cannot claim a widget test');
      const v = h.filter.v_compute();
      if (v == null || v.length < n) throw Error('filter vector shorter than the table: ' + (v == null ? null : v.length));
      const mask = Array.from(v, Boolean).slice(0, n);
      const bins = Array.from(h.get_column('bin_count'));
      return {mask, bins};
    }"""


def _state(page):
    """Integration (Claude20): LazyIntersectionFilter.v_compute() returns a vector padded beyond the table (32 entries
    for 12 rows here); the first n entries are the row mask, as engine/runner.py `__riH.selected` reads it."""
    return page.evaluate(STATE_JS, len(X))


def _unfiltered_state(page):
    """Parameter-only page: no selection filter is a *required* production contract.

    Do not use _state(): that helper intentionally rejects missing filters for
    selection-widget scenarios.  A missing HistoNdCDS model is still fatal.
    """
    return page.evaluate("""() => {
      const h = window.__riH.model('h', ['HistoNdCDS']);
      if (!h) throw Error('h missing; unfiltered parameter page invalid');
      if (h.filter !== null)
        throw Error('parameter-only histogram acquired an unexpected filter');
      return {filter_is_null:h.filter === null,
              bins:Array.from(h.get_column('bin_count'))};
    }""")


def _range_value(page):
    return page.evaluate("""() => {
      const w = window.__riH.slider('gate');
      if (!w) throw Error('RangeSlider gate missing or ambiguous');
      return Array.from(w.value);
    }""")


def _move_range(page, value):
    done = page.evaluate("""async (value) => {
      const w = window.__riH.slider('gate');
      if (!w) throw Error('RangeSlider gate missing or ambiguous');
      w.value = value;
      return await window.__riH.settle([['h','bin_count',['HistoNdCDS']]], 50);
    }""", list(value))
    assert done.get("settled"), done
    assert np.allclose(_range_value(page), value), "widget property did not change"


def _move_multi(page, selected, title="bits"):
    out = page.evaluate("""async (args) => {
      const {values, title} = args;
      const ms = window.__riH.models().filter(m => m.type === 'MultiSelect' && m.title === title);
      if (ms.length !== 1) throw Error('MultiSelect bits missing/ambiguous: ' + ms.length);
      ms[0].value = values;
      const done = await window.__riH.settle([['h','bin_count',['HistoNdCDS']]], 50);
      const fs = window.__riH.models().filter(m => m.type === 'MultiSelectFilter');
      if (fs.length !== 1) throw Error('MultiSelectFilter count != 1: ' + fs.length);
      return {done, value: Array.from(ms[0].value), linked: Array.from(fs[0].selected)};
    }""", {"values": list(selected), "title": title})
    assert out["done"].get("settled"), out
    assert out["value"] == list(selected), out
    # This is a real wiring assertion, not merely calling v_compute on a filter.
    assert out["linked"] == list(selected), "MultiSelect.value -> filter.selected callback disconnected"


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.parametrize("states", [
    pytest.param([(-2.0, 2.0), (-1.0, 0.5), (0.5, 1.5), (1.8, 1.8),
                  (-2.0, 2.0)], id="narrow-empty-restore"),
    pytest.param([(-2.0, 2.0), (-2.0, 0.0), (0.0, 2.0), (-2.0, 2.0)],
                 id="two-complementary-ranges"),
])
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_node_range_slider_selection_and_weighted_histogram(node_session, tmp_path, states):
    df = frame()
    page, report = _node_page(node_session, tmp_path, df, "range")
    initial = _range_value(page)
    assert_state(df, _state(page), oracle_range(df, *initial))
    for state in states:
        if state == (1.8, 1.8):
            assert not oracle_range(df, *state).any(), "declared empty interval is not empty"
        _move_range(page, state)
        assert_state(df, _state(page), oracle_range(df, *state))
    assert not report.page_errors and not report.console_errors, (
        report.page_errors, report.console_errors)


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.parametrize("how,selections", [
    ("any", [("A", "B", "C"), ("A",), ("B", "C"), (), ("A", "B", "C")]),
    ("all", [(), ("A", "B"), ("C",), ("A", "B", "C"), ()]),
], ids=["bitmask-any", "bitmask-all"])
@pytest.mark.feature("HIST.histogram_nd")
def test_node_multiselect_real_widget_callback_and_histogram(node_session, tmp_path, how, selections):
    df = frame()
    page, report = _node_page(node_session, tmp_path, df, "multiselect", how)
    selected = page.evaluate("""() => {
      const m = window.__riH.models().filter(m => m.type === 'MultiSelect' && m.title === 'bits');
      if (m.length !== 1) throw Error('MultiSelect not found at load');
      return Array.from(m[0].value);
    }""")
    assert_state(df, _state(page), oracle_multiselect(df, selected, how))
    for selected in selections:
        _move_multi(page, selected)
        assert_state(df, _state(page), oracle_multiselect(df, selected, how))
    assert not report.page_errors and not report.console_errors, (
        report.page_errors, report.console_errors)


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("HIST.histogram_nd")
def test_node_categorical_multiselect_selection_and_histogram(node_session, tmp_path):
    """Ordinary multiSelect on a categorical column (not only bitmask widgets)."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, df, "categorical")
    title = "category"
    initial = page.evaluate("""() => {
      const m = window.__riH.models().filter(m => m.type === 'MultiSelect' && m.title === 'category');
      if (m.length !== 1) throw Error('category MultiSelect not found at load');
      return Array.from(m[0].value);
    }""")
    assert_state(df, _state(page), oracle_categories(df, initial))
    for selected in (("A",), ("A", "C"), (), ("B",), ("A", "B", "C")):
        _move_multi(page, selected, title=title)
        assert_state(df, _state(page), oracle_categories(df, selected))
    assert not report.page_errors and not report.console_errors, (
        report.page_errors, report.console_errors)


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.profile")
@pytest.mark.feature("HIST.histogram_nd")
def test_node_range_slider_updates_nd_profile_after_each_move(node_session, headless_out):
    """Full scenario-engine path: slider -> filter -> N-D histogram -> profile.

    The profile oracle uses the *binned* y centers, not raw-row mean (the
    distinction identified in the PHASE_4_5 review). The scenario builder
    and step engine are unmodified; results come from the live production
    HistoNdCDS and HistoNdProfile models.
    """
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.data import generate
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
        Labels, Page, Scenario, TableSpec, Tables, auto, hist, load, range_,
        responds, set_
    )

    source = TableSpec(n=400, seed=2704, columns={
        "x": "uniform(0,4)", "y": "uniform(0,3)", "gate": "normal(0,1)"
    })
    bounds = ((-2.0, 2.0), (-1.0, 1.0), (-0.25, 0.25),
              (-1.0, 1.0), (-2.0, 2.0))
    sc = Scenario(
        id="widgets/gpt27_slider_profile_binned_mean",
        data=Tables(main=source),
        page=Page(
            histograms=(hist("hnd", ["x", "y"], [4, 3], [[0, 4], [0, 3]]),),
            widgets=(range_("gate", bounds[0]),),
            profiles=("hnd_1",),
        ),
        steps=(load(), *(set_("gate", b) for b in bounds[1:])),
        checks=(auto(), responds("hnd_1", "mean", moved=1, restored=4)),
        labels=Labels(
            family="widgets",
            models=("RangeFilter", "HistoNdCDS", "HistoNdProfile"),
            verified=("RangeFilter", "HistoNdCDS", "HistoNdProfile"),
            features=("HIST.histogram_nd",),
            technique="independent binned-moment oracle after each widget transition",
        ),
    )
    result = run(node_session, sc, headless_out, test_ref=__file__)
    assert result.ok, result.text()
    df = generate(source)
    for i, (low, high) in enumerate(bounds):
        assert str(i) in result.probe_values, (i, result.probe_values)
        # Runner probes hnd.bin_count first and hnd_1.mean second.
        hist_values, profile_values = json.loads(result.probe_values[str(i)])
        mask = df["gate"].between(low, high, inclusive="both").to_numpy()
        cells, _ = np.histogramdd(df.loc[mask, ["x", "y"]].to_numpy(),
                                  bins=[4, 3], range=[[0, 4], [0, 3]])
        np.testing.assert_allclose(np.asarray(hist_values, dtype=float),
                                   cells.ravel(order="F"), atol=0, rtol=0)
        totals = cells.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = (cells @ np.asarray([0.5, 1.5, 2.5])) / totals
        means[totals == 0] = np.nan
        np.testing.assert_allclose(np.asarray(profile_values, dtype=float), means,
                                   atol=1e-12, rtol=1e-12, equal_nan=True,
                                   err_msg=f"binned profile after widget step {i}")


@pytest.mark.headless_unit
@pytest.mark.backend("node")
@pytest.mark.layer("unit")
def test_oracle_and_assertions_have_controls_that_fail():
    df = frame()
    mask = oracle_range(df, -0.5, 1.5)
    good = {"mask": mask.tolist(), "bins": oracle_bins(df, mask).tolist()}
    assert_state(df, good, mask)  # healthy control
    bad_bin = {"mask": list(good["mask"]), "bins": list(good["bins"])}
    bad_bin["bins"][0] += 0.25
    with pytest.raises(AssertionError, match="weighted-bin"):
        assert_state(df, bad_bin, mask)
    bad_selection = {"mask": list(good["mask"]), "bins": list(good["bins"])}
    bad_selection["mask"][0] = not bad_selection["mask"][0]
    with pytest.raises(AssertionError, match="widget -> selection"):
        assert_state(df, bad_selection, mask)
    assert not np.array_equal(oracle_multiselect(df, ["A"], "any"),
                              oracle_multiselect(df, ["A", "B"], "all"))


def _open_browser_page(headless_session, html: Path):
    """Integration (Claude20): the page's external scripts (pako, vis, onnxruntime-web) are served from the pinned
    local copies and every other request is blocked - the engine's own rule (engine/runner.py `_browser_run`). Without
    it the page loads them from the internet (not reproducible) or, offline, does not load at all."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine import vendor
    routes = {glob.replace("**/", "/"): path for glob, path in vendor.routes().items()}
    ctx = headless_session.browser.new_context()
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda err: errors.append("pageerror: " + str(err)))
    page.on("console", lambda msg: errors.append("console.error: " + msg.text) if msg.type == "error" else None)

    def route(r):
        url = r.request.url
        if url.startswith("file:"):
            return r.continue_()
        for suffix, path in routes.items():
            if url.split("?")[0].endswith(suffix):
                return r.fulfill(path=str(path), content_type="text/javascript")
        errors.append("blocked request: " + url)
        return r.abort()
    page.route("**/*", route)
    page.add_init_script(HELPER_JS)
    page.goto(html.resolve().as_uri())
    return ctx, page, errors


def build_browser_page(tmp_path: Path, df: pd.DataFrame):
    """Browser path uses exactly the same production page builder and fixture."""
    from bokeh.plotting import output_file

    path = tmp_path / "widget_keyboard.html"
    output_file(str(path), mode="inline")
    _from_array(df, "range")
    return path


@pytest.mark.browser_widgets
@pytest.mark.backend("browser")
@pytest.mark.layer("integration")
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_chromium_real_keyboard_on_range_slider_updates_histogram(headless_session, tmp_path):
    """A *real keyboard event*, not a JS model.value assignment.

    Bokeh 3.3 RangeSlider uses noUiSlider handles. If upstream Bokeh changes
    the DOM, the selector MUST be updated after inspecting that pinned version;
    do not silently replace this with a programmatic property change.
    """
    df = frame()
    html = build_browser_page(tmp_path, df)
    context, page, errors = _open_browser_page(headless_session, html)
    try:
        page.wait_for_function("window.Bokeh && Bokeh.documents.length > 0 && Bokeh.documents[0].is_idle")
        page.wait_for_function("""() => Boolean(window.__riH.model('h', ['HistoNdCDS']))""")
        # One actual keyboard event on the physical UI control.
        handle = page.locator(".noUi-handle").first
        assert handle.count() == 1, "Bokeh RangeSlider handle not found; inspect pinned Chromium DOM"
        before = page.evaluate("() => Array.from(window.__riH.slider('gate').value)")
        handle.focus()
        page.keyboard.press("ArrowRight")
        page.wait_for_function("""old => {
          const m = window.__riH.slider('gate');
          return m && JSON.stringify(Array.from(m.value)) !== JSON.stringify(old);
        }""", arg=before)
        done = page.evaluate("p => window.__riH.settle(p, 50)", PROBES)
        assert done["settled"], done
        after = page.evaluate("() => Array.from(window.__riH.slider('gate').value)")
        assert after != before, "keyboard action did not change RangeSlider model"
        state = page.evaluate(STATE_JS, len(X))
        assert_state(df, state, oracle_range(df, *after))
        assert not errors, errors
    finally:
        context.close()


# =============================== v0.2a REVISED ADDITIONS ===============================
# The previous eight variants are retained. The new checks exercise the other
# five widget types and parameterArray; they do not claim every Bokeh option.


def _assert_no_errors(report):
    assert not report.page_errors and not report.console_errors, (
        report.page_errors, report.console_errors)


def _move_model(page, model_type, prop, value, *, title=None, label=None):
    """Change a REAL production Bokeh model, let its serialized callbacks run.

    Never directly modify a RangeFilter, MultiSelectFilter, histogram, or legend.
    A lone matching model is required; ambiguity is a hard failure, not a skip.
    """
    out = page.evaluate("""async (args) => {
      const {kind, prop, value, title, label} = args;
      const matches = window.__riH.models().filter(m => m.type === kind &&
         (title == null || m.title === title) &&
         (label == null || m.label === label));
      if (matches.length !== 1)
        throw Error('widget model missing/ambiguous ' + kind + ' title=' + title +
                    ' label=' + label + ' count=' + matches.length);
      matches[0][prop] = value;
      const done = await window.__riH.settle([['h','bin_count',['HistoNdCDS']]], 50);
      return {done, actual: matches[0][prop]};
    }""", {"kind": model_type, "prop": prop, "value": value, "title": title, "label": label})
    assert out["done"].get("settled"), out
    assert out["actual"] == value, (model_type, value, out)


def _filter_detail(page):
    """Read the concrete filter inside production LazyIntersectionFilter.

    Important: HistoNdCDS.filter is the intersection wrapper, NOT a direct
    RangeFilter/MultiSelectFilter. A test that inspects only h.filter.range
    accidentally tests a nonexistent property and misses callback wiring.
    """
    return page.evaluate("""() => {
      const h = window.__riH.model('h', ['HistoNdCDS']);
      if (!h || !h.filter) throw Error('histogram/filter missing');
      const outer = h.filter;
      if (outer.type !== 'LazyIntersectionFilter' || outer.filters.length !== 1)
          throw Error('expected one production nested filter, got ' + outer.type);
      const f = outer.filters[0];
      return {kind:f.type, active:Boolean(f.active),
              range:f.range ? Array.from(f.range) : null,
              selected:f.selected ? Array.from(f.selected) : null};
    }""")


def _raw_bins(df, mask, nx):
    positions = df.loc[np.asarray(mask, dtype=bool), ["x", "y"]].to_numpy()
    weights = df.loc[np.asarray(mask, dtype=bool), "w"].to_numpy(dtype=float)
    bins, _ = np.histogramdd(positions, bins=[nx, 3], range=RANGES, weights=weights)
    return bins.ravel(order="F")


def assert_unfiltered_state(df, state, nx=4):
    """Independent all-row weighted-bin oracle, never a disguised filter oracle.

    This is used both on live parameter-only graphs and on a Python mutation
    control.  It *rejects* a spurious filter as well as corrupt numerical bins.
    """
    assert isinstance(state, dict), state
    assert state.get("filter_is_null") is True, "unfiltered histogram unexpectedly filtered"
    assert "bins" in state, "unfiltered histogram bins missing"
    actual = np.asarray(state["bins"], dtype=float)
    expected = _raw_bins(df, np.ones(len(df), dtype=bool), nx)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12,
                               err_msg="unfiltered HistoNdCDS all-row weighted-bin mismatch")


def _check_parameter_bins(df, page, nx):
    """Independent oracle for widget -> histogram binning parameter."""
    state = page.evaluate("""() => {
      const h = window.__riH.model('h',['HistoNdCDS']);
      if (!h) throw Error('HistoNdCDS h absent');
      if (h.filter !== null)
        throw Error('parameter-only histogram acquired an unexpected filter');
      return {nbins:Array.from(h.nbins), count:Array.from(h.get_column('bin_count'))};
    }""")
    assert state["nbins"] == [nx, 3], (nx, state)
    np.testing.assert_allclose(np.asarray(state["count"], dtype=float),
                               _raw_bins(df, np.ones(len(df), dtype=bool), nx),
                               rtol=1e-12, atol=1e-12,
                               err_msg="widget -> parameterArray -> HistoNdCDS bins")


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_node_scalar_slider_selection_and_histogram(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, df, "slider_selection")
    for value in (0.0, -1.0, 1.0, 0.0):
        _move_model(page, "Slider", "value", value, title="gate")
        detail = _filter_detail(page)
        assert detail["kind"] == "RangeFilter", detail
        np.testing.assert_allclose(detail["range"], [value - 0.25, value + 0.25])
        assert_state(df, _state(page), oracle_range(df, value - 0.25, value + 0.25))
    _assert_no_errors(report)


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("HIST.histogram_nd")
def test_node_select_selection_callback_and_histogram(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, df, "select_selection")
    for value in ("B", "A", "C", "A"):
        _move_model(page, "Select", "value", value, title="category")
        detail = _filter_detail(page)
        assert detail["kind"] == "MultiSelectFilter" and detail["selected"] == [value], detail
        assert_state(df, _state(page), oracle_categories(df, [value]))
    _assert_no_errors(report)


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_node_spinner_range_each_endpoint_callback_and_histogram(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, df, "spinner_range")
    initial = _filter_detail(page)
    assert initial["kind"] == "RangeFilter" and len(initial["range"]) == 2, initial
    steps = [("min(gate)", -1.0, (-1.0, initial["range"][1])),
             ("max(gate)", 0.5, (-1.0, 0.5)),
             ("min(gate)", -2.0, (-2.0, 0.5)),
             ("max(gate)", 2.0, (-2.0, 2.0))]
    for title, value, bounds in steps:
        _move_model(page, "Spinner", "value", value, title=title)
        detail = _filter_detail(page)
        np.testing.assert_allclose(detail["range"], bounds,
                                   err_msg="spinnerRange endpoint -> RangeFilter.range callback")
        assert_state(df, _state(page), oracle_range(df, *bounds))
    _assert_no_errors(report)


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_node_toggleable_range_activation_and_restore(node_session, tmp_path):
    df = frame()
    failure = _load_failure(node_session, build_document(tmp_path, df, "toggleable_range"))      # integration: WG-D1
    regression(failure == WG_D1, "WG-D1: the page raises at load")
    assert failure is None, f"neither a page nor the WG-D1 signature: {failure}"
    page, report = _node_page(node_session, tmp_path, df, "toggleable_range")
    # Bokeh's extra 'disable' Toggle initially active: the filter is disabled.
    initial = _filter_detail(page)
    assert not initial["active"], initial
    assert_state(df, _state(page), np.ones(len(df), dtype=bool))
    _move_model(page, "Toggle", "active", False, label="disable")
    assert _filter_detail(page)["active"]
    _move_range(page, (-1.0, 0.5))
    assert_state(df, _state(page), oracle_range(df, -1.0, 0.5))
    _move_model(page, "Toggle", "active", True, label="disable")
    assert not _filter_detail(page)["active"]
    assert_state(df, _state(page), np.ones(len(df), dtype=bool))
    _assert_no_errors(report)


@pytest.mark.parametrize("kind,model_type,states", [
    pytest.param("parameter_slider", "Slider", [6, 2, 4], id="slider-binning"),
    pytest.param("parameter_select", "Select", ["6", "2", "4"], id="select-binning"),
    pytest.param("parameter_spinner", "Spinner", [6, 2, 4], id="spinner-binning"),
])
@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("HIST.histogram_nd")
def test_node_parameter_widget_updates_nd_histogram_nbins(node_session, tmp_path, kind, model_type, states):
    df = frame()
    page, report = _node_page(node_session, tmp_path, df, kind)
    _check_parameter_bins(df, page, 4)
    for value in states:
        _move_model(page, model_type, "value", value, title="nbinsX")
        _check_parameter_bins(df, page, int(value))
    _assert_no_errors(report)


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("HIST.histogram_nd")
def test_node_toggle_parameter_updates_legend_and_keeps_histogram(node_session, tmp_path):
    df = frame()
    page, report = _node_page(node_session, tmp_path, df, "parameter_toggle")
    baseline = _unfiltered_state(page)
    assert_unfiltered_state(df, baseline)
    for value in (False, True, False, True):
        _move_model(page, "Toggle", "active", value, label="show legend")
        seen = page.evaluate("""() => {
           const legends = window.__riH.models().filter(m => m.type === 'Legend');
           if (legends.length !== 1) throw Error('exactly one Legend required: ' + legends.length);
           return Boolean(legends[0].visible);
        }""")
        assert seen is value, "toggle -> parameterArray -> Bokeh Legend.visible disconnected"
        assert_unfiltered_state(df, _unfiltered_state(page))
    _assert_no_errors(report)


@pytest.mark.headless_unit
@pytest.mark.backend("node")
@pytest.mark.layer("unit")
def test_v02_oracle_controls_cover_scalar_param_and_restore():
    df = frame()
    all_rows = np.ones(len(df), dtype=bool)
    good = _raw_bins(df, all_rows, 4)
    assert len(good) == 12
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(good, _raw_bins(df, all_rows, 6))
    # A selector that leaves mask unchanged would be detected for chosen states.
    assert not np.array_equal(oracle_range(df, -0.25, 0.25),
                              oracle_range(df, -1.25, -0.75))
    assert not np.array_equal(oracle_categories(df, ["A"]), oracle_categories(df, ["B"]))
    assert not np.array_equal(oracle_range(df, -2.0, 2.0), oracle_range(df, -1.0, 0.5))
    assert not np.array_equal(oracle_multiselect(df, ["A"], "any"),
                              oracle_multiselect(df, ["B"], "any"))
    # v0.2a C0: healthy unfiltered histogram and two falsifiers.
    healthy = {"filter_is_null": True, "bins": good.tolist()}
    assert_unfiltered_state(df, healthy)
    wrong_bin = {"filter_is_null": True, "bins": good.tolist()}
    wrong_bin["bins"][0] += 0.25
    with pytest.raises(AssertionError, match="unfiltered HistoNdCDS"):
        assert_unfiltered_state(df, wrong_bin)
    with pytest.raises(AssertionError, match="unexpectedly filtered"):
        assert_unfiltered_state(df, {"filter_is_null": False, "bins": good.tolist()})


# Browser: a distinct Chromium UI control for every advertised type.
# The 'range' keyboard case above remains part of the suite. The parametrized
# browser cases below are full-suite targets, not added to browser_smoke to
# avoid importing/compiling eight browser documents on the fast CI path.
# Bokeh 3.3.2 selectors are verified only by an actual alma2 run.


def _browser_variant(headless_session, tmp_path, kind, how="any"):
    from bokeh.plotting import output_file
    html = tmp_path / ("widget_" + kind + "_" + how + ".html")
    output_file(str(html), mode="inline")
    df = frame()
    _from_array(df, kind, how)
    ctx, page, errors = _open_browser_page(headless_session, html)
    page.wait_for_function("window.Bokeh && Bokeh.documents.length > 0 && Bokeh.documents[0].is_idle")
    page.wait_for_function("() => Boolean(window.__riH.model('h', ['HistoNdCDS']))")
    return df, ctx, page, errors


def _browser_sync(page):
    done = page.evaluate("p => window.__riH.settle(p, 50)", PROBES)
    assert done["settled"], done


def _browser_glyph_reference(page):
    # Verify the plotted glyph renderer references the SAME histogram source
    # that the numerical oracle checks. A stale detached plot would be rejected.
    return page.evaluate("""() => {
      const h = window.__riH.model('h', ['HistoNdCDS']);
      if (!h) throw Error('HistoNdCDS h missing');
      const renderers = window.__riH.models().filter(m => m.type === 'GlyphRenderer');
      const reaches = src => {
        const seen = new Set();
        while (src && !seen.has(src)) {
          if (src === h) return true;
          seen.add(src);
          src = src.source || null;
        }
        return false;
      };
      return renderers.some(r => reaches(r.data_source));
    }""")


@pytest.mark.browser_widgets
@pytest.mark.backend("browser")
@pytest.mark.layer("integration")
@pytest.mark.parametrize("kind,how", [
    pytest.param("slider_selection", "any", id="slider-keyboard"),
    pytest.param("select_selection", "any", id="select-keyboard"),
    pytest.param("categorical", "any", id="multiselect-native"),
    pytest.param("multiselect", "any", id="bitmask-any-native"),
    pytest.param("multiselect", "all", id="bitmask-all-native"),
    pytest.param("spinner_range", "any", id="spinner-range-both-endpoints"),
    pytest.param("parameter_spinner", "any", id="spinner-parameter-input"),
    pytest.param("parameter_toggle", "any", id="toggle-click"),
])
@pytest.mark.feature("HIST.histogram_nd")
def test_browser_distinct_widget_input_updates_dependent_state(headless_session, tmp_path, kind, how):
    """Genuine Playwright widget actions, never direct Bokeh model assignments.

    Native select_option drives DOM input/change; slider uses keyboard, toggle
    physical click, spinner inputs use fill+Tab. Distinguish native DOM select
    event from actual keyboard and button events in the review report.
    """
    df, ctx, page, errors = _browser_variant(headless_session, tmp_path, kind, how)
    try:
        assert _browser_glyph_reference(page), "plot glyph detached from tested histogram"
        if kind == "slider_selection":
            control = page.locator(".noUi-handle").first
            assert control.count() == 1
            old = page.evaluate("() => window.__riH.models().find(m => m.type === 'Slider' && m.title === 'gate').value")
            control.focus()
            page.keyboard.press("ArrowRight")
            page.wait_for_function("old => window.__riH.models().some(m => m.type === 'Slider' && m.title === 'gate' && m.value !== old)", arg=old)
            _browser_sync(page)
            val = page.evaluate("() => window.__riH.models().find(m => m.type === 'Slider' && m.title === 'gate').value")
            assert val != old
            assert_state(df, _state(page), oracle_range(df, val - 0.25, val + 0.25))
        elif kind == "select_selection":
            control = page.locator("select:not([multiple])").first
            assert control.count() == 1
            old = page.evaluate("() => window.__riH.models().find(m => m.type === 'Select' && m.title === 'category').value")
            control.focus()
            page.keyboard.press("ArrowDown")
            page.keyboard.press("Enter")
            page.wait_for_function("old => window.__riH.models().some(m => m.type === 'Select' && m.title === 'category' && m.value !== old)", arg=old)
            _browser_sync(page)
            val = page.evaluate("() => window.__riH.models().find(m => m.type === 'Select' && m.title === 'category').value")
            assert_state(df, _state(page), oracle_categories(df, [val]))
        elif kind in {"categorical", "multiselect"}:
            control = page.locator("select[multiple]").first
            assert control.count() == 1
            if kind == "categorical":
                values = ["A", "C"]
                control.select_option(values)
                expected = oracle_categories(df, values)
            else:
                values = ["A", "B"] if how == "any" else ["A", "C"]
                control.select_option(values)
                expected = oracle_multiselect(df, values, how)
            _browser_sync(page)
            actual = page.evaluate("() => Array.from(window.__riH.models().find(m => m.type === 'MultiSelect').value)")
            assert actual == values
            assert_state(df, _state(page), expected)
        elif kind == "spinner_range":
            controls = page.locator("input.bk-input")              # Bokeh 3.3.2: type="text" (measured)
            assert controls.count() == 2, "both spinnerRange endpoint DOM controls required"
            for i, value in ((0, "-1.0"), (1, "0.5")):
                controls.nth(i).fill(value)
                controls.nth(i).press("Tab")
                _browser_sync(page)
            assert_state(df, _state(page), oracle_range(df, -1.0, 0.5))
        elif kind == "parameter_spinner":
            control = page.locator("input.bk-input")
            assert control.count() == 1
            control.fill("6")
            control.press("Tab")
            _browser_sync(page)
            _check_parameter_bins(df, page, 6)
        elif kind == "parameter_toggle":
            # Healthy baseline before any physical click: no selection filter.
            assert_unfiltered_state(df, _unfiltered_state(page))
            control = page.get_by_role("button", name="show legend")
            assert control.count() == 1
            for expected in (False, True):
                control.click()
                _browser_sync(page)
                seen = page.evaluate("""() => {
                   const legend = window.__riH.models().filter(m => m.type === 'Legend');
                   if (legend.length !== 1) throw Error('Legend not unique');
                   return Boolean(legend[0].visible);
                }""")
                assert seen is expected, "Toggle did not propagate to visible legend"
                assert_unfiltered_state(df, _unfiltered_state(page))
        else:
            raise AssertionError("unhandled browser case " + kind)
        assert _browser_glyph_reference(page), "plot source disconnected after interaction"
        assert not errors, errors
    finally:
        ctx.close()


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_node_scratch_muted_widget_callback_is_caught(node_session, tmp_path):
    """Test-only graph mutation; falsifies the callback path, not just NumPy.

    Production Python/TS remains untouched. The same builder yields two Bokeh
    documents: a healthy one and a scratch copy with RangeSlider's serialized
    JS link removed. The *same* numerical checker must reject the muted page.
    """
    df = frame()
    healthy, good_report = _node_page(node_session, tmp_path, df, "range")
    _move_range(healthy, (-1.0, 0.5))
    assert_state(df, _state(healthy), oracle_range(df, -1.0, 0.5))
    _assert_no_errors(good_report)

    broken_doc = build_document(tmp_path, df, "range", disconnect=True)
    report = ScenarioResult(id="widgets/scratch-muted-callback [node]", outdir=str(tmp_path))
    page = node_session.new_page(report)
    page.goto(broken_doc)
    page.wait_for_function(page.load_condition)
    assert_state(df, _state(page), oracle_range(df, -2.0, 2.0))
    _move_range(page, (-1.0, 0.5))
    with pytest.raises(AssertionError, match="widget -> selection mismatch"):
        assert_state(df, _state(page), oracle_range(df, -1.0, 0.5))
    _assert_no_errors(report)


# =============================== integration (Claude20.RI): WG-D1, WG-D2 ===============================
# A toggleable range widget (`{"toggleable": True}`) adds a "disable" toggle, ON at page load: the range starts disabled.
# WG-D1 - alone on its table, it breaks the page: HistoNdCDS.initialize -> change_selection ->
#         LazyIntersectionFilter.v_compute reads `this.counts.length`, but `counts` is created only when an ACTIVE
#         filter is processed (LazyIntersectionFilter.ts update_counts / v_compute). No widget is shown.
#         Control: with a second filter active from load on, the page loads.
# WG-D2 - with a second filter, the page loads, but once the range is enabled and moved it still selects every row
#         (in both orders of the two filters), although the filter is active and holds the moved range.
#         Control: the same two-filter pages with a plain (not toggleable) range filter correctly.
WG_D1 = ("TypeError", "Cannot read properties of undefined (reading 'length')",
         ["LazyIntersectionFilter.v_compute", "HistoNdCDS.change_selection", "HistoNdCDS.initialize"])

LOAD_JS = """(text) => {
  try { Bokeh.require("document").Document.from_json(JSON.parse(text)); return {ok: true}; }
  catch (e) {
    const frames = String(e && e.stack).split("\\n").slice(1, 4).map(l => { const m = l.trim().match(/^at (\\S+)/); return m ? m[1] : l.trim(); });
    return {ok: false, name: String(e && e.name), message: String(e && e.message), frames};
  }
}"""


def _load_failure(node_session, path):
    """None if Bokeh's deserializer builds the document in Node, else (error kind, message, first three frames)."""
    r = node_session.request("eval", code=LOAD_JS, arg=Path(path).read_text(encoding="utf-8"))
    assert r.get("ok"), r
    out = decode(r["result"])
    return None if out["ok"] else (out["name"], out["message"], out["frames"])


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_control_wg_d1_with_a_second_active_filter_the_page_loads(node_session, tmp_path):
    """Control: the same toggleable range plus an active range on x: the page loads, and at load every row is
    selected (the range is disabled, x keeps every row) with the right weighted bins."""
    df = frame()
    assert _load_failure(node_session, build_document(tmp_path, df, "toggleable_plus_range")) is None
    page, report = _node_page(node_session, tmp_path, df, "toggleable_plus_range")
    assert_state(df, _state(page), np.ones(len(df), dtype=bool))
    _assert_no_errors(report)


ORDERS = [pytest.param(("toggleable_plus_range", "plain_plus_range"), id="gate-first"),
          pytest.param(("range_plus_toggleable", "range_plus_plain"), id="gate-second")]


def _enable_and_move(node_session, tmp_path, kind):
    """Enable the toggleable gate range, move it to (-1, 0.5); the selection, and the gate filter's state."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, df, kind)
    _move_model(page, "Toggle", "active", False, label="disable")
    _move_range(page, (-1.0, 0.5))
    gate = page.evaluate("""() => {
      const fs = window.__riH.model('h', ['HistoNdCDS']).filter.filters.filter(f => f.field === 'gate');
      if (fs.length !== 1) throw Error('gate filters: ' + fs.length);
      return {active: Boolean(fs[0].active), range: Array.from(fs[0].range)};
    }""")
    _assert_no_errors(report)
    return df, _state(page), gate


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.parametrize("kinds", ORDERS)
def test_strict_wg_d2_an_enabled_toggleable_range_selects_its_rows(node_session, tmp_path, kinds):
    """WG-D2 (fixed in PHASE_5 Batch 1): enabled and moved to (-1, 0.5), the gate selects its 4 of 12 rows. On master
    the filter was active and held the range, but every row stayed selected."""
    df, state, gate = _enable_and_move(node_session, tmp_path, kinds[0])
    assert oracle_range(df, -1.0, 0.5).sum() == 4
    if np.array_equal(np.asarray(state["mask"], dtype=bool), oracle_range(df, -1.0, 0.5)):
        assert_state(df, state, oracle_range(df, -1.0, 0.5))                 # the bins must follow as well
        return
    regression(gate == {"active": True, "range": [-1.0, 0.5]} and all(state["mask"]), "WG-D2: every row selected")
    raise AssertionError(f"WG-D2: neither the oracle nor the WG-D2 signature: {state}, {gate}")


@pytest.mark.node_widgets
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.parametrize("kinds", ORDERS)
def test_control_wg_d2_the_same_range_without_toggle_selects_its_rows(node_session, tmp_path, kinds):
    """Control: the same two-filter page with a plain gate range, moved to (-1, 0.5): exactly its 4 rows and their
    weighted bins - so the two filters, their order and the harness are not the cause of WG-D2."""
    df = frame()
    page, report = _node_page(node_session, tmp_path, df, kinds[1])
    _move_range(page, (-1.0, 0.5))
    assert_state(df, _state(page), oracle_range(df, -1.0, 0.5))
    _assert_no_errors(report)


@pytest.mark.browser_widgets
@pytest.mark.backend("browser")
@pytest.mark.layer("integration")
@pytest.mark.feature("FILTER.range")
def test_chromium_wg_d1_the_page_loads_and_shows_the_widget(headless_session, tmp_path):
    """WG-D1 in Chromium (fixed in PHASE_5 Batch 1). On master: exactly one page error with the WG-D1 message, and no
    slider rendered. Now: no page error, and the range slider is rendered with its two handles."""
    from bokeh.plotting import output_file
    html = tmp_path / "widget_toggleable_range.html"
    output_file(str(html), mode="inline")
    _from_array(frame(), "toggleable_range")
    ctx, page, errors = _open_browser_page(headless_session, html)
    try:
        page.wait_for_load_state("load")
        for _ in range(100):                                            # up to 10 s for the error to arrive
            if errors:
                break
            page.wait_for_timeout(100)
        page.wait_for_timeout(500)                                      # and nothing else follows
        regression(errors == ["pageerror: " + WG_D1[1]], "WG-D1: the page raises at load")
        assert errors == [], errors
        assert page.locator(".noUi-handle").count() == 2
    finally:
        ctx.close()
