"""PHASE_4_6 D3 — BokehVisJSGraph3D test-only candidate, GPT26.RI.

Source basis: RI 2026-11-09 LLMBUNDLE, not the pinned receiving HEAD.
All production tests are PENDING execution on alma2.  No production edits.
The Node harness does not render views: Node claims are limited to Bokeh model
semantics, source mapping, and updates. Chromium owns vis.Graph3d rendering.

Integration (Claude20.RI, 2026-11-11), all test-only. Targets node_d3 (Node) and browser_d3 (Chromium).
- The setData probe lets through the Array call that vis 4.16.1's own Graph3d.setOptions() makes
  (`this.dataTable && this.setData(this.dataTable)`; the view calls setOptions after every data change) and still
  records and checks only the view's own vis.DataSet calls. As delivered, it threw on that call.
- Added: the graph on a public-builder page under a range filter (Node: the graph's source gives exactly the selected
  rows; Chromium: the view hands vis.js exactly the reference points at load and after the filter). The builder's
  graph reads a DownsamplerCDS, whose columns can be longer than get_length() (C2 defect DS-D1); the view reads
  the first get_length() entries, and so do these tests. The widget value is set through the model: real slider
  input is tested by A3.
- Two defects, each with an exact signature, a strict expected-failure partner and a healthy control:
    G3-D1  a 3D graph without a colour variable cannot be built: the builder passes style=None, the Python model
           requires a string, bokehDrawSA.fromArray raises ValueError (".style: expected a value of type str,
           got None of type NoneType").
    G3-D2  when a filter selects no rows, the graph keeps drawing the previous points: the view hands vis.js an empty
           DataSet, vis 4.16.1 ignores empty data (Graph3d._dataInitialize), its dataPoints keep the old count.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


NODE = pytest.mark.node_d3
BROWSER = pytest.mark.browser_d3
UNIT = pytest.mark.headless_unit


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect (integration, Claude20.RI)."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


STRICT = dict(strict=True, raises=KnownDefect)
G3_D1_START = "failed to validate BokehVisJSGraph3D(id="
G3_D1_END = ").style: expected a value of type str, got None of type NoneType"


def _frame():
    # Unambiguous, non-collinear values; no random state or boundary ambiguities.
    x = np.asarray([0., 1., 2., 3., 4., 5., 6., 7., 8.])
    return pd.DataFrame({
        "x": x, "y": 2*x+1, "z": np.asarray([2., 1., 4., 3., 7., 5., 8., 6., 9.]),
        "c": np.asarray([4., 2., 8., 6., 14., 10., 16., 12., 18.]),
    })


def _mapping_ref(df, names=("x", "y", "z", "c")):
    """Independent per-point reference; never reads production JS view output."""
    return [{"x": float(a), "y": float(b), "z": float(c), "style": float(d)}
            for a, b, c, d in df[list(names)].itertuples(index=False, name=None)]


def _assert_points(actual, expected):
    assert len(actual) == len(expected), (len(actual), len(expected))
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert set(a) == set(e) == {"x", "y", "z", "style"}, (i, a, e)
        for key in e:
            assert np.isfinite(a[key]), (i, key, a[key])
            np.testing.assert_allclose(a[key], e[key], atol=0, rtol=0,
                                       err_msg=f"3D point {i} property {key}")


def _make_page(df, tmp_path: Path, *, html=False, name="graph3d"):
    """Use the real public page builder; no handcrafted replacement graph model."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    from bokeh.document import Document
    # Suppress opening a browser while building the serializable document.
    original = sa.show
    sa.show = lambda *args, **kwargs: None
    try:
        obj = sa.bokehDrawSA.fromArray(
            df, None,
            [[["x"], ["y"], ["z"], ["c"], {"name": name}]], [],
            layout=[[0]], widgetLayout=[], sourceArray=[], parameterArray=[],
            aliasArray=[], nPointRender=1000,
            arrayCompression=[(".*", [("code", 0), ("zip", 0), ("base64", 0)])],
        )
    finally:
        sa.show = original
    graph_type = "BokehVisJSGraph3D"
    graphs = [m for m in obj.pAll.references() if type(m).__name__ == graph_type]
    assert len(graphs) == 1, f"public builder must create one {graph_type}: {len(graphs)}"
    graph = graphs[0]
    assert [graph.x, graph.y, graph.z, graph.style] == ["x", "y", "z", "c"]
    assert graph.data_source is not None
    doc = Document()
    doc.add_root(obj.pAll)
    tmp_path.mkdir(parents=True, exist_ok=True)
    if html:
        from bokeh.embed import file_html
        from bokeh.resources import INLINE
        target = tmp_path / "d3_visual.html"
        target.write_text(file_html(doc, INLINE, "RI D3 3D regression"), encoding="utf8")
    else:
        target = tmp_path / "d3_document.json"
        target.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf8")
    return target


# The source model's typed arrays can be inspected without a DOM view. This is
# deliberately NOT described as proving BokehVisJSGraph3DView.get_data().
SOURCE_JS = r"""() => {
  const graphs = window.__riH.models().filter(m => m.type === 'BokehVisJSGraph3D');
  if (graphs.length !== 1) throw Error('BokehVisJSGraph3D count=' + graphs.length);
  const g = graphs[0], src = g.data_source;
  if (!src || typeof src.get_column !== 'function') throw Error('3D data source absent');
  const names = [g.x, g.y, g.z, g.style];
  if (names.some(n => typeof n !== 'string')) throw Error('3D field names invalid');
  const cols = names.map(n => src.get_column(n));
  if (cols.some(x => x == null)) throw Error('3D field absent on source: ' + names.join(','));
  const n = src.get_length();
  if (n == null || !cols.every(c => c.length === n)) throw Error('3D column length mismatch');
  return {names, sourceType: src.type, n, options: {...g.options3D},
          points: Array.from({length:n}, (_,i)=>({x:cols[0][i], y:cols[1][i],
                                               z:cols[2][i], style:cols[3][i]}))};
}"""


@UNIT
def test_d3_python_reference_has_nonvacuous_fault_controls():
    df = _frame()
    good = _mapping_ref(df)
    _assert_points(good, good)
    with pytest.raises(AssertionError):
        _assert_points(good[:-1], good)
    bad = [dict(p) for p in good]
    bad[2]["z"] += 1
    with pytest.raises(AssertionError):
        _assert_points(bad, good)
    swapped = [dict(p) for p in good]
    swapped[0]["style"] = good[0]["z"]
    with pytest.raises(AssertionError):
        _assert_points(swapped, good)
    assert not np.array_equal(df["z"], df["c"])


@UNIT
def test_d3_python_reference_exact_row_order_and_mutation():
    df = _frame()
    rev = df.iloc[::-1].reset_index(drop=True)
    want = _mapping_ref(rev)
    assert want[0]["x"] == 8.0 and want[-1]["x"] == 0.0
    with pytest.raises(AssertionError):
        _assert_points(_mapping_ref(df), want)
    changed = df.copy()
    changed.loc[3, "z"] = 34.0
    assert _mapping_ref(changed)[3]["z"] == 34
    with pytest.raises(AssertionError):
        _assert_points(_mapping_ref(changed), _mapping_ref(df))


@NODE
@pytest.mark.feature("DISPLAY.graph3d")
def test_d3_node_public_page_maps_four_fields_to_production_source(node_session, tmp_path):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
    df = _frame()
    res = ScenarioResult(id="d3/source-mapping", outdir=str(tmp_path))
    page = node_session.new_page(res)
    page.goto(_make_page(df, tmp_path))
    got = page.evaluate(SOURCE_JS)
    assert got["names"] == ["x", "y", "z", "c"]
    assert got["n"] == len(df)
    _assert_points(got["points"], _mapping_ref(df))
    assert not res.page_errors and not res.console_errors, (res.page_errors, res.console_errors)


@NODE
@pytest.mark.feature("DISPLAY.graph3d")
def test_d3_node_public_page_mapping_under_row_reorder(node_session, tmp_path):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
    df = _frame().iloc[::-1].reset_index(drop=True)
    res = ScenarioResult(id="d3/reorder", outdir=str(tmp_path))
    page = node_session.new_page(res)
    page.goto(_make_page(df, tmp_path))
    got = page.evaluate(SOURCE_JS)
    _assert_points(got["points"], _mapping_ref(df))
    assert got["points"][0]["x"] == 8
    assert not res.page_errors and not res.console_errors


@NODE
@pytest.mark.feature("DISPLAY.graph3d")
def test_d3_node_options_and_model_identity(node_session, tmp_path):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
    df = _frame()
    res = ScenarioResult(id="d3/options", outdir=str(tmp_path))
    page = node_session.new_page(res)
    page.goto(_make_page(df, tmp_path))
    got = page.evaluate(SOURCE_JS)
    assert got["n"] == 9
    assert got["options"].get("width") == "99%"
    assert got["options"].get("height") == "99%"
    _assert_points(got["points"], _mapping_ref(df))
    assert not res.page_errors and not res.console_errors



def _make_direct_visual_page(df, tmp_path):
    """A production BokehVisJSGraph3D with an editable ordinary CDS for view updates."""
    from bokeh.document import Document
    from bokeh.models import ColumnDataSource
    from bokeh.embed import file_html
    from bokeh.resources import INLINE
    from RootInteractive.InteractiveDrawing.bokeh.bokehVisJS3DGraph import BokehVisJSGraph3D
    source = ColumnDataSource(df.copy(), name="d3_editable_source")
    graph = BokehVisJSGraph3D(
        data_source=source, x="x", y="y", z="z", style="c",
        options3D={"style":"dot-color", "width":"700px", "height":"500px"},
        width=700, height=500, name="d3_rendered_graph")
    doc = Document()
    doc.add_root(graph)
    target = tmp_path / "d3_editable_visual.html"
    target.write_text(file_html(doc, INLINE, "RI D3 production view regression"), encoding="utf8")
    return target

def _open_pinned_browser_page(session, html, res):
    """A3 browser policy: only local pinned vis/pako/ort scripts, block internet."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine import vendor
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import HELPER_JS
    ctx = session.browser.new_context(viewport={"width": 900, "height": 750})
    page = ctx.new_page()
    res["page_errors"] = []
    res["console_errors"] = []
    res["blocked"] = []
    res["served"] = []
    page.on("pageerror", lambda e: res["page_errors"].append(str(e)))
    page.on("console", lambda m: res["console_errors"].append(m.text) if m.type == "error" else None)
    routes = {glob.replace("**/", "/"): path for glob, path in vendor.routes().items()}
    def route(request):
        url = request.request.url
        if url.startswith("file:"):
            return request.continue_()
        for suffix, f in routes.items():
            if url.split("?")[0].endswith(suffix):
                res["served"].append(url)
                return request.fulfill(path=str(f), content_type="text/javascript")
        res["blocked"].append(url)
        return request.abort()
    page.route("**/*", route)
    page.add_init_script(HELPER_JS)
    page.goto(html.resolve().as_uri())
    page.wait_for_function("window.Bokeh && Bokeh.documents.length > 0 && Bokeh.documents[0].is_idle")
    page.wait_for_function("() => window.__riH.models().some(m => m.type === 'BokehVisJSGraph3D')")
    return ctx, page


# Bokeh 3.3.2's LayoutDOMView renders in an open shadow root.  Bokeh.index
# contains the root view for our *direct-model* page.  Attach a test-only marker
# to its vis.Graph3d canvas, so Playwright can use its shadow-piercing locator.
# Never use document.querySelectorAll("canvas") to decide readiness.
VIEW_READY_JS = r"""() => {
  if (!window.vis || typeof window.vis.Graph3d !== 'function') return false;
  if (!window.__riH || !window.Bokeh || !Bokeh.index) return false;
  const models=window.__riH.models().filter(m=>m.type==='BokehVisJSGraph3D' && m.name==='d3_rendered_graph');
  if (models.length !== 1) throw Error('D3 root model identity: ' + models.length);
  const graph=models[0];
  const idx=Bokeh.index;
  const views=idx instanceof Map ? Array.from(idx.values()) : Object.values(idx);
  const matches=views.filter(v=>v?.model?.id===graph.id);
  if (matches.length !== 1) return false;  // view is not mounted yet
  const view=matches[0], visGraph=view._graph;
  const root=view.shadow_el || view.el?.shadowRoot;
  if (!visGraph || typeof visGraph.setData !== 'function' || !root) return false;
  const direct=visGraph.frame?.canvas;
  const canvases=Array.from(root.querySelectorAll('canvas')).filter(c=>c.width>10 && c.height>10);
  const canvas=direct && root.contains(direct) && direct.width>10 && direct.height>10
    ? direct : (canvases.length===1 ? canvases[0] : null);
  if (!canvas || !canvas.isConnected) return false;
  canvas.setAttribute('data-ri-d3-canvas', graph.id);
  window.__riD3={graph,view,visGraph,canvas,setDataCalls:[]};
  return true;
}"""

# Source-field inspection is independent of the view pipeline.  This separate
# test-only probe calls the production VIEW's get_data() (not SOURCE_JS) and
# verifies the complete x/y/z/style array passed to the Graph3d conversion.
VIEW_INITIAL_DATA_JS = r"""() => {
  const d3=window.__riD3;
  if (!d3 || typeof d3.view.get_data !== 'function') throw Error('production D3 view.get_data absent');
  const dataset=d3.view.get_data();
  if (!dataset || typeof dataset.get !== 'function') throw Error('production D3 view did not return vis.DataSet');
  const rows=dataset.get();
  if (!Array.isArray(rows)) throw Error('vis.DataSet.get() did not return an array');
  return rows.map(r=>({x:r.x,y:r.y,z:r.z,style:r.style}));
}"""

# Capture the *real* vis.Graph3d.setData call made by the production view on a
# Bokeh ColumnDataSource change.  Instrument only this disposable page, do not
# replace the view's get_data(), its source, or any library implementation.
INSTALL_SETDATA_SPY_JS = r"""() => {
  const d3=window.__riD3;
  if (!d3 || !d3.visGraph || d3.spyInstalled) throw Error('D3 spy installation/identity');
  const graph=d3.visGraph, original=graph.setData;
  if (typeof original !== 'function') throw Error('vis.Graph3d.setData absent');
  d3.setDataCalls=[];
  d3.reapplied=0;
  graph.setData=function(dataset, ...rest) {
    // vis 4.16.1 Graph3d.setOptions() ends with this.setData(this.dataTable), an Array:
    // pass it through and count it, but record only the view's own vis.DataSet calls.
    if (Array.isArray(dataset)) { d3.reapplied++; return original.call(this,dataset,...rest); }
    if (!dataset || typeof dataset.get !== 'function') throw Error('Graph3d.setData input is not vis.DataSet');
    const rows=dataset.get();
    if (!Array.isArray(rows)) throw Error('Graph3d.setData DataSet.get() is not an array');
    const points=rows.map(r=>({x:r.x,y:r.y,z:r.z,style:r.style}));
    const result=original.call(this,dataset,...rest);
    d3.setDataCalls.push(points);
    return result;
  };
  d3.spyInstalled=true;
  return {installed:true,graphName:d3.graph.name};
}"""

# This wait is explicitly tied to the TWO changed rows and to the live
# Graph3d.setData payload, not to untouched z[0].  A source-only update cannot
# satisfy it if the Bokeh view's callback is disconnected.
AFTER_UPDATE_READY_JS = r"""(expected) => {
  const d3=window.__riD3;
  if (!d3 || d3.graph?.name!=='d3_rendered_graph') return false;
  const z=d3.graph.data_source?.get_column('z');
  const {n, first, second}=expected;
  if (!z || z.length!==n || z[first.index]!==first.value || z[second.index]!==second.value) return false;
  const calls=d3.setDataCalls || [];
  if (calls.length===0) return false;
  const points=calls[calls.length-1];
  return points.length===n && points[first.index].z===first.value && points[second.index].z===second.value;
}"""

SETDATA_CAPTURE_JS = r"""() => {
  const d3=window.__riD3;
  if (!d3 || !d3.spyInstalled || !d3.setDataCalls.length) throw Error('no real vis.Graph3d.setData call');
  const rows=d3.setDataCalls[d3.setDataCalls.length-1];
  return {calls:d3.setDataCalls.length,points:rows};
}"""


def _stable_canvas_screenshot(page, canvas, *, tries=12, interval_ms=100):
    """Bounded screenshot polling.  Redraw/camera motion is NOT a pixel oracle."""
    previous = None
    for _ in range(tries):
        current = canvas.screenshot()
        if previous is not None and current == previous:
            return current
        previous = current
        page.wait_for_timeout(interval_ms)
    raise AssertionError("D3 canvas did not settle to two identical consecutive frames")


@BROWSER
@pytest.mark.feature("DISPLAY.graph3d")
def test_d3_browser_real_vis_render_source_change_and_mouse_drag(headless_session, tmp_path):
    """Only Chromium owns rendering and physical mouse interaction; no Node substitute."""
    df = _frame()
    res = {}
    ctx, page = _open_pinned_browser_page(headless_session, _make_direct_visual_page(df, tmp_path), res)
    try:
        # The Bokeh root view owns this specific vis.Graph3d instance and
        # its shadow-DOM canvas.  Playwright locators pierce open shadow roots.
        page.wait_for_function(VIEW_READY_JS, timeout=20000)
        canvases = page.locator('canvas[data-ri-d3-canvas]')
        assert canvases.count() == 1, "expected exactly one canvas in the real D3 view"
        canvas = canvases.first
        assert canvas.is_visible(), "3D canvas must be visibly attached"
        # SOURCE_JS checks the Bokeh model; VIEW_INITIAL_DATA_JS independently
        # exercises the real production BokehVisJSGraph3DView.get_data().
        got = page.evaluate(SOURCE_JS)
        _assert_points(got["points"], _mapping_ref(df))
        view_points = page.evaluate(VIEW_INITIAL_DATA_JS)
        _assert_points(view_points, _mapping_ref(df))
        rect = canvas.bounding_box()
        assert rect and rect["width"] > 30 and rect["height"] > 30
        px = rect["x"] + rect["width"] * .5
        py = rect["y"] + rect["height"] * .5
        before = _stable_canvas_screenshot(page, canvas)
        # Real user interaction: pointer events, not a JS camera model change.
        page.mouse.move(px, py)
        page.mouse.down()
        page.mouse.move(px+32, py+22, steps=6)
        page.mouse.up()
        after = _stable_canvas_screenshot(page, canvas)
        assert before != after, "physical mouse drag did not change the D3 view canvas"
        # After camera motion is stable, install a test-only spy on the exact
        # vis.Graph3d object owned by this Bokeh view.  We then mutate the
        # *real* ColumnDataSource and require its callback to reach setData.
        spy = page.evaluate(INSTALL_SETDATA_SPY_JS)
        assert spy == {"installed": True, "graphName": "d3_rendered_graph"}
        changed = df.copy()
        changed.loc[3, "z"] = 100.0
        changed.loc[5, "z"] = -80.0
        assert changed.loc[3, "z"] != df.loc[3, "z"]
        assert changed.loc[5, "z"] != df.loc[5, "z"]
        before_update = _stable_canvas_screenshot(page, canvas)
        updated = page.evaluate("""(arr) => {
          const d3=window.__riD3;
          if (!d3 || d3.graph?.name!=='d3_rendered_graph') throw Error('graph missing/ambiguous');
          const src=d3.graph.data_source;
          if (src.type !== 'ColumnDataSource') throw Error('expected editable source, got ' + src.type);
          src.data = {...src.data, z: arr};
          src.change.emit();
          return Array.from(src.get_column('z'));
        }""", changed["z"].tolist())
        np.testing.assert_array_equal(updated, changed["z"].tolist())
        # D3-F1: both changed indices and the actual vis.Graph3d.setData
        # payload must be observed; untouched z[0] is never used as a wait.
        expected_update = {"n": len(changed),
                           "first": {"index": 3, "value": float(changed.loc[3, "z"])},
                           "second": {"index": 5, "value": float(changed.loc[5, "z"])}}
        assert expected_update["first"]["value"] == 100.0
        assert expected_update["second"]["value"] == -80.0
        page.wait_for_function(AFTER_UPDATE_READY_JS, arg=expected_update, timeout=20000)
        captured = page.evaluate(SETDATA_CAPTURE_JS)
        assert captured["calls"] >= 1
        _assert_points(captured["points"], _mapping_ref(changed))
        # F3 falsifier: a z/style swap in the *captured view dataset* must
        # fail the same comparator as the positive production setData check.
        swapped = [dict(p) for p in captured["points"]]
        swapped[3]["z"], swapped[3]["style"] = swapped[3]["style"], swapped[3]["z"]
        assert swapped[3]["z"] != captured["points"][3]["z"]
        with pytest.raises(AssertionError):
            _assert_points(swapped, _mapping_ref(changed))
        after_update = _stable_canvas_screenshot(page, canvas)
        assert before_update != after_update, "CDS change did not visibly redraw the 3D canvas"
        observed = page.evaluate(SOURCE_JS)
        _assert_points(observed["points"], _mapping_ref(changed))
        assert any("vis.min.js" in url for url in res["served"]), "vis must be pinned/local"
        assert not res["blocked"] and not res["page_errors"] and not res["console_errors"], res
    finally:
        ctx.close()



# ============================================================ integration (Claude20.RI): filter, G3-D1, G3-D2
GATE = np.asarray([-1.5, -.5, .5, 1.5, -1., 0., 1., 2., .25])
EMPTY = (0.3, 0.4)        # no gate value lies in it
PART = (-1., 1.)          # selects rows 1, 2, 4, 5, 6, 8


def _gated():
    df = _frame()
    df["gate"] = GATE
    return df


def _make_filtered_page(df, tmp_path, *, html=False, figures=None, name="filtered"):
    """The public builder with a range widget on gate (and compression), as a user page has it."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    from bokeh.document import Document
    original = sa.show
    sa.show = lambda *args, **kwargs: None
    try:
        obj = sa.bokehDrawSA.fromArray(
            df, None, figures or [[["x"], ["y"], ["z"], ["c"], {"name": "graph3d"}]],
            [["range", ["gate", -1.5, 2.]]],
            layout=[[0]], widgetLayout=[[0]], sourceArray=[], parameterArray=[],
            aliasArray=[], nPointRender=1000,
            arrayCompression=[(".*", [("code", 0), ("zip", 0), ("base64", 0)])],
        )
    finally:
        sa.show = original
    doc = Document()
    doc.add_root(obj.pAll)
    tmp_path.mkdir(parents=True, exist_ok=True)
    if html:
        from bokeh.embed import file_html
        from bokeh.resources import INLINE
        target = tmp_path / f"{name}.html"
        target.write_text(file_html(doc, INLINE, "RI D3 filtered page"), encoding="utf8")
    else:
        target = tmp_path / f"{name}.json"
        target.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf8")
    return target


# The graph's source as the view reads it: the first get_length() entries of each column (BokehVisJSGraph3DView.get_data).
SOURCE_FIRST_N_JS = r"""() => {
  const graphs = window.__riH.models().filter(m => m.type === 'BokehVisJSGraph3D');
  if (graphs.length !== 1) throw Error('BokehVisJSGraph3D count=' + graphs.length);
  const g = graphs[0], src = g.data_source, n = src.get_length();
  const cols = [g.x, g.y, g.z, g.style].map(k => src.get_column(k));
  if (cols.some(c => c == null || c.length < n)) throw Error('3D column shorter than get_length()');
  return {sourceType: src.type, n, points: Array.from({length:n}, (_,i)=>({x:cols[0][i], y:cols[1][i], z:cols[2][i], style:cols[3][i]}))};
}"""


@NODE
@pytest.mark.feature("DISPLAY.graph3d")
def test_d3_node_public_page_graph_follows_a_range_filter(node_session, tmp_path):
    """Integration: after each range move the graph's source gives exactly the selected rows, in row order."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import SET_JS, SETTLE_ROUNDS
    df = _gated()
    res = ScenarioResult(id="d3/filter", outdir=str(tmp_path))
    page = node_session.new_page(res)
    page.goto(_make_filtered_page(df, tmp_path))
    got = page.evaluate(SOURCE_FIRST_N_JS)
    assert got["sourceType"] == "DownsamplerCDS", got["sourceType"]
    _assert_points(got["points"], _mapping_ref(df))
    for rng in (PART, EMPTY, (-1.5, 2.)):
        done = page.evaluate(SET_JS, ["gate", list(rng), [], SETTLE_ROUNDS])
        assert not done.get("error"), done
        want = df[df.gate.between(*rng)].reset_index(drop=True)
        _assert_points(page.evaluate(SOURCE_FIRST_N_JS)["points"], _mapping_ref(want))
    assert not res.page_errors and not res.console_errors, (res.page_errors, res.console_errors)


# ---------------------------------------------- G3-D1: a 3D graph without a colour variable (Python builder)
def _build_error(figures, tmp_path):
    try:
        _make_filtered_page(_gated(), tmp_path, figures=figures, name="nocolour")
    except Exception as exc:                 # the builder's own exception, kept exactly
        return f"{type(exc).__name__}: {exc}"
    return None


def _is_g3_d1(error) -> bool:
    return (error is not None and error.startswith("ValueError: " + G3_D1_START) and error.endswith(G3_D1_END))


@UNIT
@pytest.mark.feature("DISPLAY.graph3d")
def test_signature_g3_d1_a_3d_graph_without_colour_cannot_be_built(tmp_path):
    """G3-D1, exact: x, y and z without a colour variable -> ValueError on BokehVisJSGraph3D.style (None)."""
    error = _build_error([[["x"], ["y"], ["z"], {"name": "graph3d"}]], tmp_path)
    assert _is_g3_d1(error), error


@UNIT
@pytest.mark.feature("DISPLAY.graph3d")
def test_control_g3_d1_the_same_graph_with_a_colour_variable_is_built(tmp_path):
    """G3-D1's control: the same page with the colour variable builds; its graph maps x, y, z and c."""
    assert _build_error([[["x"], ["y"], ["z"], ["c"], {"name": "graph3d"}]], tmp_path) is None


@UNIT
@pytest.mark.feature("DISPLAY.graph3d")
@pytest.mark.xfail(**STRICT, reason="G3-D1: a 3D graph without a colour variable cannot be built")
def test_strict_a_3d_graph_without_colour_is_built(tmp_path):
    """The builder accepts a 3D graph without a colour variable (it handles that case: varNameColor = None)."""
    error = _build_error([[["x"], ["y"], ["z"], {"name": "graph3d"}]], tmp_path)
    known_defect(_is_g3_d1(error), "G3-D1: ValueError on BokehVisJSGraph3D.style = None")
    assert error is None, error


# ---------------------------------------------- G3-D2: an empty selection leaves the old points (Chromium)
# The 3D view of a public page is a child view: walk Bokeh.index through child_views to find it.
PUBLIC_VIEW_READY_JS = r"""() => {
  if (!window.vis || typeof window.vis.Graph3d !== 'function') return false;
  if (!window.__riH || !window.Bokeh || !Bokeh.index) return false;
  const models=window.__riH.models().filter(m=>m.type==='BokehVisJSGraph3D');
  if (models.length !== 1) throw Error('D3 public model identity: ' + models.length);
  const found=[], seen=new Set();
  const walk=v=>{ if(!v||seen.has(v)) return; seen.add(v); if(v.model===models[0]) found.push(v);
                  for (const c of (v.child_views||[])) walk(c); };
  const idx=Bokeh.index;
  for (const v of (idx instanceof Map ? idx.values() : Object.values(idx))) walk(v);
  if (found.length !== 1) return false;
  const view=found[0], visGraph=view._graph, root=view.shadow_el || view.el?.shadowRoot;
  if (!visGraph || typeof visGraph.setData !== 'function' || !root) return false;
  const canvases=Array.from(root.querySelectorAll('canvas')).filter(c=>c.width>10 && c.height>10);
  if (canvases.length !== 1 || !canvases[0].isConnected) return false;
  canvases[0].setAttribute('data-ri-d3-canvas', models[0].id);
  window.__riD3={graph:models[0], view, visGraph, canvas:canvases[0], setDataCalls:[]};
  return true;
}"""

VIS_STATE_JS = r"""() => {
  const d3=window.__riD3;
  const dp=d3.visGraph.dataPoints;
  return {viewPoints:d3.view.get_data().get().map(r=>({x:r.x,y:r.y,z:r.z,style:r.style})),
          visCount: Array.isArray(dp) ? dp.length : null,
          calls: d3.setDataCalls.length,
          last: d3.setDataCalls.length ? d3.setDataCalls[d3.setDataCalls.length-1] : null};
}"""


def _public_graph_after(headless_session, tmp_path, rng):
    """Load the public page in Chromium, check the 9 points at load, set the range through the model, wait for the
    view's own setData call with the selected number of rows, and return the state before and after."""
    df = _gated()
    want = df[df.gate.between(*rng)].reset_index(drop=True)
    res = {}
    ctx, page = _open_pinned_browser_page(headless_session, _make_filtered_page(df, tmp_path, html=True), res)
    try:
        page.wait_for_function(PUBLIC_VIEW_READY_JS, timeout=20000)
        canvases = page.locator('canvas[data-ri-d3-canvas]')
        assert canvases.count() == 1 and canvases.first.is_visible()
        before = page.evaluate(VIS_STATE_JS)
        page.evaluate(INSTALL_SETDATA_SPY_JS)
        page.evaluate("(r) => { const s=window.__riH.slider('gate'); if (!s) throw Error('gate slider'); s.value=r; }",
                      list(rng))
        page.wait_for_function("(n) => { const c=window.__riD3.setDataCalls; return c.length>0 && c[c.length-1].length===n; }",
                               arg=len(want), timeout=20000)
        after = page.evaluate(VIS_STATE_JS)
        assert any("vis.min.js" in url for url in res["served"]), "vis must be pinned/local"
        return df, want, before, after, res
    finally:
        ctx.close()


def _is_g3_d2(before, after, n_before) -> bool:
    """G3-D2, exact: at load vis had all n_before points; after the empty selection the view's get_data() and its
    own setData call are empty, but vis.Graph3d still holds n_before points."""
    return (before["visCount"] == n_before and after["viewPoints"] == [] and after["last"] == []
            and after["visCount"] == n_before)


@BROWSER
@pytest.mark.feature("DISPLAY.graph3d")
def test_signature_g3_d2_an_empty_selection_leaves_the_old_points_in_vis(headless_session, tmp_path):
    df, want, before, after, res = _public_graph_after(headless_session, tmp_path, EMPTY)
    assert len(want) == 0
    _assert_points(before["viewPoints"], _mapping_ref(df))
    assert _is_g3_d2(before, after, len(df)), (before["visCount"], after)
    assert not res["blocked"] and not res["page_errors"] and not res["console_errors"], res


@BROWSER
@pytest.mark.feature("DISPLAY.graph3d")
def test_control_g3_d2_a_partial_selection_reaches_vis_exactly(headless_session, tmp_path):
    """G3-D2's control and the public-page check: 9 reference points at load; after the range (-1, 1) the view hands
    vis.js exactly the 6 selected rows, and vis holds 6 points."""
    df, want, before, after, res = _public_graph_after(headless_session, tmp_path, PART)
    assert len(want) == 6
    _assert_points(before["viewPoints"], _mapping_ref(df))
    assert before["visCount"] == len(df)
    _assert_points(after["last"], _mapping_ref(want))
    _assert_points(after["viewPoints"], _mapping_ref(want))
    assert after["visCount"] == len(want)
    assert not res["blocked"] and not res["page_errors"] and not res["console_errors"], res


@BROWSER
@pytest.mark.feature("DISPLAY.graph3d")
@pytest.mark.xfail(**STRICT, reason="G3-D2: an empty selection leaves the previous points in the 3D graph")
def test_strict_an_empty_selection_leaves_no_points_in_the_3d_graph(headless_session, tmp_path):
    df, want, before, after, res = _public_graph_after(headless_session, tmp_path, EMPTY)
    known_defect(_is_g3_d2(before, after, len(df)), "G3-D2: vis.Graph3d still holds the old points")
    assert after["visCount"] in (0, None), after
    assert not res["blocked"] and not res["page_errors"] and not res["console_errors"], res
