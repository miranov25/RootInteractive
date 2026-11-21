"""PHASE_5 group 6 - small defects whose fix is clear (Claude10.RI at the Architect's request, 2026-11-20).

In plain words. Each defect has a test of the wanted behaviour, an exact description of the wrong behaviour before
the fix (its "signature") and a healthy control, so no test is empty. The defects are FIXED (the fix commit of group 6):
the tests of the wanted behaviour are ordinary tests, and each signature is a REGRESSION check that names the defect.

The defects (closure plan of 2026-11-18, section 1; the Architect's answers to the group 6 questions G3 and G4, and
the items "settled without your answer" of the same document):
  H-RANGE   a histogram whose range has two equal ends (a range slider pulled to one value, or a column with one
            value). An explicit range [v, v] made every read of the histogram throw "Histogram range must be finite,
            ordered, and have finite width"; an automatic range of a column with one value became [v - 1, v + 1].
            NumPy's rule (L-46; answer G3 a): the range becomes [v - 0.5, v + 0.5].
  H-ERROR   the histogram's range check (HistoNdCDS.histogram) said only "Assertion error: Range minimum in
            histogram<name>is broken". It must name the histogram, the axis, the range, the value and the tolerance.
  H-CENTRE  the bin centre was computed with its own formula, min + (i + 0.5) * (max - min) / n, which differs from the
            middle of the bin's NumPy edges in the last bit for some ranges (for example [0, 1] with 3 bins). The bin
            centre is the middle of each bin's NumPy edges (L-46).
  DEBUG     nine console.time / console.timeEnd / console.debug / console.info calls in CDSAlias.ts, CDSCompress.ts
            and HistoNdCDS.ts (register R-2, the rest after group 2). Answer G4 b: they stay, behind one switch that
            is off by default (window.RootInteractiveDebug = true turns them on for performance work).
  PRINTS    two Python modules printed "x <file>.ts" when imported (bokehVisJS3DGraph.py, HistogramCDS.py), and the
            page builder printed "compressCDSPipe" for every compressed table.
  PRINTS-2  (the group 6b commits, Claude21.RI) five more modules printed "Import  <file>.ts" when imported (CDSAlias.py,
            CDSCompress.py, CDSJoin.py, CDSStack.py, DownsamplerCDS.py), and the page builder printed the number of
            options of a selection widget (a select widget, and a multi-select widget with 31 options or more).
  LENGTH    the length of a compressed table came from whichever column was decoded first, so a corrupted first
            column made every good column be refused. The length is now the table's own row count, sent with the page.
  G5        a table without rows and an automatic slider stopped the page build with an unnamed NumPy error. Answer
            G5 a: the slider gets the neutral range 0 to 1.
  G6        the dashboard template getDefaultVars without a normalization stopped with "KeyError: None". Answer G6 a:
            no normalization means a dashboard without the normalized plots (with one, nothing changes).
  G11       an alias whose variables include a parameter looked the parameter up as a column: the alias was all 0 and
            the page logged an unnamed TypeError. Answer G11 a: the variable takes the parameter's value and follows
            its widget.
  3D        a 3D graph whose table is empty when the page loads threw at every mouse move and wheel step over it
            (group 2 review G2-S1). The vis.js graph is now created when the first rows arrive.
Two more items of group 6 are tested where their tests already are: TR-D1 (answer G8 a, replacing the data of a
compressed table) in test_headless_node_d1p2_transport.py, and WR-D1 (answer G7 a, the reference side of the WeightsRef
dashboard from the values) in test_headless_oracle_weightsref.py.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def regression(condition: bool, message: str) -> None:
    """After the fix, the exact signature of a fixed defect is a regression: an ordinary failure that names it."""
    if condition:
        raise AssertionError("REGRESSION, a fixed defect is back: " + message)

UNIT = pytest.mark.headless_unit
BOKEH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BOKEH_DIR.parents[2]


def _node(target):
    """The target, backend and layer marks of a Node test. The feature marks are written out on each test, because
    scripts/generate_capability_matrix.py reads only the literal form @pytest.mark.feature("ID")."""
    marks = [target, pytest.mark.backend("node"), pytest.mark.layer("integration")]

    def wrap(f):
        for m in reversed(marks):
            f = m(f)
        return f
    return wrap


# ------------------------------------------------------------------ pages (the public builder) and the Node harness
def _page(tmp_path, tag, df, figures, histograms=(), parameters=(), widgets=(), **options):
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    before = sa.show
    sa.show = lambda *a, **kw: None
    try:
        page = sa.bokehDrawSA.fromArray(df, None, list(figures), list(widgets), parameterArray=list(parameters),
                                        histogramArray=list(histograms), layout=[list(range(len(figures)))],
                                        widgetLayout=[list(range(len(widgets)))], **options)
    finally:
        sa.show = before
    doc = Document()
    doc.add_root(page.pAll)
    out = tmp_path / f"group6_{tag}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


def _open(session, path):
    r = session.request("new_doc", path=str(path))
    assert r.get("ok") and not r.get("page_errors") and not r.get("console_errors"), r
    return r


def _eval(session, code, arg=None):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = session.request("eval", code=code, arg=arg)
    assert r.get("ok") and not r.get("page_errors"), r
    return decode(r["result"]), list(r.get("console_errors") or [])


# the columns of one histogram model; a column whose read throws is returned as "ERROR <message>"
HIST_COLUMNS = r"""(a) => {
  const ms = window.__riH.models().filter(m => m.name === a.name && (m.type === 'HistogramCDS' || m.type === 'HistoNdCDS'));
  if (ms.length !== 1) throw Error('histogram ' + a.name + ': ' + ms.length);
  const out = {};
  for (const k of a.columns) {
    try { out[k] = Array.from(ms[0].get_column(k) ?? []) } catch (e) { out[k] = 'ERROR ' + e.name + ': ' + e.message }
  }
  return out;
}"""
SET_WIDGET = r"""async (a) => {
  const ws = window.__riH.models().filter(w => 'value' in w && w.js_property_callbacks && w.name === a.name);
  if (ws.length !== 1) throw Error('widget ' + a.name + ': ' + ws.length);
  const w = ws[0], cbs = (w.js_property_callbacks || {})['change:value'] || [];
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


def _columns(session, name, columns):
    got, console = _eval(session, HIST_COLUMNS, {"name": name, "columns": list(columns)})
    return got, console


# ================================================================== H-RANGE: a range with two equal ends
X = np.array([0.3, 1.2, 2.0, 2.0, 2.0, 3.7])
RANGE_ERROR = "ERROR RangeError: Histogram range must be finite, ordered, and have finite width"


def _range_frame():
    return pd.DataFrame({"x": X, "y": X[::-1].copy(), "c": np.full(len(X), 2.0)})


def _range_page(tmp_path, tag):
    """A 1-D and a 2-D histogram whose x range follows the range slider rX, and a 1-D and a 2-D histogram of the
    column c, which has one value (2.0), with an automatic range."""
    hists = [{"name": "h1", "variables": ["x"], "nbins": 4, "range": "rX"},
             {"name": "h2", "variables": ["x", "y"], "nbins": [4, 4], "range": ["rX", [0, 4]]},
             {"name": "hc", "variables": ["c"], "nbins": 4},
             {"name": "hc2", "variables": ["c", "y"], "nbins": [4, 4], "range": [None, [0, 4]]}]
    figures = [[["bin_center"], ["bin_count"], {"source": "h1"}], [["bin_center_0"], ["bin_count"], {"source": "h2"}],
               [["bin_center"], ["bin_count"], {"source": "hc"}], [["bin_center_0"], ["bin_count"], {"source": "hc2"}]]
    return _page(tmp_path, tag, _range_frame(), figures, hists, [{"name": "rX", "value": [0, 4], "range": [0, 4]}],
                 [["range", ["rX"]]])


def _numpy_1d(values, rng):
    counts, edges = np.histogram(values, bins=4, range=rng)
    return counts.astype(float).tolist(), edges.tolist()


def _numpy_2d_axis0(x, y, rng_x):
    """bin_count of a 4 x 4 histogram in RootInteractive's order (axis 0 fastest), and the edges of axis 0."""
    counts, ex, _ = np.histogram2d(x, y, bins=[4, 4], range=[rng_x, [0, 4]])
    return counts.T.ravel().astype(float).tolist(), ex.tolist()


@_node(pytest.mark.node_histograms)
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("HIST.histogram_nd")
def test_control_h_range_an_ordinary_slider_range_gives_numpy_histograms(node_session, tmp_path):
    """Healthy control, before and after the fix: the slider at [0, 4] and at [1, 3]."""
    _open(node_session, _range_page(tmp_path, "range_control"))
    for rng in ([0, 4], [1, 3]):
        _eval(node_session, SET_WIDGET, {"name": "rX", "value": rng})
        h1, _ = _columns(node_session, "h1", ["bin_count", "bin_bottom", "bin_top"])
        counts, edges = _numpy_1d(X, tuple(rng))
        assert h1["bin_count"] == counts and h1["bin_bottom"] + h1["bin_top"][-1:] == edges, (rng, h1)
        h2, _ = _columns(node_session, "h2", ["bin_count"])
        assert h2["bin_count"] == _numpy_2d_axis0(X, X[::-1], tuple(rng))[0], (rng, h2)


@_node(pytest.mark.node_histograms)
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("HIST.histogram_nd")
def test_a_slider_range_with_two_equal_ends_gives_the_numpy_histogram(node_session, tmp_path):
    """The slider pulled to one value, [2, 2]: the histograms are NumPy's for range (2, 2), whose edges are
    1.5 ... 2.5, and nothing throws. Before the fix every read of either histogram threw RangeError."""
    _open(node_session, _range_page(tmp_path, "range_equal"))
    _eval(node_session, SET_WIDGET, {"name": "rX", "value": [2, 2]})
    h1, console1 = _columns(node_session, "h1", ["bin_count", "bin_bottom", "bin_top"])
    h2, console2 = _columns(node_session, "h2", ["bin_count", "bin_bottom_0", "bin_top_0"])
    regression(all(v == RANGE_ERROR for v in list(h1.values()) + list(h2.values())),
                 "H-RANGE: every read of the 1-D and the 2-D histogram throws " + RANGE_ERROR[6:])
    counts, edges = _numpy_1d(X, (2, 2))
    assert edges == [1.5, 1.75, 2.0, 2.25, 2.5]
    assert h1 == {"bin_count": counts, "bin_bottom": edges[:-1], "bin_top": edges[1:]}, h1
    counts2, ex = _numpy_2d_axis0(X, X[::-1], (2, 2))
    assert h2["bin_count"] == counts2, h2
    assert h2["bin_bottom_0"][:4] == ex[:-1] and h2["bin_top_0"][:4] == ex[1:], h2
    assert console1 == console2 == []


@_node(pytest.mark.node_histograms)
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("HIST.histogram_nd")
def test_the_automatic_range_of_a_column_with_one_value_is_numpys(node_session, tmp_path):
    """The column c has one value, 2.0. NumPy's automatic range is (1.5, 2.5); before the fix it was (1, 3)."""
    _open(node_session, _range_page(tmp_path, "range_auto"))
    hc, _ = _columns(node_session, "hc", ["bin_count", "bin_bottom", "bin_top"])
    hc2, _ = _columns(node_session, "hc2", ["bin_count", "bin_bottom_0", "bin_top_0"])
    regression(hc["bin_bottom"] == [1.0, 1.5, 2.0, 2.5] and hc2["bin_bottom_0"][:4] == [1.0, 1.5, 2.0, 2.5],
                 "H-RANGE: the automatic range of the one-valued column is (1, 3), not NumPy's (1.5, 2.5)")
    counts, edges = np.histogram(np.full(6, 2.0), bins=4)
    assert edges.tolist() == [1.5, 1.75, 2.0, 2.25, 2.5]
    assert hc == {"bin_count": counts.astype(float).tolist(), "bin_bottom": edges[:-1].tolist(),
                  "bin_top": edges[1:].tolist()}, hc
    assert hc2["bin_bottom_0"][:4] == edges[:-1].tolist() and hc2["bin_top_0"][:4] == edges[1:].tolist(), hc2
    counts2 = np.histogram2d(np.full(6, 2.0), X[::-1], bins=[4, 4], range=[(1.5, 2.5), (0, 4)])[0]
    assert hc2["bin_count"] == counts2.T.ravel().astype(float).tolist(), hc2


# ================================================================== H-ERROR: the text of the range check
OLD_RANGE_CHECK = "Assertion error: Range minimum in histogramh2is broken"
BREAK_TRANSFORM = r"""() => {
  const m = window.__riH.models().filter(m => m.name === 'h2' && m.type === 'HistoNdCDS')[0];
  m.get_column('bin_count');
  m._transform_origin[0] += 0.5;
  m._unweighted_histogram = null;
  m.cached_columns.clear();
  try { m.histogram(null); return null } catch (e) { return String(e.message) }
}"""


@_node(pytest.mark.node_histograms)
@pytest.mark.feature("HIST.histogram_nd")
def test_the_range_check_of_a_histogram_names_axis_range_value_and_tolerance(node_session, tmp_path):
    """A histogram whose bin transform no longer maps its range onto its bins (forced here by moving the transform
    of axis 0 by half a bin) refuses with an error that names the histogram, the axis, the range, the value and the
    tolerance."""
    _open(node_session, _range_page(tmp_path, "range_error"))
    message, _ = _eval(node_session, BREAK_TRANSFORM)
    regression(message == OLD_RANGE_CHECK, "H-ERROR: the error says only '" + OLD_RANGE_CHECK + "'")
    assert message is not None
    for part in ('"h2"', "axis 0", "[0, 4]", "-0.5", "1e-6"):
        assert part in message, (part, message)


# ================================================================== H-CENTRE: the bin centre
def _centre_page(tmp_path, tag):
    """Ranges whose old bin-centre formula differs from the middle of the NumPy edges: [0, 1] with 3 bins (bin 2)
    and [0, 1] with 10 bins (bins 1, 3, 6 and 8)."""
    df = pd.DataFrame({"x": np.linspace(0, 1, 11), "y": np.linspace(1, 0, 11)})
    hists = [{"name": "k1", "variables": ["x"], "nbins": 3, "range": [0, 1]},
             {"name": "k2", "variables": ["x", "y"], "nbins": [10, 3], "range": [[0, 1], [0, 1]]}]
    figures = [[["bin_center"], ["bin_count"], {"source": "k1"}], [["bin_center_0"], ["bin_count"], {"source": "k2"}]]
    return _page(tmp_path, tag, df, figures, hists)


def _old_centres(lo, hi, n):
    return [lo + (i + .5) * (hi - lo) / n for i in range(n)]


def _old_centres_nd(lo, hi, n):
    inv_scale = 1 / (n / (hi - lo))
    return [lo + (i + .5) * inv_scale for i in range(n)]


def _mid_centres(lo, hi, n):
    e = np.linspace(lo, hi, n + 1)
    return ((e[:-1] + e[1:]) / 2).tolist()


@UNIT
def test_control_h_centre_the_two_formulas_differ_for_these_ranges():
    """The test is not empty: for these ranges the old formula and the middle of the edges differ."""
    assert _old_centres(0, 1, 3) != _mid_centres(0, 1, 3)
    assert _old_centres_nd(0, 1, 10) != _mid_centres(0, 1, 10)


@_node(pytest.mark.node_histograms)
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("HIST.histogram_nd")
def test_the_bin_centre_is_the_middle_of_the_numpy_edges(node_session, tmp_path):
    _open(node_session, _centre_page(tmp_path, "centre"))
    k1, _ = _columns(node_session, "k1", ["bin_center", "bin_bottom", "bin_top"])
    k2, _ = _columns(node_session, "k2", ["bin_center_0", "bin_bottom_0", "bin_top_0", "bin_center_1"])
    regression(k1["bin_center"] == _old_centres(0, 1, 3) and k2["bin_center_0"][:10] == _old_centres_nd(0, 1, 10),
                 "H-CENTRE: the bin centres are min + (i + 0.5) * width / n, not the middle of the edges")
    assert k1["bin_center"] == [(a + b) / 2 for a, b in zip(k1["bin_bottom"], k1["bin_top"])], k1
    assert k1["bin_center"] == _mid_centres(0, 1, 3), k1
    assert k2["bin_center_0"] == [(a + b) / 2 for a, b in zip(k2["bin_bottom_0"], k2["bin_top_0"])], k2
    assert k2["bin_center_0"][:10] == _mid_centres(0, 1, 10), k2
    assert k2["bin_center_1"][::10] == _mid_centres(0, 1, 3), k2


# ================================================================== DEBUG: timing and debug calls behind one switch
DEBUG_CALLS = r"""(a) => {
  const counts = {time: 0, timeEnd: 0, debug: 0, info: 0};
  const consoles = [console, globalThis.console, window.console].filter((c, i, all) => c && all.indexOf(c) === i);
  const saved = [];
  for (const c of consoles) for (const k of Object.keys(counts)) { saved.push([c, k, c[k]]); c[k] = () => { counts[k]++ } }
  const before = window.RootInteractiveDebug;
  if (a.on) window.RootInteractiveDebug = true;
  try {
    const ms = window.__riH.models();
    // CDSCompress: a new table is initialized, and a column with a "linear" step is read (its decoding is not needed)
    const C = ms.filter(m => m.type === 'CDSCompress')[0].constructor;
    const probe = new C({name: 'group6_probe', inputData: {q: {array: '', decodeProgram: [['linear', {}]], dtype: 'float64'}}});
    try { probe.get_column('q') } catch (_e) {}
    // CDSAlias: a column computed from fields, invalidated and read again
    const alias = ms.filter(m => m.type === 'CDSAlias' && m.mapping && m.mapping[a.alias])[0];
    alias.invalidate_column(a.alias);
    alias.get_column(a.alias);
    // HistoNdCDS: the histogram counted again
    const h = ms.filter(m => m.type === 'HistoNdCDS')[0];
    h.change_selection();
    h.get_column('bin_count');
  } finally {
    for (const [c, k, f] of saved) c[k] = f;
    if (a.on) window.RootInteractiveDebug = before;
  }
  return counts;
}"""


def _debug_page(tmp_path, tag):
    df = pd.DataFrame({"A": np.linspace(0, 1, 50), "B": np.linspace(1, 2, 50)})
    aliases = [{"name": "AB", "variables": ["A", "B"], "func": "return A + B"}]
    hists = [{"name": "hAB", "variables": ["A", "B"], "nbins": [5, 5], "range": [[0, 1], [1, 2]]}]
    figures = [[["A"], ["AB"]], [["bin_center_0"], ["bin_count"], {"source": "hAB"}]]
    return _page(tmp_path, tag, df, figures, hists, aliasArray=aliases,
                 arrayCompression=[(".*", [("relative", 16), ("code", 0), ("zip", 0), ("base64", 0)])])


@_node(pytest.mark.node_display)
@pytest.mark.feature("ALIAS.cdsalias")
@pytest.mark.feature("HIST.histogram_nd")
def test_control_debug_with_the_switch_on_the_timing_calls_are_made(node_session, tmp_path):
    """Healthy control, before and after the fix: with window.RootInteractiveDebug = true the probe reaches the
    timing and debug calls of the three models (so the next test is not empty)."""
    _open(node_session, _debug_page(tmp_path, "debug_on"))
    counts, _ = _eval(node_session, DEBUG_CALLS, {"on": True, "alias": "AB"})
    assert counts["time"] >= 2 and counts["timeEnd"] >= 2 and counts["info"] == 1 and counts["debug"] == 1, counts


@_node(pytest.mark.node_display)
@pytest.mark.feature("ALIAS.cdsalias")
@pytest.mark.feature("HIST.histogram_nd")
def test_debug_the_models_print_nothing_unless_the_switch_is_on(node_session, tmp_path):
    _open(node_session, _debug_page(tmp_path, "debug_off"))
    counts, _ = _eval(node_session, DEBUG_CALLS, {"on": False, "alias": "AB"})
    regression(counts["time"] >= 2 and counts["timeEnd"] >= 2 and counts["info"] == 1 and counts["debug"] == 1,
                 f"DEBUG: console.time, timeEnd, debug and info are called without the switch: {counts}")
    assert counts == {"time": 0, "timeEnd": 0, "debug": 0, "info": 0}, counts


# ================================================================== PRINTS: Python prints at import and build time
PRINT_SCRIPT = r"""
import numpy as np, pandas as pd
import RootInteractive.InteractiveDrawing.bokeh.bokehVisJS3DGraph
import RootInteractive.InteractiveDrawing.bokeh.HistogramCDS
import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
sa.show = lambda *a, **k: None
print("=== build")
df = pd.DataFrame({"A": np.linspace(0, 1, 20), "B": np.linspace(1, 2, 20)})
sa.bokehDrawSA.fromArray(df, None, [[["A"], ["B"]]], [], layout=[[0]], widgetLayout=[[]],
                         arrayCompression=[(".*", [("relative", 16), ("code", 0), ("zip", 0), ("base64", 0)])])
"""


def _printed():
    r = subprocess.run([sys.executable, "-c", PRINT_SCRIPT], capture_output=True, text=True, cwd=REPO_ROOT,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.splitlines()


@UNIT
def test_prints_no_debug_line_at_import_or_when_a_compressed_page_is_built():
    """"x <file>.ts" at import and "compressCDSPipe" at build time are gone. The control is the build itself, which
    must still succeed (and the "=== build" marker is printed)."""
    lines = _printed()
    assert "=== build" in lines
    debug = [s for s in lines if re.fullmatch(r"x \S+\.ts", s) or s == "compressCDSPipe"]
    regression(sorted(debug) == ["compressCDSPipe", "x HistogramCDS.ts", "x bokehVisJS3DGraph.ts"],
                 f"PRINTS: the debug lines {debug}")
    assert debug == [], debug


PRINT_SCRIPT_2 = r"""
import numpy as np, pandas as pd
import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
sa.show = lambda *a, **k: None
print("=== build")
n = 80
df = pd.DataFrame({"A": np.linspace(0, 1, n), "B": np.linspace(1, 2, n),
                   "k": np.arange(n) % 3, "m": np.arange(n) % 40})
sa.bokehDrawSA.fromArray(df, None, [[["A"], ["B"]]], [["select", ["k"]], ["multiSelect", ["m"]]],
                         layout=[[0]], widgetLayout=[[0, 1]])
"""


@UNIT
def test_prints_no_import_line_and_no_number_of_options_when_a_page_is_built():
    """PRINTS-2: importing the page builder and building a page with a select widget (3 options) and a multi-select
    widget (40 options) prints nothing but the marker "=== build". Before, five modules printed "Import  <file>.ts" at
    import, and the page builder printed 3 and 40. The control is the build itself, which must still succeed."""
    r = subprocess.run([sys.executable, "-c", PRINT_SCRIPT_2], capture_output=True, text=True, cwd=REPO_ROOT,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = r.stdout.splitlines()
    assert "=== build" in lines
    debug = [s for s in lines if re.fullmatch(r"Import\s+\S+\.ts|\d+", s)]
    regression(sorted(debug) == ["3", "40", "Import  CDSAlias.ts", "Import  CDSCompress.ts", "Import  CDSJoin.ts",
                                 "Import  CDSStack.ts", "Import  DownsamplerCDS.ts"],
               f"PRINTS-2: the debug lines {debug}")
    assert debug == [], debug


# ================================================================== LENGTH: the length of a compressed table
def _length_page(headless_out, stem):
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_node_d1p2_transport import _corrupt_page
    return _corrupt_page(headless_out, stem)


READ_CORRUPT_FIRST = r"""() => {
  const m = window.__riH.model('cdsOrig', ['CDSCompress']);
  const read = (k) => { try { return {length: m.get_column(k).length} } catch (e) { return {error: String(e.message)} } };
  const w = read('w'), x = read('x');
  return {w, x, n: m.get_length()};
}"""


@_node(pytest.mark.node_transport)
@pytest.mark.feature("ENC.cds_compress.length_guard")
def test_length_a_corrupted_first_column_is_refused_and_the_good_columns_are_read(node_session, headless_out):
    """The 8-row table whose column w was replaced by a 7-row payload; w is read FIRST. The corrupted w is refused,
    x is read with its 8 rows, and the table has 8 rows. Before the fix, w set the length to 7: w was accepted and
    the good column x was refused."""
    r = node_session.request("new_doc", path=str(_length_page(headless_out, "group6_length")))
    assert r.get("ok"), r
    got, _ = _eval(node_session, READ_CORRUPT_FIRST)
    regression(got == {"w": {"length": 7}, "x": {"error": "Corrupted length of column x: 8 expected: 7"}, "n": 7},
                 "LENGTH: the corrupted w (7 rows) is accepted and sets the length; the good x (8 rows) is refused")
    assert got == {"w": {"error": "Corrupted length of column w: 7 expected: 8"}, "x": {"length": 8}, "n": 8}, got


# ================================================================== G5: a table without rows and an automatic slider
EMPTY_TABLE_ERROR = ("ValueError", "zero-size array to reduction operation minimum which has no identity")


def _empty_table_page(df):
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    before = sa.show
    sa.show = lambda *a, **kw: None
    try:
        return sa.bokehDrawSA.fromArray(df, None, [[["A"], ["B"]]], [["range", ["A"]], ["slider", ["B"]]],
                                        layout=[[0]], widgetLayout=[[0, 1]])
    finally:
        sa.show = before


def _sliders(page):
    from bokeh.models import RangeSlider, Slider
    return {m.name: (type(m).__name__, m.start, m.end, m.value) for m in page.pAll.references()
            if isinstance(m, (RangeSlider, Slider))}


@UNIT
@pytest.mark.feature("FILTER.range")
def test_control_g5_the_automatic_sliders_of_a_table_with_rows_span_its_values():
    df = pd.DataFrame({"A": [0.5, 1.5, 2.5], "B": [-1.0, 0.0, 3.0]})
    got = _sliders(_empty_table_page(df))
    assert got["A"] == ("RangeSlider", 0.5, 2.5, (0.5, 2.5)), got
    assert got["B"][:3] == ("Slider", -1.0, 3.0), got


@UNIT
@pytest.mark.feature("FILTER.range")
def test_g5_a_table_without_rows_and_automatic_sliders_is_built_with_a_neutral_range():
    """Answer G5 a: an empty selection is a normal state of an analysis; the automatic sliders get the range 0 to 1."""
    df = pd.DataFrame({"A": np.array([], float), "B": np.array([], float)})
    try:
        got = _sliders(_empty_table_page(df))
    except ValueError as e:
        regression((type(e).__name__, str(e)) == EMPTY_TABLE_ERROR, "G5: " + str(e))
        raise
    assert got["A"] == ("RangeSlider", 0, 1, (0, 1)), got
    assert got["B"][:3] == ("Slider", 0, 1), got


# ================================================================== G6: the dashboard template without a normalization
# the figure layout of the template with a normalization, as before group 6 (it must not change)
LAYOUT_WITH_NORMALIZATION = {
    "histoXY": [[0, 1], [2, 3], {"plot_height": 250}], "histoXYNorm": [[4, 5], [6, 7], {"plot_height": 250}],
    "histoXYZ": [[8, 9], [10, 11], {"plot_height": 250}], "histoXYNormZ": [[12, 13], [14, 15], {"plot_height": 250}],
    "histoXYNormZMedian": [[16, 17], {"plot_height": 350}], "histoXYNormZMean": [[18, 19], {"plot_height": 350}],
}


def _template(normalization):
    from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import getDefaultVars
    return getDefaultVars(normalization=normalization, variables=["A", "B", "C", "D"])


def _figure_sources(figures):
    return [f[2].get("source") for f in figures if isinstance(f, list) and len(f) == 3 and isinstance(f[2], dict)]


@UNIT
@pytest.mark.feature("DASHBOARD.histogram_only")
def test_control_g6_the_template_with_a_normalization_keeps_its_figures_and_layout():
    """Before and after the fix: with "diff" the template has its 20 plots, 4 tables and options entries, and the
    same layout (the fix builds the figure list group by group; this guards that it changed nothing here)."""
    out = _template("diff")
    figures, layout = out[6], out[7]
    assert len(figures) == 24, len(figures)
    assert {k: v for k, v in layout.items() if k.startswith("histo")} == LAYOUT_WITH_NORMALIZATION
    assert sum("Norm" in str(s) for s in _figure_sources(figures)) == 10


@UNIT
@pytest.mark.feature("DASHBOARD.histogram_only")
def test_g6_the_template_without_a_normalization_has_no_normalized_plots_and_builds_a_page():
    """Answer G6 a: no normalization means a dashboard without the normalized plots. Every layout index names one
    of its figures, no figure reads a normalized histogram, and the public builder makes the page."""
    try:
        out = _template(None)
    except KeyError as e:
        regression(e.args == (None,), "G6: getDefaultVars(normalization=None) stops with KeyError: None")
        raise
    aliases, variables, parameters, widgets, widget_layout, histograms, figures, layout = out
    assert not any("Norm" in str(s) for s in _figure_sources(figures)), _figure_sources(figures)
    assert not any("Norm" in h["name"] for h in histograms if isinstance(h, dict)), histograms
    indices = [k for v in layout.values() for row in v if isinstance(row, list) for k in row if isinstance(k, int)]
    assert indices and max(indices) < len(figures) - 1 and len(set(indices)) == len(indices), layout
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    rng = np.random.default_rng(20261120)
    df = pd.DataFrame(rng.random(size=(200, 4)), columns=list("ABCD"))
    before = sa.show
    sa.show = lambda *a, **kw: None
    try:
        sa.bokehDrawSA.fromArray(df, None, figures, widgets, layout=layout, parameterArray=parameters,
                                 widgetLayout=widget_layout, histogramArray=histograms, aliasArray=aliases)
    finally:
        sa.show = before


@UNIT
@pytest.mark.feature("DASHBOARD.histogram_only")
def test_g6_an_unknown_normalization_is_refused_by_name():
    with pytest.raises((ValueError, KeyError)) as e:
        _template("difference")
    assert "difference" in str(e.value)


# ================================================================== G11: an alias whose variables include a parameter
ALIAS_COLUMN = r"""() => Array.from(window.__riH.models()
  .filter(m => m.type === 'CDSAlias' && m.mapping && m.mapping.Acut)[0].get_column('Acut'), Number)"""
G11_ERROR = "TypeError: Cannot read properties of null (reading '0')"


def _alias_page(tmp_path, tag, variables, func):
    df = pd.DataFrame({"A": np.linspace(0, 1, 11), "B": np.full(11, 0.5)})
    aliases = [{"name": "Acut", "variables": variables, "func": func}]
    return _page(tmp_path, tag, df, [[["A"], ["Acut"]]], parameters=[{"name": "cut", "value": 0.5, "range": [0, 1]}],
                 widgets=[["slider", ["cut"]]], aliasArray=aliases)


def _expected(cut):
    return (np.linspace(0, 1, 11) > cut).astype(float).tolist()


@_node(pytest.mark.node_display)
@pytest.mark.feature("ALIAS.parameter")
def test_control_g11_an_alias_of_two_columns_is_computed(node_session, tmp_path):
    """Before and after the fix: the same alias with the column B (0.5 in every row) instead of the parameter."""
    _open(node_session, _alias_page(tmp_path, "g11_control", ["A", "B"], "return A > B"))
    got, console = _eval(node_session, ALIAS_COLUMN)
    assert got == _expected(0.5) and console == [], (got, console)


@_node(pytest.mark.node_display)
@pytest.mark.feature("ALIAS.parameter")
def test_g11_a_parameter_among_the_variables_of_an_alias_takes_its_value_and_follows_its_widget(node_session, tmp_path):
    """Answer G11 a: the variable cut takes the parameter's value (0.5), and the alias follows the slider (0.8)."""
    r = node_session.request("new_doc", path=str(_alias_page(tmp_path, "g11", ["A", "cut"], "return A > cut")))
    assert r.get("ok") and not r.get("page_errors"), r
    got, console = _eval(node_session, ALIAS_COLUMN)
    regression(got == [0.0] * 11 and console == [G11_ERROR],
                 f"G11: the alias is all 0 and the page logs '{G11_ERROR}'")
    assert got == _expected(0.5) and console == [] and not r.get("console_errors"), (got, console, r)
    _eval(node_session, SET_WIDGET, {"name": "cut", "value": 0.8})
    got, console = _eval(node_session, ALIAS_COLUMN)
    assert got == _expected(0.8) and console == [], (got, console)


# ================================================================== 3D: a graph that is empty when the page loads
DRAG_ERROR = "Cannot read properties of undefined (reading 'x')"
GRAPH_VIEW = r"""() => {
  const models = window.__riH.models().filter(m => m.type === 'BokehVisJSGraph3D' && m.name === 'd3_rendered_graph');
  const idx = Bokeh.index, views = idx instanceof Map ? Array.from(idx.values()) : Object.values(idx);
  const v = views.filter(v => v?.model?.id === models[0].id)[0];
  const r = v.el.getBoundingClientRect();
  const g = v._graph;
  return {x: r.x, y: r.y, w: r.width, h: r.height,
          points: g && Array.isArray(g.dataPoints) ? g.dataPoints.length : null};
}"""


def _drag_and_wheel(page, box):
    px, py = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    page.mouse.move(px, py)
    page.mouse.down()
    page.mouse.move(px + 30, py + 20, steps=5)
    page.mouse.up()
    page.mouse.wheel(0, 200)
    page.wait_for_timeout(300)


@pytest.mark.browser_d3
@pytest.mark.feature("DISPLAY.graph3d")
def test_3d_a_graph_without_data_can_be_dragged_and_shows_the_rows_that_arrive_later(headless_session, tmp_path):
    """The 3D graph of a table that is empty when the page loads (group 2 review G2-S1). A drag and a wheel step over
    it raise no error; when the nine rows arrive the graph draws them, and a drag still raises no error. Before the
    fix every mouse move and wheel step threw "Cannot read properties of undefined (reading 'x')"."""
    from RootInteractive.InteractiveDrawing.bokeh.headless import test_headless_visualization_d3 as D
    df = D._frame()
    res = {}
    ctx, page = D._open_pinned_browser_page(headless_session, D._make_direct_visual_page(df.iloc[0:0], tmp_path), res)
    try:
        page.wait_for_function("() => window.vis && typeof window.vis.Graph3d === 'function'", timeout=20000)
        page.wait_for_timeout(300)
        box = page.evaluate(GRAPH_VIEW)
        _drag_and_wheel(page, box)
        errors_empty = list(res["page_errors"])
        regression(len(errors_empty) > 0 and set(errors_empty) == {DRAG_ERROR},
                     f"3D: {len(errors_empty)} errors '{DRAG_ERROR}' while dragging the empty graph")
        assert errors_empty == [], errors_empty
        page.evaluate(D.SET_DIRECT_SOURCE_JS, {k: df[k].tolist() for k in df.columns})
        assert page.evaluate(GRAPH_VIEW)["points"] == len(df)
        _drag_and_wheel(page, box)
        assert res["page_errors"] == [] and res["console_errors"] == [] and not res["blocked"], res
    finally:
        ctx.close()
