"""PHASE_5 group 2 - three small defects outside the page builder (Claude21.RI, 2026-11-16).

In plain words. Each defect has a test of the wanted behaviour, an exact description of today's wrong behaviour (its
"signature") and a control that works today, so no test is empty. Before the fix the test of the wanted behaviour is
a strict expected failure; after the fix it is an ordinary test, and the signature becomes a REGRESSION check (an
ordinary failure that names the defect).

  1. The third custom function of the interactive template (register entry "funCustom2"). The template
     `getDefaultVars` (used as getDefaultVarsDiff and getDefaultVarsRatio) offers three text boxes, funCustomForm0,
     funCustomForm1 and funCustomForm2, in which the user types the body of a JavaScript function; the columns
     funCustom0, funCustom1 and funCustom2 are the results. The third column named its text box "funcCustomForm2"
     (one letter too many), so it never followed the box: the name itself was taken as the function's text.

  2. Debug printing in the browser models (register entry "R-2"). Several TypeScript models printed internal values
     to the browser console with console.log on every read or slider move (a column name, a histogram name, a range).
     A page must not print anything by itself. The two WebAssembly files that do not compile (register entry
     "W-D1/W-D2", parked) are left as they are.

  3. The plots named "Median" of the weights dashboard (Architect decision DEC-23: "a plot named Median shows the
     median"). The dashboard `test_interactiveTemplateWeights` of test_bokehClientHistogram.py drew the standard
     deviation in its four plots named "..._Median". The median of a profile is its column "quantile_<k>", where k is
     the position of 0.5 in the histogram's list of quantiles; the template's own median plots use it.

The tests of the 3D graph (register entry "G3-D2") are in test_headless_visualization_d3.py.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import CONSOLE_ALLOWED
from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4a_joins import _clean_reply, _clean_session

NODE = pytest.mark.node_aliases
BROWSER = pytest.mark.browser_d3
UNIT = pytest.mark.headless_unit

BOKEH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BOKEH_DIR.parents[2]


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


STRICT = dict(strict=True, raises=KnownDefect)


# ================================================================== 1. the third custom function of the template
# the four ways a user calls the template (getDefaultVarsDiff and getDefaultVarsRatio are both getDefaultVars)
TEMPLATES = ("getDefaultVarsDiff", "getDefaultVarsRatio", "getDefaultVarsNormAll", "getDefaultVarsRefWeights")
FUN_CUSTOM2_SIGNATURE = [("getDefaultVarsDiff", "funCustom2", "funcCustomForm2"),
                         ("getDefaultVarsRatio", "funCustom2", "funcCustomForm2")]


def _template(name):
    """The template's aliases (always first) and its parameters (the list whose entries have a name and a value)."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate as template
    out = getattr(template, name)(variables=["A", "B", "C", "D"])
    parameters = [x for x in out if isinstance(x, list) and x and all(isinstance(e, dict) and "name" in e for e in x)
                  and any("value" in e for e in x)]
    assert len(parameters) == 1, (name, len(parameters))
    return out, out[0], parameters[0]


def _function_names_without_a_text_box():
    """Every alias of a template whose function is given as one plain name (not as JavaScript text) must name a
    parameter of the same template: the parameter's text box holds the function. Returns the ones that do not."""
    wrong = []
    for name in TEMPLATES:
        _, aliases, parameters = _template(name)
        boxes = {p["name"] for p in parameters}
        for alias in aliases:
            func = alias.get("func") if isinstance(alias, dict) else None
            if isinstance(func, str) and re.fullmatch(r"\w+", func) and func not in boxes:
                wrong.append((name, alias["name"], func))
    return wrong


@UNIT
@pytest.mark.feature("DSL.custom_js_func")
def test_control_the_templates_have_three_custom_functions_each_and_the_first_two_name_their_text_boxes():
    for name in TEMPLATES:
        _, aliases, parameters = _template(name)
        funcs = {a["name"]: a["func"] for a in aliases if isinstance(a, dict) and str(a.get("name", "")).startswith("funCustom")}
        assert sorted(funcs) == ["funCustom0", "funCustom1", "funCustom2"], (name, funcs)
        assert funcs["funCustom0"] == "funCustomForm0" and funcs["funCustom1"] == "funCustomForm1", (name, funcs)
        assert {"funCustomForm0", "funCustomForm1", "funCustomForm2"} <= {p["name"] for p in parameters}, name


@UNIT
@pytest.mark.feature("DSL.custom_js_func")
@pytest.mark.xfail(**STRICT, reason="funCustom2: the third custom function of the template names a text box that does not exist")
def test_every_custom_function_of_a_template_names_one_of_its_text_boxes():
    wrong = _function_names_without_a_text_box()
    known_defect(wrong == FUN_CUSTOM2_SIGNATURE, "funCustom2: getDefaultVars (Diff and Ratio) gives funCustom2 the "
                                                 "function name 'funcCustomForm2'; the text box is 'funCustomForm2'")
    assert wrong == [], wrong


def _template_page(tmp_path):
    """A small page of the template getDefaultVarsDiff, built by the public builder as a user builds it."""
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    from RootInteractive.InteractiveDrawing.bokeh.bokehInteractiveTemplate import getDefaultVarsDiff
    rng = np.random.default_rng(20261116)
    df = pd.DataFrame(rng.random(size=(300, 4)), columns=list("ABCD"))
    aliases, variables, parameters, widgets, widget_layout, histograms, figures, layout = \
        getDefaultVarsDiff(variables=["A", "B", "C", "D"])
    before = sa.show
    sa.show = lambda *a, **kw: None
    try:
        obj = sa.bokehDrawSA.fromArray(df, None, figures, widgets, layout=layout, parameterArray=parameters,
                                       widgetLayout=widget_layout, histogramArray=histograms, aliasArray=aliases)
    finally:
        sa.show = before
    doc = Document()
    doc.add_root(obj.pAll)
    out = Path(tmp_path) / "template_page.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out, len(df)


# Type a new function into a text box (the model's value; its callbacks are awaited), then read the three columns.
TYPE_AND_READ_JS = r"""async (a) => {
  const all = window.__riH.models();
  const table = all.filter(m => m.type === 'CDSAlias' && m.name === 'cdsFull');
  if (table.length !== 1) throw Error('cdsFull: ' + table.length);
  const read = () => { const out = {};
    for (const k of a.read) {
      try { const c = table[0].get_column(k); out[k] = c == null ? null : Array.from(c); }
      catch (e) { out[k] = 'ERROR: ' + (e && e.message ? e.message : String(e)); } }
    return out; };
  const before = read();
  for (const [name, text] of a.type) {
    const boxes = all.filter(w => w.name === name && 'value' in w && w.type !== 'CustomJSNAryFunction');
    if (boxes.length !== 1) throw Error('text box ' + name + ': ' + boxes.length);
    const w = boxes[0], cbs = (w.js_property_callbacks || {})['change:value'] || [];
    w.value = text;
    for (const cb of cbs) await cb.execute(w);
  }
  await new Promise(r => setTimeout(r, 50));
  return {before, after: read()};
}"""


def _type_functions(node_session, tmp_path, typed, read=("funCustom0", "funCustom1", "funCustom2")):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    path, n = _template_page(tmp_path)
    _clean_session(node_session)
    _clean_reply(node_session.request("new_doc", path=str(path)), "open the template page")
    reply = node_session.request("eval", code=TYPE_AND_READ_JS, arg={"type": [list(t) for t in typed], "read": list(read)})
    return n, reply


def _constant(column, value, n):
    return isinstance(column, list) and len(column) == n and all(v == value for v in column)


@NODE
@pytest.mark.feature("DSL.custom_js_func")
def test_control_the_first_two_custom_functions_follow_their_text_boxes(node_session, tmp_path):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    n, reply = _type_functions(node_session, tmp_path, [("funCustomForm0", "return 7"), ("funCustomForm1", "return 9")],
                               read=("funCustom0", "funCustom1"))
    _clean_reply(reply, "type into the first two text boxes")
    got = decode(reply["result"])
    assert _constant(got["before"]["funCustom0"], 1, n) and _constant(got["before"]["funCustom1"], 1, n), got["before"]
    assert _constant(got["after"]["funCustom0"], 7, n) and _constant(got["after"]["funCustom1"], 9, n), got["after"]


@NODE
@pytest.mark.feature("DSL.custom_js_func")
@pytest.mark.xfail(**STRICT, reason="funCustom2: the third custom function of the template does not follow its text box")
def test_the_third_custom_function_follows_its_text_box(node_session, tmp_path):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    n, reply = _type_functions(node_session, tmp_path, [("funCustomForm2", "return 7")])
    assert reply.get("ok") is True, reply
    got = decode(reply["result"])
    errors = reply.get("console_errors") or []
    known_defect(_constant(got["before"]["funCustom2"], 0, n) and _constant(got["after"]["funCustom2"], 0, n)
                 and bool(errors) and set(errors) == {"ReferenceError: funcCustomForm2 is not defined"}
                 and not reply.get("page_errors"),
                 "funCustom2: the column is 0 in every row before and after typing, and the console says "
                 "'ReferenceError: funcCustomForm2 is not defined'")
    _clean_reply(reply, "type into the third text box")
    assert _constant(got["before"]["funCustom2"], 1, n), got["before"]["funCustom2"][:5]
    assert _constant(got["after"]["funCustom2"], 7, n), got["after"]["funCustom2"][:5]
    assert _constant(got["after"]["funCustom0"], 1, n) and _constant(got["after"]["funCustom1"], 1, n)


# ============================================================================== 2. debug printing in the models
# The two WebAssembly models do not compile today (register entry W-D1/W-D2, parked); they are not changed here.
PARKED_MODELS = ("CustomWasmFunc.ts", "CustomWasmModule.ts")
DEBUG_PRINT_SIGNATURE = {"CDSAlias.ts": 2, "CDSCompress.ts": 1, "DownsamplerCDS.ts": 1, "HistoNdCDS.ts": 3,
                         "MultiSelectFilter.ts": 1, "OrtFunction.ts": 4, "RangeFilter.ts": 1}


def _without_comments(text: str) -> str:
    """TypeScript text without /* ... */ blocks and without // comments (a // inside a string, as in a URL, is kept)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    out = []
    for line in text.split("\n"):
        quote, cut = None, len(line)
        for i, ch in enumerate(line):
            if quote:
                if ch == quote and line[i - 1] != "\\":
                    quote = None
            elif ch in "\"'`":
                quote = ch
            elif ch == "/" and line[i:i + 2] == "//":
                cut = i
                break
        out.append(line[:cut])
    return "\n".join(out)


def _debug_prints(folder: Path = BOKEH_DIR) -> dict:
    """File name -> number of console.log calls outside comments, for every TypeScript model of the folder."""
    found = {}
    for path in sorted(folder.glob("*.ts")):
        n = len(re.findall(r"\bconsole\s*\.\s*log\s*\(", _without_comments(path.read_text(encoding="utf-8"))))
        if n and path.name not in PARKED_MODELS:
            found[path.name] = n
    return found


@UNIT
def test_control_the_search_for_debug_printing_finds_a_call_and_skips_comments(tmp_path):
    (tmp_path / "A.ts").write_text('const u = "https://x/y.js"  // console.log(1)\nconsole.log(u)\n/* console.log(2)\n'
                                   'console.log(3) */\n  console . log (4)\n', encoding="utf-8")
    (tmp_path / "B.ts").write_text("//console.log(5)\nconst consoleXlog = 1\n", encoding="utf-8")
    (tmp_path / "CustomWasmFunc.ts").write_text("console.log(6)\n", encoding="utf-8")
    assert _debug_prints(tmp_path) == {"A.ts": 2}
    assert len(list(BOKEH_DIR.glob("*.ts"))) > 20, "the folder of the TypeScript models was not found"


@UNIT
@pytest.mark.xfail(**STRICT, reason="R-2: browser models print debug values with console.log")
def test_no_browser_model_prints_debug_values():
    found = _debug_prints()
    known_defect(found == DEBUG_PRINT_SIGNATURE, f"R-2: console.log calls in the models: {found}")
    assert found == {}, found


# What one page prints while it loads and while its range is moved twice. The page is a 3D graph with a range filter;
# its models RangeFilter, DownsamplerCDS and CDSCompress all printed: the name of every column when it is unpacked
# (x, y, z, c, gate), the name of the scatter source at every update ("default cds", three times) and the range
# property at every move (twice; the text of that object depends on the browser).
PAGE_PRINT_SIGNATURE = sorted(["x", "y", "z", "c", "gate"] + 3 * ["default cds"])


def _page_prints(headless_session, tmp_path):
    from RootInteractive.InteractiveDrawing.bokeh.headless import test_headless_visualization_d3 as d3
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine import vendor
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import HELPER_JS
    df = d3._gated()
    html = d3._make_filtered_page(df, tmp_path, html=True)
    routes = {glob.replace("**/", "/"): path for glob, path in vendor.routes().items()}
    printed, errors, blocked = [], [], []
    ctx = headless_session.browser.new_context(viewport={"width": 900, "height": 750})
    try:
        page = ctx.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: printed.append((m.type, m.text)))          # before the page opens

        def route(request):                      # only the pinned local copies of vis.js and pako; no internet
            url = request.request.url
            if url.startswith("file:"):
                return request.continue_()
            for suffix, f in routes.items():
                if url.split("?")[0].endswith(suffix):
                    return request.fulfill(path=str(f), content_type="text/javascript")
            blocked.append(url)
            return request.abort()
        page.route("**/*", route)
        page.add_init_script(HELPER_JS)
        page.goto(html.resolve().as_uri())
        page.wait_for_function(d3.PUBLIC_VIEW_READY_JS, timeout=20000)
        page.evaluate(d3.INSTALL_SETDATA_SPY_JS)
        for step, rng in enumerate((d3.PART, (-1.5, 2.))):
            n = int(df.gate.between(*rng).sum())
            page.evaluate("(r) => { window.__riH.slider('gate').value = r; }", list(rng))
            page.wait_for_function(
                "(a) => { const c=window.__riD3.setDataCalls; return c.length>a.step && c[c.length-1].length===a.n; }",
                arg={"step": step, "n": n}, timeout=20000)
        assert not blocked and not errors and not [t for k, t in printed if k == "error"], (blocked, errors, printed)
        # Bokeh's own messages start with "[bokeh]"; every other message of the kind "log" is printed by a model
        return [text for kind, text in printed if kind == "log" and not text.startswith("[bokeh]")]
    finally:
        ctx.close()


@BROWSER
@pytest.mark.xfail(**STRICT, reason="R-2: browser models print debug values with console.log")
def test_a_page_prints_nothing_while_it_loads_and_while_its_range_is_moved(headless_session, tmp_path):
    printed = _page_prints(headless_session, tmp_path)
    known = sorted(t for t in printed if t in PAGE_PRINT_SIGNATURE)
    known_defect(known == PAGE_PRINT_SIGNATURE and len(printed) == len(known) + 2,
                 f"R-2: the page printed {len(printed)} lines: {printed}")
    assert printed == [], printed


@UNIT
def test_the_onnx_model_compiles_on_its_own():
    """OrtFunction.ts lost four printing lines and is in no page of the test suite, so no other test compiles it.
    Compiled here as a page build does it (the helper of test_wasm_integration.py, a fresh process)."""
    from RootInteractive.InteractiveDrawing.bokeh.test_wasm_integration import _compile
    result = _compile("OrtFunction")
    assert result["compiled"] and result["errors"] == [], result["text"][-2000:]


# ============================================================ 3. the plots named "Median" of the weights dashboard
# The dashboard is a function of test_bokehClientHistogram.py, a module that builds its table when it is imported.
# So it runs in a child process with a small table; the builder's fromArray is replaced only to keep its arguments,
# and (for the page test) to keep the page it builds.
WEIGHTS_DASHBOARD_JS = r"""
import json, os, sys
os.environ["RI_DASHBOARD_ROWS"], os.environ["RI_DASHBOARD_SEED"] = "400", "4711"
import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
real, got = sa.bokehDrawSA.fromArray, {}
def keep(df, query, figureArray, widgetsDescription, **options):
    got["figures"], got["histograms"] = figureArray, options.get("histogramArray")
    if len(sys.argv) > 1:
        sa.show = lambda *a, **k: None
        got["page"] = real(df, query, figureArray, widgetsDescription, **options)
        return got["page"]
sa.bokehDrawSA.fromArray = staticmethod(keep)
import RootInteractive.InteractiveDrawing.bokeh.test_bokehClientHistogram as example
example.output_file = lambda *a, **k: None
example.test_interactiveTemplateWeights()
figures = [f for f in got["figures"] if isinstance(f, list) and isinstance(f[-1], dict)]
out = {"median_plots": [[f[-1]["name"], f[0], f[1], f[-1]["source"]] for f in figures
                        if str(f[-1].get("name", "")).endswith("_Median")],
       "mean_plots": [[f[-1]["name"], f[0], f[1], f[-1]["source"]] for f in figures
                      if str(f[-1].get("name", "")).endswith("_Mean")],
       "quantiles": {h["name"]: h["quantiles"] for h in got["histograms"] if "quantiles" in h}}
if len(sys.argv) > 1:
    from bokeh.document import Document
    doc = Document(); doc.add_root(got["page"].pAll)
    open(sys.argv[1], "w").write(json.dumps(doc.to_json(deferred=False)))
print("RESULT_JSON " + json.dumps(out))
"""
MEDIAN_PLOTS = ("histoNDMultiX_1_Median", "histoNDMultiX_0_Median", "histoNDMultiY_1_Median", "histoNDMultiY_0_Median")


def _weights_dashboard(document: Path = None) -> dict:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, "-c", WEIGHTS_DASHBOARD_JS] + ([str(document)] if document else [])
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, cwd=str(document.parent if document else REPO_ROOT))
    lines = [x for x in run.stdout.splitlines() if x.startswith("RESULT_JSON ")]
    assert run.returncode == 0 and len(lines) == 1, (run.returncode, run.stdout[-1500:], run.stderr[-3000:])
    return json.loads(lines[0][len("RESULT_JSON "):])


def _statistic(plot):
    """The plotted statistic of a profile plot: the variable that is not the bin centre. With the wanted name of the
    median column: quantile_<k>, k being the place of 0.5 in the quantiles of the plot's histogram."""
    name, x, y, source = plot
    stat = [v for v in x + y if not str(v).startswith("bin_center")]
    assert len(stat) == 1, plot
    return stat[0], source.rsplit("_", 1)[0]


@UNIT
@pytest.mark.feature("HIST.profile")
@pytest.mark.xfail(**STRICT, reason="DEC-23: the plots named Median of the weights dashboard draw the standard deviation")
def test_the_plots_named_median_of_the_weights_dashboard_draw_the_median():
    d = _weights_dashboard()
    assert sorted(p[0] for p in d["median_plots"]) == sorted(MEDIAN_PLOTS), d["median_plots"]
    assert [_statistic(p)[0] for p in d["mean_plots"]] == ["mean"] * 4, d["mean_plots"]          # the control
    drawn = {p[0]: _statistic(p)[0] for p in d["median_plots"]}
    known_defect(set(drawn.values()) == {"std"}, f"DEC-23: the four plots named Median draw 'std': {drawn}")
    for plot in d["median_plots"]:
        stat, histogram = _statistic(plot)
        assert stat == f"quantile_{d['quantiles'][histogram].index(0.5)}", (plot, d["quantiles"][histogram])


# the columns that the lines and markers of a page read from a profile table
PROFILE_FIELDS_JS = r"""(a) => {
  const out = {};
  for (const m of window.__riH.models()) {
    if (m.type !== 'GlyphRenderer') continue;
    let src = m.data_source, name = src && src.name;
    while (src && !a.tables.includes(name) && src.source) { src = src.source; name = src.name; }
    if (!a.tables.includes(name)) continue;
    for (const k of ['x', 'y']) {
      const f = m.glyph[k] && m.glyph[k].field;
      if (f) (out[name] = out[name] || []).push(f);
    }
  }
  const columns = {};
  for (const t of a.tables) {
    // two models carry the name: the stack of the profiles (one per selected variable) and the source of the plot
    const p = window.__riH.models().filter(m => m.name === t && m.type === 'CDSStack');
    if (p.length !== 1) throw Error('table ' + t + ': ' + p.length);
    columns[t] = {};
    for (const c of ['std', 'quantile_1']) { const v = p[0].get_column(c); columns[t][c] = v == null ? null : Array.from(v); }
  }
  return {fields: out, columns};
}"""


@NODE
@pytest.mark.feature("HIST.profile")
@pytest.mark.xfail(**STRICT, reason="DEC-23: the plots named Median of the weights dashboard draw the standard deviation")
def test_the_weights_dashboard_page_reads_the_median_column_for_its_median_plots(node_session, tmp_path):
    """The real page of the dashboard (400 rows): it loads without an error, and from each of the four profile tables
    behind the Mean and Median plots the page's lines read the mean and the median column, not the std."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    tables = ["histoNDMultiX_1", "histoNDMultiX_0", "histoNDMultiY_1", "histoNDMultiY_0"]
    document = Path(tmp_path) / "weights_dashboard.json"
    document.parent.mkdir(parents=True, exist_ok=True)
    _weights_dashboard(document)
    _clean_session(node_session)
    opened = node_session.request("new_doc", path=str(document))
    reply = node_session.request("eval", code=PROFILE_FIELDS_JS, arg={"tables": tables})
    for r, op in ((opened, "open the weights dashboard"), (reply, "read the weights dashboard")):
        # the page's only console message is the known one of the stacked weights figures (register entry O-1,
        # in the page builder): the same list of allowed messages as the scenario runner uses
        unexpected = [e for e in r.get("console_errors") or [] if not any(re.search(p, e) for p, _ in CONSOLE_ALLOWED)]
        assert r.get("ok") is True and not r.get("page_errors") and not unexpected, (op, r.get("page_errors"), unexpected[:3])
    got = decode(reply["result"])
    read = {t: set(got["fields"].get(t, [])) - {"bin_center_0", "bin_center_1"} for t in tables}
    known_defect(all(r == {"mean", "std"} for r in read.values()), f"DEC-23: the page reads {read}")
    for t in tables:
        assert read[t] == {"mean", "quantile_1"}, (t, read[t])
        std, median = np.asarray(got["columns"][t]["std"], float), np.asarray(got["columns"][t]["quantile_1"], float)
        assert np.isfinite(median).sum() > 10 and not np.allclose(std, median, equal_nan=True), t
