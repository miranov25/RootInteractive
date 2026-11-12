"""Production aliases in Node — PHASE_4_5 stage A1 (basic). Oracle and invariant tests only.

An ALIAS is a column the browser computes: from an expression (`("A", "x*2+y")`) or from a JavaScript function
(`{"name": ..., "variables": [...], "func" | "v_func" | "transform": ...}`). RootInteractive compiles it in Python
(compileVarName.py, bokehTools.py) and computes it in the browser (CDSAlias.ts, CustomJSNAryFunction.ts).

Every page here is built through the USER path — `bokehDrawSA.fromArray` with aliasArray, parameterArray, sourceArray,
jsFunctionArray, as a user writes it; only `show` is suppressed (as engine/page.py::build_document does). The page
document is loaded in the Node harness through Bokeh's deserializer and the PRODUCTION model is read: the table's
CDSAlias (`cdsFull`, or `<table>_full`) `get_column`. A widget step sets the widget's value and waits for its own
callbacks to finish (CustomJS runs asynchronously), so a stale read is never a timing accident.
Two references:
  * the ORACLE — what is correct: numpy on the same data, Python semantics (the alias language is Python syntax);
  * the PREDICTION — what production does today, for each known defect, derived from the code (see each finding).
A known-defect check passes only if production equals the prediction exactly AND the prediction differs from the
oracle; then it raises `KnownDefect`, the only failure the strict expected-failure tests accept. Any other failure —
a harness error, a missing column, a result matching neither — stays red. RootInteractive's code is NOT changed.

Findings (provisional ids):
  A-D1  an expression that reads ANOTHER ALIAS gets the whole column, not the row's value (compileVarName.py,
        visit_Attribute: an alias is returned without the row index). "A+1" gives the text of the whole column plus
        "1" in every row; "A*2" gives NaN in every row. Also a figure expression over an alias.
  A-D2  an alias whose expression is not a computation — a plain column ("x"), a constant ("5") — stops the
        processing of the alias list (bokehTools.py: `break`): that alias and EVERY LATER alias are missing; the page
        build fails "Column ... not found". Placed last, it costs only itself.
  A-D3  a column-selector parameter (a parameter with options) inside an alias expression fails the page build:
        KeyError 'cdsFull' (bokehTools.py reads the table's alias model before it exists).
  A-D4  a gather alias (`ev.ex[evIdx]`) does not send the column it gathers: unless something else shows `ev.ex`, the
        column is absent in the browser and reading the alias throws.
  A-D5  a gather of an alias of ANOTHER table (`ev.ex2[evIdx]`, ex2 an alias of ev) is compiled as a column of the
        current table: the page build fails "Column ex2 not found in data source None" (PHASE_3_1 R2).
  A-D6  a gather alias is not recomputed when the gathered table changes: it keeps the old values.
  A-D7  an alias with automatic fields (function form without "variables") is not recomputed when an alias it reads
        is recomputed (CDSAlias.invalidate_column walks `fields` as a list; for this alias it is the text "auto").
  A-D8  Python operators that mean something else in the browser: `//` truncates toward zero and wraps at 32 bits,
        `%` keeps the sign of the dividend, `~` on a comparison gives -2/-1, `1 ** inf` is NaN. The same expression
        shown directly in a figure from table columns is computed on the SERVER with numpy and is correct, so one
        expression has two results.
  A-D9  on that server path `cosh` computes sinh (compileVarName.py, math_functions: "cosh": np.sinh).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.node_aliases]   # target (PHASE_4_5 A1)


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


STRICT = dict(strict=True, raises=KnownDefect)


# ===================================================================================== harness: build, load, steps
STEPS = """async (a) => {
  const doc = Bokeh.documents[Bokeh.documents.length - 1];
  const all = Array.from(doc._all_models.values());
  const isObj = e => e !== null && typeof e === "object";
  const msg = e => (isObj(e) && "message" in e) ? String(e.message) : String(e);
  const kind = e => (isObj(e) && "name" in e) ? String(e.name) : typeof e;
  const one = (pred, what) => { const m = all.filter(pred); if (m.length !== 1) throw new Error(what + ": " + m.length); return m[0]; };
  const alias = n => one(m => m.name === n && m.constructor.__name__ === "CDSAlias", "CDSAlias models named " + n);
  const table = n => one(m => m.name === n && m.constructor.__name__ === "ColumnDataSource", "tables named " + n);
  const out = [];
  for (const s of a.steps) {
    try {
      if (s.op === "read") {
        const m = alias(s.table), r = {columns: {}, kinds: {}, throws: {}, throw_kinds: {}};
        for (const k of s.keys) {
          try { const c = m.get_column(k);
                if (c == null) { r.columns[k] = null; r.kinds[k] = "null"; continue; }
                r.columns[k] = Array.from(c);
                r.kinds[k] = Array.from(new Set(Array.from(c, v => typeof v))).sort().join(","); }
          catch (e) { r.throws[k] = msg(e); r.throw_kinds[k] = kind(e); }
        }
        out.push(r);
      } else if (s.op === "cached") {
        out.push({cached: Array.from(alias(s.table).cached_columns).sort()});
      } else if (s.op === "keys") {
        out.push({keys: Object.keys(table(s.table).data).sort()});
      } else if (s.op === "set") {
        const w = one(x => "value" in x && x.js_property_callbacks && (x.name === s.widget || (x.name == null && x.title === s.widget)),
                      "widgets named " + s.widget);
        const cbs = (w.js_property_callbacks || {})["change:value"] || [];
        const pending = [];
        for (const cb of cbs) {
          const orig = cb.execute;
          cb.execute = s.mute ? function () { const p = Promise.resolve(); pending.push(p); return p; }
                              : function (...x) { const p = orig.apply(this, x); pending.push(p); return p; };
        }
        w.value = s.value;
        const settled = await Promise.allSettled(pending);
        await new Promise(r => setTimeout(r, 0));
        for (const cb of cbs) delete cb.execute;
        out.push({callbacks: settled.length, errors: settled.filter(x => x.status === "rejected").map(x => msg(x.reason))});
      } else if (s.op === "mutate") {
        const t = table(s.table);
        t.data = Object.assign({}, t.data, s.data);
        out.push({mutated: s.table});
      } else {
        out.push({error: "unknown step " + s.op});
      }
    } catch (e) { out.push({error: msg(e)}); }
  }
  return out;
}"""


def _build(tmp_path: Path, df: pd.DataFrame, figures: list, widgets=(), **options) -> Path:
    """The page document through the user path: bokehDrawSA.fromArray, `show` suppressed (engine/page.py)."""
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        obj = sa.bokehDrawSA.fromArray(df, None, figures, list(widgets), layout=[list(range(len(figures)))],
                                       widgetLayout=[list(range(len(widgets)))], **options)
    finally:
        sa.show = shown
    doc = Document()
    doc.add_root(obj.pAll)
    path = tmp_path / "aliases_document.json"
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return path


def _build_error(tmp_path, df, figures, widgets=(), **options):
    """None if the page builds, else (exception type, message, where it was raised: file:function)."""
    import traceback
    try:
        _build(tmp_path, df, figures, widgets, **options)
    except Exception as exc:                                                # noqa: BLE001 - the signature decides
        last = traceback.extract_tb(exc.__traceback__)[-1]
        return type(exc).__name__, str(exc.args[0]) if exc.args else "", f"{Path(last.filename).name}:{last.name}"
    return None


def _run(node_session, path: Path, steps: list) -> list:
    """The production model's replies, one per step. A harness problem fails the test (never a KnownDefect)."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = node_session.request("new_doc", path=str(path))
    if not r.get("ok"):
        pytest.fail(f"Node harness could not load the document: {r}")
    r = node_session.request("eval", code=STEPS, arg={"steps": steps})
    if not r.get("ok"):
        pytest.fail(f"Node harness eval failed: {r}")
    out = decode(r["result"])
    bad = [o for o in out if "error" in o]
    if bad:
        pytest.fail(f"harness step failed: {bad}")
    return out


def read(table, *keys):
    return {"op": "read", "table": table, "keys": list(keys)}


def set_(widget, value, mute=False):
    return {"op": "set", "widget": widget, "value": value, "mute": mute}


def cached(table):
    return {"op": "cached", "table": table}


def _num(col) -> np.ndarray:
    """A production column as float64: a JS boolean as 1/0, null/undefined as NaN (the kinds say which it was)."""
    return np.array([np.nan if v is None else float(v) for v in col], dtype=float)


def _same(got, want, rtol=0.0) -> bool:
    """Equal row by row: NaN where NaN, the same infinities, finite values equal (or within rtol)."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if got.shape != want.shape:
        return False
    if not np.array_equal(np.isnan(got), np.isnan(want)):
        return False
    fin = np.isfinite(want)
    if not np.array_equal(got[~fin & ~np.isnan(want)], want[~fin & ~np.isnan(want)]):
        return False
    return bool(np.all(got[fin] == want[fin])) if rtol == 0 else bool(np.allclose(got[fin], want[fin], rtol=rtol, atol=0))


def _complete(r, keys, kinds=("number",)):
    """Every requested column read without an exception, present, numeric (or as allowed)."""
    assert not r["throws"], r["throws"]
    assert set(keys) <= set(r["columns"]), sorted(r["columns"])
    for k in keys:
        assert r["columns"][k] is not None, k
        assert set(r["kinds"][k].split(",")) <= set(kinds), (k, r["kinds"][k])


def _js_number(v: float) -> str:
    """JavaScript's String(number) for the values used here (no exponent forms)."""
    if np.isnan(v):
        return "NaN"
    if np.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if float(v).is_integer():
        return str(int(v))
    return repr(float(v))


def _js_int32(v: float) -> float:
    """JavaScript's ToInt32 (what `|0` does)."""
    if not np.isfinite(v):
        return 0.0
    n = int(np.trunc(v)) % 2 ** 32
    return float(n - 2 ** 32 if n >= 2 ** 31 else n)


# ===================================================================================== data
X = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 1.0, 2.0, 4.5, np.nan, np.inf, -np.inf])
Y = np.array([1.0, 2.0, np.nan, 4.0, -1.0, 0.5, np.inf, -np.inf, 3.0, 2.0, 0.0])
DF = pd.DataFrame({"x": X, "y": Y,
                   "evIdx": np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int32)})
EV = np.array([10.0, 20.0, 30.0])
EV_SOURCE = [{"name": "ev", "data": {"ex": EV}}]
PA = [{"name": "pa", "value": 2.0, "range": [0, 5]}]
PA_SLIDER = [["slider", ["pa"]]]


def _shown(*names):
    """One figure that shows the given columns of the main table (an alias is computed only if something uses it)."""
    return [[["x"], list(names)]]


def _expect(node_session, tmp_path, aliases, oracle: dict, rtol=0.0, kinds=("number", "boolean"), **options):
    path = _build(tmp_path, DF, _shown(*oracle), aliasArray=aliases, **options)
    r = _run(node_session, path, [read("cdsFull", *oracle)])[0]
    _complete(r, list(oracle), kinds)
    for k, want in oracle.items():
        assert _same(_num(r["columns"][k]), want, rtol), (k, r["columns"][k], list(want))
    return r


# ===================================================================================== healthy: expressions
with np.errstate(all="ignore"):
    EXPRESSIONS = {
        "x*2+y": X * 2 + Y,
        "x-y": X - Y,
        "x/y": X / Y,
        "-x": -X,
        "x**2": X ** 2,
        "abs(x)**0.5": np.abs(X) ** 0.5,
        "x>1": (X > 1).astype(float),
        "x>=y": (X >= Y).astype(float),
        "x==y": (X == Y).astype(float),
        "x!=y": (X != Y).astype(float),
        "0<=x<3": ((0 <= X) & (X < 3)).astype(float),
        "x>0 and y>0": ((X > 0) & (Y > 0)).astype(float),
        "x>0 or y>0": ((X > 0) | (Y > 0)).astype(float),
        "not x>0": (~(X > 0)).astype(float),
        "x if y>0 else -x": np.where(Y > 0, X, -X),
        "x*(y>0)": X * (Y > 0),
    }


@pytest.mark.feature("DSL.arithmetic_expr")
@pytest.mark.feature("ALIAS.cdsalias")
@pytest.mark.parametrize("expr", list(EXPRESSIONS))
def test_expression_alias_equals_numpy(node_session, tmp_path, expr):
    """Arithmetic, comparisons, chained comparisons, and/or/not, if-else: equal numpy row by row, NaN and +-inf included."""
    _expect(node_session, tmp_path, [("A", expr)], {"A": EXPRESSIONS[expr]})


with np.errstate(all="ignore"):
    MATH = {f: (getattr(np, n)(X) if n != "arctan2" else np.arctan2(Y, X))
            for f, n in [("sin", "sin"), ("cos", "cos"), ("arcsin", "arcsin"), ("arccos", "arccos"),
                         ("arctan", "arctan"), ("exp", "exp"), ("log", "log"), ("log2", "log2"), ("log10", "log10"),
                         ("sqrt", "sqrt"), ("abs", "abs"), ("sinh", "sinh"), ("cosh", "cosh"), ("arctan2", "arctan2"),
                         ("arccosh", "arccosh"), ("arcsinh", "arcsinh"), ("arctanh", "arctanh")]}
MATH_RTOL = 1e-14      # measured (Linux): 9 of the 17 functions differ from numpy, by at most 2 units in the last
                       # place (about 4e-16); 1e-14 allows other platforms' math libraries; a wrong function differs far more


@pytest.mark.feature("DSL.math_functions")
@pytest.mark.parametrize("func", list(MATH))
def test_math_function_alias_equals_numpy(node_session, tmp_path, func):
    """Every function of the alias language (but parseInt) equals numpy, outside its domain too (NaN, +-inf)."""
    expr = "arctan2(y, x)" if func == "arctan2" else f"{func}(x)"
    _expect(node_session, tmp_path, [("A", expr)], {"A": MATH[func]}, rtol=MATH_RTOL, kinds=("number",))


@pytest.mark.feature("DSL.arithmetic_expr")
def test_integer_and_boolean_columns(node_session, tmp_path):
    """Integers are exact up to 2**53; bit operations on small integers; a boolean column arrives as 1/0."""
    df = pd.DataFrame({"x": np.arange(4, dtype=float), "big": np.array([2 ** 40, -2 ** 40, 2 ** 52, 3], dtype=np.int64),
                       "b": np.array([True, False, True, False]), "k": np.array([5, 6, 7, 12], dtype=np.int32)})
    k = df.k.to_numpy()
    oracle = {"big1": (df.big + 1).to_numpy(float), "bx": (df.x * df.b).to_numpy(float),
              "kbits": ((k << 2) | 1 ^ (k & 6) >> 1).astype(float), "nb": (~(df.x > 1)).to_numpy(float)}
    path = _build(tmp_path, df, [[["x"], list(oracle)]],
                  aliasArray=[("big1", "big+1"), ("bx", "x*b"), ("kbits", "(k << 2) | 1 ^ (k & 6) >> 1"), ("nb", "not x>1")])
    r = _run(node_session, path, [read("cdsFull", *oracle)])[0]
    _complete(r, list(oracle), ("number", "boolean"))
    for k, want in oracle.items():
        assert _same(_num(r["columns"][k]), want), (k, r["columns"][k], list(want))


# ===================================================================================== healthy: function forms
V_FUNC = """if($output == null || $output.length !== x.length){ $output = new Float64Array(x.length) }
for(let i=0; i<x.length; i++){ $output[i] = x[i]*y[i] - 1 }
return $output"""


@pytest.mark.feature("DSL.custom_js_func")
@pytest.mark.feature("ALIAS.cdsalias")
def test_function_forms_agree_with_the_expression_and_numpy(node_session, tmp_path):
    """Invariant: the scalar function, the vector function, a named jsFunctionArray transform and the expression of
    the same formula give the same column, equal to numpy."""
    with np.errstate(all="ignore"):
        want = X * Y - 1
    aliases = [("E", "x*y-1"),
               {"name": "F", "variables": ["x", "y"], "func": "return x*y-1"},
               {"name": "V", "variables": ["x", "y"], "v_func": V_FUNC},
               {"name": "T", "variables": ["x", "y"], "transform": "mul1"}]
    path = _build(tmp_path, DF, _shown("E", "F", "V", "T"), aliasArray=aliases,
                  jsFunctionArray=[{"name": "mul1", "parameters": [], "fields": ["a", "b"], "func": "return a*b-1"}])
    r = _run(node_session, path, [read("cdsFull", "E", "F", "V", "T")])[0]
    _complete(r, ["E", "F", "V", "T"])
    for k in "EFVT":
        assert _same(_num(r["columns"][k]), want), (k, r["columns"][k])


@pytest.mark.feature("ALIAS.context")
def test_alias_of_a_secondary_table(node_session, tmp_path):
    """An alias declared with a table (context): computed on that table's rows."""
    path = _build(tmp_path, DF, [[["x"], ["y"]], [["ev.ex"], ["ev.ex2"]]], sourceArray=EV_SOURCE,
                  aliasArray=[("ex2", "ex*2+1", "ev")])
    r = _run(node_session, path, [read("ev_full", "ex", "ex2")])[0]
    _complete(r, ["ex", "ex2"])
    assert _same(_num(r["columns"]["ex2"]), EV * 2 + 1), r["columns"]


@pytest.mark.feature("DSL.gather_operation")
def test_gather_of_a_column_equals_numpy_indexing(node_session, tmp_path):
    """`ev.ex[evIdx]`: row i reads ev.ex at row evIdx[i]; in an expression too. (The gathered column is shown in a
    second figure, so it is sent — see A-D4.)"""
    path = _build(tmp_path, DF, [[["x"], ["g", "gx"]], [["ev.ex"], ["ev.ex"]]], sourceArray=EV_SOURCE,
                  aliasArray=[("g", "ev.ex[evIdx]"), ("gx", "ev.ex[evIdx] + x")])
    r = _run(node_session, path, [read("cdsFull", "g", "gx")])[0]
    _complete(r, ["g", "gx"])
    assert _same(_num(r["columns"]["g"]), EV[DF.evIdx]), r["columns"]["g"]
    assert _same(_num(r["columns"]["gx"]), EV[DF.evIdx] + X), r["columns"]["gx"]


@pytest.mark.feature("ALIAS.chain")
@pytest.mark.feature("DSL.custom_js_func")
def test_chain_through_the_function_form(node_session, tmp_path):
    """Control for A-D1: an alias reading another alias through the FUNCTION form ("variables") works, so CDSAlias
    chains aliases correctly; A-D1 is the expression compiler's."""
    path = _build(tmp_path, DF, _shown("A", "C"), aliasArray=[("A", "x*2+y"),
                                                                {"name": "C", "variables": ["A"], "func": "return A+1"}])
    r = _run(node_session, path, [read("cdsFull", "A", "C")])[0]
    _complete(r, ["A", "C"])
    assert _same(_num(r["columns"]["C"]), X * 2 + Y + 1), r["columns"]["C"]


# ===================================================================================== healthy: parameters and change
def _param_check(node_session, tmp_path, mute=False):
    """Parameter in an expression, in a named transform, and a function given by a text parameter: after each widget
    change the alias equals numpy with the NEW value. `mute` (control): the widget's callbacks do nothing."""
    aliases = [("E", "x*pa"), {"name": "T", "variables": ["x", "y"], "transform": "saxpy"},
               {"name": "F", "variables": ["x"], "func": "fpar"}]
    params = PA + [{"name": "fpar", "value": "return 2*x"}]
    path = _build(tmp_path, DF, _shown("E", "T", "F"), PA_SLIDER + [["text", ["fpar"], {"name": "fpar"}]],
                  aliasArray=aliases, parameterArray=params,
                  jsFunctionArray=[{"name": "saxpy", "parameters": ["pa"], "fields": ["a", "b"], "func": "return pa*a+b"}])
    out = _run(node_session, path, [read("cdsFull", "E", "T", "F"), set_("pa", 3.0, mute), set_("fpar", "return 3*x", mute),
                                    read("cdsFull", "E", "T", "F")])
    before, s1, s2, after = out
    _complete(before, ["E", "T", "F"])
    _complete(after, ["E", "T", "F"])
    assert s1["callbacks"] >= 1 and s2["callbacks"] >= 1 and not s1["errors"] and not s2["errors"], (s1, s2)
    for r, pa, f in ((before, 2.0, 2.0), (after, 3.0, 3.0)):
        assert _same(_num(r["columns"]["E"]), X * pa), ("E", pa, r["columns"]["E"])
        assert _same(_num(r["columns"]["T"]), pa * X + Y), ("T", pa, r["columns"]["T"])
        assert _same(_num(r["columns"]["F"]), f * X), ("F", f, r["columns"]["F"])


@pytest.mark.feature("ALIAS.parameter")
def test_parameter_change_recomputes_the_alias(node_session, tmp_path):
    _param_check(node_session, tmp_path)


@pytest.mark.feature("ALIAS.parameter")
def test_control_a_muted_widget_leaves_stale_values_and_the_check_sees_it(node_session, tmp_path):
    """Checks can fail: with the widget's callbacks muted the aliases keep the old values, and the same check fails."""
    with pytest.raises(AssertionError, match="'E', 3.0"):
        _param_check(node_session, tmp_path, mute=True)


@pytest.mark.feature("ALIAS.parameter")
def test_column_selector_parameter_shown_directly(node_session, tmp_path):
    """A parameter with options used as a column: switching the option switches the column."""
    path = _build(tmp_path, DF, _shown("sel"), [["select", ["sel"]]],
                  parameterArray=[{"name": "sel", "value": "x", "options": ["x", "y"]}])
    before, s, after = _run(node_session, path, [read("cdsFull", "sel"), set_("sel", "y"), read("cdsFull", "sel")])
    assert s["callbacks"] >= 1 and not s["errors"], s
    _complete(before, ["sel"])
    _complete(after, ["sel"])
    assert _same(_num(before["columns"]["sel"]), X) and _same(_num(after["columns"]["sel"]), Y), (before, after)


GRAPH = [("A", "x*2"), ("E", "x*pa"), {"name": "C", "variables": ["E"], "func": "return E+1"},
         {"name": "K", "variables": ["A"], "func": "return A-1"}]


@pytest.mark.feature("ALIAS.change_propagation")
def test_a_parameter_change_invalidates_exactly_its_dependents(node_session, tmp_path):
    """Invariant: after `pa` changes, E (uses pa) and C (uses E) are recomputed; A and K (independent of pa) keep their
    cached values; every value equals numpy with the new pa."""
    path = _build(tmp_path, DF, _shown("A", "E", "C", "K"), PA_SLIDER, aliasArray=GRAPH, parameterArray=PA)
    before, s, cache, after = _run(node_session, path, [read("cdsFull", "A", "E", "C", "K"), set_("pa", 3.0),
                                                        cached("cdsFull"), read("cdsFull", "A", "E", "C", "K")])
    assert s["callbacks"] >= 1 and not s["errors"], s
    assert {"A", "K"} <= set(cache["cached"]) and not {"E", "C"} & set(cache["cached"]), cache
    _complete(after, ["A", "E", "C", "K"])
    for k, want in {"A": X * 2, "E": X * 3, "C": X * 3 + 1, "K": X * 2 - 1}.items():
        assert _same(_num(after["columns"][k]), want), (k, after["columns"][k])


@pytest.mark.feature("ALIAS.change_propagation")
def test_a_change_of_the_table_recomputes_every_alias(node_session, tmp_path):
    """New values in the table: every alias is recomputed from them."""
    path = _build(tmp_path, DF, _shown("A", "E", "C", "K"), PA_SLIDER, aliasArray=GRAPH, parameterArray=PA)
    x2 = np.arange(len(X), dtype=float) * 10
    out = _run(node_session, path, [read("cdsFull", "A", "E", "C", "K"),
                                    {"op": "mutate", "table": "cdsOrig", "data": {"x": x2.tolist()}},
                                    read("cdsFull", "A", "E", "C", "K")])
    _complete(out[2], ["A", "E", "C", "K"])
    for k, want in {"A": x2 * 2, "E": x2 * 2, "C": x2 * 2 + 1, "K": x2 * 2 - 1}.items():
        assert _same(_num(out[2]["columns"][k]), want), (k, out[2]["columns"][k])


# ===================================================================================== A-D1 alias of an alias
A_VALUES = X * 2 + Y


def _ad1_prediction(op: str) -> list:
    """A-D1: the generated code reads the whole column `A` (an array), not `A[$i]`: `A + 1` is JavaScript text
    concatenation (String(array) + "1"), `A * 2` is NaN (a multi-element array is not a number)."""
    whole = ",".join(_js_number(v) for v in A_VALUES)
    return [whole + "1"] * len(X) if op == "+" else [float("nan")] * len(X)


def _ad1_read(node_session, tmp_path):
    path = _build(tmp_path, DF, [[["x"], ["A", "B", "M", "A*2"]]], aliasArray=[("A", "x*2+y"), ("B", "A+1"), ("M", "A*2")])
    r = _run(node_session, path, [read("cdsFull", "A", "B", "M", "A*2")])[0]
    assert not r["throws"], r["throws"]
    _complete(r, ["A"])
    assert _same(_num(r["columns"]["A"]), A_VALUES), r["columns"]["A"]
    return r


@pytest.mark.feature("ALIAS.chain")
def test_signature_a_d1_alias_of_an_alias_reads_the_whole_column(node_session, tmp_path):
    """A-D1, exact signature: B = "A+1" is the text of the whole column A plus "1" in every row; M = "A*2" and the
    figure expression "A*2" are NaN in every row. The prediction differs from numpy."""
    r = _ad1_read(node_session, tmp_path)
    assert r["columns"]["B"] == _ad1_prediction("+") and r["kinds"]["B"] == "string", r["columns"]["B"]
    for k in ("M", "A*2"):
        assert _same(_num(r["columns"][k]), _ad1_prediction("*")) and r["kinds"][k] == "number", (k, r["columns"][k])
    assert not _same(_ad1_prediction("*"), A_VALUES * 2)


@pytest.mark.feature("ALIAS.chain")
@pytest.mark.xfail(**STRICT, reason="A-D1")
def test_strict_alias_of_an_alias_equals_numpy(node_session, tmp_path):
    r = _ad1_read(node_session, tmp_path)
    good = {"B": A_VALUES + 1, "M": A_VALUES * 2, "A*2": A_VALUES * 2}
    if all(r["kinds"][k] == "number" and _same(_num(r["columns"][k]), w) for k, w in good.items()):
        return
    known_defect(r["columns"]["B"] == _ad1_prediction("+")
                 and all(_same(_num(r["columns"][k]), _ad1_prediction("*")) for k in ("M", "A*2")), "A-D1")
    raise AssertionError(f"A-D1: neither numpy nor the A-D1 signature: {r['columns']}")


# ===================================================================================== A-D2 a non-computed alias stops the list
AD2_CASES = {
    # case: (aliases, shown columns, oracle, the exact build error: type, message, raised in)
    "renamed_column_itself": ([("xx", "x"), ("A", "x*2")], ["xx", "A"], {"xx": X, "A": X * 2},
                              ("KeyError", "Column xx not found in data source None", "compileVarName.py:visit_Attribute")),
    "alias_after_a_renamed_column": ([("xx", "x"), ("A", "x*2")], ["A"], {"A": X * 2},
                                     ("KeyError", "Column A not found in data source None", "compileVarName.py:visit_Attribute")),
    "alias_after_a_constant": ([("k", "5"), ("A", "x*2")], ["A"], {"A": X * 2},
                               ("KeyError", "Column A not found in data source None", "compileVarName.py:visit_Attribute")),
}


@pytest.mark.feature("ALIAS.cdsalias")
@pytest.mark.parametrize("case", list(AD2_CASES))
def test_signature_a_d2_a_non_computed_alias_stops_the_alias_list(tmp_path, case):
    aliases, shown, _, signature = AD2_CASES[case]
    assert _build_error(tmp_path, DF, _shown(*shown), aliasArray=aliases) == signature


@pytest.mark.feature("ALIAS.cdsalias")
@pytest.mark.xfail(**STRICT, reason="A-D2")
@pytest.mark.parametrize("case", list(AD2_CASES))
def test_strict_every_alias_of_the_list_is_usable(node_session, tmp_path, case):
    aliases, shown, oracle, signature = AD2_CASES[case]
    err = _build_error(tmp_path, DF, _shown(*shown), aliasArray=aliases)
    known_defect(err == signature, f"A-D2 ({case})")
    assert err is None, f"A-D2 ({case}): neither a page nor the A-D2 signature: {err}"
    r = _run(node_session, tmp_path / "aliases_document.json", [read("cdsFull", *shown)])[0]
    _complete(r, shown)
    for k in shown:
        assert _same(_num(r["columns"][k]), oracle[k]), (k, r["columns"][k])


@pytest.mark.feature("ALIAS.cdsalias")
def test_control_a_d2_the_renamed_column_last_costs_only_itself(node_session, tmp_path):
    """Control: the same aliases with the rename LAST — A is processed before the stop and is correct."""
    _expect(node_session, tmp_path, [("A", "x*2"), ("xx", "x")], {"A": X * 2})


# ===================================================================================== A-D3 column selector in an expression
AD3_SIGNATURE = ("KeyError", "cdsFull", "bokehTools.py:bokehDrawArray")
SEL = [{"name": "sel", "value": "x", "options": ["x", "y"]}]


@pytest.mark.feature("ALIAS.parameter")
def test_signature_a_d3_column_selector_in_an_expression_fails_the_build(tmp_path):
    assert _build_error(tmp_path, DF, _shown("S"), [["select", ["sel"]]], aliasArray=[("S", "sel*2")],
                        parameterArray=SEL) == AD3_SIGNATURE


@pytest.mark.feature("ALIAS.parameter")
@pytest.mark.xfail(**STRICT, reason="A-D3")
def test_strict_column_selector_in_an_expression(node_session, tmp_path):
    err = _build_error(tmp_path, DF, _shown("S"), [["select", ["sel"]]], aliasArray=[("S", "sel*2")], parameterArray=SEL)
    known_defect(err == AD3_SIGNATURE, "A-D3")
    assert err is None, f"A-D3: neither a page nor the A-D3 signature: {err}"
    before, s, after = _run(node_session, tmp_path / "aliases_document.json",
                            [read("cdsFull", "S"), set_("sel", "y"), read("cdsFull", "S")])
    assert s["callbacks"] >= 1 and not s["errors"], s
    assert _same(_num(before["columns"]["S"]), X * 2) and _same(_num(after["columns"]["S"]), Y * 2), (before, after)


# ===================================================================================== A-D4 the gathered column is not sent
def _ad4(node_session, tmp_path):
    path = _build(tmp_path, DF, _shown("g"), sourceArray=EV_SOURCE, aliasArray=[("g", "ev.ex[evIdx]")])
    return _run(node_session, path, [read("cdsFull", "g"), {"op": "keys", "table": "ev"}])


@pytest.mark.feature("DSL.gather_operation")
def test_signature_a_d4_a_gather_alias_alone_does_not_send_the_gathered_column(node_session, tmp_path):
    r, keys = _ad4(node_session, tmp_path)
    assert "ex" not in keys["keys"], keys
    assert r["throws"] == {"g": "Cannot read properties of null (reading '0')"} and r["throw_kinds"] == {"g": "TypeError"}, r


@pytest.mark.feature("DSL.gather_operation")
@pytest.mark.xfail(**STRICT, reason="A-D4")
def test_strict_a_gather_alias_alone_equals_numpy_indexing(node_session, tmp_path):
    r, keys = _ad4(node_session, tmp_path)
    if not r["throws"] and _same(_num(r["columns"]["g"]), EV[DF.evIdx]):
        return
    known_defect("ex" not in keys["keys"] and r["throws"] == {"g": "Cannot read properties of null (reading '0')"}, "A-D4")
    raise AssertionError(f"A-D4: neither numpy nor the A-D4 signature: {r}, {keys}")


# ===================================================================================== A-D5 gather of another table's alias
AD5_SIGNATURE = ("KeyError", "Column ex2 not found in data source None", "compileVarName.py:visit_Attribute")
AD5_PAGE = dict(figures=[[["x"], ["g2"]], [["ev.ex"], ["ev.ex2"]]],
                aliases=[("ex2", "ex*2", "ev"), ("g2", "ev.ex2[evIdx]")])


@pytest.mark.feature("DSL.gather_operation")
@pytest.mark.feature("ALIAS.chain")
def test_signature_a_d5_gather_of_an_alias_of_another_table_fails_the_build(tmp_path):
    assert _build_error(tmp_path, DF, AD5_PAGE["figures"], sourceArray=EV_SOURCE,
                        aliasArray=AD5_PAGE["aliases"]) == AD5_SIGNATURE


@pytest.mark.feature("DSL.gather_operation")
@pytest.mark.feature("ALIAS.chain")
@pytest.mark.xfail(**STRICT, reason="A-D5")
def test_strict_gather_of_an_alias_of_another_table(node_session, tmp_path):
    err = _build_error(tmp_path, DF, AD5_PAGE["figures"], sourceArray=EV_SOURCE, aliasArray=AD5_PAGE["aliases"])
    known_defect(err == AD5_SIGNATURE, "A-D5")
    assert err is None, f"A-D5: neither a page nor the A-D5 signature: {err}"
    r = _run(node_session, tmp_path / "aliases_document.json", [read("cdsFull", "g2")])[0]
    _complete(r, ["g2"])
    assert _same(_num(r["columns"]["g2"]), (EV * 2)[DF.evIdx]), r["columns"]["g2"]


# ===================================================================================== A-D6 gathered table changes
EV_NEW = [100.0, 200.0, 300.0]


def _ad6(node_session, tmp_path):
    path = _build(tmp_path, DF, [[["x"], ["g"]], [["ev.ex"], ["ev.ex"]]], sourceArray=EV_SOURCE,
                  aliasArray=[("g", "ev.ex[evIdx]")])
    before, _, ev, after = _run(node_session, path, [read("cdsFull", "g"), {"op": "mutate", "table": "ev", "data": {"ex": EV_NEW}},
                                                     read("ev_full", "ex"), read("cdsFull", "g")])
    _complete(before, ["g"])
    _complete(after, ["g"])
    assert _same(_num(ev["columns"]["ex"]), EV_NEW), ev            # the gathered table has the new values
    return before, after


@pytest.mark.feature("ALIAS.change_propagation")
@pytest.mark.feature("DSL.gather_operation")
def test_signature_a_d6_a_gather_alias_keeps_old_values_after_the_gathered_table_changes(node_session, tmp_path):
    before, after = _ad6(node_session, tmp_path)
    assert _same(_num(before["columns"]["g"]), EV[DF.evIdx]) and after["columns"]["g"] == before["columns"]["g"]
    assert not _same(EV[DF.evIdx], np.array(EV_NEW)[DF.evIdx])


@pytest.mark.feature("ALIAS.change_propagation")
@pytest.mark.feature("DSL.gather_operation")
@pytest.mark.xfail(**STRICT, reason="A-D6")
def test_strict_a_gather_alias_follows_the_gathered_table(node_session, tmp_path):
    before, after = _ad6(node_session, tmp_path)
    if _same(_num(after["columns"]["g"]), np.array(EV_NEW)[DF.evIdx]):
        return
    known_defect(_same(_num(after["columns"]["g"]), EV[DF.evIdx]), "A-D6")
    raise AssertionError(f"A-D6: neither numpy nor the A-D6 signature: {after}")


# ===================================================================================== A-D7 automatic fields
U_FUNC = """if($output == null || $output.length !== E.length){ $output = new Float64Array(E.length) }
for(let i=0; i<E.length; i++){ $output[i] = E[i] + 1 }
return $output"""


def _ad7(node_session, tmp_path):
    path = _build(tmp_path, DF, _shown("E", "U"), PA_SLIDER, parameterArray=PA,
                  aliasArray=[("E", "x*pa"), {"name": "U", "v_func": U_FUNC}])
    before, s, cache, after = _run(node_session, path, [read("cdsFull", "E", "U"), set_("pa", 3.0), cached("cdsFull"),
                                                        read("cdsFull", "E", "U")])
    assert s["callbacks"] >= 1 and not s["errors"], s
    _complete(before, ["E", "U"])
    _complete(after, ["E", "U"])
    assert _same(_num(before["columns"]["U"]), X * 2 + 1), before          # correct before the change
    assert _same(_num(after["columns"]["E"]), X * 3), after                # E itself follows the parameter
    return cache, after


@pytest.mark.feature("ALIAS.change_propagation")
def test_signature_a_d7_an_alias_with_automatic_fields_is_not_invalidated(node_session, tmp_path):
    cache, after = _ad7(node_session, tmp_path)
    assert "U" in cache["cached"] and "E" not in cache["cached"], cache
    assert _same(_num(after["columns"]["U"]), X * 2 + 1), after["columns"]["U"]


@pytest.mark.feature("ALIAS.change_propagation")
@pytest.mark.xfail(**STRICT, reason="A-D7")
def test_strict_an_alias_with_automatic_fields_follows_its_input(node_session, tmp_path):
    cache, after = _ad7(node_session, tmp_path)
    if _same(_num(after["columns"]["U"]), X * 3 + 1):
        return
    known_defect("U" in cache["cached"] and _same(_num(after["columns"]["U"]), X * 2 + 1), "A-D7")
    raise AssertionError(f"A-D7: neither numpy nor the A-D7 signature: {after}")


# ===================================================================================== A-D8 / A-D9 two evaluation paths
P = np.array([-3.0, -1.0, -0.5, 0.0, 1.0, 3.0, 7.5, 6e9])
DF_P = pd.DataFrame({"x": P, "b": np.array([1.0, -1.0, 1.0, 2.0, 1.0, -1.0, 0.5, 1.0]),
                     "e": np.array([np.inf, np.inf, np.nan, 3.0, -np.inf, -np.inf, 2.0, np.nan])})
with np.errstate(all="ignore"):
    AD8 = {
        # expression: (numpy = Python semantics, today's browser result)
        "x//2": (np.floor_divide(P, 2), np.array([_js_int32(v) for v in P / 2])),
        "x%2": (np.mod(P, 2), np.fmod(P, 2)),
        "~(x>1)": ((~(P > 1)).astype(float), np.where(P > 1, -2.0, -1.0)),
        "b**e": (np.power(DF_P.b.to_numpy(), DF_P.e.to_numpy()),
                 np.where((np.abs(DF_P.b.to_numpy()) == 1) & ~np.isfinite(DF_P.e.to_numpy()), np.nan,
                          np.power(DF_P.b.to_numpy(), DF_P.e.to_numpy()))),
    }


def _two_paths(node_session, tmp_path, expr):
    """The same expression as an alias (computed in the browser) and shown directly (computed on the server)."""
    path = _build(tmp_path, DF_P, [[["x"], ["al", expr]]], aliasArray=[("al", expr)])
    r = _run(node_session, path, [read("cdsFull", "al", expr)])[0]
    _complete(r, ["al", expr], ("number", "boolean"))
    return _num(r["columns"]["al"]), _num(r["columns"][expr])


@pytest.mark.feature("DSL.server_client_parity")
@pytest.mark.parametrize("expr", list(AD8))
def test_signature_a_d8_python_operators_differ_in_the_browser(node_session, tmp_path, expr):
    """A-D8, exact signature: the alias equals the browser prediction, the server column equals numpy, and they differ."""
    numpy_, browser = AD8[expr]
    alias, server = _two_paths(node_session, tmp_path, expr)
    assert _same(alias, browser), (list(alias), list(browser))
    assert _same(server, numpy_), (list(server), list(numpy_))
    assert not _same(browser, numpy_)


@pytest.mark.feature("DSL.server_client_parity")
@pytest.mark.xfail(**STRICT, reason="A-D8")
@pytest.mark.parametrize("expr", list(AD8))
def test_strict_one_expression_one_result(node_session, tmp_path, expr):
    """Invariant: an expression gives the same column as an alias and shown directly, equal to Python semantics."""
    numpy_, browser = AD8[expr]
    alias, server = _two_paths(node_session, tmp_path, expr)
    if _same(alias, numpy_) and _same(server, numpy_):
        return
    known_defect(_same(alias, browser) and _same(server, numpy_), f"A-D8 ({expr})")
    raise AssertionError(f"A-D8 ({expr}): neither numpy nor the A-D8 signature: alias {list(alias)}, server {list(server)}")


@pytest.mark.feature("DSL.server_client_parity")
@pytest.mark.parametrize("expr", ["x*2+b", "sqrt(abs(x))", "x/b", "arctan2(b, x)"])
def test_control_where_the_semantics_agree_both_paths_agree(node_session, tmp_path, expr):
    """Control for A-D8: where Python and JavaScript agree, the alias and the server column are equal."""
    alias, server = _two_paths(node_session, tmp_path, expr)
    assert _same(alias, server, MATH_RTOL), (list(alias), list(server))


@pytest.mark.feature("DSL.math_functions")
@pytest.mark.feature("DSL.server_client_parity")
def test_signature_a_d9_server_cosh_is_sinh(node_session, tmp_path):
    alias, server = _two_paths(node_session, tmp_path, "cosh(x/1e9)")
    with np.errstate(all="ignore"):
        assert _same(server, np.sinh(P / 1e9)) and not _same(server, np.cosh(P / 1e9), MATH_RTOL), list(server)
        assert _same(alias, np.cosh(P / 1e9), MATH_RTOL), list(alias)


@pytest.mark.feature("DSL.math_functions")
@pytest.mark.feature("DSL.server_client_parity")
@pytest.mark.xfail(**STRICT, reason="A-D9")
def test_strict_server_cosh_equals_numpy(node_session, tmp_path):
    alias, server = _two_paths(node_session, tmp_path, "cosh(x/1e9)")
    with np.errstate(all="ignore"):
        if _same(server, np.cosh(P / 1e9), MATH_RTOL):
            return
        known_defect(_same(server, np.sinh(P / 1e9)), "A-D9")
    raise AssertionError(f"A-D9: neither numpy nor the A-D9 signature: {list(server)}")


# ===================================================================================== the harness itself
def test_control_the_comparison_sees_nan_position_infinity_sign_and_one_ulp():
    """Checks can fail: the row comparison rejects a NaN in another row, the other infinity, and one ulp (exact mode)."""
    a = np.array([1.0, np.nan, np.inf])
    assert _same(a, a.copy())
    assert not _same(a, np.array([np.nan, 1.0, np.inf]))
    assert not _same(a, np.array([1.0, np.nan, -np.inf]))
    assert not _same(a, np.array([np.nextafter(1.0, 2.0), np.nan, np.inf]))
    assert _same(a, np.array([np.nextafter(1.0, 2.0), np.nan, np.inf]), MATH_RTOL)
    assert not _same(a, a[:2])


def test_control_the_javascript_helpers():
    """The prediction helpers reproduce JavaScript on known values."""
    assert [_js_int32(v) for v in (3e9, -1.5, 2 ** 31, np.nan, np.inf, -7.9)] == [-1294967296.0, -1.0, -2147483648.0, 0.0, 0.0, -7.0]
    assert [_js_number(v) for v in (1.0, -0.5, 10.5, np.nan, np.inf, -np.inf, 0.0)] == ["1", "-0.5", "10.5", "NaN", "Infinity", "-Infinity", "0"]
