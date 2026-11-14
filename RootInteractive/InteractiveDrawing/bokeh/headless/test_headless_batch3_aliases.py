"""PHASE_5 Batch 3 — aliases and gathers on ADF-shaped pages, in Node. Tests only (Batch 3 CRR v0.1, Claude20.RI,
2026-11-14).

Plain words. An ALIAS is a column the browser computes: from an expression (`("h", "g*2+x")`) or from a JavaScript
function. A GATHER is an alias that reads a column of a related table through an index column: `ev.ex[evIdx]` gives,
for each track, the `ex` of its event. The ADF/dfdraw compiler (PHASE_3, on hold) will turn `adf.draw(...)` calls into
pages full of such columns: derived columns, columns of related tables, derived columns of related tables. Batch 3 of
the PHASE_5 bug fixes repairs that path. These tests come before the fix: they state what such a page must show,
compare it with numpy, and show every Batch 3 defect RED on today's code with its exact signature. No production code
is changed.

THE PAGES have the shape the relation exporter of PHASE_3_1 (step 3.1a) gives a compiled page:
  * a track table (the main table, 300 rows) with an index column `evIdx` into the event table;
  * an event table `ev` (40 events) plus one NaN FILL ROW at its end: a track without an event (an ORPHAN, 6 % here)
    points at the fill row, so everything it gathers is NaN;
  * many tracks per event (DUPLICATES: one event row read by several tracks).
Pages are built by RootInteractive's own builder (bokehDrawSA.fromArray) through the helpers of the PHASE_4 alias tests
(test_headless_node_aliases.py), and read from the production models in Node.

THE ORACLE is numpy on the same data, Python semantics. Histograms: oracles/histograms.py (rulebook RB-2, reviewed).

FAIL-CLOSED (gate G1). Each strict test sorts the page into one of three verdicts with a pure function:
  OK         the page equals the oracle;
  KNOWN      the page equals the exact signature of today's defect (raised as KnownDefect, the only failure a strict
             test accepts);
  UNEXPLAINED anything else: an ordinary failure.
The unit test `test_falsifier_every_verdict_rejects_wrong_pages` feeds each verdict wrong pages made from the oracle and
from the signature, and requires UNEXPLAINED for every one.

Defects (register PHASE_5_0_Defect_Register_v0_3; code at 492c8ff, the 2A/2B correction v0.2):
  A-D1  an expression that reads another alias gets the whole column, not the row's value (compileVarName.py 213, 232):
        "h = g*2+x" over a gathered alias g is NaN in every row (PHASE_3_1 review R2); "ex3 = ex2+1" on the event table
        is the text of the whole column ex2 plus "1".
  A-D3  a column-selector parameter inside an expression fails the page build: KeyError 'cdsFull' (bokehTools.py 495,
        512); here inside an expression that also gathers.
  A-D4  a gather alone does not send the column it gathers: reading the gather throws.
  A-D5  a gather of an alias of the related table fails the build: "Column ex3 not found in data source None"
        (compileVarName.py 283); here at the end of a chain of depth 3 over two tables.
  A-D6  a gather is not recomputed when the gathered table changes; nor is an alias that reads it.
R-4   (new, found while building these pages; not an alias defect, not fixed in Batch 3) a page whose figures all show
        the related table fails the build: KeyError 'cdsOrig' at the end of bokehDrawArray (bokehTools.py), which returns
        the main table's model although no figure made it. A strict expected failure.
A-D2, A-D7 and A-D9 keep their PHASE_4 tests (test_headless_node_aliases.py). A-D8 (operators that mean something else
in the browser) waits for the Architect's decision D-B4; its tests are added then.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import LOSSLESS
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import histograms as hist_oracle
from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_node_aliases import (
    STRICT, KnownDefect, _build_error, _js_number, _num, _run, _same, read, set_)

def _node(*marks):
    """Node test marks: the target node_aliases (PHASE_4_5 A1), backend, layer, then the given marks."""
    def wrap(test):
        for m in reversed([pytest.mark.node_aliases, pytest.mark.backend("node"), pytest.mark.layer("integration")]
                          + list(marks)):
            test = m(test)
        return test
    return wrap


def _unit(test):
    return pytest.mark.headless_unit(pytest.mark.backend("python")(pytest.mark.layer("unit")(test)))


OK, KNOWN, UNEXPLAINED = "OK", "KNOWN", "UNEXPLAINED"


def _strict(verdict: str, defect: str, detail) -> None:
    """The strict test's end: pass on OK, KnownDefect on KNOWN, an ordinary failure on anything else."""
    if verdict == OK:
        return
    if verdict == KNOWN:
        raise KnownDefect(defect)
    raise AssertionError(f"{defect}: neither the oracle nor the exact signature: {detail}")


# ===================================================================================== the ADF-shaped data
N_EV, N_TR, ORPHAN_SHARE = 40, 300, 0.06
_rng = np.random.default_rng(20261114)
EX = np.append(np.round(_rng.uniform(5.0, 65.0, N_EV), 3), np.nan)        # the event table, fill row last
EY = np.append(np.round(_rng.normal(0.0, 2.0, N_EV), 3), np.nan)
FILL = N_EV                                                                # the fill row's index
EV_IDX = _rng.integers(0, N_EV, N_TR).astype(np.int32)
EV_IDX[_rng.random(N_TR) < ORPHAN_SHARE] = FILL
TR = pd.DataFrame({"x": np.round(_rng.normal(0.0, 3.0, N_TR), 3), "y": np.round(_rng.uniform(-2.0, 2.0, N_TR), 3),
                   "evIdx": EV_IDX})
X, Y = TR.x.to_numpy(), TR.y.to_numpy()
EV_SOURCE = [{"name": "ev", "data": {"ex": EX, "ey": EY}}]
EX_NEW = np.append(np.round(EX[:N_EV] * 10 + 1, 3), np.nan)              # the event table after a change (A-D6)
CODINGS = {"plain": None, "lossless": [(p, list(a)) for p, a in LOSSLESS]}


@_unit
def test_the_data_has_orphans_duplicates_and_finite_values():
    """Non-vacuity: the pages below see orphans (gathering NaN), events read by several tracks, and finite values."""
    orphans = int(np.sum(EV_IDX == FILL))
    per_event = np.bincount(EV_IDX[EV_IDX != FILL], minlength=N_EV)
    assert 5 <= orphans <= 40, orphans
    assert per_event.max() >= 5 and np.sum(per_event >= 2) >= 30, per_event
    assert np.isnan(EX[FILL]) and np.all(np.isfinite(EX[:N_EV])) and np.all(np.isfinite(X))



def _page(node_session, tmp_path, figures, aliases, steps, widgets=(), coding="plain", **options):
    """(build error or None, the production replies). The build error is (type, message, file:function)."""
    err = _build_error(tmp_path, TR, figures, widgets, sourceArray=EV_SOURCE, aliasArray=aliases,
                       arrayCompression=CODINGS[coding], **options)
    if err is not None:
        return err, None
    return None, _run(node_session, Path(tmp_path) / "aliases_document.json", steps)


def _column(reply, key):
    """A production column as float64, and its JavaScript kinds; None for the column if it was not read or is not a
    number column."""
    if reply is None or key in reply["throws"] or reply["columns"].get(key) is None:
        return None, "missing" if reply is None else reply["kinds"].get(key, "missing")
    kinds = reply["kinds"][key]
    return (_num(reply["columns"][key]) if kinds == "number" else None), kinds


MUTATE_JS = """(a) => { const doc = Bokeh.documents[Bokeh.documents.length - 1];
  const t = Array.from(doc._all_models.values()).filter(m => m.name === a.table && m.constructor.__name__ === "ColumnDataSource");
  if (t.length !== 1) throw new Error("tables named " + a.table + ": " + t.length);
  const col = Float64Array.from(a.values, v => v === null ? NaN : v);        // JSON has no NaN: null stands for it
  t[0].data = Object.assign({}, t[0].data, {[a.column]: col});
  return col.length; }"""


def _mutate_then(node_session, table, column, values, steps):
    """New values for one column of a table of the loaded page (NaN allowed), then the production replies to steps."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_node_aliases import STEPS
    r = node_session.request("eval", code=MUTATE_JS, arg={"table": table, "column": column,
                                                          "values": [None if np.isnan(v) else float(v) for v in values]})
    assert r.get("ok") and r["result"] == len(values), r
    r = node_session.request("eval", code=STEPS, arg={"steps": steps})
    assert r.get("ok"), r
    out = decode(r["result"])
    assert not [o for o in out if "error" in o], out
    return out


EV_SHOWN = [["ev.ex"], ["ev.ex"]]          # a figure that shows ev.ex, so the column is sent (keeps A-D4 out)


# ===================================================================================== 1. gathers: orphans, duplicates
@_node(pytest.mark.feature("DSL.gather_operation"))
@pytest.mark.parametrize("coding", list(CODINGS))
def test_control_a_gather_with_orphans_and_duplicates_equals_numpy(node_session, tmp_path, coding):
    """`ev.ex[evIdx]` as an alias, as a second alias, and shown directly in a figure: each track gets its event's value;
    the tracks of one event all get the same value; an orphan gets the fill row's NaN. Plain and lossless coding."""
    err, out = _page(node_session, tmp_path, [[["x"], ["g", "gy", "ev.ex[evIdx]"]], EV_SHOWN, [["ev.ey"], ["ev.ey"]]],
                     [("g", "ev.ex[evIdx]"), ("gy", "ev.ey[evIdx]")], [read("cdsFull", "g", "gy", "ev.ex[evIdx]")],
                     coding=coding)
    assert err is None, err
    r = out[0]
    assert not r["throws"], r["throws"]
    for key, want in (("g", EX[EV_IDX]), ("gy", EY[EV_IDX]), ("ev.ex[evIdx]", EX[EV_IDX])):
        got, kinds = _column(r, key)
        assert kinds == "number" and _same(got, want), (key, kinds)


def verdict_gather_alone(ev_keys, reply) -> str:
    """A-D4 in the ADF shape: a gather shown alone. OK: equal to numpy. KNOWN: the event table has no `ex` and reading
    the gather throws exactly "Cannot read properties of null (reading '<evIdx of the first track>')" (TypeError): it
    indexes a column that is not there."""
    got, kinds = _column(reply, "g")
    if got is not None and kinds == "number" and _same(got, EX[EV_IDX]):
        return OK
    if ("ex" not in ev_keys and reply["throws"] == {"g": f"Cannot read properties of null (reading '{EV_IDX[0]}')"}
            and reply["throw_kinds"] == {"g": "TypeError"}):
        return KNOWN
    return UNEXPLAINED


@_node(pytest.mark.feature("DSL.gather_operation"))
@pytest.mark.xfail(**STRICT, reason="A-D4")
def test_strict_a_gather_alone_sends_its_column(node_session, tmp_path):
    """A compiled page gathers columns it does not show. The gather alone must work: the page sends the column."""
    err, out = _page(node_session, tmp_path, [[["x"], ["g"]]], [("g", "ev.ex[evIdx]")],
                     [read("cdsFull", "g"), {"op": "keys", "table": "ev"}])
    assert err is None, err
    _strict(verdict_gather_alone(out[1]["keys"], out[0]), "A-D4", out)


# ===================================================================================== 2. chains: A-D1, A-D5
H_ORACLE = EX[EV_IDX] * 2 + X


def verdict_alias_over_gather(reply) -> str:
    """R2 (PHASE_3_1 review): h = "g*2+x" over the gathered alias g. OK: g and h equal numpy. KNOWN (A-D1): g is right,
    h is a number column that is NaN in every row."""
    g, gk = _column(reply, "g")
    h, hk = _column(reply, "h")
    if g is None or gk != "number" or not _same(g, EX[EV_IDX]):
        return UNEXPLAINED
    if h is not None and hk == "number" and _same(h, H_ORACLE):
        return OK
    if h is not None and hk == "number" and np.all(np.isnan(h)):
        return KNOWN
    return UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.chain"), pytest.mark.feature("DSL.gather_operation"))
@pytest.mark.xfail(**STRICT, reason="A-D1")
def test_strict_alias_over_a_gathered_alias(node_session, tmp_path):
    """The PHASE_3_1 R2 case in a page: an expression over a gathered alias equals numpy, orphans NaN."""
    err, out = _page(node_session, tmp_path, [[["x"], ["g", "h"]], EV_SHOWN], [("g", "ev.ex[evIdx]"), ("h", "g*2+x")],
                     [read("cdsFull", "g", "h")])
    assert err is None, err
    _strict(verdict_alias_over_gather(out[0]), "A-D1", out)


EX2, EX3 = EX * 2, EX * 2 + 1


def _ad1_text(column) -> list:
    """A-D1 on the event table: "ex3 = ex2+1" is JavaScript's String(whole column ex2) + "1" in every row."""
    return [",".join(_js_number(v) for v in column) + "1"] * len(column)


def verdict_alias_of_alias_on_the_related_table(reply) -> str:
    """OK: ex2 and ex3 equal numpy (fill row NaN). KNOWN (A-D1): ex2 is right, ex3 is the text of the whole column ex2
    plus "1" in every row."""
    ex2, k2 = _column(reply, "ex2")
    if ex2 is None or k2 != "number" or not _same(ex2, EX2):
        return UNEXPLAINED
    ex3, k3 = _column(reply, "ex3")
    if ex3 is not None and k3 == "number" and _same(ex3, EX3):
        return OK
    if k3 == "string" and reply["columns"]["ex3"] == _ad1_text(EX2):
        return KNOWN
    return UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.chain"), pytest.mark.feature("ALIAS.context"))
@pytest.mark.xfail(**STRICT, reason="A-D1")
def test_strict_alias_of_an_alias_on_the_related_table(node_session, tmp_path):
    """Derived columns of the event table that build on each other: ex3 = ex2 + 1, ex2 = ex * 2."""
    err, out = _page(node_session, tmp_path, [[["x"], ["y"]], [["ev.ex"], ["ev.ex3"]]],
                     [("ex2", "ex*2", "ev"), ("ex3", "ex2+1", "ev")], [read("ev_full", "ex2", "ex3")])
    assert err is None, err
    _strict(verdict_alias_of_alias_on_the_related_table(out[0]), "A-D1", out)


H3_ORACLE = EX3[EV_IDX] * X
AD5_CHAIN_SIGNATURE = ("KeyError", "Column ex3 not found in data source None", "compileVarName.py:visit_Attribute")


def verdict_chain_of_depth_3(err, reply) -> str:
    """ex2 = ex*2 and ex3 = ex2+1 on the event table, g3 = ev.ex3[evIdx] and h3 = g3*x on the track table. OK: g3 and
    h3 equal numpy. KNOWN (A-D5): the build fails exactly with the A-D5 error for ex3."""
    if err is not None:
        return KNOWN if tuple(err) == AD5_CHAIN_SIGNATURE else UNEXPLAINED
    g3, gk = _column(reply, "g3")
    h3, hk = _column(reply, "h3")
    ok = (g3 is not None and gk == "number" and _same(g3, EX3[EV_IDX])
          and h3 is not None and hk == "number" and _same(h3, H3_ORACLE))
    return OK if ok else UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.chain"), pytest.mark.feature("DSL.gather_operation"))
@pytest.mark.xfail(**STRICT, reason="A-D5")
def test_strict_alias_chain_of_depth_3_across_two_tables(node_session, tmp_path):
    """A chain over two tables, as compiled ADF queries build them: two derived columns of the event table, gathered to
    the tracks, used in an expression with a track column."""
    err, out = _page(node_session, tmp_path, [[["x"], ["g3", "h3"]], [["ev.ex"], ["ev.ex3"]]],
                     [("ex2", "ex*2", "ev"), ("ex3", "ex2+1", "ev"), ("g3", "ev.ex3[evIdx]"), ("h3", "g3*x")],
                     [read("cdsFull", "g3", "h3")])
    _strict(verdict_chain_of_depth_3(err, out[0] if out else None), "A-D5", err or out)


# ===================================================================================== 3. a column selector: A-D3
AD3_SIGNATURE = ("KeyError", "cdsFull", "bokehTools.py:bokehDrawArray")
SEL = [{"name": "sel", "value": "x", "options": ["x", "y"]}]


def verdict_selector_in_a_gather(err, before, after) -> str:
    """sg = "ev.ex[evIdx]*sel", sel a column selector (x or y). OK: before the change sg = ex[evIdx]*x, after the change
    to y it is ex[evIdx]*y. KNOWN (A-D3): the build fails exactly with KeyError 'cdsFull' in bokehDrawArray."""
    if err is not None:
        return KNOWN if tuple(err) == AD3_SIGNATURE else UNEXPLAINED
    b, bk = _column(before, "sg")
    a, ak = _column(after, "sg")
    ok = (b is not None and bk == "number" and _same(b, EX[EV_IDX] * X)
          and a is not None and ak == "number" and _same(a, EX[EV_IDX] * Y))
    return OK if ok else UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.parameter"), pytest.mark.feature("DSL.gather_operation"))
@pytest.mark.xfail(**STRICT, reason="A-D3")
def test_strict_a_column_selector_inside_a_gathering_expression(node_session, tmp_path):
    """A widget-driven variable inside an expression that also gathers ($w(var)$ in a compiled query)."""
    err, out = _page(node_session, tmp_path, [[["x"], ["sg"]], EV_SHOWN], [("sg", "ev.ex[evIdx]*sel")],
                     [read("cdsFull", "sg"), set_("sel", "y"), read("cdsFull", "sg")],
                     widgets=[["select", ["sel"]]], parameterArray=SEL)
    if out is not None:
        assert out[1]["callbacks"] >= 1 and not out[1]["errors"], out[1]
    _strict(verdict_selector_in_a_gather(err, out[0] if out else None, out[2] if out else None), "A-D3", err or out)


# ===================================================================================== 4. the gathered table changes: A-D6
C_FUNC = {"name": "C", "variables": ["g"], "func": "return g+1"}       # the function form chains correctly (PHASE_4)


def verdict_gather_follows_the_table(before, ev_after, after) -> str:
    """g = ev.ex[evIdx] and C = g + 1 (function form), then the event table gets new values. OK: before and after, g
    and C equal numpy with the table's values at that time. KNOWN (A-D6): before right, the event table has the new
    values, and g and C still have exactly the old ones."""
    def same(reply, ex):
        g, gk = _column(reply, "g")
        c, ck = _column(reply, "C")
        return (g is not None and gk == "number" and _same(g, ex[EV_IDX])
                and c is not None and ck == "number" and _same(c, ex[EV_IDX] + 1))
    ev, evk = _column(ev_after, "ex")
    if not same(before, EX) or ev is None or not _same(ev, EX_NEW):
        return UNEXPLAINED
    if same(after, EX_NEW):
        return OK
    if same(after, EX):
        return KNOWN
    return UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.change_propagation"), pytest.mark.feature("DSL.gather_operation"))
@pytest.mark.xfail(**STRICT, reason="A-D6")
def test_strict_a_gather_and_its_dependents_follow_the_gathered_table(node_session, tmp_path):
    """New values in the event table: the gather and the alias built on it are recomputed."""
    err, out = _page(node_session, tmp_path, [[["x"], ["g", "C"]], EV_SHOWN], [("g", "ev.ex[evIdx]"), C_FUNC],
                     [read("cdsFull", "g", "C")])
    assert err is None, err
    ev, after = _mutate_then(node_session, "ev", "ex", EX_NEW, [read("ev_full", "ex"), read("cdsFull", "g", "C")])
    _strict(verdict_gather_follows_the_table(out[0], ev, after), "A-D6", (out, ev, after))


# ===================================================================================== 5. histograms of aliases
HISTS = [{"name": "h2", "variables": ["xs", "g"], "nbins": [6, 5], "range": [[-9, 9], [5, 65]]},
         {"name": "h1", "variables": ["g"], "nbins": 6, "range": [5, 65]}]
BIN_COUNTS_JS = """() => { const ms = Array.from(Bokeh.documents[Bokeh.documents.length - 1]._all_models.values())
  .filter(m => m.type === 'HistoNdCDS' || m.type === 'HistogramCDS');
  const out = {}; for (const m of ms) out[m.name] = Array.from(m.get_column('bin_count')); return out; }"""


@_node(pytest.mark.feature("HIST.histogram_nd"), pytest.mark.feature("HIST.histogram_1d"),
       pytest.mark.feature("DSL.gather_operation"))
def test_histograms_of_an_alias_and_a_gathered_column_equal_numpy(node_session, tmp_path):
    """The contract's RI-ADF-02 shape: a 2-D histogram over an expression alias (x + 0.25) and a gathered column, and a
    1-D histogram of the gathered column. Orphans gather NaN and are not counted (rulebook RB-2.5; B8 fixed)."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    err = _build_error(tmp_path, TR, [[["x"], ["g"]], EV_SHOWN, [["bin_center_0"], ["bin_count"], {"source": "h2"}],
                                      [["bin_center"], ["bin_count"], {"source": "h1"}]],
                       sourceArray=EV_SOURCE, aliasArray=[("g", "ev.ex[evIdx]"), ("xs", "x+0.25")], histogramArray=HISTS)
    assert err is None, err
    assert node_session.request("new_doc", path=str(Path(tmp_path) / "aliases_document.json"))["ok"]
    r = node_session.request("eval", code=BIN_COUNTS_JS)
    assert r["ok"], r
    got = {k: np.asarray(v, dtype=float) for k, v in decode(r["result"]).items()}
    g = EX[EV_IDX]
    want = {"h2": hist_oracle.counts([X + 0.25, g], [6, 5], [[-9, 9], [5, 65]]),
            "h1": hist_oracle.counts([g], [6], [[5, 65]])}
    for k in want:
        assert _same(got[k], want[k]), (k, list(got[k]), list(want[k]))
    assert want["h1"].sum() < N_TR and want["h2"].sum() > 0              # orphans not counted; the page is not empty


# ===================================================================================== a new finding: R-4
R4_SIGNATURE = ("KeyError", "cdsOrig", "bokehTools.py:bokehDrawArray")


def verdict_related_table_only(err, reply) -> str:
    """R-4 (new, found while building these pages): a page whose figures all show the related table. OK: it builds,
    and the event table's columns and its alias read right (fill row NaN). KNOWN: the build fails exactly with
    KeyError 'cdsOrig' at the end of bokehDrawArray, which returns the main table's model although no figure made it."""
    if err is not None:
        return KNOWN if tuple(err) == R4_SIGNATURE else UNEXPLAINED
    ex, k1 = _column(reply, "ex")
    ex2, k2 = _column(reply, "ex2")
    ok = ex is not None and k1 == "number" and _same(ex, EX) and ex2 is not None and k2 == "number" and _same(ex2, EX2)
    return OK if ok else UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.context"))
@pytest.mark.xfail(**STRICT, reason="R-4: a page of related-table figures only does not build")
def test_strict_a_page_of_related_table_figures_only_builds(node_session, tmp_path):
    """A compiled query that draws only event-level quantities still makes a page. (Control: every other page here has
    a figure of the track table too, and builds.)"""
    err, out = _page(node_session, tmp_path, [[["ev.ex"], ["ev.ex2"]]], [("ex2", "ex*2", "ev")],
                     [read("ev_full", "ex", "ex2")])
    verdict = verdict_related_table_only(err, out[0] if out else None)
    if verdict == KNOWN:
        raise KnownDefect("R-4")                         # not fixed in Batch 3: a strict expected failure
    _strict(verdict, "R-4", err or out)


# ===================================================================================== the verdicts can fail
def _reply(**cols):
    """A fake production reply: number columns from arrays, text columns from lists of strings."""
    r = {"columns": {}, "kinds": {}, "throws": {}, "throw_kinds": {}}
    for k, v in cols.items():
        if isinstance(v, list) and v and isinstance(v[0], str):
            r["columns"][k], r["kinds"][k] = v, "string"
        else:
            r["columns"][k] = [None if np.isnan(x) else float(x) for x in np.asarray(v, dtype=float)]
            r["kinds"][k] = "number"
    return r


def _shifted(a):
    b = np.array(a, dtype=float)
    b[np.flatnonzero(np.isfinite(b))[0]] += 1.0
    return b


@_unit
def test_falsifier_every_verdict_rejects_wrong_pages():
    """Each verdict says OK for the oracle and KNOWN for the exact signature, and UNEXPLAINED for pages that are wrong
    in another way: one value off, a column missing, the signature in a column it does not concern, orphans not NaN,
    a partial signature."""
    g = EX[EV_IDX]
    nan = np.full(N_TR, np.nan)
    # A-D4
    good = _reply(g=g)
    sig = {"columns": {}, "kinds": {}, "throws": {"g": f"Cannot read properties of null (reading '{EV_IDX[0]}')"},
           "throw_kinds": {"g": "TypeError"}}
    assert verdict_gather_alone(["ey"], good) == OK and verdict_gather_alone(["ey"], sig) == KNOWN
    assert verdict_gather_alone(["ex", "ey"], sig) == UNEXPLAINED                       # the column was sent after all
    assert verdict_gather_alone(["ey"], {**sig, "throws": {"g": "other error"}}) == UNEXPLAINED
    assert verdict_gather_alone(["ey"], _reply(g=_shifted(g))) == UNEXPLAINED
    assert verdict_gather_alone(["ey"], _reply(g=np.where(EV_IDX == FILL, 0.0, g))) == UNEXPLAINED   # orphans not NaN
    # A-D1, R2
    assert verdict_alias_over_gather(_reply(g=g, h=H_ORACLE)) == OK
    assert verdict_alias_over_gather(_reply(g=g, h=nan)) == KNOWN
    assert verdict_alias_over_gather(_reply(g=g, h=_shifted(H_ORACLE))) == UNEXPLAINED
    assert verdict_alias_over_gather(_reply(g=g, h=g * 2)) == UNEXPLAINED              # "+x" lost
    assert verdict_alias_over_gather(_reply(g=nan, h=nan)) == UNEXPLAINED               # g wrong too
    assert verdict_alias_over_gather(_reply(g=g)) == UNEXPLAINED                        # h missing
    # A-D1 on the event table
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=EX2, ex3=EX3)) == OK
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=EX2, ex3=_ad1_text(EX2))) == KNOWN
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=EX2, ex3=_ad1_text(EX))) == UNEXPLAINED
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=EX2, ex3=_shifted(EX3))) == UNEXPLAINED
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=_shifted(EX2), ex3=_ad1_text(EX2))) == UNEXPLAINED
    # A-D5, depth 3
    assert verdict_chain_of_depth_3(None, _reply(g3=EX3[EV_IDX], h3=H3_ORACLE)) == OK
    assert verdict_chain_of_depth_3(AD5_CHAIN_SIGNATURE, None) == KNOWN
    assert verdict_chain_of_depth_3(("KeyError", "Column ex2 not found in data source None",
                                     "compileVarName.py:visit_Attribute"), None) == UNEXPLAINED
    assert verdict_chain_of_depth_3(None, _reply(g3=EX3[EV_IDX], h3=nan)) == UNEXPLAINED  # A-D1 behind a fixed A-D5
    assert verdict_chain_of_depth_3(None, _reply(g3=EX2[EV_IDX], h3=EX2[EV_IDX] * X)) == UNEXPLAINED
    # A-D3
    before, after = _reply(sg=g * X), _reply(sg=g * Y)
    assert verdict_selector_in_a_gather(None, before, after) == OK
    assert verdict_selector_in_a_gather(AD3_SIGNATURE, None, None) == KNOWN
    assert verdict_selector_in_a_gather(None, before, before) == UNEXPLAINED             # the selector did nothing
    assert verdict_selector_in_a_gather(("KeyError", "cdsOrig", "bokehTools.py:bokehDrawArray"), None, None) == UNEXPLAINED
    # A-D6
    old, new = _reply(g=g, C=g + 1), _reply(g=EX_NEW[EV_IDX], C=EX_NEW[EV_IDX] + 1)
    ev_new = _reply(ex=EX_NEW)
    assert verdict_gather_follows_the_table(old, ev_new, new) == OK
    assert verdict_gather_follows_the_table(old, ev_new, old) == KNOWN
    assert verdict_gather_follows_the_table(old, _reply(ex=EX), old) == UNEXPLAINED       # the table did not change
    assert verdict_gather_follows_the_table(old, ev_new, _reply(g=EX_NEW[EV_IDX], C=g + 1)) == UNEXPLAINED  # C stale only
    assert verdict_gather_follows_the_table(old, ev_new, _reply(g=g, C=EX_NEW[EV_IDX] + 1)) == UNEXPLAINED
    # R-4
    assert verdict_related_table_only(None, _reply(ex=EX, ex2=EX2)) == OK
    assert verdict_related_table_only(R4_SIGNATURE, None) == KNOWN
    assert verdict_related_table_only(("KeyError", "cdsFull", "bokehTools.py:bokehDrawArray"), None) == UNEXPLAINED
    assert verdict_related_table_only(None, _reply(ex=EX, ex2=_shifted(EX2))) == UNEXPLAINED
