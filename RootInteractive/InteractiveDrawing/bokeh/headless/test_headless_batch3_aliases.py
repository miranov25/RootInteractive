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
        the main table's model although no figure made it. Fixed in PHASE_5 Group 4 (decision L-24): the builder
        returns None for a main table that nothing uses; the test is an ordinary test, the signature a REGRESSION.
A-D2, A-D7 and A-D9 keep their PHASE_4 tests (test_headless_node_aliases.py). A-D8 (operators that mean something else
in the browser) waits for the Architect's decision D-B4; its tests are added then.

FIX COMMIT (PHASE_5 Batch 3 fix CRR v0.1, Claude20.RI, 2026-11-14): A-D1, A-D3, A-D4, A-D5 and A-D6 are fixed. The
strict marks are off: every page must equal numpy. A page that equals an old exact signature (the verdict KNOWN) is now
a REGRESSION, an ordinary failure that names the defect; the falsifier still checks every verdict.

BATCH 3 v0.2 TESTS (review summary v0.1, GPT28.RI; plan v0.2 approved 2026-11-14). What the reviewers found, reproduced:
  * A-D11 (new; review B3-C1): an alias whose inputs are listed as a DICTIONARY ({"lhs": "g"}, {"terms": ["g", "x"]}) is
    not recomputed when an input changes; it keeps its old values. In production this form is used by ONNX model
    aliases (OrtFunction, "variables": {"float_input": [...]}); onnxruntime does not run in Node, so the tests give the
    PRODUCTION alias model (CDSAlias) small test transforms that take a dictionary, as OrtFunction does. Synchronous and
    asynchronous (a Promise, as OrtFunction returns). Strict expected failures with the exact signature.
  * B3-C2: `verdict_gather_alone` said OK when the event column `ex` was never sent. OK now needs `ex` sent, nothing
    thrown, and the right numbers.
  * B3-C5: every Node request goes through `_request`, which fails the test on any error the page reports.
Added (review B3-N1): a gather chain on a page WITHOUT an event figure, so only the alias's own dependencies send the
event column. Side effects: aliases that do not read the changed input keep their cached values; two tables whose
aliases read each other settle after a change (no endless loop) with the right values.
Rule 1 of plan v0.2: every condition of a verdict's OK has a wrong page in the falsifier that breaks only it.

CORRECTED AFTER THE TEST REVIEW. Before the fix is written, the reviewers check the test-only commit (here 0149b82;
summary by GPT28.RI). GPT27.RI found that verdicts accepted a read in which a column they do not look at threw an error
(findings G5a-F1, G5a-F2). Now a read in which ANY column threw is never OK, and never KNOWN unless that error is
itself the exact signature: every verdict asks `_clean` of every read reply it receives. The falsifier adds, for each
verdict and each of its read replies, the same page with one more column that threw.

BATCH 3 v0.2 FIX (CDSAlias.ts, field_names: the values of a dictionary of inputs are inputs too): A-D11 is fixed. Its
two tests lose their strict marks; their exact signature (the verdict KNOWN) is now a REGRESSION.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import LOSSLESS
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import histograms as hist_oracle
from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_node_aliases import (
    STEPS, STRICT, KnownDefect, _FakeSession, _build_error, _clean, _js_number, _num, _request, _run, _same, _threw, cached,
    read, set_)

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
    """The test's end: pass on OK; on KNOWN, the old exact signature of a fixed defect, a REGRESSION (fix commit); an
    ordinary failure on anything else."""
    if verdict == OK:
        return
    if verdict == KNOWN:
        raise AssertionError("REGRESSION, a fixed defect is back: " + defect)
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
    options.setdefault("sourceArray", EV_SOURCE)
    err = _build_error(tmp_path, TR, figures, widgets, aliasArray=aliases, arrayCompression=CODINGS[coding], **options)
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


def _steps(node_session, steps):
    """The production replies to steps on the loaded page; a harness problem or a page error fails the test."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    out = decode(_request(node_session, "eval", code=STEPS, arg={"steps": steps})["result"])
    assert not [o for o in out if "error" in o], out
    return out


def _mutate_then(node_session, table, column, values, steps):
    """New values for one column of a table of the loaded page (NaN allowed), then the production replies to steps.
    Batch 3 v0.2 (review B3-C5): both requests must be ok and the page healthy."""
    r = _request(node_session, "eval", code=MUTATE_JS, arg={"table": table, "column": column,
                                                           "values": [None if np.isnan(v) else float(v) for v in values]})
    assert r["result"] == len(values), r
    return _steps(node_session, steps)


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
    """A-D4 in the ADF shape: a gather shown alone. OK: the event table's column `ex` was sent, nothing threw, and g
    equals numpy (Batch 3 v0.2, review B3-C2: the right numbers alone are not enough - the test is about sending the
    column). KNOWN: the event table has no `ex` and reading the gather throws exactly "Cannot read properties of null
    (reading '<evIdx of the first track>')" (TypeError): it indexes a column that is not there."""
    got, kinds = _column(reply, "g")
    if "ex" in ev_keys and not reply["throws"] and got is not None and kinds == "number" and _same(got, EX[EV_IDX]):
        return OK
    if ("ex" not in ev_keys and reply["throws"] == {"g": f"Cannot read properties of null (reading '{EV_IDX[0]}')"}
            and reply["throw_kinds"] == {"g": "TypeError"}):
        return KNOWN
    return UNEXPLAINED


@_node(pytest.mark.feature("DSL.gather_operation"))
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
    h is a number column that is NaN in every row. Test review correction: no column threw (`_clean`), for OK and
    KNOWN alike."""
    if not _clean(reply):
        return UNEXPLAINED
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
    plus "1" in every row. Test review correction: no column threw (`_clean`)."""
    if not _clean(reply):
        return UNEXPLAINED
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
    if not _clean(reply):                                       # test review correction
        return UNEXPLAINED
    g3, gk = _column(reply, "g3")
    h3, hk = _column(reply, "h3")
    ok = (g3 is not None and gk == "number" and _same(g3, EX3[EV_IDX])
          and h3 is not None and hk == "number" and _same(h3, H3_ORACLE))
    return OK if ok else UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.chain"), pytest.mark.feature("DSL.gather_operation"))
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
    if not _clean(before, after):                               # test review correction
        return UNEXPLAINED
    b, bk = _column(before, "sg")
    a, ak = _column(after, "sg")
    ok = (b is not None and bk == "number" and _same(b, EX[EV_IDX] * X)
          and a is not None and ak == "number" and _same(a, EX[EV_IDX] * Y))
    return OK if ok else UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.parameter"), pytest.mark.feature("DSL.gather_operation"))
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
    values, and g and C still have exactly the old ones. Test review correction: no column threw in any of the
    three reads."""
    def same(reply, ex):
        g, gk = _column(reply, "g")
        c, ck = _column(reply, "C")
        return (g is not None and gk == "number" and _same(g, ex[EV_IDX])
                and c is not None and ck == "number" and _same(c, ex[EV_IDX] + 1))
    if not _clean(before, ev_after, after):
        return UNEXPLAINED
    ev, evk = _column(ev_after, "ex")
    if not same(before, EX) or ev is None or not _same(ev, EX_NEW):
        return UNEXPLAINED
    if same(after, EX_NEW):
        return OK
    if same(after, EX):
        return KNOWN
    return UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.change_propagation"), pytest.mark.feature("DSL.gather_operation"))
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


def _bin_counts(node_session, path):
    """The bin counts of every histogram of the page; Batch 3 v0.2 (review B3-C5): a page error fails the test."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    _request(node_session, "new_doc", path=str(path))
    r = _request(node_session, "eval", code=BIN_COUNTS_JS)
    return {k: np.asarray(v, dtype=float) for k, v in decode(r["result"]).items()}


@_node(pytest.mark.feature("HIST.histogram_nd"), pytest.mark.feature("HIST.histogram_1d"),
       pytest.mark.feature("DSL.gather_operation"))
def test_histograms_of_an_alias_and_a_gathered_column_equal_numpy(node_session, tmp_path):
    """The contract's RI-ADF-02 shape: a 2-D histogram over an expression alias (x + 0.25) and a gathered column, and a
    1-D histogram of the gathered column. Orphans gather NaN and are not counted (rulebook RB-2.5; B8 fixed)."""
    err = _build_error(tmp_path, TR, [[["x"], ["g"]], EV_SHOWN, [["bin_center_0"], ["bin_count"], {"source": "h2"}],
                                      [["bin_center"], ["bin_count"], {"source": "h1"}]],
                       sourceArray=EV_SOURCE, aliasArray=[("g", "ev.ex[evIdx]"), ("xs", "x+0.25")], histogramArray=HISTS)
    assert err is None, err
    got = _bin_counts(node_session, Path(tmp_path) / "aliases_document.json")
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
    if not _clean(reply):                                       # test review correction
        return UNEXPLAINED
    ex, k1 = _column(reply, "ex")
    ex2, k2 = _column(reply, "ex2")
    ok = ex is not None and k1 == "number" and _same(ex, EX) and ex2 is not None and k2 == "number" and _same(ex2, EX2)
    return OK if ok else UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.context"))
def test_strict_a_page_of_related_table_figures_only_builds(node_session, tmp_path):
    """A compiled query that draws only event-level quantities still makes a page. (Control: every other page here has
    a figure of the track table too, and builds.)"""
    err, out = _page(node_session, tmp_path, [[["ev.ex"], ["ev.ex2"]]], [("ex2", "ex*2", "ev")],
                     [read("ev_full", "ex", "ex2")])
    verdict = verdict_related_table_only(err, out[0] if out else None)
    _strict(verdict, "R-4", err or out)                  # fixed in PHASE_5 Group 4: KNOWN is a REGRESSION


# ===================================================================================== 6. inputs as a dictionary: A-D11
U_G = """if($output == null || $output.length !== g.length){ $output = new Float64Array(g.length) }
for(let i=0; i<g.length; i++){ $output[i] = g[i] + 1 }
return $output"""
DICT_PAGE = dict(figures=[[["x"], ["g", "L", "U", "K"]]],
                 aliases=[("g", "x*pa"), {"name": "L", "variables": ["g"], "func": "return g+1"},
                          {"name": "U", "v_func": U_G}, ("K", "x*2")],
                 widgets=[["slider", ["pa"]]], parameterArray=[{"name": "pa", "value": 2.0, "range": [0, 5]}])
ADD_DICT_ALIASES_JS = """(a) => {
  const doc = Bokeh.documents[Bokeh.documents.length - 1];
  const m = Array.from(doc._all_models.values()).filter(x => x.name === a.table && x.constructor.__name__ === "CDSAlias");
  if (m.length !== 1) throw new Error("CDSAlias models named " + a.table + ": " + m.length);
  const each = (n, f) => Float64Array.from({length: n}, (_, i) => f(i));
  const add = {                              // name: [fields as a dictionary, what the test transform computes]
    D1: [{lhs: "g"},          f => each(f.lhs.length, i => f.lhs[i] + 1)],
    D2: [{terms: ["g", "x"]}, f => each(f.terms[0].length, i => f.terms[0][i] + f.terms[1][i])],
    D3: [{lhs: "g", k: "3"},  f => each(f.lhs.length, i => f.lhs[i] * f.k[i])],         // "3": a constant column
    D4: [{lhs: "x"},          f => each(f.lhs.length, i => f.lhs[i] + 1)],               // does not read g
    A1: [{lhs: "g"},          f => Promise.resolve(each(f.lhs.length, i => f.lhs[i] * 10))],   // as OrtFunction
    A2: [{lhs: "A1"},         f => each(f.lhs.length, i => f.lhs[i] + 1)],
  };
  for (const [name, [fields, fn]] of Object.entries(add))
    m[0].mapping[name] = {fields, transform: {id: "ri_test_transform_" + name, parameters: {}, v_compute: fn}};
  return Object.keys(m[0].mapping).sort();
}"""
DICT_ALIASES = ["A1", "A2", "D1", "D2", "D3", "D4"]


def _dict_page(node_session, tmp_path, *step_lists):
    """The page with the dictionary aliases added to the track table's production alias model, then the replies to
    each list of steps (one request each)."""
    err = _build_error(tmp_path, TR, DICT_PAGE["figures"], DICT_PAGE["widgets"], sourceArray=EV_SOURCE,
                       aliasArray=DICT_PAGE["aliases"], parameterArray=DICT_PAGE["parameterArray"])
    assert err is None, err
    _request(node_session, "new_doc", path=str(Path(tmp_path) / "aliases_document.json"))
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    keys = decode(_request(node_session, "eval", code=ADD_DICT_ALIASES_JS, arg={"table": "cdsFull"})["result"])
    assert set(DICT_ALIASES) <= set(keys), keys
    return [_steps(node_session, steps) for steps in step_lists]


def _dict_oracle(pa):
    g = X * pa
    return {"g": g, "L": g + 1, "U": g + 1, "K": X * 2, "D1": g + 1, "D2": g + X, "D3": g * 3, "D4": X + 1}


DICT_READS_G = ("D1", "D2", "D3")


def _right(reply, want: dict, keys) -> bool:
    return all(_column(reply, k)[0] is not None and _same(_column(reply, k)[0], want[k]) for k in keys)


def verdict_dictionary_inputs(before, s, cache, after) -> str:
    """A-D11 (review B3-C1), synchronous. g = x*pa; L (inputs as a list) and U (automatic inputs) read g; D1, D2, D3
    read g through a dictionary ({"lhs": "g"}, {"terms": ["g", "x"]}, {"lhs": "g", "k": "3"}); K and D4 do not read g.
    OK: before (pa = 2) every column equals numpy; the slider's callbacks ran without error; right after the change to
    pa = 3 none of g, L, U, D1, D2, D3 is in the cache, while K and D4 still are (side effect: an alias that does not
    read the changed input is not thrown away); after it every column equals numpy with pa = 3. KNOWN (A-D11): all of
    that, except that D1, D2 and D3 are still in the cache and equal exactly their values before the change. Test
    review correction (finding G5a-F2): no column threw in either read, for OK and KNOWN alike."""
    if not _clean(before, after):
        return UNEXPLAINED
    names = list(_dict_oracle(2.0))
    if not _right(before, _dict_oracle(2.0), names) or s["callbacks"] < 1 or s["errors"]:
        return UNEXPLAINED
    now = set(cache["cached"])
    if not {"K", "D4"} <= now or {"g", "L", "U"} & now:
        return UNEXPLAINED
    if not _right(after, _dict_oracle(3.0), ["g", "L", "U", "K", "D4"]):
        return UNEXPLAINED
    if _right(after, _dict_oracle(3.0), DICT_READS_G) and not set(DICT_READS_G) & now:
        return OK
    if set(DICT_READS_G) <= now and _right(after, {k: _column(before, k)[0] for k in DICT_READS_G}, DICT_READS_G):
        return KNOWN
    return UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.change_propagation"))
def test_strict_an_alias_with_inputs_as_a_dictionary_follows_its_input(node_session, tmp_path):
    """Review B3-C1: when g changes, every alias that reads it is recomputed - also one that lists its inputs as a
    dictionary, as an ONNX model alias does; aliases that do not read g keep their cached values."""
    names = list(_dict_oracle(2.0))
    (before, s, cache, after), = _dict_page(node_session, tmp_path,
                                            [read("cdsFull", *names), set_("pa", 3.0), cached("cdsFull"),
                                             read("cdsFull", *names)])
    verdict = verdict_dictionary_inputs(before, s, cache, after)
    _strict(verdict, "A-D11", (before, s, cache, after))


def verdict_dictionary_inputs_async(first, cache, after) -> str:
    """A-D11, asynchronous. A1 = g*10 is computed by a Promise (as OrtFunction's is), A2 = A1 + 1 reads A1 through a
    dictionary. The first read of A2 comes while A1's Promise is pending, so A2 is computed from A1's placeholder
    (zeros): 1 in every row - required, it shows the asynchronous path was taken. OK: once the Promise has settled, A1
    equals numpy, A2 is no longer in the cache, and reading it gives g*10 + 1. KNOWN (A-D11): A1 right, and A2 still in
    the cache with exactly 1 in every row. Test review correction (finding G5a-F2): no column threw in either
    read."""
    if not _clean(first, after):
        return UNEXPLAINED
    a2_first, _ = _column(first, "A2")
    if a2_first is None or not _same(a2_first, np.ones(N_TR)):
        return UNEXPLAINED
    g = X * 2.0
    if not _right(after, {"g": g, "A1": g * 10}, ["g", "A1"]):
        return UNEXPLAINED
    a2, _ = _column(after, "A2")
    if a2 is not None and _same(a2, g * 10 + 1) and "A2" not in cache["cached"]:
        return OK
    if a2 is not None and _same(a2, np.ones(N_TR)) and "A2" in cache["cached"]:
        return KNOWN
    return UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.change_propagation"))
def test_strict_an_alias_reading_an_asynchronous_alias_through_a_dictionary_is_recomputed(node_session, tmp_path):
    """Review B3-C1, the asynchronous path of CDSAlias (a transform that returns a Promise, as an ONNX model does):
    when the result arrives, the aliases that read it through a dictionary are recomputed."""
    (first,), (cache, after) = _dict_page(node_session, tmp_path, [read("cdsFull", "A2")],
                                          [cached("cdsFull"), read("cdsFull", "g", "A1", "A2")])
    verdict = verdict_dictionary_inputs_async(first, cache, after)
    _strict(verdict, "A-D11", (first, cache, after))


# ===================================================================================== 7. a chain without an event figure
def verdict_chain_without_an_event_figure(ev_keys, before, after) -> str:
    """Review B3-N1: ex2 = ex*2 and ex3 = ex2+1 on the event table, g3 = ev.ex3[evIdx] on the track table, and no figure
    of the event table, so only g3's own dependencies can send `ex`. OK: `ex` was sent, nothing threw, g3 equals numpy
    (orphans NaN), and after the event table's `ex` changes g3 equals numpy with the new values."""
    if "ex" not in ev_keys or not _clean(before, after):
        return UNEXPLAINED
    b, _ = _column(before, "g3")
    a, _ = _column(after, "g3")
    ok = b is not None and _same(b, EX3[EV_IDX]) and a is not None and _same(a, (EX_NEW * 2 + 1)[EV_IDX])
    return OK if ok else UNEXPLAINED


@_node(pytest.mark.feature("ALIAS.chain"), pytest.mark.feature("DSL.gather_operation"),
       pytest.mark.feature("ALIAS.change_propagation"))
def test_a_gather_chain_without_an_event_figure_sends_and_follows_its_column(node_session, tmp_path):
    """A compiled page may show only track-level quantities computed from event-level aliases."""
    err, out = _page(node_session, tmp_path, [[["x"], ["g3"]]],
                     [("ex2", "ex*2", "ev"), ("ex3", "ex2+1", "ev"), ("g3", "ev.ex3[evIdx]")],
                     [read("cdsFull", "g3"), {"op": "keys", "table": "ev"}])
    assert err is None, err
    after, = _mutate_then(node_session, "ev", "ex", EX_NEW, [read("cdsFull", "g3")])
    verdict = verdict_chain_without_an_event_figure(out[1]["keys"], out[0], after)
    assert verdict == OK, (verdict, out, after)


# ===================================================================================== 8. two tables that read each other
_rng_run = np.random.default_rng(20261115)
N_RUN = 6
RX = np.append(np.round(_rng_run.uniform(100.0, 200.0, N_RUN), 3), np.nan)        # the run table, fill row last
RX_NEW = np.append(RX[:N_RUN] + 1000.0, np.nan)
RUN_IDX = np.append(_rng_run.integers(0, N_RUN, N_EV), N_RUN).astype(np.int32)   # the event fill row -> the run fill row
EV_FIRST = np.array([np.flatnonzero(RUN_IDX[:N_EV] == r)[0] if np.any(RUN_IDX[:N_EV] == r) else FILL
                     for r in range(N_RUN)] + [FILL], dtype=np.int32)              # a run's first event
CYCLE_SOURCES = [{"name": "ev", "data": {"ex": EX, "ey": EY, "runIdx": RUN_IDX}},
                 {"name": "run", "data": {"rx": RX, "evFirst": EV_FIRST}}]
CYCLE_ALIASES = [("er", "run.rx[runIdx]", "ev"), ("re", "ev.ex[evFirst]", "run"), ("g", "ev.ex[evIdx]"),
                 ("gr", "ev.er[evIdx]")]
CYCLE_READS = [read("cdsFull", "g", "gr"), read("ev_full", "er"), read("run_full", "re")]


def _cycle_want(ex, rx):
    er = rx[RUN_IDX]
    return {"g": ex[EV_IDX], "gr": er[EV_IDX], "er": er, "re": ex[EV_FIRST]}


@_node(pytest.mark.feature("ALIAS.change_propagation"), pytest.mark.feature("DSL.gather_operation"))
def test_two_tables_whose_aliases_read_each_other_settle_after_a_change(node_session, tmp_path):
    """Side effect of the A-D6 fix (an alias table now listens to every table its aliases read): the event table reads
    the run table (er = run.rx[runIdx]) and the run table reads the event table (re = ev.ex[evFirst]), so each listens
    to the other. A change of either must end - every request answers, no endless loop - and every alias, also the
    track table's gather of a gathered alias (gr = ev.er[evIdx]), equals numpy after each change."""
    assert len(set(EV_FIRST[:N_RUN])) >= 4 and np.sum(RUN_IDX[:N_EV] == RUN_IDX[0]) >= 2      # non-vacuous
    err, first = _page(node_session, tmp_path, [[["x"], ["g", "gr"]], [["ev.ex"], ["ev.er"]], [["run.rx"], ["run.re"]]],
                       CYCLE_ALIASES, CYCLE_READS, sourceArray=CYCLE_SOURCES)
    assert err is None, err
    second = _mutate_then(node_session, "ev", "ex", EX_NEW, CYCLE_READS)
    third = _mutate_then(node_session, "run", "rx", RX_NEW, CYCLE_READS)
    for state, out, want in (("before", first, _cycle_want(EX, RX)), ("after ev.ex changed", second, _cycle_want(EX_NEW, RX)),
                             ("after run.rx changed", third, _cycle_want(EX_NEW, RX_NEW))):
        got = {**{k: _column(out[0], k)[0] for k in ("g", "gr")}, "er": _column(out[1], "er")[0],
               "re": _column(out[2], "re")[0]}
        for k, w in want.items():
            assert got[k] is not None and _same(got[k], w), (state, k, got[k], list(w))


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
    in another way. Batch 3 v0.2 (plan v0.2, rule 1): every condition of an OK has a wrong page that breaks only that
    condition - the reviewers' pages first (B3-C2: the right numbers without the column sent)."""
    g = EX[EV_IDX]
    nan = np.full(N_TR, np.nan)
    text = ["a"] * N_TR
    # A-D4: OK = ex sent, nothing thrown, g numbers, g values
    good, keys = _reply(g=g), ["evIdx", "ex", "ey"]
    sig = {"columns": {}, "kinds": {}, "throws": {"g": f"Cannot read properties of null (reading '{EV_IDX[0]}')"},
           "throw_kinds": {"g": "TypeError"}}
    assert verdict_gather_alone(keys, good) == OK and verdict_gather_alone(["evIdx", "ey"], sig) == KNOWN
    assert verdict_gather_alone(["evIdx", "ey"], good) == UNEXPLAINED                    # B3-C2: ex never sent
    assert verdict_gather_alone(keys, {**good, "throws": {"h": "x"}, "throw_kinds": {"h": "Error"}}) == UNEXPLAINED
    assert verdict_gather_alone(keys, _reply(g=text)) == UNEXPLAINED                    # not numbers
    assert verdict_gather_alone(keys, _reply(g=EY[EV_IDX])) == UNEXPLAINED              # the other column gathered
    assert verdict_gather_alone(keys, _reply(g=_shifted(g))) == UNEXPLAINED
    assert verdict_gather_alone(keys, _reply(g=np.where(EV_IDX == FILL, 0.0, g))) == UNEXPLAINED   # orphans not NaN
    assert verdict_gather_alone(keys, sig) == UNEXPLAINED                               # the column was sent after all
    assert verdict_gather_alone(["evIdx", "ey"], {**sig, "throws": {"g": "other error"}}) == UNEXPLAINED
    assert verdict_gather_alone(["evIdx", "ey"], {**sig, "throw_kinds": {"g": "Error"}}) == UNEXPLAINED
    # A-D1, R2: OK = g right, h right
    assert verdict_alias_over_gather(_reply(g=g, h=H_ORACLE)) == OK
    assert verdict_alias_over_gather(_reply(g=g, h=nan)) == KNOWN
    assert verdict_alias_over_gather(_reply(h=H_ORACLE)) == UNEXPLAINED                 # g missing
    assert verdict_alias_over_gather(_reply(g=text, h=H_ORACLE)) == UNEXPLAINED         # g not numbers
    assert verdict_alias_over_gather(_reply(g=_shifted(g), h=H_ORACLE)) == UNEXPLAINED  # g off
    assert verdict_alias_over_gather(_reply(g=g, h=_shifted(H_ORACLE))) == UNEXPLAINED
    assert verdict_alias_over_gather(_reply(g=g, h=g * 2)) == UNEXPLAINED              # "+x" lost
    assert verdict_alias_over_gather(_reply(g=nan, h=nan)) == UNEXPLAINED               # g wrong too
    assert verdict_alias_over_gather(_reply(g=g)) == UNEXPLAINED                        # h missing
    # A-D1 on the event table: OK = ex2 right, ex3 right
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=EX2, ex3=EX3)) == OK
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=EX2, ex3=_ad1_text(EX2))) == KNOWN
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=_shifted(EX2), ex3=EX3)) == UNEXPLAINED
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex3=EX3)) == UNEXPLAINED   # ex2 missing
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=EX2)) == UNEXPLAINED   # ex3 missing
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=EX2, ex3=_ad1_text(EX))) == UNEXPLAINED
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=EX2, ex3=_shifted(EX3))) == UNEXPLAINED
    assert verdict_alias_of_alias_on_the_related_table(_reply(ex2=_shifted(EX2), ex3=_ad1_text(EX2))) == UNEXPLAINED
    # A-D5, depth 3: OK = built, g3 right, h3 right
    assert verdict_chain_of_depth_3(None, _reply(g3=EX3[EV_IDX], h3=H3_ORACLE)) == OK
    assert verdict_chain_of_depth_3(AD5_CHAIN_SIGNATURE, None) == KNOWN
    assert verdict_chain_of_depth_3(("KeyError", "Column ex2 not found in data source None",
                                     "compileVarName.py:visit_Attribute"), None) == UNEXPLAINED
    assert verdict_chain_of_depth_3(None, _reply(g3=_shifted(EX3[EV_IDX]), h3=H3_ORACLE)) == UNEXPLAINED
    assert verdict_chain_of_depth_3(None, _reply(h3=H3_ORACLE)) == UNEXPLAINED          # g3 missing
    assert verdict_chain_of_depth_3(None, _reply(g3=EX3[EV_IDX], h3=nan)) == UNEXPLAINED  # A-D1 behind a fixed A-D5
    assert verdict_chain_of_depth_3(None, _reply(g3=EX2[EV_IDX], h3=EX2[EV_IDX] * X)) == UNEXPLAINED
    # A-D3: OK = built, before right, after right
    before, after = _reply(sg=g * X), _reply(sg=g * Y)
    assert verdict_selector_in_a_gather(None, before, after) == OK
    assert verdict_selector_in_a_gather(AD3_SIGNATURE, None, None) == KNOWN
    assert verdict_selector_in_a_gather(None, before, before) == UNEXPLAINED             # the selector did nothing
    assert verdict_selector_in_a_gather(None, after, after) == UNEXPLAINED               # wrong before the change
    assert verdict_selector_in_a_gather(None, before, _reply()) == UNEXPLAINED           # missing after
    assert verdict_selector_in_a_gather(("KeyError", "cdsOrig", "bokehTools.py:bokehDrawArray"), None, None) == UNEXPLAINED
    # A-D6: OK = before right, the table changed, after right (g and C)
    old, new = _reply(g=g, C=g + 1), _reply(g=EX_NEW[EV_IDX], C=EX_NEW[EV_IDX] + 1)
    ev_new = _reply(ex=EX_NEW)
    assert verdict_gather_follows_the_table(old, ev_new, new) == OK
    assert verdict_gather_follows_the_table(old, ev_new, old) == KNOWN
    assert verdict_gather_follows_the_table(_reply(g=np.full(N_TR, -9000.0), C=g + 1), ev_new, new) == UNEXPLAINED
    assert verdict_gather_follows_the_table(_reply(g=g, C=_shifted(g + 1)), ev_new, new) == UNEXPLAINED
    assert verdict_gather_follows_the_table(old, _reply(ex=EX), old) == UNEXPLAINED       # the table did not change
    assert verdict_gather_follows_the_table(old, _reply(), new) == UNEXPLAINED            # the table not read
    assert verdict_gather_follows_the_table(old, ev_new, _reply(g=EX_NEW[EV_IDX], C=g + 1)) == UNEXPLAINED  # C stale only
    assert verdict_gather_follows_the_table(old, ev_new, _reply(g=g, C=EX_NEW[EV_IDX] + 1)) == UNEXPLAINED
    # R-4: OK = built, ex right, ex2 right
    assert verdict_related_table_only(None, _reply(ex=EX, ex2=EX2)) == OK
    assert verdict_related_table_only(R4_SIGNATURE, None) == KNOWN
    assert verdict_related_table_only(("KeyError", "cdsFull", "bokehTools.py:bokehDrawArray"), None) == UNEXPLAINED
    assert verdict_related_table_only(None, _reply(ex=_shifted(EX), ex2=EX2)) == UNEXPLAINED
    assert verdict_related_table_only(None, _reply(ex=EX, ex2=_shifted(EX2))) == UNEXPLAINED
    # A-D11, synchronous: OK = before right, callbacks ran without error, K and D4 still cached, g L U not cached,
    # g L U K D4 right after, D1 D2 D3 right after and not cached
    b, s = _reply(**_dict_oracle(2.0)), {"callbacks": 1, "errors": []}
    a, cache = _reply(**_dict_oracle(3.0)), {"cached": ["D4", "K", "x"]}
    stale = _reply(**{**_dict_oracle(3.0), **{k: _dict_oracle(2.0)[k] for k in DICT_READS_G}})
    stale_cache = {"cached": ["D1", "D2", "D3", "D4", "K", "x"]}
    assert verdict_dictionary_inputs(b, s, cache, a) == OK
    assert verdict_dictionary_inputs(b, s, stale_cache, stale) == KNOWN
    assert verdict_dictionary_inputs(_reply(**{**_dict_oracle(2.0), "D2": _shifted(_dict_oracle(2.0)["D2"])}), s, cache, a) == UNEXPLAINED
    assert verdict_dictionary_inputs(b, {"callbacks": 0, "errors": []}, cache, a) == UNEXPLAINED   # the slider did nothing
    assert verdict_dictionary_inputs(b, {"callbacks": 1, "errors": ["boom"]}, cache, a) == UNEXPLAINED
    assert verdict_dictionary_inputs(b, s, {"cached": ["D4", "x"]}, a) == UNEXPLAINED     # K thrown away (side effect)
    assert verdict_dictionary_inputs(b, s, {"cached": ["K", "x"]}, a) == UNEXPLAINED      # D4 thrown away (side effect)
    assert verdict_dictionary_inputs(b, s, {"cached": ["D4", "K", "U"]}, a) == UNEXPLAINED  # U not invalidated
    assert verdict_dictionary_inputs(b, s, cache, _reply(**{**_dict_oracle(3.0), "L": _dict_oracle(2.0)["L"]})) == UNEXPLAINED
    assert verdict_dictionary_inputs(b, s, cache, _reply(**{**_dict_oracle(3.0), "D4": _shifted(X + 1)})) == UNEXPLAINED
    assert verdict_dictionary_inputs(b, s, cache, _reply(**{**_dict_oracle(3.0), "D3": _shifted(X * 9)})) == UNEXPLAINED
    assert verdict_dictionary_inputs(b, s, {"cached": ["D1", "D4", "K"]}, a) == UNEXPLAINED  # right, yet not invalidated
    assert verdict_dictionary_inputs(b, s, stale_cache, _reply(**{**_dict_oracle(3.0), "D1": _dict_oracle(2.0)["D1"]})) == UNEXPLAINED
    assert verdict_dictionary_inputs(b, s, stale_cache, _reply(**{**_dict_oracle(3.0), **{k: _shifted(_dict_oracle(2.0)[k]) for k in DICT_READS_G}})) == UNEXPLAINED
    # A-D11, asynchronous: OK = first A2 the placeholder, g and A1 right, A2 right and not cached
    g2 = X * 2.0
    first, a_cache, a_after = _reply(A2=np.ones(N_TR)), {"cached": ["A1", "g"]}, _reply(g=g2, A1=g2 * 10, A2=g2 * 10 + 1)
    assert verdict_dictionary_inputs_async(first, a_cache, a_after) == OK
    assert verdict_dictionary_inputs_async(first, {"cached": ["A1", "A2", "g"]}, _reply(g=g2, A1=g2 * 10, A2=np.ones(N_TR))) == KNOWN
    assert verdict_dictionary_inputs_async(_reply(A2=g2 * 10 + 1), a_cache, a_after) == UNEXPLAINED   # not asynchronous
    assert verdict_dictionary_inputs_async(first, a_cache, _reply(g=g2, A1=_shifted(g2 * 10), A2=g2 * 10 + 1)) == UNEXPLAINED
    assert verdict_dictionary_inputs_async(first, a_cache, _reply(g=_shifted(g2), A1=g2 * 10, A2=g2 * 10 + 1)) == UNEXPLAINED
    assert verdict_dictionary_inputs_async(first, {"cached": ["A1", "A2", "g"]}, a_after) == UNEXPLAINED  # right, yet cached
    assert verdict_dictionary_inputs_async(first, a_cache, _reply(g=g2, A1=g2 * 10, A2=np.ones(N_TR))) == UNEXPLAINED
    assert verdict_dictionary_inputs_async(first, a_cache, _reply(g=g2, A1=g2 * 10, A2=_shifted(g2 * 10 + 1))) == UNEXPLAINED
    # B3-N1: OK = ex sent, nothing thrown, g3 right before, g3 right after the change
    nb, na = _reply(g3=EX3[EV_IDX]), _reply(g3=(EX_NEW * 2 + 1)[EV_IDX])
    assert verdict_chain_without_an_event_figure(["evIdx", "ex"], nb, na) == OK
    assert verdict_chain_without_an_event_figure(["evIdx"], nb, na) == UNEXPLAINED          # ex not sent
    assert verdict_chain_without_an_event_figure(["evIdx", "ex"], {**nb, "throws": {"h": "x"}}, na) == UNEXPLAINED
    assert verdict_chain_without_an_event_figure(["evIdx", "ex"], nb, {**na, "throws": {"h": "x"}}) == UNEXPLAINED
    assert verdict_chain_without_an_event_figure(["evIdx", "ex"], _reply(g3=EX2[EV_IDX]), na) == UNEXPLAINED
    assert verdict_chain_without_an_event_figure(["evIdx", "ex"], nb, nb) == UNEXPLAINED     # did not follow the change
    assert verdict_chain_without_an_event_figure(["evIdx", "ex"], nb, _reply()) == UNEXPLAINED
    # the test review's wrong pages (GPT27, findings G5a-F1, G5a-F2): each verdict, each of its read replies in turn,
    # one more column that threw - never OK; on a KNOWN page that is not itself an error, never KNOWN
    def each_threw(verdict, pages, fixed=(), after_fixed=()):
        """The verdict's answer for `pages` with reply i thrown, for each i."""
        return [verdict(*fixed, *[_threw(p) if j == i else p for j, p in enumerate(pages)], *after_fixed)
                for i in range(len(pages))]
    cases = [   # (name, verdict, fixed args before, read pages, fixed args after, what the healthy pages give)
        ("A-D1 R2", verdict_alias_over_gather, (), [_reply(g=g, h=H_ORACLE)], (), OK),
        ("A-D1 R2 known", verdict_alias_over_gather, (), [_reply(g=g, h=nan)], (), KNOWN),
        ("A-D1 event", verdict_alias_of_alias_on_the_related_table, (), [_reply(ex2=EX2, ex3=EX3)], (), OK),
        ("A-D1 event known", verdict_alias_of_alias_on_the_related_table, (), [_reply(ex2=EX2, ex3=_ad1_text(EX2))], (), KNOWN),
        ("A-D5", verdict_chain_of_depth_3, (None,), [_reply(g3=EX3[EV_IDX], h3=H3_ORACLE)], (), OK),
        ("A-D3", verdict_selector_in_a_gather, (None,), [before, after], (), OK),
        ("A-D6", verdict_gather_follows_the_table, (), [old, ev_new, new], (), OK),
        ("A-D6 known", verdict_gather_follows_the_table, (), [old, ev_new, old], (), KNOWN),
        ("R-4", verdict_related_table_only, (None,), [_reply(ex=EX, ex2=EX2)], (), OK),
        ("B3-N1", verdict_chain_without_an_event_figure, (["evIdx", "ex"],), [nb, na], (), OK),
        ("A-D4", verdict_gather_alone, (keys,), [good], (), OK),
    ]
    for name, verdict, fixed, pages, after_fixed, healthy in cases:
        assert verdict(*fixed, *pages, *after_fixed) == healthy, name
        assert each_threw(verdict, pages, fixed, after_fixed) == [UNEXPLAINED] * len(pages), name
    # A-D11: the read replies are before and after (synchronous), first and after (asynchronous)
    for bb, aa, cc, healthy in ((b, a, cache, OK), (b, stale, stale_cache, KNOWN)):
        assert verdict_dictionary_inputs(bb, s, cc, aa) == healthy
        assert verdict_dictionary_inputs(_threw(bb), s, cc, aa) == UNEXPLAINED                # test review: before
        assert verdict_dictionary_inputs(bb, s, cc, _threw(aa)) == UNEXPLAINED                # test review: after
    for ff, cc, aa, healthy in ((first, a_cache, a_after, OK),
                                (first, {"cached": ["A1", "A2", "g"]}, _reply(g=g2, A1=g2 * 10, A2=np.ones(N_TR)), KNOWN)):
        assert verdict_dictionary_inputs_async(ff, cc, aa) == healthy
        assert verdict_dictionary_inputs_async(_threw(ff), cc, aa) == UNEXPLAINED             # test review: first
        assert verdict_dictionary_inputs_async(ff, cc, _threw(aa)) == UNEXPLAINED             # test review: after


@_unit
def test_falsifier_a_page_error_fails_every_batch3_request_path():
    """Batch 3 v0.2 (review B3-C5): the right numbers from a page that reported an error fail the test, on every request
    path of this file - the change of a table and its read (_mutate_then), the histogram counts (_bin_counts). A healthy
    page passes."""
    ok = {"ok": True, "page_errors": [], "console_errors": []}
    mutate = [{**ok, "result": N_EV + 1}, {**ok, "result": [_reply(g=EX_NEW[EV_IDX])]}]
    bins = [{**ok, "result": {}}, {**ok, "result": {"h1": [1.0, 2.0]}}]
    assert _mutate_then(_FakeSession(mutate), "ev", "ex", EX_NEW, [read("cdsFull", "g")])[0]["kinds"]["g"] == "number"
    assert list(_bin_counts(_FakeSession(bins), "page.json")["h1"]) == [1.0, 2.0]
    for call, replies in ((lambda s: _mutate_then(s, "ev", "ex", EX_NEW, [read("cdsFull", "g")]), mutate),
                          (lambda s: _bin_counts(s, "page.json"), bins)):
        for where in range(len(replies)):
            for key, error in (("page_errors", "uncaught exception: unrelated failure"), ("console_errors", "boom")):
                bad = [dict(r) for r in replies]
                bad[where] = {**bad[where], key: [error]}
                with pytest.raises(pytest.fail.Exception, match="the page reported errors"):
                    call(_FakeSession(bad))
