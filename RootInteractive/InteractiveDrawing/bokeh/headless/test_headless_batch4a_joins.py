"""PHASE_5 Batch 4.a, part 1: what a correct join is. Tests only; RootInteractive's code is not changed.

In plain words. A join (the browser model CDSJoin) shows the columns of two tables side by side. The CHILD table holds
the pointer (each track holds the row number of its collision, or a key value); the PARENT is the table pointed to.
An INDEX join pairs rows by a row number stored in the child; a KEY join pairs rows that have equal key values.
On 2026-11-15 the Architect decided what a correct join does in the cases below (PHASE_5 decision log v0.5):
  L-32  unsorted keys are joined correctly (the join sorts by itself);
  L-33  a row number past the end of the parent table is refused by name;
  L-34  a NaN key never matches, not even another NaN;
  L-35  a missing parent (row number -1, or a key without a partner) shows NaN in floating-point columns;
  L-36  ... and NaN in integer columns, null in text columns;
  L-37  with several keys the first listed key counts first (the join must work however the tables are sorted);
  L-39  the rows of a join keep the child table's order (a proposal; the reviewers have not been heard).
The tests also read every level of the ToyMC chains (cascade -> V0 -> track -> collision), where three registered
defects show: child columns read through the parent's row numbers (J-D1), an inner index join that loses the rows
pointing to row 0 (J-D2), and a join whose input is a join coming out empty (J-D5).

How a test decides. The production model is loaded in the Node harness and its columns are read. The reply is sorted:
  OK      it equals the REFERENCE: pandas or plain Python on the source tables, never the join itself;
  KNOWN   it equals, cell by cell, what today's code is known to give (the reviewed line-by-line copy of CDSJoin.ts
          from PHASE_4, class Port), and that differs from the reference;
  anything else is an ordinary failure.
A known defect is a strict expected failure: only KNOWN is accepted. After the fix the marks come off and each
case becomes an ordinary regression check. Every sorting function has replies that are wrong in one way and must be
rejected (the last section).

FIX COMMIT (PHASE_5 Batch 4, Claude21.RI, 2026-11-15): the join is fixed in CDSJoin.ts. Every strict mark of this file
is off, so each case is now an ordinary check: the reply must equal the reference. The result of the code before the
fix (KNOWN) is an ordinary failure that says a fixed defect is back. No test and no case is removed.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

# reused from PHASE_4: today's code as a line-by-line copy (Port), and the reviewed pandas reference (_oracle)
from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_node_joins import Port, _oracle

GEN_DIR = Path(__file__).resolve().parents[3] / "Tools" / "generators"
sys.path.insert(0, str(GEN_DIR))
from toy_mc_events import generate_toy_mc_events, inject_orphan_test_rows    # noqa: E402

pytestmark = [pytest.mark.node_joins]

UNDEF, NAN, NULL = "<undefined>", "<NaN>", "<null>"    # how the page's special values arrive here, each one distinct


class KnownJoinDefect(AssertionError):
    """Raised only when production shows exactly the registered wrong result of today."""


READ = """(a) => {
  const doc = Bokeh.documents[Bokeh.documents.length - 1];
  const m = doc.get_model_by_name(a.name);
  if (!m) return {error: "no model named " + a.name};
  const tok = v => v === undefined ? "<undefined>" : v === null ? "<null>"
    : (typeof v === "number" && Number.isNaN(v)) ? "<NaN>" : v === Infinity ? "<+inf>" : v === -Infinity ? "<-inf>" : v;
  const isObj = e => e !== null && typeof e === "object";
  const out = {columns: {}, throws: {}, kinds: {}};
  const note = (k, e) => { out.throws[k] = (isObj(e) && "message" in e) ? String(e.message) : String(e);
                           out.kinds[k] = (isObj(e) && "name" in e) ? String(e.name) : typeof e; };
  try { out.length = tok(m.get_length()); } catch (e) { note("length", e); }
  for (const k of a.keys) {
    try { const c = m.get_column(k); out.columns[k] = c == null ? tok(c) : Array.from(c, tok); } catch (e) { note(k, e); }
  }
  return out;
}"""


# ============================================================================================ harness: build, load, read
def _read(node_session, tmp_path, tables, joins, name, keys):
    """The production model's reply. Any harness problem, page error or console error fails the test."""
    from bokeh.document import Document
    from bokeh.models import ColumnDataSource
    from RootInteractive.InteractiveDrawing.bokeh.CDSJoin import CDSJoin
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    models = {n: ColumnDataSource(data={k: list(v) for k, v in cols.items()}, name=n) for n, cols in tables.items()}
    for n, s in joins.items():                                          # joins in dependency order (dict order)
        models[n] = CDSJoin(left=models[s["left"]], right=models[s["right"]], on_left=s.get("on_left", []),
                            on_right=s.get("on_right", []), how=s.get("how", "inner"),
                            prefix_left=s.get("prefix_left", ""), prefix_right=s.get("prefix_right", ""), name=n)
    doc = Document()
    for n in joins:
        doc.add_root(models[n])
    path = tmp_path / "batch4a_join_document.json"
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    _clean_session(node_session)
    for op, kw in (("new_doc", dict(path=str(path))), ("eval", dict(code=READ, arg={"name": name, "keys": list(keys)}))):
        reply = node_session.request(op, **kw)
        _clean_reply(reply, op)
    out = decode(reply["result"])
    assert "error" not in out, out
    return out


def _clean_session(session):
    """Errors the Node session collected while it started. They are stored on the session, not in a reply, so a
    clean reply alone does not show them (review finding of GPT25.RI)."""
    for kind in ("startup_page_errors", "startup_console_errors"):
        assert not getattr(session, kind, []), f"the Node session started with {kind}: {getattr(session, kind)}"


def _clean_reply(reply, op):
    assert reply.get("ok") is True, f"{op}: the Node harness failed: {reply}"
    assert not reply.get("page_errors"), f"{op}: page errors: {reply['page_errors']}"
    assert not reply.get("console_errors"), f"{op}: console errors: {reply['console_errors']}"


# ============================================================================================ sorting a reply
def _cell_equal(got, want):
    if isinstance(want, str) or isinstance(got, str) or isinstance(want, bool) or isinstance(got, bool):
        return type(got) is type(want) and got == want                 # text and the special-value marks: exact
    return isinstance(got, (int, float)) and isinstance(want, (int, float)) and \
        math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)


def _same(got, want, ordered=True):
    """Equal columns: the same names, the same length, the same cells. `ordered=False`: the same rows in any order."""
    if set(got) != set(want) or len({len(c) for c in got.values()} | {len(c) for c in want.values()}) != 1:
        return False
    keys = sorted(want)
    rows_g, rows_w = list(zip(*(got[k] for k in keys))), list(zip(*(want[k] for k in keys)))
    if not ordered:
        rows_g, rows_w = sorted(rows_g, key=repr), sorted(rows_w, key=repr)
    return all(_cell_equal(a, b) for rg, rw in zip(rows_g, rows_w) for a, b in zip(rg, rw))


def _verdict(out, keys, want, today, ordered=True):
    """OK, KNOWN, or an AssertionError. A reply with an exception, a missing column or a wrong length is never KNOWN."""
    assert out.get("throws") == {}, f"a read threw: {out.get('throws')}"
    assert set(out["columns"]) == set(keys), f"columns missing or extra: {sorted(out['columns'])}"
    got = {k: out["columns"][k] for k in keys}
    assert all(isinstance(c, list) for c in got.values()), f"a column is not a list: {got}"
    assert all(len(c) == out.get("length") for c in got.values()), \
        f"length {out.get('length')} against columns {[len(c) for c in got.values()]}"
    if _same(got, want, ordered):
        return "OK"
    if today is not None and not _same(today, want, ordered) and _same(got, today):
        return "KNOWN"
    raise AssertionError(f"neither the reference nor today's registered result:\n got   {got}\n want  {want}\n today {today}")


def _is_refusal(out, keys, join, column, value):
    """L-33: every read (the length and each column) throws a plain Error that names the join and the column in
    double quotes and the offending value as `row number <value>`. The value is matched in that place only: the same
    digits elsewhere in the message, for example in the size of the parent table, do not count (review finding of
    GPT25.RI, GPT27.RI and GPT28.RI)."""
    reads = {"length", *keys}
    if set(out.get("throws", {})) != reads or out.get("columns") or "length" in out:
        return False
    number = re.compile(rf"\brow number {re.escape(str(value))}(?![\d.])")
    return all(out["kinds"].get(r) == "Error" and f'"{join}"' in out["throws"][r] and f'"{column}"' in out["throws"][r]
               and number.search(out["throws"][r]) is not None for r in reads)


def _today(tables, joins, name, keys):
    """What today's code gives, from the PHASE_4 copy. Its None is JavaScript's undefined; a source NaN stays NaN."""
    port = Port(tables, joins)
    mark = lambda v: UNDEF if v is None else NAN if isinstance(v, float) and math.isnan(v) else v
    return {k: [mark(v) for v in port.column(name, k)] for k in keys}


def _reference_chain(tables, joins, name, keys):
    """The reference of an index-join chain: pandas gathers on the source tables. No parent: NaN, or null for text."""
    df = _oracle(tables, joins, name)
    text = {k for cols in tables.values() for k, v in cols.items() if any(isinstance(x, str) for x in v)}
    out = {}
    for k in keys:
        base = k.split("_", 1)[1] if k.split("_", 1)[0] in ("pos", "neg") else k
        fill = NULL if base in text else NAN
        out[k] = [fill if v is None or (isinstance(v, float) and math.isnan(v)) else v for v in df[k]]
    return out


def _reference_pairs(child_keys, parent_keys, how, tol=1e-5):
    """Row pairs (child row, parent row) of a key join by the decisions; -1 means no partner. Plain Python, no pandas:
    pandas `merge` pairs NaN with NaN, which L-34 forbids. Rows keep the child's order (L-39); parent rows without a
    child follow."""
    match = lambda a, b: all(not math.isnan(x) and not math.isnan(y) and abs(x - y) <= tol for x, y in zip(a, b))
    pairs, used = [], set()
    for i, a in enumerate(child_keys):
        js = [j for j, b in enumerate(parent_keys) if match(a, b)]
        used.update(js)
        pairs += [(i, j) for j in js] if js else ([(i, -1)] if how in ("left", "outer") else [])
    if how in ("right", "outer"):
        pairs += [(-1, j) for j in range(len(parent_keys)) if j not in used]
    return pairs


def _run(node_session, tmp_path, tables, joins, name, keys, want, ordered=True):
    out = _read(node_session, tmp_path, tables, joins, name, keys)
    return _verdict(out, keys, want, _today(tables, joins, name, keys), ordered)


def _strict(verdict, label):
    """Since the fix commit: today's-before-the-fix result is a regression, an ordinary failure that names the case."""
    if verdict == "KNOWN":
        raise KnownJoinDefect("REGRESSION, a fixed defect is back: " + label)


# ============================================================================================ small tables
PARENT = {"pid": [0, 1, 2], "x": [0.5, 1.5, 2.5], "n": [7, 8, 9], "s": ["a", "b", "c"]}


def _child(refs):
    return {"cid": list(range(len(refs))), "ref": list(refs)}


def _index_join(how):
    return {"J": dict(left="CH", right="PA", on_left=["ref"], on_right=[], how=how)}


@pytest.mark.feature("JOIN.cdsjoin.basic")
def test_strict_a_missing_parent_shows_nan_and_null_in_text(node_session, tmp_path):
    tables = {"CH": _child([2, -1, 0, -1]), "PA": PARENT}
    want = {"x": [2.5, NAN, 0.5, NAN], "n": [9, NAN, 7, NAN], "s": ["c", NULL, "a", NULL]}
    _strict(_run(node_session, tmp_path, tables, _index_join("left"), "J", ["x", "n", "s"], want),
            "a missing parent gives undefined in every column")


@pytest.mark.feature("JOIN.cdsjoin.basic")
def test_control_every_parent_present_reads_floats_integers_and_text(node_session, tmp_path):
    """The healthy side of the test above: with no missing parent the three kinds of column are right today."""
    tables = {"CH": _child([2, 1, 0, 1]), "PA": PARENT}
    want = {"x": [2.5, 1.5, 0.5, 1.5], "n": [9, 8, 7, 8], "s": ["c", "b", "a", "b"]}
    assert _run(node_session, tmp_path, tables, _index_join("left"), "J", ["x", "n", "s"], want) == "OK"


@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.parametrize("how", ["left", "inner"])
def test_strict_a_row_number_past_the_end_is_refused_by_name(node_session, tmp_path, how):
    tables = {"CH": _child([1, 3, 2]), "PA": PARENT}                     # 3 is past the end of a 3-row parent
    out = _read(node_session, tmp_path, tables, _index_join(how), "J", ["x"])
    if _is_refusal(out, ["x"], "J", "ref", 3):
        return
    today = _today(tables, _index_join(how), "J", ["x"])                 # left: undefined in that row; inner: row dropped
    assert out.get("throws") == {} and out.get("columns") == today and out.get("length") == len(today["x"]), \
        f"neither a refusal by name nor today's registered result: {out}"
    raise KnownJoinDefect("the row number past the end is accepted silently")


# -------------------------------------------------------------------------------------------- a parent that lost rows
# Decided by the Architect on 2026-11-15: an inner join may drop rows of the parent table (here tracks without a
# collision). A row number held by the child (here a V0) still means the ORIGINAL parent table, not the position in
# the shortened one, and the child row is kept only if its parent row is still there.
GRAND = {"cRow": [0, 1, 2], "cVz": [0.5, 1.5, 2.5]}
MIDDLE = {"tRow": [0, 1, 2, 3, 4], "tEventIdx": [-1, 0, -1, 1, 2], "tPt": [10.0, 11.0, 12.0, 13.0, 14.0]}   # rows 0, 2 lost
SMALL = {"vRow": [0, 1, 2, 3, 4, 5], "vPos": [1, 0, 3, 2, 4, 3], "vMass": [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]}
LOST = {"CO": GRAND, "TR": MIDDLE, "V0": SMALL}
LOST_KEYS = ["vRow", "vMass", "tRow", "tPt", "cVz"]


def _lost_joins(how_child):
    return {"TE": dict(left="TR", right="CO", on_left=["tEventIdx"], on_right=[], how="inner"),
            "VTE": dict(left="V0", right="TE", on_left=["vPos"], on_right=[], how=how_child)}


def _lost_reference(tables, how_child, position_in_the_shortened_table=False):
    """Plain Python on the ORIGINAL row numbers; it does not use the reused pandas reference, which renumbers the
    rows after dropping some. With `position_in_the_shortened_table` it gives the WRONG page a join would show if
    it took the child's row number as a position in the shortened parent - the mistake the test must catch."""
    tr, co, v0 = tables["TR"], tables["CO"], tables["V0"]
    kept = [t for t, e in enumerate(tr["tEventIdx"]) if e != -1]                # the track rows that survive
    out = {k: [] for k in LOST_KEYS}
    for row, pointer in enumerate(v0["vPos"]):
        if position_in_the_shortened_table:
            track = kept[pointer] if 0 <= pointer < len(kept) else None
        else:
            track = pointer if pointer in kept else None
        if track is None and how_child == "inner":
            continue
        out["vRow"].append(v0["vRow"][row]); out["vMass"].append(v0["vMass"][row])
        out["tRow"].append(NAN if track is None else tr["tRow"][track])
        out["tPt"].append(NAN if track is None else tr["tPt"][track])
        out["cVz"].append(NAN if track is None else co["cVz"][tr["tEventIdx"][track]])
    return out


@pytest.mark.feature("JOIN.recursive")
@pytest.mark.parametrize("how_child", ["inner", "left"])
def test_strict_a_row_number_means_the_original_parent_table(node_session, tmp_path, how_child):
    """Track rows 0 and 2 have no collision, so the inner track table loses its first row and one in the middle."""
    want = _lost_reference(LOST, how_child)
    _strict(_run(node_session, tmp_path, LOST, _lost_joins(how_child), "VTE", LOST_KEYS, want), f"lost rows, {how_child}")


def test_the_lost_rows_reference_uses_original_row_numbers():
    inner = _lost_reference(LOST, "inner")
    assert inner["vRow"] == [0, 2, 4, 5] and inner["tRow"] == [1, 3, 4, 3] and inner["cVz"] == [0.5, 1.5, 2.5, 1.5]
    left = _lost_reference(LOST, "left")
    assert left["vRow"] == SMALL["vRow"] and left["tRow"] == [1, NAN, 3, NAN, 4, 3]
    for how_child in ("inner", "left"):                           # the wrong page, by position, must be rejected
        want, shifted = _lost_reference(LOST, how_child), _lost_reference(LOST, how_child, True)
        today = _today(LOST, _lost_joins(how_child), "VTE", LOST_KEYS)
        assert not _same(shifted, want) and not _same(shifted, today) and not _same(today, want)
        good = {"length": len(want["vRow"]), "columns": want, "throws": {}}
        assert _verdict(good, LOST_KEYS, want, today) == "OK"
        with pytest.raises(AssertionError):
            _verdict({"length": len(shifted["vRow"]), "columns": shifted, "throws": {}}, LOST_KEYS, want, today)


# -------------------------------------------------------------------------------------------- key joins
def _key_tables(child_keys, parent_keys):
    """Child and parent with one column per key, an id each, and a value each. Keys are tuples."""
    n_keys = len(child_keys[0])
    ch = {"lid": list(range(len(child_keys))), "a": [10.0 + i for i in range(len(child_keys))]}
    pa = {"rid": list(range(len(parent_keys))), "b": [20.0 + j for j in range(len(parent_keys))]}
    for q in range(n_keys):
        ch[f"k{q}"] = [float(k[q]) for k in child_keys]
        pa[f"j{q}"] = [float(k[q]) for k in parent_keys]
    joins = lambda how: {"J": dict(left="CH", right="PA", on_left=[f"k{q}" for q in range(n_keys)],
                                   on_right=[f"j{q}" for q in range(n_keys)], how=how)}
    return {"CH": ch, "PA": pa}, joins


def _pair_columns(tables, pairs):
    ch, pa = tables["CH"], tables["PA"]
    return {"lid": [ch["lid"][i] if i >= 0 else NAN for i, _ in pairs], "a": [ch["a"][i] if i >= 0 else NAN for i, _ in pairs],
            "rid": [pa["rid"][j] if j >= 0 else NAN for _, j in pairs], "b": [pa["b"][j] if j >= 0 else NAN for _, j in pairs]}


nan = float("nan")
T = lambda *ks: [k if isinstance(k, tuple) else (k,) for k in ks]       # keys as tuples
KEY_CASES = {   # name: (child keys, parent keys, the decision it tests)
    "unsorted child":             (T(1, 3, 2), T(1, 2, 3), "L-32"),
    "unsorted parent":            (T(1, 2, 3), T(3, 1, 2), "L-32"),
    "both unsorted":              (T(2, 3, 1), T(3, 1, 2), "L-32"),
    "unsorted with a repeated key": (T(2, 1, 2, 3), T(3, 2, 1), "L-32"),
    "NaN key in the child":       (T(1, nan, 3), T(1, 2, 3), "L-34"),
    "NaN key in the parent":      (T(1, 2, 3), T(1, nan, 3), "L-34"),
    "NaN keys on both sides":     (T(1, nan, 3), T(1, nan, 3), "L-34"),
    "a key without a partner inside": (T(1, 2, 3, 4), T(1, 3, 4), "J-D4"),
    "two keys sorted by the first, a gap inside": (T((1, 0), (1, 1), (2, 0), (2, 1)), T((1, 1), (2, 1)), "L-37, J-D4"),
    "two keys sorted by the second, a gap inside": (T((1, 0), (2, 0), (1, 1), (2, 1)), T((1, 1), (2, 1)), "L-37, J-D4"),
    "two keys, the parent starts later": (T((1, 1), (2, 0)), T((2, 0),), "L-37"),
    "keys equal within the tolerance": (T(1.0, 2.0, 3.0), T(1.000004, 2.000004, 2.999996), "tolerance"),
    "keys apart by more than the tolerance": (T(1.0, 2.0), T(1.0001, 2.0001), "tolerance"),
}
KEY_MODES = ["inner", "left", "right", "outer"]
KEY_COLUMNS = ["lid", "a", "rid", "b"]                                   # two child columns, two parent columns
# The cases that are right today, measured on commit 7396c2f; all the others are strict expected failures.
EQUAL, APART = "keys equal within the tolerance", "keys apart by more than the tolerance"
RIGHT_TODAY_ANY_ORDER = {(EQUAL, "inner"), (EQUAL, "left"), (EQUAL, "right"), (EQUAL, "outer"), (APART, "inner")}
RIGHT_TODAY_CHILD_ORDER = {(EQUAL, "inner"), (EQUAL, "left"), (APART, "inner")}


def _key_case(node_session, tmp_path, case, how, ordered):
    child_keys, parent_keys, _ = KEY_CASES[case]
    tables, joins = _key_tables(child_keys, parent_keys)
    want = _pair_columns(tables, _reference_pairs(child_keys, parent_keys, how))
    return _run(node_session, tmp_path, tables, joins(how), "J", KEY_COLUMNS, want, ordered)


@pytest.mark.feature("JOIN.cdsjoin.outer")
@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.parametrize("case,how", [(c, h) for c in KEY_CASES for h in KEY_MODES if (c, h) not in RIGHT_TODAY_ANY_ORDER])
def test_strict_key_join_pairs_the_right_rows(node_session, tmp_path, case, how):
    """Child and parent columns together, the rows in any order: which child row is paired with which parent row."""
    _strict(_key_case(node_session, tmp_path, case, how, ordered=False), f"{case}, {how}")


@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.parametrize("case,how", sorted(RIGHT_TODAY_ANY_ORDER))
def test_control_key_join_pairs_that_are_right_today(node_session, tmp_path, case, how):
    """Sorted keys that all find their partner, or none: right today, and they must stay right after the fix."""
    assert _key_case(node_session, tmp_path, case, how, ordered=False) == "OK"


@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.parametrize("case,how", [(c, h) for c in KEY_CASES for h in ("inner", "left")
                                      if (c, h) not in RIGHT_TODAY_CHILD_ORDER])
def test_strict_key_join_rows_keep_the_childs_order(node_session, tmp_path, case, how):
    _strict(_key_case(node_session, tmp_path, case, how, ordered=True), f"{case}, {how}")


@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.parametrize("case,how", sorted(RIGHT_TODAY_CHILD_ORDER))
def test_control_key_join_child_order_right_today(node_session, tmp_path, case, how):
    assert _key_case(node_session, tmp_path, case, how, ordered=True) == "OK"


# ============================================================================================ the ToyMC chains
@pytest.fixture(scope="module")
def physics():
    return generate_toy_mc_events(n_collisions=10, seed=42)


def _sources(tables):
    c, t, v, ca = (tables[k] for k in ("collisions", "tracks", "v0s", "cascades"))
    col = lambda frame, name: [x.item() if isinstance(x, np.generic) else x for x in frame[name].to_numpy()]
    return {"CO": {"cRow": col(c, "rowId"), "cVz": col(c, "vz"), "cNTracks": col(c, "nTracks")},
            "TR": {"tRow": col(t, "rowId"), "tEventIdx": col(t, "collisionIdx"), "tPt": col(t, "pt"),
                   "tCharge": col(t, "charge"), "tOrigin": col(t, "origin")},
            "V0": {"vRow": col(v, "rowId"), "vPos": col(v, "posTrackIdx"), "vNeg": col(v, "negTrackIdx"),
                   "vMass": col(v, "mass"), "vOrigin": col(v, "origin")},
            "CA": {"caRow": col(ca, "rowId"), "caV0": col(ca, "v0Idx"), "caBachelor": col(ca, "bachelorTrackIdx"),
                   "caMass": col(ca, "mass")}}


def _link(left, right, key, how, **kw):
    return dict(left=left, right=right, on_left=[key], on_right=[], how=how, **kw)


def _chains(how):
    """Every chain of decision L-23. Right-nested: the parent side is a join. Left-nested: the child side is a join."""
    te = {"TE": _link("TR", "CO", "tEventIdx", how)}
    return {
        "track -> collision": (te, "TE", ["tRow", "tPt", "tOrigin", "cVz", "cNTracks"]),
        "V0 -> (track -> collision)": (
            {**te, "VPE": _link("V0", "TE", "vPos", how)},
            "VPE", ["vRow", "vMass", "tRow", "tPt", "tCharge", "cVz"]),
        "cascade -> (V0 -> (track -> collision))": (
            {**te, "VNE": _link("V0", "TE", "vNeg", how), "CVNE": _link("CA", "VNE", "caV0", how)},
            "CVNE", ["caRow", "caMass", "vRow", "vMass", "tRow", "tPt", "cVz"]),
        "cascade -> (bachelor -> collision)": (
            {**te, "CBE": _link("CA", "TE", "caBachelor", how)},
            "CBE", ["caRow", "caMass", "tRow", "tPt", "cVz"]),
        "(V0 -> track) -> collision": (
            {"VT": _link("V0", "TR", "vPos", how), "VTE": _link("VT", "CO", "tEventIdx", how)},
            "VTE", ["vRow", "vMass", "tRow", "tPt", "cVz"]),
        "V0 with both daughters and the collision": (
            {"VP": _link("V0", "TR", "vPos", how, prefix_right="pos_"),
             "VPN": _link("VP", "TR", "vNeg", how, prefix_right="neg_"),
             "VPNE": _link("VPN", "CO", "pos_tEventIdx", how)},
            "VPNE", ["vRow", "vMass", "pos_tRow", "pos_tPt", "neg_tRow", "neg_tPt", "cVz"]),
        "((cascade -> V0) -> track) -> collision": (
            {"CV": _link("CA", "V0", "caV0", how), "CVT": _link("CV", "TR", "vPos", how),
             "CVTE": _link("CVT", "CO", "tEventIdx", how)},
            "CVTE", ["caRow", "caMass", "vRow", "vMass", "tRow", "tPt", "cVz"]),
    }


CHAINS = list(_chains("left"))
RIGHT_NESTED_WITH_DROPPED_ROWS = {"V0 -> (track -> collision)", "cascade -> (V0 -> (track -> collision))",
                                  "cascade -> (bachelor -> collision)"}


def _chain(node_session, tmp_path, tables, chain, how):
    joins, target, keys = _chains(how)[chain]
    src = _sources(tables)
    return _run(node_session, tmp_path, src, joins, target, keys, _reference_chain(src, joins, target, keys))


@pytest.mark.feature("JOIN.recursive")
@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("how", ["left", "inner"])
def test_strict_every_level_of_a_chain_is_right(node_session, tmp_path, physics, chain, how):
    """Every table of the chain has a parent here, so `inner` and `left` must give the same rows."""
    _strict(_chain(node_session, tmp_path, physics, chain, how), f"{chain}, {how}")


@pytest.mark.feature("JOIN.recursive")
@pytest.mark.parametrize("chain,how", [(c, "left") for c in CHAINS]
                         + [(c, "inner") for c in CHAINS if c not in RIGHT_NESTED_WITH_DROPPED_ROWS])
def test_strict_every_level_of_a_chain_with_missing_parents(node_session, tmp_path, physics, chain, how):
    """The ToyMC with rows that have no parent (-1), also deeper in the chain. `left` keeps them, filled with NaN;
    `inner` drops them. An inner join INTO an inner join that dropped rows is left out HERE, because the reused
    pandas reference renumbers the rows after dropping some and is wrong for that case. It is tested with a
    reference of its own in `test_strict_a_row_number_means_the_original_parent_table`."""
    _strict(_chain(node_session, tmp_path, inject_orphan_test_rows(physics), chain, how), f"{chain}, {how}, missing parents")


# ============================================================================================ no Node: the checks themselves
def test_the_reference_of_a_chain_equals_a_direct_numpy_gather(physics):
    """The reference is not trusted blindly: for the deepest chain it equals plain NumPy indexing."""
    src = _sources(physics)
    joins, target, keys = _chains("left")["cascade -> (V0 -> (track -> collision))"]
    ref = _reference_chain(src, joins, target, keys)
    cv, neg, ev = (np.array(src[t][k]) for t, k in (("CA", "caV0"), ("V0", "vNeg"), ("TR", "tEventIdx")))
    assert len(cv) > 0 and ref["cVz"] == np.array(src["CO"]["cVz"])[ev[neg[cv]]].tolist()
    assert ref["tPt"] == np.array(src["TR"]["tPt"])[neg[cv]].tolist()
    assert ref["vMass"] == np.array(src["V0"]["vMass"])[cv].tolist()
    assert ref["caMass"] == src["CA"]["caMass"]
    orphans = _sources(inject_orphan_test_rows(physics))
    ref = _reference_chain(orphans, joins, target, ["cVz", "vMass"])
    assert NAN in ref["cVz"] and UNDEF not in ref["cVz"] and len(ref["cVz"]) == len(orphans["CA"]["caRow"])


def test_the_key_join_reference_follows_the_decisions():
    assert _reference_pairs(T(1, 3, 2), T(1, 2, 3), "inner") == [(0, 0), (1, 2), (2, 1)]            # L-32, child order
    assert _reference_pairs(T(1, nan, 3), T(1, nan, 3), "outer") == [(0, 0), (1, -1), (2, 2), (-1, 1)]   # L-34
    assert _reference_pairs(T((1, 1), (2, 0)), T((2, 0),), "inner") == [(1, 0)]                      # L-37
    assert _reference_pairs(T(1.0), T(1.000004), "inner") == [(0, 0)] and _reference_pairs(T(1.0), T(1.0001), "inner") == []


GOOD = {"length": 2, "columns": {"x": [0.5, NAN]}, "throws": {}, "kinds": {}}
WANT, TODAY = {"x": [0.5, NAN]}, {"x": [0.5, UNDEF]}


def test_the_verdict_accepts_only_the_reference_or_todays_exact_result():
    assert _verdict(GOOD, ["x"], WANT, TODAY) == "OK"
    assert _verdict({**GOOD, "columns": {"x": [0.5, UNDEF]}}, ["x"], WANT, TODAY) == "KNOWN"
    wrong = [{**GOOD, "columns": {"x": [0.5, NULL]}},                       # another mark for "missing"
             {**GOOD, "columns": {"x": [0.5, "<+inf>"]}},                   # infinity is not "missing"
             {**GOOD, "columns": {"x": [0.6, UNDEF]}},                      # today's mark, one cell changed
             {**GOOD, "columns": {"x": [0.5]}},                             # a row lost
             {**GOOD, "length": 3},                                        # the length disagrees with the columns
             {**GOOD, "columns": {}},                                      # the column is missing
             {**GOOD, "columns": {"x": [0.5, NAN], "y": [1, 2]}},           # an extra column
             {**GOOD, "throws": {"x": "boom"}},                            # right numbers, but a read threw
             {**GOOD, "columns": {"x": NULL}}]                             # not a list
    for reply in wrong:
        with pytest.raises(AssertionError):
            _verdict(reply, ["x"], WANT, TODAY)
    with pytest.raises(AssertionError):                                    # KNOWN needs a registered result ...
        _verdict({**GOOD, "columns": {"x": [0.5, UNDEF]}}, ["x"], WANT, None)
    with pytest.raises(AssertionError):                                    # ... that differs from the reference
        _verdict({**GOOD, "columns": {"x": [0.7, NAN]}}, ["x"], WANT, WANT)


def test_the_verdict_tells_row_order_from_row_content():
    swapped = {"length": 2, "columns": {"a": [2, 1], "b": [20.0, 10.0]}, "throws": {}}
    want = {"a": [1, 2], "b": [10.0, 20.0]}
    assert _verdict(swapped, ["a", "b"], want, None, ordered=False) == "OK"
    with pytest.raises(AssertionError):
        _verdict(swapped, ["a", "b"], want, None, ordered=True)
    with pytest.raises(AssertionError):                                    # rows mixed up between columns
        _verdict({**swapped, "columns": {"a": [2, 1], "b": [10.0, 20.0]}}, ["a", "b"], want, None, ordered=False)


def test_the_refusal_check_accepts_only_a_refusal_by_name():
    msg = 'CDSJoin "J": row number 3 in column "ref" is outside the parent table (3 rows)'
    good = {"columns": {}, "throws": {"length": msg, "x": msg}, "kinds": {"length": "Error", "x": "Error"}}
    assert _is_refusal(good, ["x"], "J", "ref", 3)
    wrong = [{**good, "kinds": {"length": "TypeError", "x": "TypeError"}},                 # an accident, not a refusal
             {**good, "throws": {"length": msg, "x": "something went wrong"}},            # does not say why
             {**good, "throws": {"length": msg.replace('"ref"', '"other"'), "x": msg}},   # names another column
             {**good, "throws": {"x": msg}, "kinds": {"x": "Error"}},                     # the length still answers
             {**good, "columns": {"x": [0.5]}},                                           # a column still answers
             {**good, "length": 3},
             {"columns": {"x": [1.5, UNDEF, 2.5]}, "throws": {}, "kinds": {}, "length": 3}]   # today's silent result
    for other in ("13", "30", "500", "3.5", "-3"):                 # an error about ANOTHER row number, same table size
        text = msg.replace("row number 3 ", f"row number {other} ")
        assert text != msg and "(3 rows)" in text
        wrong.append({**good, "throws": {"length": text, "x": text}})
    wrong.append({**good, "throws": {"length": msg.replace("row number 3 ", "a row number "), "x": msg}})   # no value
    assert not any(_is_refusal(reply, ["x"], "J", "ref", 3) for reply in wrong)
    assert _is_refusal({**good, "throws": {"length": msg.replace("(3 rows)", "(13 rows)"), "x": msg}}, ["x"], "J", "ref", 3)


class _Session:
    startup_page_errors, startup_console_errors = [], []


def test_errors_at_the_start_of_the_session_fail_a_test():
    """A reply can be clean although the page reported an error while it started; that must not count as healthy."""
    _clean_session(_Session())
    for kind in ("startup_page_errors", "startup_console_errors"):
        broken = _Session()
        setattr(broken, kind, ["ReferenceError: something failed while the page was loading"])
        with pytest.raises(AssertionError, match=kind):
            _clean_session(broken)
