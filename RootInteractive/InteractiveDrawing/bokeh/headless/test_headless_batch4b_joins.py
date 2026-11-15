"""PHASE_5 Batch 4.b: tests that come WITH the fix of the join (CDSJoin.ts). They cover what the fix adds and what the
tests written before it (Batch 4.a) did not reach.

In plain words (the words are those of test_headless_batch4a_joins.py: a join shows two tables side by side; the CHILD
holds the pointer, the PARENT is pointed to; an INDEX join pairs rows by a row number, a KEY join by equal keys):
  1. the reverse direction of an index join (the row numbers are held by the table listed on the right) on the ToyMC
     tables: it must show the same rows as the same relation written the usual way round;
  2. a row number pointing into a chain of joins that lost rows at TWO levels still means the original table
     (Architect, 2026-11-15), and a row number past the end of that original table is refused by name (L-33);
  3. values that are not a row number at all;
  4. a join whose pointer or key column does not exist;
  5. a key join on a real dashboard page, built by bokehDrawSA as a user builds it.

Decided, and not decided. Points 1 and 2 follow decisions. In points 3 and 4 only "-1 means no parent" is decided.
The rest is what the fix does today, so that a later change is seen; each such test says so, and the open questions
are listed in the change report of the fix:
  * NaN or no value as a row number: no parent (the same rule as a NaN key, L-34: it cannot be associated);
  * another negative number, or a fraction, as a row number: no parent (what a left join gave before the fix);
  * a pointer or key column that does not exist: an empty join and a console warning (as before the fix for an index
    join; a key join gave rows paired by accident).

The reference is plain Python on the source tables, never the join itself.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4a_joins import (
    NAN, PARENT, _child, _clean_reply, _clean_session, _index_join, _is_refusal, _link, _read, _reference_chain, _same,
    _sources, _verdict)
from toy_mc_events import generate_toy_mc_events, inject_orphan_test_rows    # the path is set by the import above

pytestmark = [pytest.mark.node_joins]
nan = float("nan")


@pytest.fixture(scope="module")
def physics():
    return generate_toy_mc_events(n_collisions=10, seed=42)


# ============================================================================================ 1. the reverse direction
REVERSE_KEYS = ["tRow", "tPt", "tOrigin", "cRow", "cVz", "cNTracks"]         # three track and three collision columns


@pytest.mark.feature("JOIN.cdsjoin.reverse_index")
@pytest.mark.parametrize("rows", ["every track has a collision", "a track without a collision"])
@pytest.mark.parametrize("how", ["inner", "right"])
def test_the_reverse_index_join_equals_the_same_relation_written_forward(node_session, tmp_path, physics, rows, how):
    """Tracks hold the row number of their collision. Forward: tracks on the left. Reverse: collisions on the left,
    tracks on the right, `on_right`. Both must show one row per track, in the track table's order. `right` keeps a
    track without a collision (collision columns NaN), as `left` does forward; `inner` drops it."""
    src = _sources(physics if rows == "every track has a collision" else inject_orphan_test_rows(physics))
    forward = {"TE": _link("TR", "CO", "tEventIdx", "left" if how == "right" else "inner")}
    want = _reference_chain(src, forward, "TE", REVERSE_KEYS)
    reverse = {"ET": dict(left="CO", right="TR", on_left=[], on_right=["tEventIdx"], how=how)}
    out = _read(node_session, tmp_path, src, reverse, "ET", REVERSE_KEYS)
    assert _verdict(out, REVERSE_KEYS, want, None) == "OK"
    if rows == "a track without a collision":                            # the two modes really differ here
        n = len(src["TR"]["tRow"])
        assert src["TR"]["tEventIdx"].count(-1) == 1 and out["length"] == (n if how == "right" else n - 1)
        assert (NAN in want["cVz"]) == (how == "right")


# ============================================================================================ 2. rows lost at two levels
# V0 1 has no positive track (lost by the first inner join). V0 3 points to track 2, which has no collision (lost by
# the second inner join). So the chain (V0 -> track) -> collision keeps the V0 rows 0, 2, 4, 5 - at the positions
# 0, 1, 2, 3. The cascades point to V0 rows by their ORIGINAL numbers.
CO = {"cRow": [0, 1, 2], "cVz": [0.5, 1.5, 2.5]}
TR = {"tRow": [0, 1, 2, 3, 4], "tEventIdx": [-1, 0, -1, 1, 2], "tPt": [10.0, 11.0, 12.0, 13.0, 14.0]}
V0 = {"vRow": [0, 1, 2, 3, 4, 5], "vPos": [1, -1, 3, 2, 4, 3], "vMass": [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]}
CA = {"caRow": [0, 1, 2, 3, 4], "caV0": [5, 1, 3, 2, 0], "caMass": [1.30, 1.31, 1.32, 1.33, 1.34]}
TWO_LEVELS = {"CO": CO, "TR": TR, "V0": V0, "CA": CA}
TWO_KEYS = ["caRow", "caMass", "vRow", "vMass", "tRow", "tPt", "cVz"]


def _two_level_joins(how):
    return {"VT": _link("V0", "TR", "vPos", "inner"), "VTE": _link("VT", "CO", "tEventIdx", "inner"),
            "CVTE": _link("CA", "VTE", "caV0", how)}


def _two_level_reference(tables, how, by_position=False):
    """Plain Python on the original row numbers. `by_position`: the WRONG page of a join that takes the cascade's row
    number as a position in the shortened table."""
    co, tr, v0, ca = (tables[k] for k in ("CO", "TR", "V0", "CA"))
    kept = [v for v, t in enumerate(v0["vPos"]) if t != -1 and tr["tEventIdx"][t] != -1]
    out = {k: [] for k in TWO_KEYS}
    for row, pointer in enumerate(ca["caV0"]):
        if by_position:
            v = kept[pointer] if 0 <= pointer < len(kept) else None
        else:
            v = pointer if pointer in kept else None
        if v is None and how == "inner":
            continue
        t = None if v is None else v0["vPos"][v]
        out["caRow"].append(ca["caRow"][row]); out["caMass"].append(ca["caMass"][row])
        out["vRow"].append(NAN if v is None else v0["vRow"][v]); out["vMass"].append(NAN if v is None else v0["vMass"][v])
        out["tRow"].append(NAN if v is None else tr["tRow"][t]); out["tPt"].append(NAN if v is None else tr["tPt"][t])
        out["cVz"].append(NAN if v is None else co["cVz"][tr["tEventIdx"][t]])
    return out


@pytest.mark.feature("JOIN.recursive")
@pytest.mark.parametrize("how", ["inner", "left"])
def test_a_row_number_means_the_original_table_through_two_levels_of_lost_rows(node_session, tmp_path, how):
    want = _two_level_reference(TWO_LEVELS, how)
    out = _read(node_session, tmp_path, TWO_LEVELS, _two_level_joins(how), "CVTE", TWO_KEYS)
    assert _verdict(out, TWO_KEYS, want, None) == "OK"


@pytest.mark.feature("JOIN.recursive")
def test_a_row_number_past_the_end_of_the_original_table_is_refused_with_its_size(node_session, tmp_path):
    """The shortened chain has 4 rows, the original V0 table 6. Row number 4 is inside the original table (V0 4 is
    kept): no refusal. Row number 6 is past its end: refused, and the message gives the original size."""
    inside = {**TWO_LEVELS, "CA": {**CA, "caV0": [4, 4, 4, 4, 4]}}
    out = _read(node_session, tmp_path, inside, _two_level_joins("inner"), "CVTE", ["vRow"])
    assert out.get("throws") == {} and out["columns"]["vRow"] == [4, 4, 4, 4, 4]
    past = {**TWO_LEVELS, "CA": {**CA, "caV0": [5, 1, 6, 2, 0]}}
    out = _read(node_session, tmp_path, past, _two_level_joins("inner"), "CVTE", ["vRow"])
    assert _is_refusal(out, ["vRow"], "CVTE", "caV0", 6), out
    assert all("(6 rows)" in message for message in out["throws"].values()), out["throws"]


def test_the_two_level_reference_uses_original_row_numbers():
    inner = _two_level_reference(TWO_LEVELS, "inner")
    assert inner["caRow"] == [0, 3, 4] and inner["vRow"] == [5, 2, 0] and inner["tRow"] == [3, 3, 1]
    assert inner["cVz"] == [1.5, 1.5, 0.5]
    left = _two_level_reference(TWO_LEVELS, "left")
    assert left["caRow"] == CA["caRow"] and left["vRow"] == [5, NAN, NAN, 2, 0]
    for how in ("inner", "left"):                                 # the wrong page, by position, is another page
        assert not _same(_two_level_reference(TWO_LEVELS, how, by_position=True), _two_level_reference(TWO_LEVELS, how))


# ============================================================================================ 3. not a row number
NOT_A_ROW_NUMBER = [2, -1, -5, nan, None, 1.5, 0]            # only the first and the last are row numbers


@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.parametrize("how", ["left", "inner"])
def test_a_value_that_is_not_a_row_number_means_no_parent(node_session, tmp_path, how):
    """-1 is the agreed mark of "no parent". NaN and no value follow the rule of a NaN key. -5 and 1.5 are NOT decided:
    the fix gives "no parent" for them (see the top of this file). None is refused, and none is an exception."""
    tables = {"CH": _child(NOT_A_ROW_NUMBER), "PA": PARENT}
    if how == "left":
        want = {"cid": [0, 1, 2, 3, 4, 5, 6], "x": [2.5, NAN, NAN, NAN, NAN, NAN, 0.5], "n": [9, NAN, NAN, NAN, NAN, NAN, 7]}
    else:
        want = {"cid": [0, 6], "x": [2.5, 0.5], "n": [9, 7]}
    out = _read(node_session, tmp_path, tables, _index_join(how), "J", ["cid", "x", "n"])
    assert _verdict(out, ["cid", "x", "n"], want, None) == "OK"


# ============================================================================================ 4. a column that is not there
@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.parametrize("kind", ["index join", "key join, child key", "key join, parent key"])
def test_a_join_on_a_column_that_does_not_exist_is_empty(node_session, tmp_path, kind):
    """NOT decided (see the top of this file): what the fix does today. The join is empty, no read throws, and the
    browser console gets a warning. Before the fix a key join paired rows here by accident."""
    tables = {"CH": _child([2, 1, 0]), "PA": PARENT}
    on = {"index join": (["nowhere"], []), "key join, child key": (["nowhere"], ["pid"]),
          "key join, parent key": (["cid"], ["nowhere"])}[kind]
    joins = {"J": dict(left="CH", right="PA", on_left=on[0], on_right=on[1], how="left")}
    out = _read(node_session, tmp_path, tables, joins, "J", ["cid", "x"])
    assert _verdict(out, ["cid", "x"], {"cid": [], "x": []}, None) == "OK"


# ============================================================================================ 5. a real dashboard page
DASHBOARD_READ = """(a) => {
  const doc = Bokeh.documents[Bokeh.documents.length - 1];
  const of = (kind, name) => Array.from(doc._all_models.values()).filter(m => m.constructor.__name__ === kind && m.name === name);
  const joins = of("CDSJoin", "joinA"), histograms = of("HistogramCDS", "histo0");
  if (joins.length !== 1 || histograms.length !== 1) return {error: "the page must hold one join joinA and one histogram histo0"};
  const join = joins[0], histogram = histograms[0];
  const out = {length: join.get_length(), columns: {}, sides: [join.left.constructor.__name__, join.right.constructor.__name__]};
  for (const k of a.keys) out.columns[k] = Array.from(join.get_column(k));
  out.bin_count = Array.from(histogram.get_column("bin_count"));
  return out;
}"""


@pytest.mark.feature("JOIN.cdsjoin.basic")
def test_a_key_join_on_a_real_dashboard_page(node_session, tmp_path):
    """The page of test_ClientSideJoin.py: a second table df2 (X = 0 ... 9999, Y = sqrt(X)) joined by key to the main
    table (A = the squares 0, 1, 4, ... 9801), and a histogram of Y weighted by A that reads the join. bokehDrawSA
    builds it, so both sides of the join are alias tables (CDSAlias), not plain tables. The join must have the 100
    rows whose X is a square, each with its own Y and A. Before the fix it had 2 rows: X = 2 has no partner, and the
    merge then moved on in the wrong table (J-D4)."""
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    x = np.arange(10000)
    squares = np.square(np.arange(100))
    sources = [{"name": "df2", "data": {"X": x, "Y": np.sqrt(x)}},
               {"name": "joinA", "left": "df2", "right": None, "left_on": ["X"], "right_on": ["A"]},
               {"name": "histo0", "variables": ["Y"], "weights": "A", "source": "joinA"}]
    figures = [[["df2.X"], ["df2.Y"]], [["joinA.X"], ["joinA.df2.X"]], [["histo0.bin_center"], ["histo0.bin_count"]]]
    shown = sa.show
    sa.show = lambda *a, **k: None                       # build only: nothing written, nothing compiled
    try:
        page = sa.bokehDrawSA.fromArray(pd.DataFrame({"A": squares}), None, figures, [["range", ["A"]]], sourceArray=sources,
                                        layout=[[0, 1, {"height": 200}], [2, {"height": 50}], {"sizing_mode": "scale_width"}],
                                        widgetLayout=[[0]])
    finally:
        sa.show = shown
    doc = Document()
    doc.add_root(page.pAll)
    path = tmp_path / "dashboard_with_a_join.json"
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    _clean_session(node_session)
    keys = ["X", "df2.X", "Y", "A"]
    for op, kw in (("new_doc", dict(path=str(path))), ("eval", dict(code=DASHBOARD_READ, arg={"keys": keys}))):
        reply = node_session.request(op, **kw)
        _clean_reply(reply, op)
    out = decode(reply["result"])
    assert "error" not in out, out
    assert out["sides"] == ["CDSAlias", "CDSAlias"]                    # the page really joins two alias tables
    want = {"X": squares.tolist(), "df2.X": squares.tolist(), "Y": np.arange(100.0).tolist(), "A": squares.tolist()}
    assert out["length"] != 2, "REGRESSION, a fixed defect is back: J-D4 on the dashboard page (2 rows)"
    assert out["length"] == 100 and _same(out["columns"], want), out
    assert sum(out["bin_count"]) == int(squares.sum()) == 328350       # the histogram counts every row of the join
