"""Production CDSJoin in Node — J0 (mechanism) and J1 (single-level joins). Oracle and invariant tests only.

Each test builds source tables and CDSJoin models in Python (RootInteractive's own model class), loads the document in
the Node harness through Bokeh's deserializer, and reads the PRODUCTION model: its row pairs (`_indices_left`,
`_indices_right`), its columns (`get_column`) and its length (`get_length`). Every table carries a row-id column, so
the result can be compared with an independent oracle: pandas `merge`. Contract C-1 … C-5 of the PHASE_4_3 review
summary. RootInteractive's code is NOT changed (the Architect's rule): defects found are recorded below as exact
expected failures, each with a healthy control and a strict check that turns red the day the defect is fixed.

Findings (provisional ids, to receive B-numbers in the bug report):
  J-D1  left columns are gathered with the RIGHT table's row numbers (CDSJoin.ts get_column, `this._indices_right`
        for every column) — key joins and index joins, every mode.
  J-D2  an inner index join (the left table holds row numbers into the right) drops row 0 (`x > 0`, lines 92/103).
  J-D3  the reverse index join (the right table holds row numbers) is unreachable (line 75 catches on_left == []);
        the indices stay unset and every column read throws.
  J-H1  precondition: a key join is a sort-merge join ("So far, we assume sorted keys"); keys must be sorted
        ascending — unsorted input gives a silently wrong result, not an error.
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.node_joins]   # target (PHASE_4_3)

READ = """(a) => {
  const doc = Bokeh.documents[Bokeh.documents.length - 1];
  const m = doc.get_model_by_name(a.name);
  if (!m) return {error: "no model named " + a.name};
  const arr = x => (x == null) ? null : Array.from(x, v => (v === undefined || Number.isNaN(v)) ? null : v);
  const out = {il: arr(m._indices_left), ir: arr(m._indices_right), columns: {}, throws: {}};
  try { out.length = m.get_length(); } catch (e) { out.throws.length = String(e.message); }
  for (const k of a.keys) {
    try { out.columns[k] = arr(m.get_column(k)); } catch (e) { out.throws[k] = String(e.message); }
  }
  return out;
}"""


def _document(tmp_path: Path, tables: dict, joins: dict) -> Path:
    from bokeh.document import Document
    from bokeh.models import ColumnDataSource
    from RootInteractive.InteractiveDrawing.bokeh.CDSJoin import CDSJoin
    models = {n: ColumnDataSource(data={k: list(v) for k, v in cols.items()}, name=n) for n, cols in tables.items()}
    for n, spec in joins.items():
        models[n] = CDSJoin(left=models[spec["left"]], right=models[spec["right"]], on_left=spec.get("on_left", []),
                            on_right=spec.get("on_right", []), how=spec.get("how", "inner"), name=n)
    doc = Document()
    for n in joins:
        doc.add_root(models[n])
    path = tmp_path / "joins_document.json"
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return path


def _read(node_session, tables: dict, joins: dict, tmp_path: Path, keys, name="J") -> dict:
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = node_session.request("new_doc", path=str(_document(tmp_path, tables, joins)))
    assert r.get("ok"), r
    r = node_session.request("eval", code=READ, arg={"name": name, "keys": list(keys)})
    assert r.get("ok"), r
    out = decode(r["result"])
    assert "error" not in out, out
    return out


def _ids(xs):
    return [None if x is None or x < 0 else int(x) for x in xs]


def _oracle_pairs(left: dict, right: dict, on_left, on_right, how: str) -> Counter:
    """pandas merge as the oracle: the multiset of (left row, right row), None for the missing side."""
    m = pd.merge(pd.DataFrame(left), pd.DataFrame(right), left_on=on_left, right_on=on_right, how=how)
    return Counter(zip(_ids(m["lid"].fillna(-1)), _ids(m["rid"].fillna(-1))))


def _gather(column, rows):
    """column[row] for each joined row number (Node returns them as floats); None for a missing side or out of range."""
    return [None if r is None or r < 0 or r >= len(column) else column[int(r)] for r in rows]


# Non-trivial key tables, sorted ascending by key (precondition J-H1): a 2 x 1 and a 1 x 2 fan-out, row 0 of each
# side joined, an orphan on each side, and value columns that differ from the row ids.
KL = {"lid": [0, 1, 2, 3, 4], "k": [0, 1, 1, 2, 5], "a": [10.0, 11.0, 12.0, 13.0, 14.0]}
KR = {"rid": [0, 1, 2, 3, 4], "kr": [0, 1, 2, 2, 3], "b": [20.0, 21.0, 22.0, 23.0, 24.0]}
KEY = dict(left="L", right="R", on_left=["k"], on_right=["kr"])
MODES = ["inner", "left", "right", "outer"]


def _key_join(node_session, tmp_path, how, left=KL, right=KR):
    return _read(node_session, {"L": left, "R": right}, {"J": {**KEY, "how": how}}, tmp_path, ["lid", "rid", "a", "b"])


# ---------------------------------------------------------------------------------------------- J0 / C-1, C-4
@pytest.mark.parametrize("how", MODES)
def test_c1_key_join_row_pairs_equal_pandas(node_session, tmp_path, how):
    """C-1: the production join's row pairs equal pandas merge for every mode (fan-out, row 0, orphans)."""
    out = _key_join(node_session, tmp_path, how)
    got = Counter(zip(_ids(out["il"]), _ids(out["ir"])))
    assert got == _oracle_pairs(KL, KR, ["k"], ["kr"], how), (dict(got), dict(_oracle_pairs(KL, KR, ["k"], ["kr"], how)))


@pytest.mark.parametrize("how", MODES)
def test_c4_length_and_every_column_agree(node_session, tmp_path, how):
    """C-4: get_length equals the number of joined pairs, and every column has that length."""
    out = _key_join(node_session, tmp_path, how)
    n = sum(_oracle_pairs(KL, KR, ["k"], ["kr"], how).values())
    assert out["length"] == n and all(len(c) == n for c in out["columns"].values()), (out["length"], n)


def test_c5_row_order_within_equal_keys_does_not_change_the_join(node_session, tmp_path):
    """C-5 (invariant): permuting rows inside a run of equal keys (still sorted) leaves the joined id pairs unchanged."""
    base = _key_join(node_session, tmp_path, "inner")
    perm = {c: [v[0], v[2], v[1], v[3], v[4]] for c, v in KL.items()}                 # swap the two k == 1 rows
    out = _key_join(node_session, tmp_path, "inner", left=perm)
    to_ids = lambda o, tl: Counter(zip([tl["lid"][i] for i in _ids(o["il"])], _ids(o["ir"])))
    assert to_ids(out, perm) == to_ids(base, KL)


# ---------------------------------------------------------------------------------------------- C-2 values
@pytest.mark.parametrize("how", MODES)
def test_c2_right_columns_are_the_right_table_at_the_joined_rows(node_session, tmp_path, how):
    """C-2: every right column equals the right table at the joined right rows (missing for an orphan)."""
    out = _key_join(node_session, tmp_path, how)
    for col in ("rid", "b"):
        assert out["columns"][col] == _gather(KR[col], out["ir"]), col


@pytest.mark.parametrize("how", MODES)
def test_known_defect_j_d1_left_columns_use_the_right_row_numbers(node_session, tmp_path, how):
    """J-D1, exact signature: each left column equals the LEFT table read at the RIGHT row numbers — and that differs
    from the correct values (the left table at the left row numbers) on these tables."""
    out = _key_join(node_session, tmp_path, how)
    for col in ("lid", "a"):
        signature, correct = _gather(KL[col], out["ir"]), _gather(KL[col], out["il"])
        assert out["columns"][col] == signature and signature != correct, (col, out["columns"][col], correct)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="J-D1: left columns gathered with the right row numbers")
@pytest.mark.parametrize("how", MODES)
def test_left_columns_are_the_left_table_at_the_joined_rows(node_session, tmp_path, how):
    """The correct behaviour (C-2) — fails today because of J-D1; turns red (strict XPASS) the day it is fixed."""
    out = _key_join(node_session, tmp_path, how)
    assert all(out["columns"][c] == _gather(KL[c], out["il"]) for c in ("lid", "a"))


def test_healthy_control_j_d1_cannot_show_when_both_sides_align(node_session, tmp_path):
    """Control: a one-to-one join of identical keys gives equal left and right row numbers, so J-D1 cannot show and
    the left columns are right — the check is not failing for another reason."""
    left = {"lid": [0, 1, 2], "k": [1, 2, 3], "a": [10.0, 11.0, 12.0]}
    right = {"rid": [0, 1, 2], "kr": [1, 2, 3], "b": [20.0, 21.0, 22.0]}
    out = _key_join(node_session, tmp_path, "inner", left=left, right=right)
    assert out["il"] == out["ir"] and out["columns"]["a"] == left["a"] and out["columns"]["b"] == right["b"]


# ---------------------------------------------------------------------------------------------- J-H1 precondition
def test_precondition_j_h1_unsorted_keys_give_a_silently_wrong_join(node_session, tmp_path):
    """J-H1: the key join assumes keys sorted ascending. Unsorted input raises nothing and differs from pandas.
    If the join learns to handle unsorted input, this test turns red so the record is updated."""
    right = {"rid": [0, 1, 2, 3, 4], "kr": [2, 0, 3, 1, 2], "b": [20.0, 21.0, 22.0, 23.0, 24.0]}
    out = _key_join(node_session, tmp_path, "inner", right=right)
    got = Counter(zip(_ids(out["il"]), _ids(out["ir"])))
    assert not out["throws"] and got != _oracle_pairs(KL, right, ["k"], ["kr"], "inner")


# ---------------------------------------------------------------------------------------------- C-3 index joins
IL = {"lid": [0, 1, 2, 3, 4, 5], "idx": [0, 2, -1, 4, 1, 7], "a": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}
IR = {"rid": [0, 1, 2, 3, 4], "b": [20.0, 21.0, 22.0, 23.0, 24.0]}
INDEX = dict(left="L", right="R", on_left=["idx"], on_right=[])


def _index_join(node_session, tmp_path, how, left=IL):
    return _read(node_session, {"L": left, "R": IR}, {"J": {**INDEX, "how": how}}, tmp_path, ["lid", "rid", "a", "b"])


def _index_oracle(left, how):
    """Row i of the left table joins right row idx[i] when 0 <= idx[i] < len(right); otherwise it is an orphan."""
    n = len(IR["rid"])
    pairs = [(i, x if 0 <= x < n else None) for i, x in enumerate(left["idx"])]
    return [p for p in pairs if p[1] is not None] if how == "inner" else pairs


def test_c3_left_index_join_keeps_every_left_row_with_its_right_values(node_session, tmp_path):
    """C-3, left join: one row per left row; the right values follow idx (row 0 included), missing for -1 and 7."""
    out = _index_join(node_session, tmp_path, "left")
    expected_right = [r for _, r in _index_oracle(IL, "left")]
    assert out["length"] == len(IL["lid"]) and out["columns"]["rid"] == _gather(IR["rid"], [-1 if r is None else r for r in expected_right])


def test_known_defect_j_d2_inner_index_join_drops_row_0(node_session, tmp_path):
    """J-D2, exact signature: the inner index join keeps the valid right rows except row 0."""
    out = _index_join(node_session, tmp_path, "inner")
    correct = Counter(r for _, r in _index_oracle(IL, "inner"))
    signature = correct - Counter({0: correct[0]})
    assert Counter(out["columns"]["rid"]) == signature and signature != correct, (out["columns"]["rid"], dict(correct))


def test_healthy_control_j_d2_without_a_reference_to_row_0(node_session, tmp_path):
    """Control: with no index pointing at row 0, the inner index join gives exactly the valid right rows."""
    left = {**IL, "idx": [3, 2, -1, 4, 1, 7]}
    out = _index_join(node_session, tmp_path, "inner", left=left)
    assert Counter(out["columns"]["rid"]) == Counter(r for _, r in _index_oracle(left, "inner"))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="J-D2: the inner index join drops row 0")
def test_inner_index_join_keeps_row_0(node_session, tmp_path):
    out = _index_join(node_session, tmp_path, "inner")
    assert Counter(out["columns"]["rid"]) == Counter(r for _, r in _index_oracle(IL, "inner"))


def test_known_defect_j_d1_in_an_index_join(node_session, tmp_path):
    """J-D1 in the left index join: the left columns are the left table at the idx values, not each row itself."""
    out = _index_join(node_session, tmp_path, "left")
    signature = _gather(IL["lid"], out["ir"])
    assert out["columns"]["lid"] == signature and signature != IL["lid"]


def test_known_defect_j_d3_reverse_index_join_throws(node_session, tmp_path):
    """J-D3, exact signature: with the row numbers in the right table, every column read throws (indices unset)."""
    left = {"lid": [0, 1, 2, 3, 4], "a": [10.0, 11.0, 12.0, 13.0, 14.0]}
    right = {"rid": [0, 1, 2, 3, 4, 5], "idx": [0, 2, -1, 4, 1, 7]}
    out = _read(node_session, {"L": left, "R": right}, {"J": dict(left="L", right="R", on_left=[], on_right=["idx"])},
                tmp_path, ["lid", "rid"])
    assert out["il"] is None and out["ir"] is None
    assert set(out["throws"]) >= {"lid", "rid"} and all("map" in m for k, m in out["throws"].items() if k != "length")


# ---------------------------------------------------------------------------------------------- the oracle can fail
def test_the_oracle_catches_a_wrong_join():
    """The checks must be able to fail: a dropped duplicate or a shifted row id differs from pandas."""
    expected = _oracle_pairs(KL, KR, ["k"], ["kr"], "inner")
    dropped = expected.copy()
    dropped[next(iter(dropped))] -= 1
    shifted = Counter({(l, (r + 1) % 5): n for (l, r), n in expected.items()})
    assert +dropped != expected and shifted != expected
