"""Production CDSJoin in Node — J0, J1 (revised) and J2 (recursive joins). Oracle and invariant tests only.

Source tables and CDSJoin models are built in Python with RootInteractive's own model class, loaded in the Node harness
through Bokeh's deserializer, and the PRODUCTION model is read: row pairs (`_indices_left`, `_indices_right`),
columns (`get_column`), length (`get_length`). Two independent references:
  * the ORACLE — what is correct: pandas `merge` / explicit row-number gathers;
  * the PORT — what production does today: a line-by-line Python port of CDSJoin.ts (compute_indices, get_column,
    join_column, get_length), defects included. It predicts the exact wrong result of every known defect.
A known-defect check passes only if production equals the port exactly AND the port differs from the oracle; then it
raises `KnownDefect`, the only failure the strict expected-failure tests accept (R-4). Any other failure — a Node
error, a missing column, a result matching neither — stays red. RootInteractive's code is NOT changed.

Findings (provisional ids, B-numbers in the bug report):
  J-D1  every column, left ones included, is gathered with the right row numbers (get_column, `_indices_right`).
  J-D2  an inner index join drops row 0 (`x > 0`).
  J-D3  the reverse index join (the right table holds row numbers) is unreachable; indices unset, every read throws.
  J-D4  the key join's merge advances the WRONG side when keys differ (left key larger -> left row skipped), so an
        unmatched key inside the tables loses matches; only unmatched keys at the ends are handled correctly.
  J-H1  the key join assumes sorted keys; unsorted input is silently wrong (Q-1 open; tested as (b): join correctly
        or refuse loudly).
  (multi-key) the merge compares the LAST key column first, but no independent effect of the sort order was found
        (4 000 random fixtures with the same keys on both sides join correctly in either order); multi-key joins
        share J-D4 when a key is unmatched inside the tables.
  J-D5  a join whose input is a JOIN reads that input's key column through `get_array`, which returns only the inner
        join's lazily cached `data` — an empty array at load (Bokeh stores it). An INDEX join on it is therefore EMPTY
        (V0 -> track -> event); a KEY join on it takes the lengths from the inputs but compares undefined values, whose
        NaN differences pass the tolerance test as "equal", so rows are paired BY POSITION, not by key. Nothing is
        recomputed when the inner column is computed later. A join whose key column lies in a plain table — e.g.
        V0 x (track x event) — is not affected.
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.node_joins]   # target (PHASE_4_3)


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect (R-4)."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


# ===================================================================================== harness: build, load, read
READ = """(a) => {
  const doc = Bokeh.documents[Bokeh.documents.length - 1];
  for (const mu of (a.mutate || [])) { const t = doc.get_model_by_name(mu.table); t.data = Object.assign({}, t.data, mu.data); }
  const m = doc.get_model_by_name(a.name);
  if (!m) return {error: "no model named " + a.name};
  const arr = x => (x == null) ? null : Array.from(x, v => (v === undefined || Number.isNaN(v)) ? null : v);
  const out = {il: arr(m._indices_left), ir: arr(m._indices_right), columns: {}, throws: {}};
  try { out.length = m.get_length(); } catch (e) { out.throws.length = String(e.message); }
  for (const k of a.keys) { try { out.columns[k] = arr(m.get_column(k)); } catch (e) { out.throws[k] = String(e.message); } }
  return out;
}"""


def _document(tmp_path: Path, tables: dict, joins: dict) -> Path:
    from bokeh.document import Document
    from bokeh.models import ColumnDataSource
    from RootInteractive.InteractiveDrawing.bokeh.CDSJoin import CDSJoin
    models = {n: ColumnDataSource(data={k: list(v) for k, v in cols.items()}, name=n) for n, cols in tables.items()}
    for n, s in joins.items():                                       # joins in dependency order (dict order)
        models[n] = CDSJoin(left=models[s["left"]], right=models[s["right"]], on_left=s.get("on_left", []),
                            on_right=s.get("on_right", []), how=s.get("how", "inner"),
                            prefix_left=s.get("prefix_left", ""), prefix_right=s.get("prefix_right", ""), name=n)
    doc = Document()
    for n in joins:
        doc.add_root(models[n])
    path = tmp_path / "joins_document.json"
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return path


def _read(node_session, tables, joins, tmp_path, name, keys, mutate=()):
    """The production model's reply. A harness problem fails the test (never a KnownDefect)."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = node_session.request("new_doc", path=str(_document(tmp_path, tables, joins)))
    if not r.get("ok"):
        pytest.fail(f"Node harness could not load the document: {r}")
    r = node_session.request("eval", code=READ, arg={"name": name, "keys": list(keys), "mutate": list(mutate)})
    if not r.get("ok"):
        pytest.fail(f"Node harness eval failed: {r}")
    out = decode(r["result"])
    if "error" in out:
        pytest.fail(out["error"])
    return out


def _complete(out, keys, length):
    """C-4 without a vacuous pass (R-3): every requested column present, no read exception, the expected length."""
    assert not out["throws"], out["throws"]
    assert set(out["columns"]) == set(keys), sorted(out["columns"])
    assert out["length"] == length and all(len(out["columns"][k]) == length for k in keys), (out["length"], length)


# ===================================================================================== the PORT (production today)
class Port:
    """Line-by-line port of CDSJoin.ts as it is (compute_indices, get_column, join_column, get_length)."""
    TOL = 1e-5

    def __init__(self, tables, joins):
        self.t, self.j = tables, joins

    def column(self, m, key):
        if m in self.t:
            return list(self.t[m][key]) if key in self.t[m] else None
        s = self.j[m]
        il, ir = self.indices(m)
        if il == "UNSET":
            raise RuntimeError("Cannot read properties of undefined (reading 'map')")        # J-D3
        pl, pr = s.get("prefix_left", ""), s.get("prefix_right", "")
        if pl and key.startswith(pl):
            c = self.column(s["left"], key.replace(pl, "", 1))
        elif pr and key.startswith(pr):
            c = self.column(s["right"], key.replace(pr, "", 1))
        else:
            c = self.column(s["left"], key)
            if c is None:
                c = self.column(s["right"], key)
        if c is None:
            return None
        if ir is None:
            return c
        return [c[int(x)] if x is not None and 0 <= x < len(c) else None for x in ir]           # J-D1: always ir

    def length(self, m):
        if m in self.t:
            return len(next(iter(self.t[m].values())))
        il, ir = self.indices(m)
        if il not in (None, "UNSET"):
            return len(il)
        if ir not in (None, "UNSET"):
            return len(ir)
        return self.length(self.j[m]["left"])

    def array(self, m, key):
        """get_array: a table's column; a JOIN's lazily cached data, empty at load (J-D5)."""
        return self.column(m, key) if m in self.t else []

    @staticmethod
    def at(a, i):
        """JavaScript a[i]: undefined (NaN in arithmetic) outside the array; None also counts as undefined."""
        v = a[i] if 0 <= i < len(a) else None
        return float("nan") if v is None else v

    def indices(self, m):
        s = self.j[m]
        on_l, on_r, how = s.get("on_left", []), s.get("on_right", []), s.get("how", "inner")
        if not on_l:
            return (None, None) if not on_r else ("UNSET", "UNSET")                               # J-D3
        if len(on_l) == 1 and not on_r:
            idx, n = self.array(s["left"], on_l[0]), self.length(s["right"])
            return None, ([x for x in idx if x is not None and 0 < x < n] if how == "inner" else idx)   # J-D2
        L = [self.array(s["left"], k) for k in on_l]
        R = [self.array(s["right"], k) for k in on_r]
        return self._merge(L, R, how, self.length(s["left"]), self.length(s["right"]))   # lengths: left.length

    def _merge(self, L, R, how, nl, nr):
        lj, rj = how in ("left", "outer"), how in ("right", "outer")
        at = self.at
        dl = dr = 0
        il, ir = [], []

        def cmp(i, j):                                                # reduceRight: the LAST key decides first (J-H2)
            acc = 0
            for k in reversed(range(len(L))):
                r = (at(L[k], i) - at(R[k], j)) if acc == 0 else acc
                acc = r if abs(r) > self.TOL else 0                   # NaN fails the test -> "equal" (J-D5)
            return acc

        def run_end(A, start, n):
            up = start
            while True:
                up += 1
                if up >= n:
                    return up
                diff = 0
                for k in reversed(range(len(A))):
                    diff = (at(A[k], up) - at(A[k], start)) if diff == 0 else diff
                if diff != 0:                                         # NaN != 0: a run of one row (J-D5)
                    return up

        while dl < nl and dr < nr:
            c = cmp(dl, dr)
            if c > 0:                                                  # J-D4: the left key is larger, yet left advances
                if lj:
                    il.append(dl); ir.append(-1)
                dl += 1
            elif c < 0:
                if rj:
                    il.append(-1); ir.append(dr)
                dr += 1
            else:
                ul, ur = run_end(L, dl, nl), run_end(R, dr, nr)
                for a in range(dl, ul):
                    for b in range(dr, ur):
                        il.append(a); ir.append(b)
                dl, dr = ul, ur
        if lj:
            il += list(range(dl, nl)); ir += [-1] * (nl - dl)
        if rj:
            il += [-1] * (nr - dr); ir += list(range(dr, nr))
        return il, ir


# ===================================================================================== the ORACLE (what is correct)
def _oracle(tables, joins, m) -> pd.DataFrame:
    """Correct joined records: index joins gather by row number; key joins are pandas merges."""
    if m in tables:
        return pd.DataFrame(tables[m]).astype(object)
    s = joins[m]
    L, R = _oracle(tables, joins, s["left"]), _oracle(tables, joins, s["right"])
    L = L.rename(columns=lambda c: s.get("prefix_left", "") + c)
    R = R.rename(columns=lambda c: s.get("prefix_right", "") + c)
    on_l, on_r, how = s.get("on_left", []), s.get("on_right", []), s.get("how", "inner")

    def gather(holder, target, col):
        rows = []
        for x in holder[col]:
            ok = x is not None and not pd.isna(x) and 0 <= int(x) < len(target)
            rows.append(target.iloc[int(x)].to_dict() if ok else {c: None for c in target.columns})
        return pd.DataFrame(rows, columns=target.columns).astype(object), [r is not None and not pd.isna(r) and 0 <= int(r) < len(target) for r in holder[col]]

    if len(on_l) == 1 and not on_r:                                   # the left table holds right row numbers
        G, valid = gather(L, R, s.get("prefix_left", "") + on_l[0])
        out = pd.concat([L.reset_index(drop=True), G], axis=1)
        return out[valid].reset_index(drop=True) if how == "inner" else out
    if not on_l and len(on_r) == 1:                                   # the right table holds left row numbers
        G, valid = gather(R, L, s.get("prefix_right", "") + on_r[0])
        out = pd.concat([G, R.reset_index(drop=True)], axis=1)
        return out[valid].reset_index(drop=True) if how == "inner" else out
    return pd.merge(L, R, left_on=[s.get("prefix_left", "") + k for k in on_l],
                    right_on=[s.get("prefix_right", "") + k for k in on_r], how=how).astype(object)


def _norm(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)


def _records(cols: dict, keys) -> Counter:
    return Counter(tuple(_norm(v) for v in row) for row in zip(*(cols[k] for k in keys)))


def _oracle_records(tables, joins, m, keys) -> Counter:
    df = _oracle(tables, joins, m)
    return _records({k: list(df[k]) for k in keys}, keys)


def _port_columns(tables, joins, m, keys) -> dict:
    p = Port(tables, joins)
    return {k: p.column(m, k) for k in keys}


def _check(node_session, tables, joins, tmp_path, m, keys, label, mutate=(), mtables=None):
    """Correct (oracle) -> pass; exactly today's port result != oracle -> KnownDefect(label); anything else -> red."""
    out = _read(node_session, tables, joins, tmp_path, m, keys, mutate)
    after = mtables or tables
    assert not out["throws"], out["throws"]
    assert len({len(out["columns"][k]) for k in keys}) == 1, {k: len(out["columns"][k]) for k in keys}
    got = _records(out["columns"], keys)
    if got == _oracle_records(after, joins, m, keys):
        return out
    port = _port_columns(after, joins, m, keys)
    known_defect({k: [_norm(v) for v in out["columns"][k]] for k in keys} == {k: [_norm(v) for v in port[k]] for k in keys}
                 and _records(port, keys) != _oracle_records(after, joins, m, keys), label)
    raise AssertionError(f"{label}: production matches neither the oracle nor the port: {out['columns']}")


def _signature(node_session, tables, joins, tmp_path, m, keys):
    """The exact-signature check: production equals the port exactly, and the port differs from the oracle."""
    out = _read(node_session, tables, joins, tmp_path, m, keys)
    port = _port_columns(tables, joins, m, keys)
    assert {k: [_norm(v) for v in out["columns"][k]] for k in keys} == {k: [_norm(v) for v in port[k]] for k in keys}, out
    assert _records(port, keys) != _oracle_records(tables, joins, m, keys)
    assert out["length"] == Port(tables, joins).length(m)


STRICT = dict(strict=True, raises=KnownDefect)

# ===================================================================================== J1 fixtures
# key tables sorted ascending (J-H1); fan-out 2x1 and 1x2, row 0 joined, orphans only at the ENDS (J-D4 cannot show)
KL = {"lid": [0, 1, 2, 3, 4], "k": [0, 1, 1, 2, 5], "a": [10.0, 11.0, 12.0, 13.0, 14.0]}
KR = {"rid": [0, 1, 2, 3, 4], "kr": [0, 1, 2, 2, 3], "b": [20.0, 21.0, 22.0, 23.0, 24.0]}
# the same with unmatched keys INSIDE the tables on both sides (k = 1.5 left, kr = 0.5 right)
KLI = {"lid": [0, 1, 2, 3, 4, 5], "k": [0, 1, 1, 1.5, 2, 5], "a": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}
KRI = {"rid": [0, 1, 2, 3, 4, 5], "kr": [0, 0.5, 1, 2, 2, 3], "b": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]}
MODES = ["inner", "left", "right", "outer"]


def _kj(how, left="L", right="R", **kw):
    return {"J": dict(left=left, right=right, on_left=["k"], on_right=["kr"], how=how, **kw)}


def _pairs(out):
    assert len(out["il"]) == len(out["ir"]), (len(out["il"]), len(out["ir"]))                  # R-3: no silent zip
    ids = lambda xs: [None if x is None or x < 0 else int(x) for x in xs]
    return Counter(zip(ids(out["il"]), ids(out["ir"])))


def _oracle_pairs(left, right, how, on_left=("k",), on_right=("kr",)):
    m = pd.merge(pd.DataFrame(left), pd.DataFrame(right), left_on=list(on_left), right_on=list(on_right), how=how)
    ids = lambda s: [None if pd.isna(x) else int(x) for x in s]
    return Counter(zip(ids(m["lid"]), ids(m["rid"])))


# ===================================================================================== J1: key joins
@pytest.mark.parametrize("how", MODES)
def test_c1_key_join_row_pairs_equal_pandas(node_session, tmp_path, how):
    """C-1: on sorted keys with unmatched keys only at the ends, the row pairs equal pandas in every mode."""
    out = _read(node_session, {"L": KL, "R": KR}, _kj(how), tmp_path, "J", [])
    assert _pairs(out) == _oracle_pairs(KL, KR, how)


@pytest.mark.parametrize("how", MODES)
def test_c4_length_and_every_column(node_session, tmp_path, how):
    """C-4 (R-3): every requested column present, no read exception, each the join's length."""
    keys = ["lid", "rid", "a", "b"]
    out = _read(node_session, {"L": KL, "R": KR}, _kj(how), tmp_path, "J", keys)
    _complete(out, keys, sum(_oracle_pairs(KL, KR, how).values()))


@pytest.mark.parametrize("side", ["left", "right"])
def test_c5_row_order_within_equal_keys_does_not_change_the_join(node_session, tmp_path, side):
    """C-5 (R-5): permuting rows inside runs of equal keys, of either input, leaves the id pairs unchanged. General
    permutations are outside the sorted-key precondition (J-H1)."""
    base = _pairs(_read(node_session, {"L": KL, "R": KR}, _kj("outer"), tmp_path, "J", []))
    if side == "left":
        L, R = {c: [v[0], v[2], v[1], v[3], v[4]] for c, v in KL.items()}, KR
    else:
        L, R = KL, {c: [v[0], v[1], v[3], v[2], v[4]] for c, v in KR.items()}
    out = _read(node_session, {"L": L, "R": R}, _kj("outer"), tmp_path, "J", [])
    to_ids = lambda o, tl, tr: Counter((None if a is None else tl["lid"][a], None if b is None else tr["rid"][b])
                                       for a, b in _pairs(o).elements())
    assert to_ids(out, L, R) == to_ids(_read(node_session, {"L": KL, "R": KR}, _kj("outer"), tmp_path, "J", []), KL, KR)
    assert base == _pairs(_read(node_session, {"L": KL, "R": KR}, _kj("outer"), tmp_path, "J", []))


@pytest.mark.parametrize("how", MODES)
def test_c2_right_columns_correct(node_session, tmp_path, how):
    """C-2: right columns equal the right table at the joined right rows (missing for an orphan)."""
    keys = ["rid", "b"]
    out = _read(node_session, {"L": KL, "R": KR}, _kj(how), tmp_path, "J", keys)
    for k in keys:
        assert out["columns"][k] == [None if r is None or r < 0 else KR[k][int(r)] for r in out["ir"]], k


@pytest.mark.parametrize("how", MODES)
def test_signature_j_d1_left_columns(node_session, tmp_path, how):
    """J-D1, exact signature: production equals the port (left columns at the right row numbers) and differs from pandas."""
    _signature(node_session, {"L": KL, "R": KR}, _kj(how), tmp_path, "J", ["lid", "rid", "a", "b"])


@pytest.mark.xfail(**STRICT, reason="J-D1")
@pytest.mark.parametrize("how", MODES)
def test_strict_key_join_columns_are_correct(node_session, tmp_path, how):
    _check(node_session, {"L": KL, "R": KR}, _kj(how), tmp_path, "J", ["lid", "rid", "a", "b"], "J-D1")


def test_control_j_d1_aligned_rows(node_session, tmp_path):
    """Control: identical one-to-one keys give equal row numbers on both sides; J-D1 cannot show; columns correct."""
    L = {"lid": [0, 1, 2], "k": [1, 2, 3], "a": [10.0, 11.0, 12.0]}
    R = {"rid": [0, 1, 2], "kr": [1, 2, 3], "b": [20.0, 21.0, 22.0]}
    _check(node_session, {"L": L, "R": R}, _kj("inner"), tmp_path, "J", ["lid", "rid", "a", "b"], "control")


@pytest.mark.parametrize("how", MODES)
def test_signature_j_d4_unmatched_key_inside_the_tables(node_session, tmp_path, how):
    """J-D4, exact signature: an unmatched key inside sorted tables makes the merge advance the wrong side."""
    out = _read(node_session, {"L": KLI, "R": KRI}, _kj(how), tmp_path, "J", [])
    port = Port({"L": KLI, "R": KRI}, _kj(how)).indices("J")
    assert ([None if x is None else int(x) for x in out["il"]], [None if x is None else int(x) for x in out["ir"]]) == port
    ids = lambda xs: [None if x < 0 else x for x in xs]
    assert Counter(zip(ids(port[0]), ids(port[1]))) != _oracle_pairs(KLI, KRI, how)


@pytest.mark.xfail(**STRICT, reason="J-D4")
@pytest.mark.parametrize("how", MODES)
def test_strict_key_join_with_an_unmatched_key_inside(node_session, tmp_path, how):
    out = _read(node_session, {"L": KLI, "R": KRI}, _kj(how), tmp_path, "J", [])
    if _pairs(out) == _oracle_pairs(KLI, KRI, how):
        return
    port = Port({"L": KLI, "R": KRI}, _kj(how)).indices("J")
    known_defect(([None if x is None else int(x) for x in out["il"]], [None if x is None else int(x) for x in out["ir"]]) == port, "J-D4")
    raise AssertionError("J-D4: production matches neither pandas nor the port")


# ----- J-H1: the sorted-key precondition (R-2; Q-1 open, tested as (b): join correctly or refuse loudly)
UNSORTED_R = {"rid": [0, 1, 2, 3, 4], "kr": [2, 0, 3, 1, 2], "b": [20.0, 21.0, 22.0, 23.0, 24.0]}


def test_j_h1_exact_result_on_unsorted_keys_and_its_sorted_control(node_session, tmp_path):
    """J-H1, exact: the unsorted fixture gives exactly the port's pairs (not just 'different'); the same rows sorted
    give exactly pandas (the paired control)."""
    out = _read(node_session, {"L": KL, "R": UNSORTED_R}, _kj("inner"), tmp_path, "J", [])
    port = Port({"L": KL, "R": UNSORTED_R}, _kj("inner")).indices("J")
    assert not out["throws"] and ([int(x) for x in out["il"]], [int(x) for x in out["ir"]]) == port
    assert Counter(zip(*port)) != _oracle_pairs(KL, UNSORTED_R, "inner")
    order = sorted(range(5), key=lambda i: UNSORTED_R["kr"][i])
    sorted_r = {c: [v[i] for i in order] for c, v in UNSORTED_R.items()}
    ctl = _read(node_session, {"L": KL, "R": sorted_r}, _kj("inner"), tmp_path, "J", [])
    as_ids = Counter((KL["lid"][a], sorted_r["rid"][b]) for a, b in _pairs(ctl).elements())
    assert as_ids == _oracle_pairs(KL, sorted_r, "inner")


@pytest.mark.xfail(**STRICT, reason="J-H1 (Q-1 (b))")
def test_strict_unsorted_keys_join_correctly_or_refuse_loudly(node_session, tmp_path):
    out = _read(node_session, {"L": KL, "R": UNSORTED_R}, _kj("inner"), tmp_path, "J", ["lid"])
    if out["throws"] or _pairs(out) == _oracle_pairs(KL, UNSORTED_R, "inner"):
        return
    known_defect(([int(x) for x in out["il"]], [int(x) for x in out["ir"]]) == Port({"L": KL, "R": UNSORTED_R}, _kj("inner")).indices("J"), "J-H1")
    raise AssertionError("J-H1: silent result matches neither pandas nor the port")


# ----- multi-key joins (R-8): which key column is most significant
ML = {"lid": [0, 1, 2, 3], "k1": [0, 1, 0, 2], "k2": [0, 0, 1, 1]}           # sorted with k2 most significant;
MR = {"rid": [0, 1, 2, 3, 4], "j1": [0, 1, 1, 0, 2], "j2": [0, 0, 0, 1, 1]}    # the same key set, a duplicate (1, 0)


def _mk(how="inner"):
    return {"J": dict(left="L", right="R", on_left=["k1", "k2"], on_right=["j1", "j2"], how=how)}


@pytest.mark.parametrize("order", ["last_key_major", "first_key_major"])
def test_multi_key_join_equals_pandas_in_both_sort_orders(node_session, tmp_path, order):
    """Multi-key (R-8): with the same keys on both sides, the two-key join equals pandas whichever key is major."""
    key = (lambda t, a, b: (lambda i: (t[b][i], t[a][i]))) if order == "last_key_major" else (lambda t, a, b: (lambda i: (t[a][i], t[b][i])))
    o1, o2 = sorted(range(4), key=key(ML, "k1", "k2")), sorted(range(5), key=key(MR, "j1", "j2"))
    L, R = {c: [v[i] for i in o1] for c, v in ML.items()}, {c: [v[i] for i in o2] for c, v in MR.items()}
    out = _read(node_session, {"L": L, "R": R}, _mk("outer"), tmp_path, "J", [])
    got = Counter((None if a is None else L["lid"][a], None if b is None else R["rid"][b]) for a, b in _pairs(out).elements())
    assert got == _oracle_pairs(ML, MR, "outer", ("k1", "k2"), ("j1", "j2"))


def test_signature_j_d4_multi_key_unmatched_key_inside(node_session, tmp_path):
    """J-D4 with two keys, exact: an unmatched key (1, 1) inside the left table gives exactly the port's pairs."""
    L = {"lid": [0, 1, 2, 3, 4], "k1": [0, 1, 0, 1, 2], "k2": [0, 0, 1, 1, 1]}
    out = _read(node_session, {"L": L, "R": MR}, _mk("outer"), tmp_path, "J", [])
    port = Port({"L": L, "R": MR}, _mk("outer")).indices("J")
    assert ([None if x is None else int(x) for x in out["il"]], [None if x is None else int(x) for x in out["ir"]]) == port
    ids = Counter(zip([None if x < 0 else L["lid"][x] for x in port[0]], [None if x < 0 else MR["rid"][x] for x in port[1]]))
    assert ids != _oracle_pairs(L, MR, "outer", ("k1", "k2"), ("j1", "j2"))


def test_prefixed_names_route_to_their_side(node_session, tmp_path):
    """R-8, prefixes: "l_a" reads the left table's a, "r_b" the right table's b; values per the port (J-D1 applies)."""
    joins = _kj("inner", prefix_left="l_", prefix_right="r_")
    keys = ["l_lid", "r_rid", "l_a", "r_b"]
    out = _read(node_session, {"L": KL, "R": KR}, joins, tmp_path, "J", keys)
    _complete(out, keys, sum(_oracle_pairs(KL, KR, "inner").values()))
    assert {k: [_norm(v) for v in out["columns"][k]] for k in keys} == \
        {k: [_norm(v) for v in c] for k, c in _port_columns({"L": KL, "R": KR}, joins, "J", keys).items()}
    assert out["columns"]["r_b"] == [KR["b"][int(r)] for r in out["ir"]]


# ===================================================================================== J1: index joins
IL = {"lid": [0, 1, 2, 3, 4, 5], "idx": [0, 2, -1, 4, 1, 7], "a": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}
IR = {"rid": [0, 1, 2, 3, 4], "b": [20.0, 21.0, 22.0, 23.0, 24.0]}


def _ij(how, left="L", right="R"):
    return {"J": dict(left=left, right=right, on_left=["idx"], on_right=[], how=how)}


def test_c3_left_index_join_right_values(node_session, tmp_path):
    """C-3: the left index join keeps every left row; the right values follow idx, row 0 included, missing for -1/7."""
    out = _read(node_session, {"L": IL, "R": IR}, _ij("left"), tmp_path, "J", ["rid", "b"])
    _complete(out, ["rid", "b"], 6)
    assert out["columns"]["b"] == [IR["b"][x] if 0 <= x < 5 else None for x in IL["idx"]]


def test_signature_j_d2_and_j_d1_inner_index_join(node_session, tmp_path):
    """J-D2 + J-D1, exact: the inner index join drops row 0 and reads the left columns at the idx values."""
    _signature(node_session, {"L": IL, "R": IR}, _ij("inner"), tmp_path, "J", ["lid", "rid", "a", "b"])


@pytest.mark.xfail(**STRICT, reason="J-D2 / J-D1")
def test_strict_inner_index_join_is_correct(node_session, tmp_path):
    _check(node_session, {"L": IL, "R": IR}, _ij("inner"), tmp_path, "J", ["lid", "rid", "a", "b"], "J-D2/J-D1")


def test_control_j_d2_without_a_reference_to_row_0(node_session, tmp_path):
    """Control: no index points at row 0, so J-D2 cannot show; the right rows are exactly the valid ones."""
    left = {**IL, "idx": [3, 2, -1, 4, 1, 7]}
    out = _read(node_session, {"L": left, "R": IR}, _ij("inner"), tmp_path, "J", ["rid"])
    assert Counter(out["columns"]["rid"]) == Counter(x for x in left["idx"] if 0 <= x < 5)


# ----- J-D3, the reverse index join (R-1): oracle, signature, healthy control, strict check
RL = {"lid": [0, 1, 2, 3, 4], "a": [10.0, 11.0, 12.0, 13.0, 14.0]}
RR = {"rid": [0, 1, 2, 3, 4, 5], "idx": [0, 2, -1, 4, 1, 7], "b": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]}
REV = {"J": dict(left="L", right="R", on_left=[], on_right=["idx"], how="inner")}


def test_signature_j_d3_reverse_index_join(node_session, tmp_path):
    """J-D3, exact: both index arrays unset and every column read throws the 'map' error the port predicts."""
    out = _read(node_session, {"L": RL, "R": RR}, REV, tmp_path, "J", ["lid", "rid"])
    assert out["il"] is None and out["ir"] is None
    assert set(out["throws"]) >= {"lid", "rid"} and all("reading 'map'" in out["throws"][k] for k in ("lid", "rid"))
    with pytest.raises(RuntimeError, match="reading 'map'"):
        Port({"L": RL, "R": RR}, REV).column("J", "lid")


def test_control_j_d3_the_same_relation_in_the_supported_direction(node_session, tmp_path):
    """Control: the same relation with the row numbers in the LEFT table (forward index join, a left join so J-D2
    cannot show) gives the reverse oracle's left values — the oracle and data are sound."""
    fwd = {"J": dict(left="R", right="L", on_left=["idx"], on_right=[], how="left")}
    out = _read(node_session, {"L": RL, "R": RR}, fwd, tmp_path, "J", ["a"])
    expected = list(_oracle({"L": RL, "R": RR}, {"J": dict(REV["J"], how="left")}, "J")["a"])
    assert [_norm(v) for v in out["columns"]["a"]] == [_norm(v) for v in expected]


@pytest.mark.xfail(**STRICT, reason="J-D3")
def test_strict_reverse_index_join_is_correct(node_session, tmp_path):
    out = _read(node_session, {"L": RL, "R": RR}, REV, tmp_path, "J", ["lid", "rid", "a", "b"])
    if not out["throws"] and _records(out["columns"], ["lid", "rid", "a", "b"]) == _oracle_records({"L": RL, "R": RR}, REV, "J", ["lid", "rid", "a", "b"]):
        return
    known_defect(out["il"] is None and out["ir"] is None and all("reading 'map'" in m for m in out["throws"].values()), "J-D3")
    raise AssertionError(f"J-D3: unexpected result {out}")


# ===================================================================================== J2: recursive joins (AO2D-like)
# events; tracks -> event row; V0 -> two daughter track rows; cascade -> V0 row and a bachelor track row.
# Two V0s share track 2 as negative daughter; row 0 referenced at every level; an orphan at every level (-1) and an
# out-of-range reference (cascade 3 -> V0 7); track 4 has no event (-1).
E = {"eid": [0, 1, 2, 3], "vz": [0.5, 1.5, 2.5, 3.5]}
T = {"tid": [0, 1, 2, 3, 4, 5], "ev": [0, 0, 1, 3, -1, 2], "pt": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
V0 = {"vid": [0, 1, 2, 3, 4], "pos": [0, 1, 3, 5, -1], "neg": [2, 2, 4, 0, 1], "mass": [0.49, 0.50, 0.51, 0.52, 0.53]}
C = {"cid": [0, 1, 2, 3], "v0": [0, 2, 4, 7], "bach": [3, 0, 5, 1], "cmass": [1.32, 1.33, 1.34, 1.35]}
AO2D = {"E": E, "T": T, "V0": V0, "C": C}


def _chain_v0(how):
    return {"VT": dict(left="V0", right="T", on_left=["pos"], on_right=[], how=how),
            "VTE": dict(left="VT", right="E", on_left=["ev"], on_right=[], how=how)}


def _chain_cascade(how):
    return {"CV": dict(left="C", right="V0", on_left=["v0"], on_right=[], how=how),
            "CVT": dict(left="CV", right="T", on_left=["pos"], on_right=[], how=how),
            "CVTE": dict(left="CVT", right="E", on_left=["ev"], on_right=[], how=how)}


V0_KEYS = ["vid", "mass", "tid", "pt", "eid", "vz"]
C_KEYS = ["cid", "cmass", "vid", "mass", "tid", "pt", "eid", "vz"]


@pytest.mark.parametrize("how", ["inner", "left"])
def test_c6_signature_v0_track_event(node_session, tmp_path, how):
    """C-6, J2, exact signature: V0 -> track -> event gives exactly the port's result (J-D1 at both levels, J-D2 for
    inner), which differs from the step-by-step oracle."""
    _signature(node_session, AO2D, _chain_v0(how), tmp_path, "VTE", V0_KEYS)


@pytest.mark.xfail(**STRICT, reason="J-D1 / J-D2 through the chain")
@pytest.mark.parametrize("how", ["inner", "left"])
def test_c6_strict_v0_track_event_is_correct(node_session, tmp_path, how):
    _check(node_session, AO2D, _chain_v0(how), tmp_path, "VTE", V0_KEYS, "J-D1/J-D2 chain")


@pytest.mark.parametrize("how", ["inner", "left"])
def test_c6_signature_cascade_v0_track_event(node_session, tmp_path, how):
    """C-6, J2, exact signature: cascade -> V0 -> track -> event (three levels)."""
    _signature(node_session, AO2D, _chain_cascade(how), tmp_path, "CVTE", C_KEYS)


@pytest.mark.xfail(**STRICT, reason="J-D1 / J-D2 through the chain")
@pytest.mark.parametrize("how", ["inner", "left"])
def test_c6_strict_cascade_v0_track_event_is_correct(node_session, tmp_path, how):
    _check(node_session, AO2D, _chain_cascade(how), tmp_path, "CVTE", C_KEYS, "J-D1/J-D2 chain")


def _right_nested(how):
    return {"TE": dict(left="T", right="E", on_left=["ev"], on_right=[], how="left"),
            "VTE2": dict(left="V0", right="TE", on_left=["pos"], on_right=[], how=how)}


def test_c6_control_right_nested_identity_references(node_session, tmp_path):
    """Control: V0 x (track x event) — the outer key lies in a plain table, so J-D5 cannot show; references equal to
    the row numbers (left joins) make J-D1/J-D2 impossible: the two-level chain equals the oracle exactly."""
    t = {"E": {"eid": [0, 1, 2, 3, 4, 5], "vz": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]}, "T": {**T, "ev": [0, 1, 2, 3, 4, 5]},
         "V0": {**V0, "pos": [0, 1, 2, 3, 4]}}                    # identity references at every level
    _check(node_session, t, _right_nested("left"), tmp_path, "VTE2", V0_KEYS, "control")


def test_signature_j_d5_left_nested_chain_is_empty_and_stays_empty(node_session, tmp_path):
    """J-D5, exact: V0 -> track -> event is empty at load (length 0), and stays empty after the inner join's key
    column has been computed; only an explicit recomputation gives the rows."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = node_session.request("new_doc", path=str(_document(tmp_path, AO2D, _chain_v0("left"))))
    assert r.get("ok"), r
    code = """() => { const d = Bokeh.documents[Bokeh.documents.length-1], VT = d.get_model_by_name("VT"), VTE = d.get_model_by_name("VTE");
      const out = {at_load: VTE.get_length(), inner_ev: Array.from(VT.get_column("ev"), v => v === undefined ? null : v)};
      out.after_inner_computed = VTE.get_length(); VTE.compute_indices(); out.after_recompute = VTE.get_length(); return out; }"""
    r = node_session.request("eval", code=code, arg=None)
    assert r.get("ok"), r
    out = decode(r["result"])
    assert out == {"at_load": 0, "inner_ev": [0, 0, 3, 2, None], "after_inner_computed": 0, "after_recompute": 5}, out


def test_c6_two_daughters_join_of_two_joins_signature(node_session, tmp_path):
    """C-6, a join whose two inputs are joins: (V0 x positive track) joined with (V0 x negative track) on the V0 id —
    exactly the port's result."""
    joins = {"VP": dict(left="V0", right="T", on_left=["pos"], on_right=[], how="left", prefix_right="p_"),
             "VN": dict(left="V0", right="T", on_left=["neg"], on_right=[], how="left", prefix_right="n_"),
             "VPN": dict(left="VP", right="VN", on_left=["vid"], on_right=["vid"], how="inner")}
    keys = ["vid", "mass", "p_tid", "p_pt", "n_tid", "n_pt"]
    out = _read(node_session, AO2D, joins, tmp_path, "VPN", keys)
    port = _port_columns(AO2D, joins, "VPN", keys)
    assert not out["throws"] and {k: [_norm(v) for v in out["columns"][k]] for k in keys} == {k: [_norm(v) for v in port[k]] for k in keys}


def test_signature_j_d5_key_join_of_joins_pairs_rows_by_position(node_session, tmp_path):
    """J-D5, exact: when the two joined joins list the V0s at different positions (the negative-daughter join is an
    inner join that lost a row), the key join pairs rows by position, not by V0 id — exactly the port's result."""
    joins = {"VP": dict(left="V0", right="T", on_left=["pos"], on_right=[], how="left", prefix_right="p_"),
             "VN": dict(left="V0", right="T", on_left=["neg"], on_right=[], how="inner", prefix_right="n_"),
             "VPN": dict(left="VP", right="VN", on_left=["vid"], on_right=["vid"], how="inner")}
    keys = ["vid", "p_tid", "n_tid"]
    _signature(node_session, AO2D, joins, tmp_path, "VPN", keys)


@pytest.mark.parametrize("how", ["inner"])
def test_c7_associativity_v0_track_event(node_session, tmp_path, how):
    """C-7 (invariant): (V0 x T) x E and V0 x (T x E) — the oracle gives the same rows both ways; production gives
    each way exactly the port's result."""
    right_nested = _right_nested(how)
    assert _oracle_records(AO2D, _chain_v0(how), "VTE", V0_KEYS) == _oracle_records(AO2D, right_nested, "VTE2", V0_KEYS)
    for joins, m in ((_chain_v0(how), "VTE"), (right_nested, "VTE2")):
        out = _read(node_session, AO2D, joins, tmp_path, m, V0_KEYS)
        assert {k: [_norm(v) for v in out["columns"][k]] for k in V0_KEYS} == \
            {k: [_norm(v) for v in c] for k, c in _port_columns(AO2D, joins, m, V0_KEYS).items()}


@pytest.mark.parametrize("change", ["event_values", "track_to_event_mapping"])
def test_c8_a_change_of_the_innermost_table_reaches_the_outermost_join(node_session, tmp_path, change):
    """C-8: after changing the innermost table in Node, the outermost join's event values equal a fresh computation
    on the new tables — through two levels (V0 x (track x event), where J-D5 cannot show), with no stale cached
    column. The event values are gathered with the right row numbers, so J-D1 does not touch them: a full oracle check."""
    if change == "event_values":
        mutate, new = [{"table": "E", "data": {"vz": [9.5, 8.5, 7.5, 6.5]}}], {**AO2D, "E": {**E, "vz": [9.5, 8.5, 7.5, 6.5]}}
    else:
        mutate, new = [{"table": "T", "data": {"ev": [3, 2, 1, 0, 1, 0]}}], {**AO2D, "T": {**T, "ev": [3, 2, 1, 0, 1, 0]}}
    out = _read(node_session, AO2D, _right_nested("left"), tmp_path, "VTE2", ["vz", "eid"], mutate)
    assert not out["throws"]
    expected = [_norm(v) for v in Port(new, _right_nested("left")).column("VTE2", "vz")]
    oracle = [_norm(v) for v in _oracle(new, _right_nested("left"), "VTE2")["vz"]]
    assert [_norm(v) for v in out["columns"]["vz"]] == expected == oracle


# ===================================================================================== the checks can fail
def test_the_references_catch_a_wrong_join():
    """The oracle and the port must disagree with deliberately wrong results: a dropped pair, a shifted id."""
    expected = _oracle_pairs(KL, KR, "inner")
    dropped = expected.copy()
    dropped[next(iter(dropped))] -= 1
    assert +dropped != expected and Counter({(l, (r + 1) % 5): n for (l, r), n in expected.items()}) != expected
    port = Port({"L": KL, "R": KR}, _kj("inner")).indices("J")
    assert Counter(zip(*port)) == expected                     # the port reproduces the correct pairs on sorted keys
