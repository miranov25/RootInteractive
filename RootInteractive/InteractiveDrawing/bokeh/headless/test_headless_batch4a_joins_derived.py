"""PHASE_5 Batch 4.a, part 2: what is built ON a join. Tests only; RootInteractive's code is not changed.

In plain words. Part 1 (test_headless_batch4a_joins.py) reads the columns of a join. A dashboard does more with them:
  * a DERIVED COLUMN computed in the browser from joined columns (the model CDSAlias) - here the mass of a two-track
    decay (V0), computed from the momenta of its two daughter tracks, which the join brings into the V0 table;
  * a HISTOGRAM of a joined column (the model HistogramCDS);
  * a CHANGE of a source table after the page has already read the join: the join must follow, not keep old values.
The same three registered defects show here as in part 1: child columns read through the parent's row numbers
(J-D1), an inner index join that loses the rows pointing to row 0 (J-D2), a join whose input is a join coming out
empty (J-D5). The words, the way a test decides (OK, KNOWN, anything else fails) and the helper functions are those
of part 1; a known defect is a strict expected failure.

The V0 mass uses the generator's own formula and its own mass table (toy_mc_events.py: MASS, V0_DAUGHTERS); nothing
is invented here. A test without the browser checks that this formula reproduces the mass the generator stored.
FIX COMMIT (PHASE_5 Batch 4, Claude21.RI, 2026-11-15): the join is fixed in CDSJoin.ts. Every strict mark of this file
is off, so each case is now an ordinary check: the reply must equal the reference. The result of the code before the
fix (KNOWN) is an ordinary failure that says a fixed defect is back. No test and no case is removed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4a_joins import (
    NAN, UNDEF, _clean_reply, _clean_session, _link, _reference_chain, _strict, _today, _verdict)

GEN_DIR = Path(__file__).resolve().parents[3] / "Tools" / "generators"
sys.path.insert(0, str(GEN_DIR))
from toy_mc_events import MASS, V0_DAUGHTERS, generate_toy_mc_events    # noqa: E402

pytestmark = [pytest.mark.node_joins]

READ = """(a) => {
  const doc = Bokeh.documents[Bokeh.documents.length - 1];
  const m = doc.get_model_by_name(a.name);
  if (!m) return {error: "no model named " + a.name};
  const tok = v => v === undefined ? "<undefined>" : v === null ? "<null>"
    : (typeof v === "number" && Number.isNaN(v)) ? "<NaN>" : v === Infinity ? "<+inf>" : v === -Infinity ? "<-inf>" : v;
  const isObj = e => e !== null && typeof e === "object";
  const read = () => {
    const out = {columns: {}, throws: {}, kinds: {}};
    const note = (k, e) => { out.throws[k] = (isObj(e) && "message" in e) ? String(e.message) : String(e);
                             out.kinds[k] = (isObj(e) && "name" in e) ? String(e.name) : typeof e; };
    try { out.length = tok(m.get_length()); } catch (e) { note("length", e); }
    for (const k of a.keys) {
      try { const c = m.get_column(k); out.columns[k] = c == null ? tok(c) : Array.from(c, tok); }
      catch (e) { note(k, e); }
    }
    return out;
  };
  if (!a.change) return {only: read()};
  const before = read();                      // the page reads first, so its caches are filled ...
  for (const ch of a.change) { const t = doc.get_model_by_name(ch.table); t.data = Object.assign({}, t.data, ch.data); }
  return {before, after: read()};            // ... and only then the source table changes
}"""


def _read(node_session, tmp_path, tables, joins, name, keys, more=None, change=None):
    """Like part 1's reader, with two additions: `more(models)` adds models built on the joins, and `change`
    replaces columns of source tables AFTER a first read. Returns the reply, or (before, after) with a change."""
    from bokeh.document import Document
    from bokeh.models import ColumnDataSource
    from RootInteractive.InteractiveDrawing.bokeh.CDSJoin import CDSJoin
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    models = {n: ColumnDataSource(data={k: list(v) for k, v in cols.items()}, name=n) for n, cols in tables.items()}
    for n, s in joins.items():
        models[n] = CDSJoin(left=models[s["left"]], right=models[s["right"]], on_left=s.get("on_left", []),
                            on_right=s.get("on_right", []), how=s.get("how", "inner"),
                            prefix_left=s.get("prefix_left", ""), prefix_right=s.get("prefix_right", ""), name=n)
    roots = dict(more(models)) if more else {n: models[n] for n in joins}
    doc = Document()
    for model in roots.values():
        doc.add_root(model)
    path = tmp_path / "batch4a_derived_document.json"
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    _clean_session(node_session)
    arg = {"name": name, "keys": list(keys), "change": change}
    for op, kw in (("new_doc", dict(path=str(path))), ("eval", dict(code=READ, arg=arg))):
        reply = node_session.request(op, **kw)
        _clean_reply(reply, op)
    out = decode(reply["result"])
    assert "error" not in out, out
    return (out["before"], out["after"]) if change else out["only"]


@pytest.fixture(scope="module")
def physics():
    return generate_toy_mc_events(n_collisions=10, seed=42)


def _sources(tables):
    """The four ToyMC tables with the columns these tests need. `vMPos`, `vMNeg`: the generator's mass hypotheses of
    the two daughters of each V0."""
    c, t, v = (tables[k] for k in ("collisions", "tracks", "v0s"))
    col = lambda frame, name: [x.item() if isinstance(x, np.generic) else x for x in frame[name].to_numpy()]
    return {"CO": {"cRow": col(c, "rowId"), "cVz": col(c, "vz")},
            "TR": {"tRow": col(t, "rowId"), "tEventIdx": col(t, "collisionIdx"), "tPt": col(t, "pt"),
                   "tPx": col(t, "px"), "tPy": col(t, "py"), "tPz": col(t, "pz")},
            "V0": {"vRow": col(v, "rowId"), "vPos": col(v, "posTrackIdx"), "vNeg": col(v, "negTrackIdx"),
                   "vMass": col(v, "mass"), "vPt": col(v, "pt"),
                   "vMPos": [MASS[V0_DAUGHTERS[int(p)][0]] for p in v["pdg"]],
                   "vMNeg": [MASS[V0_DAUGHTERS[int(p)][1]] for p in v["pdg"]]}}


# ============================================================================================ a derived column on a join
MASS_FIELDS = ["pos_tPx", "pos_tPy", "pos_tPz", "vMPos", "neg_tPx", "neg_tPy", "neg_tPz", "vMNeg"]
MASS_JS = ("const e = Math.sqrt(px1*px1 + py1*py1 + pz1*pz1 + m1*m1) + Math.sqrt(px2*px2 + py2*py2 + pz2*pz2 + m2*m2);"
           " const x = px1 + px2, y = py1 + py2, z = pz1 + pz2; return Math.sqrt(e*e - x*x - y*y - z*z)")
FRACTION_FIELDS, FRACTION_JS = ["pos_tPt", "vPt"], "return ptDaughter / ptV0"
MOMENTUM_FIELDS, MOMENTUM_JS = ["pos_tPx", "pos_tPy", "pos_tPz"], "return Math.sqrt(px*px + py*py + pz*pz)"


def _mass(px1, py1, pz1, m1, px2, py2, pz2, m2):
    """The generator's formula: the invariant mass of two tracks, each with its mass hypothesis."""
    e = math.sqrt(px1 * px1 + py1 * py1 + pz1 * pz1 + m1 * m1) + math.sqrt(px2 * px2 + py2 * py2 + pz2 * pz2 + m2 * m2)
    x, y, z = px1 + px2, py1 + py2, pz1 + pz2
    return math.sqrt(e * e - x * x - y * y - z * z)


DERIVED = {   # name: (fields, JavaScript argument names, JavaScript body, the same function in Python)
    "mass": (MASS_FIELDS, ["px1", "py1", "pz1", "m1", "px2", "py2", "pz2", "m2"], MASS_JS, _mass),
    "fraction": (FRACTION_FIELDS, ["ptDaughter", "ptV0"], FRACTION_JS, lambda a, b: a / b),
    "momentum": (MOMENTUM_FIELDS, ["px", "py", "pz"], MOMENTUM_JS, lambda x, y, z: math.sqrt(x * x + y * y + z * z)),
}


def _daughter_joins(how):
    """The V0 table with both daughters: (V0 -> positive track) -> negative track. The second join's child is a join."""
    return {"VP": _link("V0", "TR", "vPos", how, prefix_right="pos_"),
            "VPN": _link("VP", "TR", "vNeg", how, prefix_right="neg_")}


def _apply(function, columns, fields):
    """The derived column from given input columns. A missing input (undefined or NaN) gives NaN, as in JavaScript."""
    rows = zip(*(columns[f] for f in fields))
    return [NAN if any(isinstance(x, str) for x in row) else function(*row) for row in rows]


def _derived(node_session, tmp_path, tables, joins, on, which):
    from RootInteractive.InteractiveDrawing.bokeh.CDSAlias import CDSAlias
    from RootInteractive.InteractiveDrawing.bokeh.CustomJSNAryFunction import CustomJSNAryFunction
    fields, names, body, function = DERIVED[which]
    def more(models):
        transform = CustomJSNAryFunction(fields=names, func=body)
        return {"AL": CDSAlias(source=models[on], mapping={which: {"fields": fields, "transform": transform}},
                               includeOrigColumns=True, name="AL")}
    out = _read(node_session, tmp_path, tables, joins, "AL", [which], more=more)
    want = {which: _apply(function, _reference_chain(tables, joins, on, fields), fields)}
    today = {which: _apply(function, _today(tables, joins, on, fields), fields)}
    return _verdict(out, [which], want, today)


@pytest.mark.feature("JOIN.recursive")
@pytest.mark.parametrize("how", ["left", "inner"])
def test_strict_v0_mass_from_both_joined_daughters(node_session, tmp_path, physics, how):
    """The physics check of the whole chain: the browser computes the V0 mass from the joined daughters' momenta."""
    src = _sources(physics)
    _strict(_derived(node_session, tmp_path, src, _daughter_joins(how), "VPN", "mass"), f"V0 mass, {how}")


@pytest.mark.feature("JOIN.cdsjoin.basic")
def test_strict_derived_column_from_a_child_and_a_parent_column(node_session, tmp_path, physics):
    """The share of the V0's transverse momentum carried by its positive daughter: V0 column over track column."""
    joins = {"VP": _link("V0", "TR", "vPos", "left", prefix_right="pos_")}
    _strict(_derived(node_session, tmp_path, _sources(physics), joins, "VP", "fraction"), "pt fraction")


@pytest.mark.feature("JOIN.cdsjoin.basic")
def test_control_derived_column_from_parent_columns_is_right_today(node_session, tmp_path, physics):
    """The healthy side: a derived column that reads only parent columns of one join is right today."""
    joins = {"VP": _link("V0", "TR", "vPos", "left", prefix_right="pos_")}
    assert _derived(node_session, tmp_path, _sources(physics), joins, "VP", "momentum") == "OK"


# ============================================================================================ a histogram of a joined column
NBINS = 8


def _counts(values, lo, hi):
    return np.histogram(np.array(values, dtype=float), bins=NBINS, range=(lo, hi))[0].tolist()


def _histogram(node_session, tmp_path, tables, joins, on, column):
    """Bin counts of a joined column. The range is wider than the data, so no value lies on an outer edge."""
    from RootInteractive.InteractiveDrawing.bokeh.HistogramCDS import HistogramCDS
    ref = _reference_chain(tables, joins, on, [column])[column]
    lo, hi = min(ref) - 1.0, max(ref) + 1.0
    more = lambda models: {"H": HistogramCDS(source=models[on], sample=column, nbins=NBINS, range=[lo, hi], name="H")}
    out = _read(node_session, tmp_path, tables, joins, "H", ["bin_count"], more=more)
    today = _today(tables, joins, on, [column])[column]
    assert not any(isinstance(x, str) for x in ref + today), "this test is for columns without missing values"
    return _verdict(out, ["bin_count"], {"bin_count": _counts(ref, lo, hi)}, {"bin_count": _counts(today, lo, hi)})


TE = {"TE": _link("TR", "CO", "tEventIdx", "left")}


@pytest.mark.feature("JOIN.cdsjoin.basic")
def test_control_histogram_of_a_parent_column_is_right_today(node_session, tmp_path, physics):
    assert _histogram(node_session, tmp_path, _sources(physics), TE, "TE", "cVz") == "OK"


@pytest.mark.feature("JOIN.cdsjoin.basic")
def test_strict_histogram_of_a_child_column(node_session, tmp_path, physics):
    _strict(_histogram(node_session, tmp_path, _sources(physics), TE, "TE", "tPt"), "histogram of the track pt")


# ============================================================================================ a change after a read
def _changed(tables, table, column, values):
    return {**tables, table: {**tables[table], column: list(values)}}


def _follows(node_session, tmp_path, tables, joins, name, keys, table, column, values):
    """Read, change one source column, read again. Both reads are sorted; the second must differ from the first."""
    after_tables = _changed(tables, table, column, values)
    want_before, want_after = (_reference_chain(t, joins, name, keys) for t in (tables, after_tables))
    assert want_before != want_after, "the change must be visible in the columns read"
    before, after = _read(node_session, tmp_path, tables, joins, name, keys,
                          change=[{"table": table, "data": {column: list(values)}}])
    verdicts = {_verdict(before, keys, want_before, _today(tables, joins, name, keys)),
                _verdict(after, keys, want_after, _today(after_tables, joins, name, keys))}
    return "KNOWN" if "KNOWN" in verdicts else "OK"


CHAIN = {"TE": _link("TR", "CO", "tEventIdx", "left"), "VPE": _link("V0", "TE", "vPos", "left")}


@pytest.mark.feature("JOIN.change_propagation")
@pytest.mark.parametrize("name", ["TE", "VPE"])
def test_control_a_changed_pointer_is_followed_in_parent_columns(node_session, tmp_path, physics, name):
    """Every track is moved to the next collision after the first read. Parent columns are right today, and follow."""
    src = _sources(physics)
    moved = [(e + 1) % len(src["CO"]["cRow"]) for e in src["TR"]["tEventIdx"]]
    assert _follows(node_session, tmp_path, src, CHAIN, name, ["cVz", "cRow"], "TR", "tEventIdx", moved) == "OK"


@pytest.mark.feature("JOIN.change_propagation")
@pytest.mark.parametrize("table,column,read", [("TR", "tPt", "tPt"), ("V0", "vMass", "vMass"),
                                               ("TR", "tEventIdx", "tRow")])
def test_strict_a_change_after_a_read_is_followed_in_child_columns(node_session, tmp_path, physics, table, column, read):
    """A value of the middle table, a value of the child table, and the pointer itself, each changed after a read."""
    src = _sources(physics)
    old = src[table][column]
    new = [(e + 1) % len(src["CO"]["cRow"]) for e in old] if column == "tEventIdx" else [x + 0.25 for x in old]
    _strict(_follows(node_session, tmp_path, src, CHAIN, "VPE", [read, "cVz"], table, column, new), f"{table}.{column}")


# ============================================================================================ no Node: the checks themselves
def test_the_mass_formula_reproduces_the_generators_stored_mass(physics):
    """The reference of the mass test is the generator's own number: the formula on the track momenta equals the
    stored V0 mass. Tolerance 1e-9 (relative): the generator builds its four-vectors from pt, eta and phi, this
    formula from px, py and pz, and the subtraction E^2 - p^2 loses a few digits."""
    src = _sources(physics)
    joins = _daughter_joins("left")
    cols = _reference_chain(src, joins, "VPN", MASS_FIELDS + ["vMass"])
    mass = _apply(_mass, cols, MASS_FIELDS)
    assert len(mass) == len(src["V0"]["vRow"]) > 0
    assert all(math.isclose(m, stored, rel_tol=1e-9) for m, stored in zip(mass, cols["vMass"]))
    assert len({round(m, 3) for m in mass}) > 5, "the masses must differ from row to row, or the test proves little"


def test_a_missing_input_gives_nan_and_a_wrong_row_changes_the_result():
    cols = {"pos_tPt": [1.0, UNDEF, 3.0], "vPt": [2.0, 4.0, NAN]}
    assert _apply(DERIVED["fraction"][3], cols, FRACTION_FIELDS) == [0.5, NAN, NAN]
    good = {"length": 2, "columns": {"fraction": [0.5, 0.25]}, "throws": {}}
    assert _verdict(good, ["fraction"], {"fraction": [0.5, 0.25]}, {"fraction": [0.25, 0.5]}) == "OK"
    with pytest.raises(AssertionError):                                # a value from another row is not accepted
        _verdict({**good, "columns": {"fraction": [0.5, 0.26]}}, ["fraction"], {"fraction": [0.5, 0.25]}, None)
    empty = {"length": 0, "columns": {"fraction": []}, "throws": {}}
    with pytest.raises(AssertionError):                                # an empty answer needs a registered defect
        _verdict(empty, ["fraction"], {"fraction": [0.5, 0.25]}, None)
    assert _verdict(empty, ["fraction"], {"fraction": [0.5, 0.25]}, {"fraction": []}) == "KNOWN"


def test_a_stale_answer_after_a_change_is_rejected():
    """The second read of `_follows` is compared with the values AFTER the change: an old answer cannot pass."""
    old, new = {"x": [1.0, 2.0]}, {"x": [1.25, 2.25]}
    stale = {"length": 2, "columns": dict(old), "throws": {}}
    assert _verdict(stale, ["x"], old, None) == "OK"
    with pytest.raises(AssertionError):
        _verdict(stale, ["x"], new, None)
    with pytest.raises(AssertionError):                                # also not as "today's known result"
        _verdict(stale, ["x"], new, {"x": [9.0, 9.0]})
