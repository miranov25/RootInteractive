"""PHASE_5_1: Node-backed PRODUCTION CDSJoin tests using four-table ToyMC physics.

This is an ADDITIVE test file: no RI/ADF production changes.  It uses the
PHASE_4 Node session fixture and actual Bokeh CDSJoin deserialization.  Only
known J-D1/J-D2/J-D5 signatures are strict expected failures; any other wrong
result, harness error or exception is a normal failure.  The Python oracle is
calculated from the original four DataFrames, not from the CDSJoin output.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

GEN_DIR = Path(__file__).resolve().parents[3] / "Tools" / "generators"
sys.path.insert(0, str(GEN_DIR))
from toy_mc_events import (generate_toy_mc_events, independent_gather_oracle,
                           inject_orphan_test_rows, validate_relations)

pytestmark = [pytest.mark.node_joins]
EXPECTED_SOURCE_MD5 = "10a9546d2b1c2632c1496d1be267197d"


class KnownJoinDefect(AssertionError):
    """The precise known production signature, not a generic assertion failure."""


STRICT = dict(strict=True, raises=KnownJoinDefect)


@pytest.fixture(scope="module")
def physics():
    path = GEN_DIR / "toy_mc_events.py"
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if actual != EXPECTED_SOURCE_MD5:
        pytest.fail(f"ToyMC generator MD5={actual}; expected {EXPECTED_SOURCE_MD5}; update test only after review")
    raw = generate_toy_mc_events(n_collisions=10, seed=42)
    assert validate_relations(raw)["collisions"] == 10
    return raw


def _sources(tables):
    c, t, v, ca = (tables[k] for k in ("collisions", "tracks", "v0s", "cascades"))
    def arr(frame, col):
        return [None if x is None or (isinstance(x, (float, np.floating)) and math.isnan(x))
                else x.item() if isinstance(x, np.generic) else x for x in frame[col].to_numpy()]
    return {
        "CO": {"cRow": arr(c, "rowId"), "cVz": arr(c, "vz")},
        "TR": {"tRow": arr(t, "rowId"), "tEventIdx": arr(t, "collisionIdx"), "tPt": arr(t, "pt")},
        "V0": {"vRow": arr(v, "rowId"), "vPos": arr(v, "posTrackIdx"),
               "vNeg": arr(v, "negTrackIdx"), "vMass": arr(v, "mass")},
        "CA": {"caRow": arr(ca, "rowId"), "caV0": arr(ca, "v0Idx"),
               "caBachelor": arr(ca, "bachelorTrackIdx"), "caMass": arr(ca, "mass")},
    }


def _relation(left, right, foreign_key, how="left"):
    return dict(left=left, right=right, on_left=[foreign_key], on_right=[], how=how)


def _single(which, how="left"):
    cases = {"TE": ("TR", "CO", "tEventIdx"),
             "VP": ("V0", "TR", "vPos"),
             "VN": ("V0", "TR", "vNeg"),
             "CV": ("CA", "V0", "caV0"),
             "CB": ("CA", "TR", "caBachelor")}
    return {which: _relation(*cases[which], how=how)}


def _right_nested():
    return {
        "TE": _relation("TR", "CO", "tEventIdx"),
        "VPE": _relation("V0", "TE", "vPos"),
        "VNE": _relation("V0", "TE", "vNeg"),
        "CBE": _relation("CA", "TE", "caBachelor"),
        "CVPE": _relation("CA", "VPE", "caV0"),
        "CVNE": _relation("CA", "VNE", "caV0"),
    }


# Read production CDSJoin through the existing PHASE_4 Node/Bokeh harness.
READ = """(a) => {
  const doc = Bokeh.documents[Bokeh.documents.length - 1];
  const m = doc.get_model_by_name(a.name);
  if (!m) return {error: "missing model " + a.name};
  // Preserve both infinities. Only an undefined/NaN missing value becomes null.
  const arr = xs => xs == null ? null : Array.from(xs, v =>
    v === undefined || Number.isNaN(v) ? null : v);
  const capture = () => {
    const out = {il: arr(m._indices_left), ir: arr(m._indices_right), columns: {}, throws: {}};
    try { out.length = m.get_length(); } catch (e) { out.throws.length = String(e); }
    for (const k of a.keys) {
      try { out.columns[k] = arr(m.get_column(k)); }
      catch (e) { out.throws[k] = String(e); }
    }
    return out;
  };
  const mutate = () => {
    for (const mu of (a.mutate || [])) {
      const t = doc.get_model_by_name(mu.table);
      t.data = Object.assign({}, t.data, mu.data);
    }
  };
  if (a.warm_mutate) {
    const before = capture();  // Populate lazy caches in THIS document before any mutation.
    mutate();
    return {before, after: capture()};
  }
  mutate();
  return capture();
}"""


def _assert_clean_node_reply(reply, operation):
    """T1: a usable numeric reply is not enough when the page emitted an error."""
    assert reply.get("ok") is True, f"{operation}: Node harness failed: {reply}"
    assert not reply.get("page_errors", []), f"{operation}: page errors: {reply['page_errors']}"
    assert not reply.get("console_errors", []), f"{operation}: console errors: {reply['console_errors']}"


def _assert_no_infinity(obj, location="reply"):
    """T2: do not let +/-Infinity pass as null, even in index arrays."""
    if isinstance(obj, dict):
        for k, value in obj.items():
            _assert_no_infinity(value, f"{location}.{k}")
    elif isinstance(obj, (tuple, list)):
        for i, value in enumerate(obj):
            _assert_no_infinity(value, f"{location}[{i}]")
    elif isinstance(obj, (float, np.floating)):
        assert not math.isinf(obj), f"{location}: infinite value is not a missing reference"


def _classify_j_d2_reply(out, fk):
    """T3: require the left-index shape for both healthy and known-bug signatures."""
    assert out["throws"] == {}, f"J-D2: unexpected exception {out['throws']}"
    assert out["il"] is None, f"J-D2: left indices must be unset, not {out['il']}"
    assert out["columns"] == {}, f"J-D2: unexpected columns {out['columns']}"
    if out["ir"] == fk and out["length"] == len(fk):
        return "correct"
    defect = [p for p in fk if p > 0]
    if out["ir"] == defect and out["length"] == len(defect) and fk.count(0) > 0 and len(defect) < len(fk):
        return "J-D2"
    raise AssertionError(f"J-D2: not the correct result or the exact documented signature: {out}")



def _read(node_session, tmp_path, tables, joins, target, keys, mutate=(), warm_mutate=False):
    from bokeh.document import Document
    from bokeh.models import ColumnDataSource
    from RootInteractive.InteractiveDrawing.bokeh.CDSJoin import CDSJoin
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode

    models = {name: ColumnDataSource(data=columns, name=name)
              for name, columns in _sources(tables).items()}
    for name, cfg in joins.items():
        models[name] = CDSJoin(left=models[cfg["left"]], right=models[cfg["right"]],
                               on_left=cfg["on_left"], on_right=cfg["on_right"],
                               how=cfg["how"], prefix_left="", prefix_right="", name=name)
    doc = Document()
    for name in joins:
        doc.add_root(models[name])
    path = tmp_path / "toymc_join_doc.json"
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    assert not getattr(node_session, "startup_page_errors", []), "Node startup page errors"
    assert not getattr(node_session, "startup_console_errors", []), "Node startup console errors"
    reply = node_session.request("new_doc", path=str(path))
    _assert_clean_node_reply(reply, "new_doc")
    reply = node_session.request("eval", code=READ,
                                 arg={"name": target, "keys": list(keys), "mutate": list(mutate),
                                      "warm_mutate": warm_mutate})
    _assert_clean_node_reply(reply, "eval")
    result = decode(reply["result"])
    assert "error" not in result, result
    _assert_no_infinity(result)
    return result


def _check(out, field, expected, label):
    assert out["throws"] == {}, f"{label}: exceptions are NEVER an expected defect: {out['throws']}"
    assert set(out["columns"]) == {field}, f"{label}: incomplete response {out['columns']}"
    values = out["columns"][field]
    assert len(values) == out["length"] == len(expected), (label, len(values), out["length"], len(expected))
    _assert_no_infinity(out)
    for i, (got, want) in enumerate(zip(values, expected)):
        if want is None:
            assert got is None, (label, i, got, want)
        elif isinstance(want, (float, np.floating)):
            assert isinstance(got, (float, int)) and math.isclose(got, want, rel_tol=1e-12, abs_tol=2e-12), \
                (label, i, got, want)
        else:
            assert got == want, (label, i, got, want)


def _take(values, refs):
    out = []
    for p in refs:
        out.append(None if p is None or not 0 <= p < len(values) else values[p])
    return out


@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.parametrize("join,field,oracle", [
    ("TE", "cVz", "tracks.collisionVz"),
    ("VP", "tPt", "v0s.posPt"),
    ("VN", "tPt", "v0s.negPt"),
    ("CV", "vMass", "cascades.v0Mass"),
    ("CB", "tPt", "cascades.bachelorPt"),
])
def test_tmc_five_single_gathers(node_session, tmp_path, physics, join, field, oracle):
    """Actual production source, not a Python emulation or JavaScript prototype."""
    out = _read(node_session, tmp_path, physics, _single(join), join, [field])
    _check(out, field, list(independent_gather_oracle(physics)[oracle]), join)


@pytest.mark.feature("JOIN.recursive")
@pytest.mark.parametrize("target,source_key,join_key", [
    ("VPE", "v0s", "posTrackIdx"),
    ("VNE", "v0s", "negTrackIdx"),
    ("CBE", "cascades", "bachelorTrackIdx"),
    ("CVPE", "cascades", "v0Idx"),
    ("CVNE", "cascades", "v0Idx"),
])
def test_tmc_right_nested_collision_vz(node_session, tmp_path, physics, target, source_key, join_key):
    """Collision vz is a RIGHT-hand field at every join, avoiding J-D1 by construction.

    This tests actual recursive navigation, not just a copied direct collisionIdx.
    """
    t, v, ca, c = (physics[k] for k in ("tracks", "v0s", "cascades", "collisions"))
    ev = t.collisionIdx.tolist()
    vz = c.vz.tolist()
    if target == "VPE":
        want = _take(vz, _take(ev, v.posTrackIdx.tolist()))
    elif target == "VNE":
        want = _take(vz, _take(ev, v.negTrackIdx.tolist()))
    elif target == "CBE":
        want = _take(vz, _take(ev, ca.bachelorTrackIdx.tolist()))
    elif target == "CVPE":
        want = _take(vz, _take(_take(ev, v.posTrackIdx.tolist()), ca.v0Idx.tolist()))
    else:
        want = _take(vz, _take(_take(ev, v.negTrackIdx.tolist()), ca.v0Idx.tolist()))
    assert len(want) == len(physics[source_key]) and len(want) > 0
    out = _read(node_session, tmp_path, physics, _right_nested(), target, ["cVz"])
    _check(out, "cVz", want, f"{target} {join_key}")


@pytest.mark.feature("JOIN.change_propagation")
def test_tmc_mutate_collision_vz_reaches_recursive_cascades(node_session, tmp_path, physics):
    changed = (physics["collisions"].vz + 7.25).tolist()
    out = _read(node_session, tmp_path, physics, _right_nested(), "CVPE", ["cVz"],
                mutate=[{"table": "CO", "data": {"cVz": changed}}])
    t, v, ca = (physics[x] for x in ("tracks", "v0s", "cascades"))
    refs = _take(_take(t.collisionIdx.tolist(), v.posTrackIdx.tolist()), ca.v0Idx.tolist())
    _check(out, "cVz", _take(changed, refs), "collision vz mutation")


@pytest.mark.feature("JOIN.cdsjoin.basic")
def test_tmc_orphan_left_gather_nan_fill(node_session, tmp_path, physics):
    orphans = inject_orphan_test_rows(physics)
    validate_relations(orphans, allow_orphans=True)
    oracle = independent_gather_oracle(orphans)
    out = _read(node_session, tmp_path, orphans, _single("TE"), "TE", ["cVz"])
    want = [None if np.isnan(x) else float(x) for x in oracle["tracks.collisionVz"]]
    _check(out, "cVz", want, "orphan track(-1) -> missing collision")
    assert want[-1] is None and out["columns"]["cVz"][-1] is None


@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.xfail(**STRICT, reason="J-D1: left field gathered using right-side indices")
def test_tmc_strict_left_column_must_use_track_row(node_session, tmp_path, physics):
    src = _sources(physics)
    out = _read(node_session, tmp_path, physics, _single("TE"), "TE", ["tRow"])
    want = src["TR"]["tRow"]
    if not out["throws"] and out["length"] == len(want) and out["columns"].get("tRow") == want:
        return
    defect = _take(src["TR"]["tRow"], src["TR"]["tEventIdx"])
    if not out["throws"] and out["length"] == len(want) and out["columns"].get("tRow") == defect and defect != want:
        raise KnownJoinDefect("J-D1 signature: TR.tRow gathered by collision index")
    pytest.fail(f"J-D1 test returned neither correct rows nor exact registered defect: {out}")


@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.xfail(**STRICT, reason="J-D2: INNER index join incorrectly drops parent row 0")
def test_tmc_strict_inner_keeps_event_zero(node_session, tmp_path, physics):
    src = _sources(physics)
    out = _read(node_session, tmp_path, physics, _single("TE", how="inner"), "TE", [])
    fk = src["TR"]["tEventIdx"]
    verdict = _classify_j_d2_reply(out, fk)
    if verdict == "J-D2":
        raise KnownJoinDefect("J-D2 signature: exactly event-zero tracks dropped")


@pytest.mark.feature("JOIN.recursive")
@pytest.mark.xfail(**STRICT, reason="J-D5: left-nested CDSJoin reads empty cached join key")
def test_tmc_strict_left_nested_v0_track_collision(node_session, tmp_path, physics):
    joins = {"VT": _relation("V0", "TR", "vPos"),
             "VTE": _relation("VT", "CO", "tEventIdx")}
    out = _read(node_session, tmp_path, physics, joins, "VTE", ["cVz"])
    v = physics["v0s"]
    t, c = physics["tracks"], physics["collisions"]
    want = _take(c.vz.tolist(), _take(t.collisionIdx.tolist(), v.posTrackIdx.tolist()))
    if out["throws"] == {} and out["length"] == len(want) and out["columns"].get("cVz") == want:
        return
    if (out["throws"] == {} and out["length"] == 0
            and out["columns"].get("cVz") == [] and len(want) > 0):
        raise KnownJoinDefect("J-D5 signature: left-nested join cold-load length zero")
    pytest.fail(f"J-D5: unclassified result: {out}")


def test_tmc_oracle_catches_shift_and_wrong_branch(physics):
    """No Node needed: falsify the fixture/reference, not the product."""
    c, t, v, ca = (physics[x] for x in ("collisions", "tracks", "v0s", "cascades"))
    correct = _take(c.vz.tolist(), t.collisionIdx.tolist())
    shifted = t.collisionIdx.to_numpy().copy()
    shifted[0] = len(c) - 1
    assert correct != _take(c.vz.tolist(), shifted.tolist())
    pos = _take(t.pt.tolist(), v.posTrackIdx.tolist())
    neg = _take(t.pt.tolist(), v.negTrackIdx.tolist())
    assert pos != neg
    assert len(ca) > 0 and len(v) > 0


# v0.4 Node-harness controls: all must work without the installed Bokeh product.
def test_tmc_node_page_and_console_errors_rejected_even_with_good_numbers():
    """T1: a numeric answer alongside a JavaScript error is a hard failure."""
    good = {"ok": True, "result": {"columns": {"cVz": [1.0]}}}
    _assert_clean_node_reply(good, "eval")
    for key in ("page_errors", "console_errors"):
        with pytest.raises(AssertionError, match="errors"):
            _assert_clean_node_reply({**good, key: ["uncaught JavaScript error"]}, "eval")
        with pytest.raises(AssertionError, match="errors"):
            _assert_clean_node_reply({**good, key: ["uncaught JavaScript error"]}, "new_doc")
    with pytest.raises(AssertionError, match="failed"):
        _assert_clean_node_reply({**good, "ok": False}, "eval")


def test_tmc_infinity_must_not_equal_missing_orphan():
    """T2: missing null is accepted; both infinite values are distinct and forbidden."""
    good = {"length": 2, "columns": {"cVz": [1.25, None]}, "throws": {}, "il": None, "ir": [0, -1]}
    _check(good, "cVz", [1.25, None], "good orphan control")
    for infinite in (float("inf"), -float("inf")):
        for field in ("columns", "ir"):
            corrupt = {**good, "columns": {"cVz": [1.25, infinite]}} if field == "columns" else {**good, "ir": [0, infinite]}
            with pytest.raises(AssertionError, match="infinite"):
                _assert_no_infinity(corrupt)
            with pytest.raises(AssertionError, match="infinite"):
                _check(corrupt, "cVz", [1.25, None], "corrupt orphan")


def test_tmc_known_j_d2_signature_rejects_corrupted_left_indices():
    """T3: unrelated index corruption must be FAIL, never strict XFAIL."""
    fk = [0, 0, 1, 2]
    healthy = {"il": None, "ir": list(fk), "length": len(fk), "throws": {}, "columns": {}}
    assert _classify_j_d2_reply(healthy, fk) == "correct"
    defect = {**healthy, "ir": [1, 2], "length": 2}
    assert _classify_j_d2_reply(defect, fk) == "J-D2"
    for sample in (healthy, defect):
        for broken in ([999] * len(sample["ir"]), [], [-1]):
            with pytest.raises(AssertionError, match="left indices"):
                _classify_j_d2_reply({**sample, "il": broken}, fk)


def _branch_joins():
    """V0 -> positive/negative tracks; cascade -> each daughter role."""
    return {"VP": _relation("V0", "TR", "vPos"),
            "VN": _relation("V0", "TR", "vNeg"),
            "CVP": _relation("CA", "VP", "caV0"),
            "CVN": _relation("CA", "VN", "caV0"),
            "CB": _relation("CA", "TR", "caBachelor")}


def _branch_pt_reference(tables, target):
    """T5: NumPy positional gather from physical track pt; not from CDSJoin."""
    t, v, ca = (tables[n] for n in ("tracks", "v0s", "cascades"))
    pt = t.pt.to_numpy()
    pos, neg, bach, cv = (x.to_numpy() for x in
                          (v.posTrackIdx, v.negTrackIdx, ca.bachelorTrackIdx, ca.v0Idx))
    if target == "VP":
        return pt[pos].tolist()
    if target == "VN":
        return pt[neg].tolist()
    if target == "CVP":
        return pt[pos[cv]].tolist()
    if target == "CVN":
        return pt[neg[cv]].tolist()
    if target == "CB":
        return pt[bach].tolist()
    raise ValueError(target)


@pytest.mark.feature("JOIN.recursive")
@pytest.mark.parametrize("target", ["VP", "VN", "CVP", "CVN", "CB"])
def test_tmc_track_pt_distinguishes_every_branch(node_session, tmp_path, physics, target):
    """T5: track pt from actual model, not common collision-z across all paths."""
    want = _branch_pt_reference(physics, target)
    assert len(want) == len(physics["v0s"] if target in ("VP", "VN") else physics["cascades"])
    assert len(want) > 0
    out = _read(node_session, tmp_path, physics, _branch_joins(), target, ["tPt"])
    _check(out, "tPt", want, f"{target} track pt")


def test_tmc_wrong_daughter_and_bachelor_branches_are_falsified(physics):
    """T5 negative control: collision ancestry alone would NOT detect these swaps."""
    for a, b in (("VP", "VN"), ("CVP", "CVN"), ("CVP", "CB"), ("CVN", "CB")):
        correct, wrong = (_branch_pt_reference(physics, name) for name in (a, b))
        assert correct != wrong, (a, b, "degenerate fixture")
        with pytest.raises(AssertionError):
            _check({"length": len(correct), "columns": {"tPt": wrong}, "throws": {}},
                   "tPt", correct, f"mutant: swap {a} for {b}")


@pytest.mark.feature("JOIN.change_propagation")
def test_tmc_warm_cache_recursive_cascade_update_on_same_document(node_session, tmp_path, physics):
    """T4: read initial value, THEN mutate and read again within one deserialized page."""
    from copy import deepcopy
    changed = (physics["collisions"].vz + 7.25).tolist()
    t, v, ca, c = (physics[n] for n in ("tracks", "v0s", "cascades", "collisions"))
    refs = _take(_take(t.collisionIdx.tolist(), v.posTrackIdx.tolist()), ca.v0Idx.tolist())
    old = _take(c.vz.tolist(), refs)
    new = _take(changed, refs)
    assert old != new and len(new) > 0
    out = _read(node_session, tmp_path, physics, _right_nested(), "CVPE", ["cVz"],
                mutate=[{"table": "CO", "data": {"cVz": changed}}], warm_mutate=True)
    assert set(out) == {"before", "after"}, out
    _check(out["before"], "cVz", old, "warm-cache original")
    _check(out["after"], "cVz", new, "warm-cache after mutation")


def test_tmc_stale_warm_cache_reply_fails_independent_updated_oracle(physics):
    """T4 negative control: a cached pre-mutation answer cannot pass as updated."""
    t, v, ca, c = (physics[n] for n in ("tracks", "v0s", "cascades", "collisions"))
    refs = _take(_take(t.collisionIdx.tolist(), v.posTrackIdx.tolist()), ca.v0Idx.tolist())
    old = _take(c.vz.tolist(), refs)
    new = _take((c.vz + 7.25).tolist(), refs)
    assert old != new
    stale = {"length": len(old), "columns": {"cVz": old}, "throws": {}}
    _check(stale, "cVz", old, "initial cache healthy")
    with pytest.raises(AssertionError):
        _check(stale, "cVz", new, "mutant stale cache after change")
