"""PHASE_5 Batch 4.c, step 1: the cascade chain on a real dashboard page. Tests only; RootInteractive's code is not
changed.

In plain words. The page is built by bokehDrawSA, as a user builds it, from the four ToyMC tables: the cascades are
the main table; collisions, tracks and V0s are named tables. Joins by row number lead from a cascade to its V0, to
the V0's positive and negative track, to the cascade's bachelor track, and from each track to its collision:
    TE    track -> collision
    VPE   V0 -> (positive track -> collision)       VNE   the same for the negative track
    CVPE  cascade -> (V0 -> (positive track -> collision))      CVNE  the same for the negative track
    CBE   cascade -> (bachelor track -> collision)
The tests read every column of the three cascade joins from the page's own models and compare them, row by row, with
two references that use no join:
  * a plain gather on the ToyMC tables, written here;
  * the reviewers' table of expected values for the 10 cascades (GPT25.RI; installed byte-identical under oracles/,
    SHA256 checked below).
The page is read in Node and, with the same reading code, in a real browser (Chromium).

Two defects of the dashboard show here; each is a strict expected failure that accepts only its exact signature:
  B1  the page builder does not send the columns a join needs (the pointer columns and the columns read through the
      join). A column arrives only if a figure of its own table uses it. Without such figures the join track ->
      collision is empty and the cascade joins refuse with "(0 rows)".
  B2  on a dashboard every table is wrapped in an alias table, and a row number pointing into an alias table around
      an inner join that lost rows is taken as a position, not as the row of the original table (review finding F1
      of the join fix). The result: a wrong row without any message, or a refusal of a row that exists.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4a_joins import (
    NAN, _clean_reply, _clean_session, _is_refusal, _same, _verdict)
from toy_mc_events import generate_toy_mc_events, inject_orphan_test_rows    # the path is set by the import above

HERE = Path(__file__).resolve().parent
ORACLE_PATH = HERE / "oracles" / "PHASE_5_Batch4b_Cascade_Collision_Independent_Oracle_seed42.json"
ORACLE_SHA256 = "1279710bed8d127413c815bcf8e4d20136c8679f3bc7255f0b940c5e57c582f9"      # GPT25.RI, 2026-11-15
GENERATOR_SHA256 = "9ced93f0dfc394574a778c7d70f5c4bc81d90bfc543771703bbd454d9a9b2ad5"   # the generator the oracle names

OWN = {"CA": ["caRow", "caMass"], "V0": ["vRow", "vMass"], "TR": ["tRow", "tPt"], "CO": ["cRow", "cVz"]}
VIEWS = {"CVPE": ("caV0", "vPos"), "CVNE": ("caV0", "vNeg"), "CBE": ("caBachelor",)}
KEYS = {"CVPE": OWN["CA"] + OWN["V0"] + OWN["TR"] + OWN["CO"], "CVNE": OWN["CA"] + OWN["V0"] + OWN["TR"] + OWN["CO"],
        "CBE": OWN["CA"] + OWN["TR"] + OWN["CO"]}
# the reviewers' column names for the same values
ORACLE_NAMES = {
    "CVPE": {"caMass": "cascade_mass", "vRow": "cascade_v0_original_row", "vMass": "v0_mass",
             "tRow": "v0_pos_track_original_row", "tPt": "pos_track_pt", "cRow": "pos_collision_original_row",
             "cVz": "pos_collision_vz"},
    "CVNE": {"caMass": "cascade_mass", "vRow": "cascade_v0_original_row", "vMass": "v0_mass",
             "tRow": "v0_neg_track_original_row", "tPt": "neg_track_pt", "cRow": "neg_collision_original_row",
             "cVz": "neg_collision_vz"},
    "CBE": {"caMass": "cascade_mass", "tRow": "bachelor_track_original_row", "tPt": "bachelor_track_pt",
            "cRow": "bachelor_collision_original_row", "cVz": "bachelor_collision_vz"},
}


class KnownDashboardDefect(AssertionError):
    """Raised only when the page shows the exact signature of B1 or B2 (see the top of this file)."""


STRICT = dict(strict=True, raises=KnownDashboardDefect)


@pytest.fixture(scope="module")
def physics():
    return generate_toy_mc_events(n_collisions=10, seed=42)


# ============================================================================================ the tables and the page
def _tables(raw):
    """The four tables as plain columns, with the column names used on the page."""
    c, t, v, ca = (raw[k] for k in ("collisions", "tracks", "v0s", "cascades"))
    col = lambda frame, name: [x.item() if isinstance(x, np.generic) else x for x in frame[name].to_numpy()]
    return {"CO": {"cRow": col(c, "rowId"), "cVz": col(c, "vz")},
            "TR": {"tRow": col(t, "rowId"), "tEventIdx": col(t, "collisionIdx"), "tPt": col(t, "pt")},
            "V0": {"vRow": col(v, "rowId"), "vPos": col(v, "posTrackIdx"), "vNeg": col(v, "negTrackIdx"),
                   "vMass": col(v, "mass")},
            "CA": {"caRow": col(ca, "rowId"), "caV0": col(ca, "v0Idx"), "caBachelor": col(ca, "bachelorTrackIdx"),
                   "caMass": col(ca, "mass")}}


def _page(tables, how, own_figures):
    """The page through the user path, bokehDrawSA.fromArray; `show` is suppressed, nothing is written.
    `own_figures`: also show, in figures of their own tables, every column the chain needs (see B1)."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    arrays = lambda name: {k: np.asarray(v) for k, v in tables[name].items()}
    join = lambda name, left, right, key: {"name": name, "left": left, "right": right, "left_on": [key],
                                           "right_on": [], "how": how}
    sources = [{"name": n, "data": arrays(n)} for n in ("CO", "TR", "V0")] + [
        join("TE", "TR", "CO", "tEventIdx"), join("VPE", "V0", "TE", "vPos"), join("VNE", "V0", "TE", "vNeg"),
        join("CVPE", None, "VPE", "caV0"), join("CVNE", None, "VNE", "caV0"), join("CBE", None, "TE", "caBachelor")]
    figures = [[["caRow"], ["caMass"]],
               [["CVPE.vMass"], ["CVPE.tPt"]], [["CVPE.caMass"], ["CVPE.cVz"]],
               [["CVNE.vMass"], ["CVNE.tPt"]], [["CBE.tPt"], ["CBE.cVz"]]]
    if own_figures:
        figures += [[["caV0"], ["caBachelor"]], [["CO.cRow"], ["CO.cVz"]], [["TR.tRow"], ["TR.tEventIdx"]],
                    [["TR.tRow"], ["TR.tPt"]], [["V0.vRow"], ["V0.vPos"]], [["V0.vNeg"], ["V0.vMass"]]]
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        return sa.bokehDrawSA.fromArray(pd.DataFrame(arrays("CA")), None, figures, [["range", ["caMass"]]],
                                        sourceArray=sources, layout=[list(range(i, min(i + 4, len(figures)))) for i in range(0, len(figures), 4)],
                                        widgetLayout=[[0]]).pAll
    finally:
        sa.show = shown


READ = """(a) => {
  const doc = Bokeh.documents[Bokeh.documents.length - 1];
  const joins = name => Array.from(doc._all_models.values()).filter(m => m.constructor.__name__ === "CDSJoin" && m.name === name);
  const tok = v => v === undefined ? "<undefined>" : v === null ? "<null>"
    : (typeof v === "number" && Number.isNaN(v)) ? "<NaN>" : v === Infinity ? "<+inf>" : v === -Infinity ? "<-inf>" : v;
  const isObj = e => e !== null && typeof e === "object";
  if (a.change) {                                      // replace one column of a named table, on this page
    const m = joins(a.change.join);
    if (m.length !== 1) return {error: "no single join named " + a.change.join};
    const table = m[0][a.change.side].source;
    table.data = Object.assign({}, table.data, {[a.change.column]: a.change.values});
  }
  const out = {};
  for (const name of a.names) {
    const m = joins(name);
    if (m.length !== 1) return {error: "the page must hold exactly one join named " + name};
    const o = {columns: {}, throws: {}, kinds: {}, sides: [m[0].left.constructor.__name__, m[0].right.constructor.__name__]};
    const note = (k, e) => { o.throws[k] = (isObj(e) && "message" in e) ? String(e.message) : String(e);
                             o.kinds[k] = (isObj(e) && "name" in e) ? String(e.name) : typeof e; };
    try { o.length = tok(m[0].get_length()); } catch (e) { note("length", e); }
    for (const k of a.keys[name]) {
      try { const c = m[0].get_column(k); o.columns[k] = c == null ? tok(c) : Array.from(c, tok); } catch (e) { note(k, e); }
    }
    out[name] = o;
  }
  return out;
}"""


def _node(node_session, tmp_path, tables, how, own_figures=True, names=tuple(VIEWS), change=None):
    """The joins of the page, read in Node. Any harness problem, page error or console error fails the test."""
    from bokeh.document import Document
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    doc = Document()
    doc.add_root(_page(tables, how, own_figures))
    path = tmp_path / "cascade_dashboard.json"
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    _clean_session(node_session)
    arg = {"names": list(names), "keys": {n: KEYS.get(n, []) for n in names}, "change": change}
    for op, kw in (("new_doc", dict(path=str(path))), ("eval", dict(code=READ, arg=arg))):
        reply = node_session.request(op, **kw)
        _clean_reply(reply, op)
    out = decode(reply["result"])
    assert "error" not in out, out
    return out


# ============================================================================================ the reference: a plain gather
class _Refused(Exception):
    """The by-position model of B2 refuses: join, column, row number, rows of the shortened parent."""


def _rows(tables, name):
    """The table's own columns, row by row. A NaN of the table itself arrives from the page as the NaN mark."""
    cols = OWN[name]
    mark = lambda v: NAN if isinstance(v, float) and math.isnan(v) else v
    return [dict(zip(cols, map(mark, values))) for values in zip(*(tables[name][c] for c in cols))]


def _level(own, pointers, parent, parent_keys, how, by_position=None):
    """One join by row number. `parent` is listed by ORIGINAL row; None marks a row an inner join dropped.
    `by_position` = (join, column): the WRONG model of B2 - the row number is taken as a position in the shortened
    parent, and a number past its end is refused."""
    if by_position:
        parent = [r for r in parent if r is not None]
    out = []
    for row, p in zip(own, pointers):
        if by_position and p is not None and p >= len(parent):
            raise _Refused(*by_position, p, len(parent))
        partner = parent[p] if p is not None and 0 <= p < len(parent) else None
        if partner is None:
            out.append(None if how == "inner" else {**row, **{k: NAN for k in parent_keys}})
        else:
            out.append({**row, **partner})
    return out


def _reference(tables, how, by_position=False):
    """Every cascade join as columns, from the four tables alone. With `by_position`: what B2 shows instead; a
    refusal is returned as the _Refused exception in place of the columns."""
    t = tables
    mark = lambda join, column: (join, column) if by_position else None
    track_keys = OWN["TR"] + OWN["CO"]
    out = {}
    te = _level(_rows(t, "TR"), t["TR"]["tEventIdx"], _rows(t, "CO"), OWN["CO"], how)     # the parent is a plain table
    for name, links in VIEWS.items():
        try:
            if len(links) == 1:                                                              # cascade -> bachelor track
                rows = _level(_rows(t, "CA"), t["CA"][links[0]], te, track_keys, how, mark(name, links[0]))
            else:
                v0 = _level(_rows(t, "V0"), t["V0"][links[1]], te, track_keys, how, mark("VPE" if name == "CVPE" else "VNE", links[1]))
                rows = _level(_rows(t, "CA"), t["CA"][links[0]], v0, OWN["V0"] + track_keys, how, mark(name, links[0]))
            out[name] = {k: [r[k] for r in rows if r is not None] for k in KEYS[name]}
        except _Refused as refusal:
            out[name] = refusal
    return out


def _oracle():
    assert hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest() == ORACLE_SHA256, "the reviewers' table was changed"
    return json.loads(ORACLE_PATH.read_text(encoding="utf-8"))


def _all_right(out, tables, how):
    want = _reference(tables, how)
    for name in VIEWS:
        assert out[name]["sides"][1] == "CDSAlias", out[name]["sides"]         # the parent side is an alias table
        assert _verdict(out[name], KEYS[name], want[name], None) == "OK", name
    return want


# ============================================================================================ the values
@pytest.mark.node_joins
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.parametrize("how", ["left", "inner"])
def test_node_every_column_of_the_cascade_chain_equals_the_plain_gather(node_session, tmp_path, physics, how):
    """Every table has its parent here, so `left` and `inner` give the same rows. Every column of every level, in the
    cascade table's order."""
    tables = _tables(physics)
    want = _all_right(_node(node_session, tmp_path, tables, how), tables, how)
    assert want == _reference(tables, "left") and len(want["CVPE"]["caRow"]) == len(tables["CA"]["caRow"]) == 10


@pytest.mark.node_joins
@pytest.mark.feature("JOIN.recursive")
def test_node_the_cascade_chain_equals_the_reviewers_table(node_session, tmp_path, physics):
    """The second reference, not the coder's: GPT25.RI's table of expected values, one row per cascade."""
    oracle = _oracle()
    out = _node(node_session, tmp_path, _tables(physics), "left")
    for name, names in ORACLE_NAMES.items():
        want = {k: [row[theirs] for row in oracle["rows"]] for k, theirs in names.items()}
        got = {k: out[name]["columns"][k] for k in names}
        assert out[name]["throws"] == {} and _same(got, want), (name, got, want)
    assert out["CVPE"]["columns"]["caRow"] == [row["cascade_row"] for row in oracle["rows"]]


@pytest.mark.node_joins
@pytest.mark.feature("JOIN.recursive")
def test_node_rows_without_a_parent_show_nan_in_a_left_chain(node_session, tmp_path, physics):
    """The ToyMC with rows that have no parent (-1) at every level; `left` keeps them, filled with NaN."""
    tables = _tables(inject_orphan_test_rows(physics))
    want = _all_right(_node(node_session, tmp_path, tables, "left"), tables, "left")
    assert all(NAN in want[name]["cVz"] for name in VIEWS) and len(want["CVPE"]["caRow"]) == len(tables["CA"]["caRow"])


@pytest.mark.node_joins
@pytest.mark.feature("JOIN.change_propagation")
def test_node_a_change_of_the_collision_table_reaches_the_cascade_joins(node_session, tmp_path, physics):
    """Read the page, then replace the collisions' vz on the same page: every cascade join shows the new values."""
    tables = _tables(physics)
    moved = [z + 7.25 for z in tables["CO"]["cVz"]]
    change = {"join": "TE", "side": "right", "column": "cVz", "values": moved}
    out = _node(node_session, tmp_path, tables, "left", change=change)
    after = {**tables, "CO": {**tables["CO"], "cVz": moved}}
    want = _all_right(out, after, "left")
    assert not _same(want["CVPE"], _reference(tables, "left")["CVPE"])                 # the change is visible


# ============================================================================================ B1: columns not sent
@pytest.mark.node_joins
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.xfail(**STRICT, reason="B1: the page builder does not send the columns a join needs")
def test_strict_the_chain_needs_no_extra_figures(node_session, tmp_path, physics):
    """The same page without the figures that show the pointer columns in their own tables. It must give the same
    values. Today the tracks' pointer column is not on the page: the join track -> collision has no rows, and every
    cascade join refuses its first row number as outside a parent table of 0 rows."""
    tables = _tables(physics)
    out = _node(node_session, tmp_path, tables, "left", own_figures=False, names=("TE",) + tuple(VIEWS))
    want = _reference(tables, "left")
    if all(out[n]["throws"] == {} and _same({k: out[n]["columns"][k] for k in KEYS[n]}, want[n]) for n in VIEWS):
        return
    first = {"CVPE": ("caV0", tables["CA"]["caV0"][0]), "CVNE": ("caV0", tables["CA"]["caV0"][0]),
             "CBE": ("caBachelor", tables["CA"]["caBachelor"][0])}
    empty = out["TE"].get("length") == 0 and out["TE"]["throws"] == {}
    refused = all(_is_refusal(out[n], KEYS[n], n, *first[n]) and all("(0 rows)" in m for m in out[n]["throws"].values())
                  for n in VIEWS)
    if empty and refused:
        raise KnownDashboardDefect("B1: the join's columns are not sent to the page")
    raise AssertionError(f"neither the reference nor the registered signature of B1: {out}")


# ============================================================================================ B2: the original row
def _lose_the_first_track(tables):
    """Track 0 loses its collision, so an inner join track -> collision drops the FIRST track: every later track
    stands one position earlier than its original row number."""
    pointers = list(tables["TR"]["tEventIdx"])
    pointers[0] = -1
    return {**tables, "TR": {**tables["TR"], "tEventIdx": pointers}}


@pytest.mark.node_joins
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.xfail(**STRICT, reason="B2: a row number into an alias table around an inner join that lost rows is "
                                    "taken as a position (review finding F1 of the join fix)")
def test_strict_an_inner_chain_that_lost_a_row_keeps_the_original_row_numbers(node_session, tmp_path, physics):
    """Decided by the Architect on 2026-11-15: a row number means the row of the ORIGINAL table, and the child row is
    kept only if that row is still there. Today, through the alias table, each join shows the by-position page, or
    refuses a row number that is past the end of the SHORTENED table."""
    tables = _lose_the_first_track(_tables(physics))
    out = _node(node_session, tmp_path, tables, "inner")
    want, today = _reference(tables, "inner"), _reference(tables, "inner", by_position=True)
    assert any(isinstance(today[n], _Refused) or not _same(today[n], want[n]) for n in VIEWS)   # the case can tell
    if all(out[n]["throws"] == {} and _same({k: out[n]["columns"][k] for k in KEYS[n]}, want[n]) for n in VIEWS):
        return
    for n in VIEWS:
        if isinstance(today[n], _Refused):
            join, column, value, count = today[n].args
            known = _is_refusal(out[n], KEYS[n], join, column, value) and \
                all(f"({count} rows)" in m for m in out[n]["throws"].values())
        else:
            known = out[n]["throws"] == {} and _same({k: out[n]["columns"][k] for k in KEYS[n]}, today[n])
        assert known, f"{n}: neither the reference nor the registered signature of B2: {out[n]}"
    raise KnownDashboardDefect("B2: row numbers are taken as positions in the shortened table")


# ============================================================================================ the real browser
def _open(headless_session, html):
    """The page in Chromium with the pinned local copies of its external scripts; every other request is blocked
    (as in test_headless_a4_multitable.py)."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine import vendor
    routes = {glob.replace("**/", "/"): path for glob, path in vendor.routes().items()}
    ctx = headless_session.browser.new_context()
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda err: errors.append("pageerror: " + str(err)))
    page.on("console", lambda msg: errors.append("console.error: " + msg.text) if msg.type == "error" else None)

    def route(r):
        url = r.request.url
        if url.startswith("file:"):
            return r.continue_()
        for suffix, path in routes.items():
            if url.split("?")[0].endswith(suffix):
                return r.fulfill(path=str(path), content_type="text/javascript")
        errors.append("blocked request: " + url)
        return r.abort()
    page.route("**/*", route)
    page.goto(html.resolve().as_uri())
    page.wait_for_function("() => window.Bokeh && Bokeh.documents.length > 0 && Bokeh.documents[0]._all_models.size > 0",
                           timeout=60_000)
    return ctx, page, errors


@pytest.mark.browser_a4
@pytest.mark.backend("browser")
@pytest.mark.layer("integration")
@pytest.mark.feature("JOIN.recursive")
def test_browser_the_cascade_chain_equals_the_plain_gather_and_the_reviewers_table(headless_session, tmp_path, physics):
    """The same page as a standalone HTML file in Chromium, read with the same code as in Node: every column of the
    three cascade joins equals the plain gather and the reviewers' table, and the page reports no error. The Node
    tests above compare with the same two references, so the browser and Node show the same values."""
    from bokeh.io import save
    from bokeh.resources import INLINE
    tables = _tables(physics)
    html = tmp_path / "cascade_dashboard.html"
    save(_page(tables, "left", True), filename=str(html), resources=INLINE, title="cascade -> V0 -> tracks -> collision")
    ctx, page, errors = _open(headless_session, html)
    try:
        out = page.evaluate(READ, {"names": list(VIEWS), "keys": KEYS, "change": None})
    finally:
        ctx.close()
    assert not errors, errors
    assert "error" not in out, out
    _all_right(out, tables, "left")
    oracle = _oracle()
    for name, names in ORACLE_NAMES.items():
        want = {k: [row[theirs] for row in oracle["rows"]] for k, theirs in names.items()}
        assert _same({k: out[name]["columns"][k] for k in names}, want), name


# ============================================================================================ no Node: the checks themselves
@pytest.mark.node_joins
def test_the_reviewers_table_is_the_installed_one_and_fits_the_generator(physics):
    oracle = _oracle()
    generator = Path(generate_toy_mc_events.__code__.co_filename)
    assert hashlib.sha256(generator.read_bytes()).hexdigest() == GENERATOR_SHA256 == oracle["generator_sha256"]
    assert (oracle["seed"], oracle["n_collisions"], len(oracle["rows"])) == (42, 10, len(physics["cascades"]))


@pytest.mark.node_joins
def test_the_plain_gather_equals_the_reviewers_table_and_numpy(physics):
    """The two references agree with each other, and with direct NumPy indexing, before either is used on the page."""
    tables, oracle = _tables(physics), _oracle()
    want = _reference(tables, "left")
    for name, names in ORACLE_NAMES.items():
        assert _same({k: want[name][k] for k in names}, {k: [row[o] for row in oracle["rows"]] for k, o in names.items()})
    v0, pos, ev = (np.array(tables[t][k]) for t, k in (("CA", "caV0"), ("V0", "vPos"), ("TR", "tEventIdx")))
    assert want["CVPE"]["cVz"] == np.array(tables["CO"]["cVz"])[ev[pos[v0]]].tolist()
    assert want["CBE"]["tPt"] == np.array(tables["TR"]["tPt"])[np.array(tables["CA"]["caBachelor"])].tolist()
    assert 0 in tables["TR"]["tEventIdx"] and 0 in tables["CA"]["caBachelor"]            # row 0 is pointed to


@pytest.mark.node_joins
def test_the_by_position_model_differs_from_the_reference_exactly_where_a_row_was_lost(physics):
    tables = _tables(physics)
    assert all(_same(_reference(tables, "inner", by_position=True)[n], _reference(tables, "inner")[n]) for n in VIEWS)
    lost = _lose_the_first_track(tables)
    want, wrong = _reference(lost, "inner"), _reference(lost, "inner", by_position=True)
    assert all(isinstance(wrong[n], _Refused) or not _same(wrong[n], want[n]) for n in VIEWS)
    with pytest.raises(AssertionError):                                # the wrong page is never accepted as right
        for n in VIEWS:
            cols = wrong[n] if not isinstance(wrong[n], _Refused) else {k: [] for k in KEYS[n]}
            _verdict({"length": len(cols[KEYS[n][0]]), "columns": cols, "throws": {}}, KEYS[n], want[n], None)
