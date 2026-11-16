"""PHASE_5 Batch 4.c, stage A of the selections plan: the ToyMC dashboard through joins, display only. Tests only;
RootInteractive's code is not changed. The page is built here without its selection tabs (selections=False); the
selections are tested in test_headless_batch4c_toymc_joins_dashboard_selections.py.

In plain words. The page (Tools/generators/toy_mc_joins_dashboard.py) is built by bokehDrawSA, as a user builds it,
from the four ToyMC tables. Its figures read parents through joins by row number:
    TrkJ   a track and its collision
    V0J    a V0, its positive and negative track (Pos, Neg) and the collision of each
    CascJ  a cascade, everything of its V0, and its bachelor track
The tests read the page's own models, in Node and with the same reading code in a real browser (Chromium), and compare
with references that use no join and no RootInteractive code:
  * every joined column, row by row, with plain NumPy indexing of the ToyMC tables (and, where the generator has its
    own reference gather, with that one too);
  * every histogram with numpy.histogram of those columns;
  * the profile with means computed by NumPy per bin;
  * the mass computed in the browser from the joined tracks with the generator's own mass functions and with the
    stored mass (the consistency check of the page);
  * the five AliasDataFrame queries the Architect ran on alma2 on 2026-11-16 (27,500 collisions, seed 42): the
    number of entries, means, standard deviations and correlations that AliasDataFrame printed.
A cut of a query is applied here, in the test, to the columns read from the page.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4a_joins import (
    NAN, _clean_reply, _clean_session)
from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4c_cascade_dashboard import _open
import toy_mc_events as gen                                 # the path is set by the import above
import toy_mc_joins_dashboard as dash

# the columns read from the page, per join (through the table the figures use, "<join>_full")
KEYS = {
    "TrkJ": ["trk_pt", "trk_eta", "trk_collisionIdx", "coll_vz", "trk_tgl"],
    "V0J": ["v0_mass", "v0_pdg", "v0_posTrackIdx", "v0_negTrackIdx", "pos_pt", "neg_pt", "pos_collisionIdx",
            "neg_collisionIdx", "posColl_vz", "negColl_vz", "v0_deltaMass", "v0_tgl", "v0_invPt", "v0_posPtOverPt",
            "v0_collisionVzDiff", "v0_massFromTracks", "v0_massDiff"],
    "CascJ": ["casc_mass", "casc_v0Idx", "casc_bachelorTrackIdx", "v0_mass", "pos_pt", "neg_pt", "posColl_vz",
              "negColl_vz", "bach_pt", "bachColl_vz", "casc_deltaMass", "casc_massFromV0AndBachelor", "casc_massDiff",
              "casc_collisionVzDiff"],
}
# computed in the browser as a small difference of large numbers: compared within MASS_TOLERANCE, not cell by cell
MASS_KEYS = {"v0_massFromTracks", "v0_massDiff", "casc_massFromV0AndBachelor", "casc_massDiff"}
MASS_TOLERANCE = 1e-9                                       # GeV/c2
# histograms of such a difference: it is rounding noise around 0, so its sign is not the same in the browser and in
# Python. Compared by the number of entries, all of them in the two bins next to 0.
NOISE_HISTOGRAMS = ("hV0MassDiff", "hCascMassDiff")
# what AliasDataFrame printed on alma2 on 2026-11-16 (27,500 collisions, seed 42; std with ddof=0)
ADF = {
    "tracks: pt, selection abs(Collision.vz) < 5": {"n": 206247, "mean": 1.2898175970995127, "std": 0.9403200871751285},
    "v0s: Pos.pt against Neg.pt, selection abs(mass - pdgMass) < 0.02":
        {"n": 36834, "mean_x": 0.7237911840640718, "mean_y": 0.7272380527863817, "corr": -0.22636226988213012},
    "v0s: mass, selection abs(Pos.Collision.vz) < 5": {"n": 46585, "mean": 1.524704878731037, "std": 0.9856118101723016},
    "cascades: mass, selection abs(V0.Pos.Collision.vz) < 5":
        {"n": 14403, "mean": 2.242643529866271, "std": 1.3353172497666632},
    "v0s: Pos.pt/pt against 1/pt, selection abs(pz/pt) < 1":
        {"n": 62805, "mean_x": 0.8834310172847665, "mean_y": 0.6004499016894418, "corr": 0.2021174592654435},
}

pytestmark = [pytest.mark.feature("JOIN.recursive")]


@pytest.fixture(scope="module")
def physics():
    return gen.generate_toy_mc_events(n_collisions=10, seed=42)


@pytest.fixture(scope="module")
def physics_300():
    return gen.generate_toy_mc_events(n_collisions=300, seed=42)


# ============================================================================================ the reference: no join
def _take(values, rows):
    """Plain indexing by row number; a row number of -1 (no parent) gives NaN."""
    values, rows = np.asarray(values, dtype=float), np.asarray(rows, dtype=float)
    ok = np.isfinite(rows) & (rows >= 0)
    out = np.full(len(rows), np.nan)
    out[ok] = values[rows[ok].astype(np.int64)]
    return out


def _mass(parts):
    """The invariant mass of a sum of four-momenta, row by row, with the generator's own functions.
    `parts`: a list of (pt, eta, phi, mass) columns. A missing part gives NaN."""
    out = []
    for row in zip(*(zip(*p) for p in parts)):
        if any(math.isnan(x) for part in row for x in part):
            out.append(np.nan)
        else:
            out.append(gen.inv_mass(sum(gen.p4_from_pt_eta_phi(*part) for part in row)))
    return np.array(out)


def _reference(raw):
    """Every column of KEYS from the four ToyMC tables alone (a `left` join: a missing parent gives NaN)."""
    c, t, v, ca = (raw[k] for k in ("collisions", "tracks", "v0s", "cascades"))
    n = lambda frame, name: frame[name].to_numpy()
    f = lambda frame, name: frame[name].to_numpy(dtype=float)
    trk = {"trk_pt": f(t, "pt"), "trk_eta": f(t, "eta"), "trk_collisionIdx": n(t, "collisionIdx"),
           "coll_vz": _take(c["vz"], n(t, "collisionIdx")), "trk_tgl": np.sinh(f(t, "eta"))}
    pos, neg = n(v, "posTrackIdx"), n(v, "negTrackIdx")
    v0 = {"v0_mass": f(v, "mass"), "v0_pdg": n(v, "pdg"), "v0_posTrackIdx": pos, "v0_negTrackIdx": neg,
          "pos_pt": _take(t["pt"], pos), "neg_pt": _take(t["pt"], neg),
          "pos_collisionIdx": _take(t["collisionIdx"], pos), "neg_collisionIdx": _take(t["collisionIdx"], neg)}
    v0["posColl_vz"] = _take(c["vz"], v0["pos_collisionIdx"])
    v0["negColl_vz"] = _take(c["vz"], v0["neg_collisionIdx"])
    v0["v0_deltaMass"] = f(v, "mass") - f(v, "pdgMass")
    v0["v0_tgl"] = f(v, "pz") / f(v, "pt")
    v0["v0_invPt"] = 1 / f(v, "pt")
    v0["v0_posPtOverPt"] = v0["pos_pt"] / f(v, "pt")
    v0["v0_collisionVzDiff"] = v0["posColl_vz"] - v0["negColl_vz"]
    daughters = [gen.V0_DAUGHTERS[int(p)] for p in n(v, "pdg")]
    v0["v0_massFromTracks"] = _mass([
        (v0["pos_pt"], _take(t["eta"], pos), _take(t["phi"], pos), [gen.MASS[d[0]] for d in daughters]),
        (v0["neg_pt"], _take(t["eta"], neg), _take(t["phi"], neg), [gen.MASS[d[1]] for d in daughters])])
    v0["v0_massDiff"] = v0["v0_massFromTracks"] - f(v, "mass")
    cv, cb = n(ca, "v0Idx"), n(ca, "bachelorTrackIdx")
    casc = {"casc_mass": f(ca, "mass"), "casc_v0Idx": cv, "casc_bachelorTrackIdx": cb,
            "bach_pt": _take(t["pt"], cb), "bachColl_vz": _take(c["vz"], _take(t["collisionIdx"], cb)),
            "casc_deltaMass": f(ca, "mass") - f(ca, "pdgMass")}
    for key in ("v0_mass", "pos_pt", "neg_pt", "posColl_vz", "negColl_vz"):
        casc[key] = _take(v0[key], cv)
    # the generator adds the bachelor to the V0's four-momentum; the V0's energy follows from its stored mass
    v0_px, v0_py, v0_pz, v0_m = (_take(v[k], cv) for k in ("px", "py", "pz", "mass"))
    bach = [gen.p4_from_pt_eta_phi(*row) if not any(math.isnan(x) for x in row) else np.full(4, np.nan)
            for row in zip(casc["bach_pt"], _take(t["eta"], cb), _take(t["phi"], cb),
                           [gen.MASS[gen.CASCADE_DAUGHTERS[int(p)][1]] for p in n(ca, "pdg")])]
    total = np.column_stack([np.sqrt(v0_m ** 2 + v0_px ** 2 + v0_py ** 2 + v0_pz ** 2), v0_px, v0_py, v0_pz]) + np.array(bach).reshape(-1, 4)
    casc["casc_massFromV0AndBachelor"] = np.array([np.nan if np.isnan(p).any() else gen.inv_mass(p) for p in total])
    casc["casc_massDiff"] = casc["casc_massFromV0AndBachelor"] - f(ca, "mass")
    casc["casc_collisionVzDiff"] = casc["posColl_vz"] - casc["bachColl_vz"]
    out = {"TrkJ": trk, "V0J": v0, "CascJ": casc}
    assert all(set(out[j]) == set(KEYS[j]) for j in KEYS)
    return out


def _histograms(ref, raw, selected=None):
    """numpy.histogram for every histogram of the page, from the reference columns (NaN is not counted).
    `selected`: {table or join: boolean mask of its rows}; only these rows are counted."""
    own = {"Collisions": {"coll_vz": raw["collisions"]["vz"].to_numpy(dtype=float),
                          "coll_nTracks": raw["collisions"]["nTracks"].to_numpy(dtype=float)}}
    out = {}
    for name, (column, source, nbins, low, high, _, _) in dash.HISTOGRAMS.items():
        values = np.asarray({**own, **ref}[source][column], dtype=float)
        if selected is not None:
            values = values[np.asarray(selected[source], dtype=bool)]
        out[name] = np.histogram(values[np.isfinite(values)], bins=nbins, range=(low, high))[0].tolist()
    return out


def _profile(ref):
    """The profile of the page: per bin of 1/pt and of the particle type, the entries and the mean of Pos.pt/pt."""
    p = dash.PROFILE
    x, y, z = (np.asarray(ref["V0J"][k], dtype=float) for k in p["variables"])
    (x0, x1), (y0, y1), (z0, z1) = p["range"]
    nx, _, nz = p["nbins"]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)
    ix = np.minimum(np.searchsorted(np.linspace(x0, x1, nx + 1), x[ok], side="right") - 1, nx - 1)
    iz = np.minimum(np.searchsorted(np.linspace(z0, z1, nz + 1), z[ok], side="right") - 1, nz - 1)
    out = {}
    for b_z in range(nz):
        for b_x in range(nx):
            values = y[ok][(ix == b_x) & (iz == b_z)]
            out[(b_x, b_z)] = (len(values), float(values.mean()) if len(values) else None)
    return out


# ============================================================================================ reading the page
READ = """(a) => {
  const doc = Bokeh.documents[Bokeh.documents.length - 1];
  const all = Array.from(doc._all_models.values());
  const one = (classes, name) => { const m = all.filter(x => classes.includes(x.constructor.__name__) && x.name === name);
    if (m.length !== 1) throw new Error("the page must hold exactly one " + classes + " named " + name + ", found " + m.length);
    return m[0]; };
  const tok = v => v === undefined ? "<undefined>" : v === null ? "<null>"
    : (typeof v === "number" && Number.isNaN(v)) ? "<NaN>" : v === Infinity ? "<+inf>" : v === -Infinity ? "<-inf>" : v;
  const out = {columns: {}, lengths: {}, histograms: {}, profile: {}, throws: {}};
  try {
    for (const [join, keys] of Object.entries(a.keys)) {
      const table = one(["CDSAlias"], join + "_full");                    // the table the figures of this join use
      out.lengths[join] = one(["CDSJoin"], join).get_length();
      out.columns[join] = {};
      for (const k of keys) {
        try { out.columns[join][k] = Array.from(table.get_column(k), tok); } catch (e) { out.throws[join + "." + k] = String(e && e.message || e); }
      }
    }
    for (const name of a.histograms) {
      try { out.histograms[name] = Array.from(one(["HistogramCDS"], name).get_column("bin_count"), tok); } catch (e) { out.throws[name] = String(e && e.message || e); }
    }
    if (a.profile) {
      const p = one(["HistoNdProfile"], a.profile);
      for (const k of ["bin_center_0", "bin_center_2", "entries", "mean"]) out.profile[k] = Array.from(p.get_column(k), tok);
    }
    out.tabs = all.filter(m => m.constructor.__name__ === "LazyTabs" || m.constructor.__name__ === "Tabs").map(t => t.tabs.map(p => p.title));
    out.sliders = all.filter(m => ["RangeSlider", "Slider", "Select", "MultiSelect", "MultiChoice", "TextInput", "Spinner", "Toggle"].includes(m.constructor.__name__)).length;
  } catch (e) { out.error = String(e && e.message || e); }
  return out;
}"""


def _arg(keys=KEYS, histograms=True, profile=True):
    return {"keys": keys, "histograms": list(dash.HISTOGRAMS) if histograms else [],
            "profile": dash.PROFILE["name"] + "_1" if profile else None}


def _node(node_session, tmp_path, raw, how="left", arg=None):
    """The page read in Node. Any harness problem, page error or console error fails the test."""
    from bokeh.document import Document
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    doc = Document()
    doc.add_root(dash.toy_mc_joins_dashboard(raw, how, selections=False))
    path = tmp_path / "toy_mc_joins_dashboard.json"
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    _clean_session(node_session)
    for op, kw in (("new_doc", dict(path=str(path))), ("eval", dict(code=READ, arg=arg or _arg()))):
        reply = node_session.request(op, **kw)
        _clean_reply(reply, op)
    out = decode(reply["result"])
    assert "error" not in out and out["throws"] == {}, {k: out[k] for k in ("error", "throws") if k in out}
    return out


def _assert_histograms(out, ref, raw, selected=None):
    want = _histograms(ref, raw, selected)
    assert set(out["histograms"]) == set(want)
    for name, counts in want.items():
        got = out["histograms"][name]
        if name in NOISE_HISTOGRAMS:
            middle = len(counts) // 2
            assert sum(got) == sum(counts) == sum(got[middle - 1:middle + 1]) == sum(counts[middle - 1:middle + 1]), name
        else:
            assert got == counts, name
    return want


def _number(cell):
    return np.nan if cell == NAN else cell


def _assert_columns(out, ref, rows=None):
    """Every column of every join: the right length, and the right value in every row."""
    for join, keys in KEYS.items():
        assert set(out["columns"][join]) == set(keys), join
        for key in keys:
            got = out["columns"][join][key]
            assert all(cell == NAN or (isinstance(cell, (int, float)) and not isinstance(cell, bool)) for cell in got), (join, key)
            got, want = np.array([_number(x) for x in got], dtype=float), np.asarray(ref[join][key], dtype=float)
            assert len(got) == len(want) == out["lengths"][join] == (rows or {}).get(join, len(want)), (join, key)
            if key in MASS_KEYS:
                same = np.allclose(got, want, rtol=0, atol=MASS_TOLERANCE, equal_nan=True)
            else:
                same = np.allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)
            assert same, f"{join}.{key}: the page differs from the plain NumPy gather"


# ============================================================================================ the values, in Node
@pytest.mark.node_joins
@pytest.mark.parametrize("how", ["left", "inner"])
def test_node_every_joined_column_equals_plain_numpy_indexing(node_session, tmp_path, physics, how):
    """Every table has its parent here, so `left` and `inner` give the same rows: every column of the three joins,
    the pointer columns included, in the order of the child table."""
    ref = _reference(physics)
    out = _node(node_session, tmp_path, physics, how)
    _assert_columns(out, ref, {"TrkJ": len(physics["tracks"]), "V0J": len(physics["v0s"]), "CascJ": len(physics["cascades"])})
    _assert_histograms(out, ref, physics)


@pytest.mark.node_joins
def test_node_rows_without_a_parent_show_nan_and_are_not_counted(node_session, tmp_path, physics):
    """The ToyMC with rows that have no parent (-1) at every level. A `left` join keeps them: the parent's columns
    and everything computed from them are NaN, and a histogram does not count them."""
    raw = gen.inject_orphan_test_rows(physics)
    ref = _reference(raw)
    assert all(np.isnan(np.asarray(ref[j][k], dtype=float)).any() for j, k in
               (("TrkJ", "coll_vz"), ("V0J", "pos_pt"), ("V0J", "v0_massFromTracks"), ("CascJ", "bach_pt"), ("CascJ", "v0_mass")))
    out = _node(node_session, tmp_path, raw, "left")
    _assert_columns(out, ref, {"TrkJ": len(raw["tracks"]), "V0J": len(raw["v0s"]), "CascJ": len(raw["cascades"])})
    want = _assert_histograms(out, ref, raw)
    assert sum(want["hTrkCollVz"]) == len(raw["tracks"]) - 1 and sum(want["hV0MassDiff"]) == len(raw["v0s"]) - 1


@pytest.mark.node_joins
def test_node_histograms_and_profile_equal_numpy_on_three_thousand_tracks(node_session, tmp_path, physics_300):
    """300 collisions (3,237 tracks, 727 V0s, 246 cascades): every histogram equals numpy.histogram of the reference
    column, and the profile equals the entries and means NumPy gives per bin."""
    assert tuple(len(physics_300[k]) for k in gen.TABLES) == (300, 3237, 727, 246)
    ref = _reference(physics_300)
    out = _node(node_session, tmp_path, physics_300)
    want = _assert_histograms(out, ref, physics_300)
    assert sum(want["hTrkPt"]) > 3000 and sum(want["hV0Mass"]) > 600 and sum(want["hCascMass"]) > 150      # not empty
    profile, p = _profile(ref), out["profile"]
    nx, _, nz = dash.PROFILE["nbins"]
    assert len(p["mean"]) == len(p["entries"]) == nx * nz
    x_centers, z_centers = sorted(set(p["bin_center_0"])), sorted(set(p["bin_center_2"]))
    filled = 0
    for cx, cz, entries, mean in zip(p["bin_center_0"], p["bin_center_2"], p["entries"], p["mean"]):
        want_entries, want_mean = profile[(x_centers.index(cx), z_centers.index(cz))]
        assert entries == want_entries, (cx, cz, entries, want_entries)
        if want_entries:
            filled += 1
            assert math.isclose(mean, want_mean, rel_tol=1e-12), (cx, cz, mean, want_mean)
    assert filled > 30 and sum(p["entries"]) == sum(n for n, _ in profile.values()) > 600


@pytest.mark.node_joins
def test_node_the_mass_from_the_joined_tracks_equals_the_stored_mass(node_session, tmp_path, physics_300):
    """The consistency check of the page. The browser computes the mass of every V0 from its joined positive and
    negative track, and of every cascade from its joined V0 and bachelor track; both equal the mass the generator
    stored, for every row."""
    out = _node(node_session, tmp_path, physics_300, arg=_arg(histograms=False, profile=False))
    for join, key, stored, rows in (("V0J", "v0_massFromTracks", physics_300["v0s"]["mass"], 727),
                                    ("CascJ", "casc_massFromV0AndBachelor", physics_300["cascades"]["mass"], 246)):
        got = np.array(out["columns"][join][key], dtype=float)
        assert len(got) == rows and np.isfinite(got).all()
        assert np.abs(got - stored.to_numpy()).max() < MASS_TOLERANCE, (join, key)
    assert all(v == 0 for v in out["columns"]["V0J"]["v0_collisionVzDiff"])       # both tracks of a V0: one collision
    assert all(v == 0 for v in out["columns"]["CascJ"]["casc_collisionVzDiff"])   # the bachelor track too


@pytest.mark.node_joins
def test_node_the_five_aliasdataframe_queries_of_the_architect(node_session, tmp_path):
    """The page with 27,500 collisions (302,883 tracks, 68,207 V0s, 21,159 cascades). The columns read from the page,
    cut here as each query cuts, give the entries, means, standard deviations and correlations that AliasDataFrame
    printed for the same query on alma2."""
    raw = gen.generate_toy_mc_events(n_collisions=27500, seed=42)
    assert tuple(len(raw[k]) for k in gen.TABLES) == (27500, 302883, 68207, 21159)
    keys = {"TrkJ": ["trk_pt", "coll_vz"], "CascJ": ["casc_mass", "posColl_vz"],
            "V0J": ["v0_mass", "v0_deltaMass", "pos_pt", "neg_pt", "posColl_vz", "v0_tgl", "v0_invPt", "v0_posPtOverPt"]}
    out = _node(node_session, tmp_path, raw, arg=_arg(keys=keys, histograms=False, profile=False))
    col = lambda join, key: np.array([_number(x) for x in out["columns"][join][key]], dtype=float)
    one = lambda x: {"n": len(x), "mean": x.mean(), "std": x.std()}
    two = lambda x, y: {"n": len(x), "mean_x": x.mean(), "mean_y": y.mean(), "corr": np.corrcoef(x, y)[0, 1]}
    s1 = np.abs(col("TrkJ", "coll_vz")) < 5
    s2 = np.abs(col("V0J", "v0_deltaMass")) < 0.02
    s3 = np.abs(col("V0J", "posColl_vz")) < 5
    s4 = np.abs(col("CascJ", "posColl_vz")) < 5
    s6 = np.abs(col("V0J", "v0_tgl")) < 1
    got = [one(col("TrkJ", "trk_pt")[s1]), two(col("V0J", "neg_pt")[s2], col("V0J", "pos_pt")[s2]),
           one(col("V0J", "v0_mass")[s3]), one(col("CascJ", "casc_mass")[s4]),
           two(col("V0J", "v0_invPt")[s6], col("V0J", "v0_posPtOverPt")[s6])]
    for (query, want), result in zip(ADF.items(), got):
        assert result["n"] == want["n"], (query, result, want)
        for key in want:
            assert math.isclose(result[key], want[key], rel_tol=1e-9), (query, key, result[key], want[key])


# ============================================================================================ the real browser
@pytest.mark.browser_a4
@pytest.mark.backend("browser")
@pytest.mark.layer("integration")
def test_browser_the_page_shows_the_same_values_in_tabs(headless_session, tmp_path, physics):
    """The same page as a standalone HTML file in Chromium, read with the same code as in Node: the same columns, the
    same histograms, the five figure tabs, no selection control on this page, and no page error."""
    from bokeh.io import save
    from bokeh.resources import INLINE
    html = tmp_path / "toy_mc_joins_dashboard.html"
    save(dash.toy_mc_joins_dashboard(physics, selections=False), filename=str(html), resources=INLINE, title="ToyMC through joins")
    ctx, page, errors = _open(headless_session, html)
    try:
        out = page.evaluate(READ, _arg())
    finally:
        ctx.close()
    assert not errors, errors
    assert "error" not in out and out["throws"] == {}, out
    ref = _reference(physics)
    _assert_columns(out, ref)
    _assert_histograms(out, ref, physics)
    assert out["tabs"] == [["Collisions", "Tracks", "V0s", "Cascades", "Consistency"]] and out["sliders"] == 0


# ============================================================================================ no page: the checks themselves
@pytest.mark.node_joins
def test_the_reference_equals_the_generators_own_gather(physics):
    """The reference of this file agrees with the generator's independent gather, with and without missing parents,
    before it is used on the page."""
    for raw in (physics, gen.inject_orphan_test_rows(physics)):
        ref, theirs = _reference(raw), gen.independent_gather_oracle(raw)
        for (join, key), name in {("TrkJ", "coll_vz"): "tracks.collisionVz", ("V0J", "pos_pt"): "v0s.posPt",
                                  ("V0J", "neg_pt"): "v0s.negPt", ("V0J", "pos_collisionIdx"): "v0s.posCollisionIdx",
                                  ("V0J", "neg_collisionIdx"): "v0s.negCollisionIdx", ("CascJ", "v0_mass"): "cascades.v0Mass",
                                  ("CascJ", "bach_pt"): "cascades.bachelorPt"}.items():
            assert np.array_equal(np.asarray(ref[join][key], dtype=float), theirs[name], equal_nan=True), (join, key)
    ref = _reference(physics)
    assert np.abs(ref["V0J"]["v0_massDiff"]).max() < 1e-12 and np.abs(ref["CascJ"]["casc_massDiff"]).max() < 1e-12


@pytest.mark.node_joins
def test_the_page_description_is_complete_and_keeps_pointer_columns_as_integers(physics):
    """Every figure is in exactly one tab; every join, histogram and derived column names things that exist; every
    pointer column is an integer column of its child table (integer columns are not lossy-compressed)."""
    d = dash.page_description(physics)
    used = [i for rows in d["figure_layout"].values() for row in rows if isinstance(row, list) for i in row]
    assert sorted(used) == list(range(len(d["figures"]))) and list(d["figure_layout"]) == ["Collisions", "Tracks", "V0s", "Cascades", "Consistency"]
    assert set(d["tables"]) == set(dash.ROLES) and dash.MAIN in dash.ROLES
    reachable = {role: set(columns) for role, columns in d["tables"].items()}
    for name, (child, parent, pointer) in dash.JOINS.items():                # the joins are listed parents first
        assert child in reachable and parent in reachable and pointer in reachable[child], name
        reachable[name] = reachable[child] | reachable[parent]
    for name, (child, _, pointer) in dash.JOINS.items():
        owner = next(role for role in dash.ROLES if pointer in d["tables"][role])
        assert np.issubdtype(d["tables"][owner][pointer].dtype, np.integer), (name, pointer)
    for alias in dash.ALIASES:
        reachable[alias["context"]].add(alias["name"])
    for name, (column, source, *_rest) in dash.HISTOGRAMS.items():
        assert column in reachable[source], name
    assert all(v in reachable[dash.PROFILE["source"]] for v in dash.PROFILE["variables"])
    prefixes = [prefix for _, prefix, _ in dash.ROLES.values()]
    assert len(set(prefixes)) == len(prefixes)                               # one prefix per role: unique column names
    columns = [c for role in d["tables"].values() for c in role]
    assert len(set(columns)) == len(columns)
