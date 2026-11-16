"""PHASE_5 Batch 4.c, stage B of the selections plan: the selection tabs of the ToyMC dashboard. Tests only;
RootInteractive's code is not changed.

In plain words. Under its figures the page (Tools/generators/toy_mc_joins_dashboard.py) has one tab per table:
Collisions, Tracks, Pos, Neg, V0s, Cascades. Each tab holds range sliders and switches. The tests move the sliders
and the switches of the page itself, in Node and in a real browser (Chromium; there also with the mouse and the
keyboard), and compare what the page then shows with references that use no join and no RootInteractive code.

The rule the page must follow (the Architect, 2026-11-16):
  * the mask of a table is the logical AND of the cuts in its tab, and it can be switched ON and OFF;
  * a row is shown only if its parents are selected: a track needs its collision; a V0 needs its positive and its
    negative track, each with its collision; a cascade needs its V0 and its bachelor track with its collision;
  * the Tracks tab cuts every track, in whatever role; the Pos tab cuts only the positive track of a V0, the Neg
    tab only the negative one;
  * a row without a parent is hidden while a mask of that parent is ON, and shown when it is OFF;
  * nothing acts upwards (a cut on tracks does not hide collisions).

The references:
  * `_selected` below: the rule written level by level (collisions, then tracks, then V0s, then cascades) with plain
    NumPy indexing. The page computes it differently: one flat list of cuts per join;
  * the generator's own selection of the descendants of chosen collisions (select_collision_descendants);
  * the five AliasDataFrame queries the Architect ran on alma2 on 2026-11-16 (27,500 collisions, seed 42).
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4a_joins import _clean_reply, _clean_session
from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4c_cascade_dashboard import _open
from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4c_toymc_joins_dashboard import (
    ADF, NOISE_HISTOGRAMS, _histograms, _reference, _take)
import toy_mc_events as gen                                 # the path is set by the imports above
import toy_mc_joins_dashboard as dash

INF = float("inf")
TABS = ("Collisions", "Tracks", "Pos", "Neg", "V0s", "Cascades")
# every slider of the page: its name on the page -> (tab, column of the tab's own table)
SLIDERS = {"selection_coll_vz": ("Collisions", "vz"), "selection_trk_pt": ("Tracks", "pt"), "selection_trk_eta": ("Tracks", "eta"),
           "selection_pos_pt": ("Pos", "pt"), "selection_pos_eta": ("Pos", "eta"),
           "selection_neg_pt": ("Neg", "pt"), "selection_neg_eta": ("Neg", "eta"),
           "selection_v0_mass": ("V0s", "mass"), "selection_v0_deltaMass": ("V0s", "deltaMass"),
           "selection_v0_tgl": ("V0s", "tgl"), "selection_v0_pt": ("V0s", "pt"),
           "selection_casc_mass": ("Cascades", "mass"), "selection_casc_deltaMass": ("Cascades", "deltaMass"),
           "selection_casc_pt": ("Cascades", "pt")}
ROWS = {"Collisions": "collisions", "TrkJ": "tracks", "V0J": "v0s", "CascJ": "cascades"}       # target -> ToyMC table
# one column of each scatter figure's source, read to see which rows the scatter figures hold. A scatter source is
# computed only while its figure tab is open, so the reading code opens that tab first, as a click on it does.
SCATTER_KEY = {"TrkJ": "trk_pt", "V0J": "pos_pt", "CascJ": "casc_mass"}
FIGURE_TAB = {"TrkJ": "Tracks", "V0J": "V0s", "CascJ": "Cascades"}

pytestmark = [pytest.mark.feature("JOIN.recursive")]


@pytest.fixture(scope="module")
def physics_300():
    return gen.generate_toy_mc_events(n_collisions=300, seed=42)


# ============================================================================================ the reference: the rule
def _selected(raw, cuts=None, off=()):
    """Boolean masks of the selected rows of the four tables, level by level, with plain NumPy indexing.
    `cuts`: {slider name: (low, high)} for the sliders that are not at full range. `off`: the names of the switches
    that are OFF: a tab name, or a slider name."""
    cuts = cuts or {}
    c, t, v, ca = (raw[k] for k in ("collisions", "tracks", "v0s", "cascades"))
    own_columns = {
        "Collisions": {"vz": c["vz"]}, "Tracks": {"pt": t["pt"], "eta": t["eta"]},
        "V0s": {"mass": v["mass"], "deltaMass": v["mass"] - v["pdgMass"], "tgl": v["pz"] / v["pt"], "pt": v["pt"]},
        "Cascades": {"mass": ca["mass"], "deltaMass": ca["mass"] - ca["pdgMass"], "pt": ca["pt"]}}
    own_columns["Pos"] = own_columns["Neg"] = own_columns["Tracks"]             # the same table in another role

    def active(tab):
        """The cuts of a tab that are ON: [(column, low, high)]. A tab switched OFF has none."""
        if tab in off:
            return []
        return [(col, *cuts.get(name, (-INF, INF))) for name, (its_tab, col) in SLIDERS.items()
                if its_tab == tab and name not in off]

    def own(tab):
        """The mask of the tab on its own table: the AND of its cuts. A missing value (NaN) fails a cut that is ON."""
        mask = np.ones(len(next(iter(own_columns[tab].values()))), dtype=bool)
        for col, low, high in active(tab):
            x = np.asarray(own_columns[tab][col], dtype=float)
            with np.errstate(invalid="ignore"):
                mask &= (x >= low) & (x <= high)
        return mask

    def through(parent_selected, pointers, masks_on):
        """A child row passes if its parent is selected. Without a parent (-1): only if no mask of the parent is ON."""
        pointers = np.asarray(pointers, dtype=np.int64)
        out = np.full(len(pointers), not masks_on)
        has = pointers >= 0
        out[has] = parent_selected[pointers[has]]
        return out

    on = {tab: bool(active(tab)) for tab in TABS}
    collisions = own("Collisions")
    # a track as a parent: the Tracks mask and its collision; the role masks (Pos, Neg) come on top of it
    track = own("Tracks") & through(collisions, t["collisionIdx"], on["Collisions"])
    track_on = on["Tracks"] or on["Collisions"]
    v0 = (own("V0s") & through(track & own("Pos"), v["posTrackIdx"], track_on or on["Pos"])
          & through(track & own("Neg"), v["negTrackIdx"], track_on or on["Neg"]))
    v0_on = on["V0s"] or track_on or on["Pos"] or on["Neg"]
    cascade = own("Cascades") & through(v0, ca["v0Idx"], v0_on) & through(track, ca["bachelorTrackIdx"], track_on)
    return {"Collisions": collisions, "TrkJ": track, "V0J": v0, "CascJ": cascade}


# ============================================================================================ driving and reading the page
# 1. every control back to its start, then the wanted state; 2. (a separate request, after the page reacted) read
SET = """(a) => {
  const all = Array.from(Bokeh.documents[Bokeh.documents.length - 1]._all_models.values());
  const controls = all.filter(m => typeof m.name === "string" && m.name.startsWith("selection_"));
  const one = n => { const r = controls.filter(m => m.name === n); if (r.length !== 1) throw new Error(n + " found " + r.length + " times"); return r[0]; };
  for (const m of controls) {
    if (m.constructor.__name__ === "Toggle" && !m.active) m.active = true;
    if (m.constructor.__name__ === "RangeSlider" && (m.value[0] !== m.start || m.value[1] !== m.end)) m.value = [m.start, m.end];
  }
  for (const [name, value] of a.cuts) one(name).value = value;
  for (const name of a.off) one(name.startsWith("selection_") ? name.replace("selection_", "selection_switch_") : "selection_switch_" + name).active = false;
  if (a.cuts.length === 0 && a.off.length === 0) {       // nothing to change: one switch OFF and ON again, so that the counter is written
    const s = one("selection_switch_Cascades"); s.active = false; s.active = true;
  }
  return {sliders: controls.filter(m => m.constructor.__name__ === "RangeSlider").map(m => m.name).sort(),
          switches: controls.filter(m => m.constructor.__name__ === "Toggle").map(m => m.name).sort()};
}"""
READ = """(a) => {
  const all = Array.from(Bokeh.documents[Bokeh.documents.length - 1]._all_models.values());
  const one = (cls, n) => { const r = all.filter(m => m.constructor.__name__ === cls && m.name === n);
    if (r.length !== 1) throw new Error(cls + " " + n + " found " + r.length + " times"); return r[0]; };
  const out = {selected: {}, histograms: {}, scatter: {}, labels: {}, throws: {}};
  for (const target of a.targets) {
    try { out.selected[target] = Array.from(one("LazyIntersectionFilter", "selection_mask_" + target).get_indices()); } catch (e) { out.throws[target] = String(e && e.message || e); }
  }
  for (const name of a.histograms) {
    try { out.histograms[name] = Array.from(one("HistogramCDS", name).get_column("bin_count")); } catch (e) { out.throws[name] = String(e && e.message || e); }
  }
  const figure_tabs = all.filter(m => m.constructor.__name__ === "LazyTabs");
  for (const [target, key] of Object.entries(a.scatter)) {
    try {
      if (figure_tabs.length !== 1) throw new Error("the page must hold exactly one set of figure tabs");
      const index = figure_tabs[0].tabs.map(p => p.title).indexOf(a.figure_tab[target]);
      if (index < 0) throw new Error("no figure tab " + a.figure_tab[target]);
      figure_tabs[0].active = index;
      out.scatter[target] = Array.from(one("DownsamplerCDS", target).get_column(key));
    } catch (e) { out.throws["scatter " + target] = String(e && e.message || e); }
  }
  out.counter = one("Div", "selection_counter").text;
  for (const m of all) if (m.constructor.__name__ === "Toggle" && m.name && m.name.startsWith("selection_switch_")) out.labels[m.name] = m.label;
  out.tabs = one("Tabs", "selection_tabs").tabs.map(p => p.title);
  return out;
}"""


def _read_arg(histograms=True, scatter=True):
    return {"targets": list(ROWS), "histograms": list(dash.HISTOGRAMS) if histograms else [],
            "scatter": SCATTER_KEY if scatter else {}, "figure_tab": FIGURE_TAB}


def _node_page(node_session, tmp_path, raw):
    """The page with its selection tabs in Node. Returns state(cuts, off) -> what the page shows in that state."""
    from bokeh.document import Document
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    doc = Document()
    doc.add_root(dash.toy_mc_joins_dashboard(raw))
    path = tmp_path / "toy_mc_joins_dashboard_selections.json"
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    _clean_session(node_session)
    _clean_reply(node_session.request("new_doc", path=str(path)), "new_doc")

    def state(cuts=None, off=(), histograms=True, scatter=True):
        arg = {"cuts": [[k, list(v)] for k, v in (cuts or {}).items()], "off": list(off)}
        for code, a in ((SET, arg), (READ, _read_arg(histograms, scatter))):
            reply = node_session.request("eval", code=code, arg=a)
            _clean_reply(reply, "eval")
        out = decode(reply["result"])
        assert out["throws"] == {}, out["throws"]
        return out
    return state


def _open_ended(cuts):
    """The cut a slider value means: a value at the end of the slider's scale is open there."""
    scale = {f"selection_{prefix}{col}": (start, end) for prefix, sliders in dash.SELECTIONS.values()
             for col, _, start, end, _ in sliders}
    return {name: (-INF if low <= scale[name][0] else low, INF if high >= scale[name][1] else high)
            for name, (low, high) in (cuts or {}).items()}


def _assert_state(out, raw, ref, cuts=None, off=(), histograms=True, scatter=True, counter=True):
    """The page in this state shows exactly the rows the rule selects: in its masks, its counter, its histograms and
    the sources of its scatter figures."""
    want = _selected(raw, _open_ended(cuts), off)
    for target, table in ROWS.items():
        assert out["selected"][target] == np.flatnonzero(want[target]).tolist(), \
            f"{target}: the page selects {len(out['selected'][target])} rows, the rule {int(want[target].sum())}"
    if counter:                                    # the page writes its counter at the first change of a control
        numbers = [int(x) for x in re.findall(r"(\d+) of \d+", out["counter"])]
        assert numbers == [int(want[t].sum()) for t in ROWS], out["counter"]
    if histograms:
        expected = _histograms(ref, raw, want)
        for name, counts in expected.items():
            if name in NOISE_HISTOGRAMS:
                assert sum(out["histograms"][name]) == sum(counts), name
            else:
                assert out["histograms"][name] == counts, name
    if scatter:
        for target, key in SCATTER_KEY.items():
            values = np.asarray(ref[target][key], dtype=float)[want[target]]
            got = np.array(out["scatter"][target], dtype=float)
            assert len(got) == len(values) and np.allclose(got, values, rtol=1e-12, atol=1e-12, equal_nan=True), target
    return want


# ============================================================================================ the rule, in Node
@pytest.mark.node_joins
def test_node_at_the_start_every_row_is_selected_and_every_control_is_there(node_session, tmp_path, physics_300):
    state = _node_page(node_session, tmp_path, physics_300)
    out = state()
    want = _assert_state(out, physics_300, _reference(physics_300))
    assert all(mask.all() for mask in want.values())
    assert out["tabs"] == list(TABS) == list(dash.SELECTIONS)
    assert set(out["labels"]) == {"selection_switch_" + t for t in TABS} | {n.replace("selection_", "selection_switch_") for n in SLIDERS}


@pytest.mark.node_joins
def test_node_a_cut_on_collisions_acts_on_their_tracks_v0s_and_cascades(node_session, tmp_path, physics_300):
    """Collision vz in [-5, 5]. The rows the page selects are also exactly the generator's own descendants of the
    selected collisions (its all-paths rule)."""
    state = _node_page(node_session, tmp_path, physics_300)
    cuts = {"selection_coll_vz": (-5.0, 5.0)}
    want = _assert_state(state(cuts), physics_300, _reference(physics_300), cuts)
    theirs = gen.select_collision_descendants(physics_300, np.flatnonzero(want["Collisions"]).tolist())
    for target, table in ROWS.items():
        assert np.flatnonzero(want[target]).tolist() == theirs[table].tolist(), target
        assert 0.4 * len(want[target]) < want[target].sum() < 0.8 * len(want[target]), target       # a real cut


@pytest.mark.node_joins
def test_node_pos_and_neg_are_separate_selections(node_session, tmp_path, physics_300):
    """A cut on Pos.pt cuts V0s by their positive track only, a cut on Neg.pt by their negative track only; neither
    cuts the tracks themselves or the collisions. The cascades follow their V0."""
    state = _node_page(node_session, tmp_path, physics_300)
    ref = _reference(physics_300)
    pos = _assert_state(state({"selection_pos_pt": (1.0, 10.0)}), physics_300, ref, {"selection_pos_pt": (1.0, 10.0)})
    neg = _assert_state(state({"selection_neg_pt": (1.0, 10.0)}), physics_300, ref, {"selection_neg_pt": (1.0, 10.0)})
    for want, mine, other in ((pos, "pos_pt", "neg_pt"), (neg, "neg_pt", "pos_pt")):
        assert want["Collisions"].all() and want["TrkJ"].all()
        assert (np.asarray(ref["V0J"][mine])[want["V0J"]] >= 1.0).all() and (np.asarray(ref["V0J"][other])[want["V0J"]] < 1.0).any()
    assert not np.array_equal(pos["V0J"], neg["V0J"]) and not np.array_equal(pos["CascJ"], neg["CascJ"])
    assert 0 < pos["CascJ"].sum() < len(pos["CascJ"])


@pytest.mark.node_joins
def test_node_a_cut_on_tracks_acts_on_both_tracks_of_a_v0_and_on_the_bachelor(node_session, tmp_path, physics_300):
    state = _node_page(node_session, tmp_path, physics_300)
    ref, cuts = _reference(physics_300), {"selection_trk_pt": (0.3, 10.0), "selection_trk_eta": (-0.8, 0.8)}
    want = _assert_state(state(cuts), physics_300, ref, cuts)
    assert want["Collisions"].all()                                                    # nothing acts upwards
    v0 = ref["V0J"]
    assert (np.asarray(v0["pos_pt"])[want["V0J"]] >= 0.3).all() and (np.asarray(v0["neg_pt"])[want["V0J"]] >= 0.3).all()
    assert (np.asarray(ref["CascJ"]["bach_pt"])[want["CascJ"]] >= 0.3).all()
    assert all(0 < want[t].sum() < len(want[t]) for t in ("TrkJ", "V0J", "CascJ"))


@pytest.mark.node_joins
def test_node_a_v0_needs_the_collisions_of_both_tracks_and_a_cascade_also_that_of_its_bachelor(node_session, tmp_path, physics_300):
    """In the ToyMC every track of a V0 or a cascade comes from one collision, so a cut on collisions cannot tell
    whether the page looks at one track or at all of them. Here three tracks are moved, in a copy, to a collision
    outside the cut: the negative track of one V0, the positive track of another, and the bachelor of a cascade.
    The page hides exactly these candidates (and the cascades of the two V0s), as the rule says."""
    raw = {k: frame.copy(deep=True) for k, frame in physics_300.items()}
    vz = raw["collisions"]["vz"].to_numpy()
    inside, outside = np.flatnonzero(np.abs(vz) < 4), np.flatnonzero(np.abs(vz) > 6)
    tracks, v0s, cascades = raw["tracks"], raw["v0s"], raw["cascades"]
    in_cut = np.isin(tracks["collisionIdx"].to_numpy(), inside)
    good_v0 = np.flatnonzero(in_cut[v0s["posTrackIdx"].to_numpy()] & in_cut[v0s["negTrackIdx"].to_numpy()])
    used_by_cascades = set(cascades["v0Idx"].tolist())
    free_v0 = [i for i in good_v0 if i not in used_by_cascades]
    v0_neg, v0_pos = free_v0[0], free_v0[1]
    good_cascade = np.flatnonzero(np.isin(cascades["v0Idx"].to_numpy(), good_v0) & in_cut[cascades["bachelorTrackIdx"].to_numpy()])
    cascade = int(good_cascade[0])
    moved = [int(v0s["negTrackIdx"].iloc[v0_neg]), int(v0s["posTrackIdx"].iloc[v0_pos]), int(cascades["bachelorTrackIdx"].iloc[cascade])]
    assert len(set(moved)) == 3
    before = _selected(raw, {"selection_coll_vz": (-5.0, 5.0)})
    assert before["V0J"][v0_neg] and before["V0J"][v0_pos] and before["CascJ"][cascade]
    tracks.loc[moved, "collisionIdx"] = int(outside[0])
    state = _node_page(node_session, tmp_path, raw)
    cuts = {"selection_coll_vz": (-5.0, 5.0)}
    want = _assert_state(state(cuts), raw, _reference(raw), cuts)
    assert not want["V0J"][v0_neg] and not want["V0J"][v0_pos] and not want["CascJ"][cascade]
    assert before["V0J"].sum() - want["V0J"].sum() == 2 and before["CascJ"].sum() - want["CascJ"].sum() == 1
    assert not want["TrkJ"][moved].any()


@pytest.mark.node_joins
def test_node_cuts_on_v0s_also_on_derived_columns_act_on_cascades(node_session, tmp_path, physics_300):
    """The V0 mass window and pz/pt are computed in the browser; the cascades read them through their V0."""
    state = _node_page(node_session, tmp_path, physics_300)
    cuts = {"selection_v0_deltaMass": (-0.02, 0.02), "selection_v0_tgl": (-1.0, 1.0)}
    want = _assert_state(state(cuts), physics_300, _reference(physics_300), cuts)
    assert want["TrkJ"].all() and all(0 < want[t].sum() < len(want[t]) for t in ("V0J", "CascJ"))


@pytest.mark.node_joins
def test_node_the_switches_of_a_table_and_of_a_slider(node_session, tmp_path, physics_300):
    """A table switched OFF: its cuts do nothing although the sliders keep their values. A slider switched OFF: only
    that cut does nothing. Switched ON again, the cut is back. The labels say ON or OFF."""
    state = _node_page(node_session, tmp_path, physics_300)
    ref = _reference(physics_300)
    cuts = {"selection_coll_vz": (-5.0, 5.0), "selection_v0_tgl": (-1.0, 1.0), "selection_v0_pt": (0.5, 10.0)}
    both = _assert_state(state(cuts), physics_300, ref, cuts)
    out = state(cuts, off=["Collisions"])
    table_off = _assert_state(out, physics_300, ref, cuts, off=["Collisions"])
    assert out["labels"]["selection_switch_Collisions"] == "Selection on Collisions: OFF" and table_off["TrkJ"].all()
    assert out["labels"]["selection_switch_V0s"] == "Selection on V0s: ON"
    out = state(cuts, off=["selection_v0_tgl"])
    slider_off = _assert_state(out, physics_300, ref, cuts, off=["selection_v0_tgl"])
    assert out["labels"]["selection_switch_v0_tgl"] == "OFF" and out["labels"]["selection_switch_v0_pt"] == "ON"
    assert both["V0J"].sum() < table_off["V0J"].sum() and both["V0J"].sum() < slider_off["V0J"].sum()
    again = _assert_state(state(cuts), physics_300, ref, cuts)
    assert all(np.array_equal(again[t], both[t]) for t in ROWS)


@pytest.mark.node_joins
def test_node_rows_without_a_parent_are_hidden_while_a_mask_of_the_parent_is_on(node_session, tmp_path):
    """The ToyMC with rows that have no parent: a track without a collision, a V0 without its positive track,
    cascades without a V0 or a bachelor. With every mask ON they are hidden; with the parents' masks OFF they are
    shown."""
    raw = gen.inject_orphan_test_rows(gen.generate_toy_mc_events(n_collisions=10, seed=42))
    orphan_track, orphan_v0 = len(raw["tracks"]) - 1, len(raw["v0s"]) - 1
    assert raw["tracks"]["collisionIdx"].iloc[orphan_track] == -1 and raw["v0s"]["posTrackIdx"].iloc[orphan_v0] == -1
    state = _node_page(node_session, tmp_path, raw)
    ref = _reference(raw)
    start = _assert_state(state(), raw, ref)
    assert not start["TrkJ"][orphan_track] and not start["V0J"][orphan_v0] and start["TrkJ"].sum() == len(raw["tracks"]) - 1
    off = ["Collisions"]
    no_collisions = _assert_state(state(off=off), raw, ref, off=off)
    assert no_collisions["TrkJ"].all() and not no_collisions["V0J"][orphan_v0]            # its track is still missing
    off = ["Collisions", "Tracks", "Pos", "Neg", "V0s"]
    parents_off = _assert_state(state(off=off), raw, ref, off=off)
    assert parents_off["TrkJ"].all() and parents_off["V0J"][orphan_v0]                    # now it is shown
    off = list(TABS)
    all_off = _assert_state(state(off=off), raw, ref, off=off)
    assert all(mask.all() for mask in all_off.values())


@pytest.mark.node_joins
def test_node_twelve_combinations_of_cuts_and_switches_equal_the_rule(node_session, tmp_path, physics_300):
    """One page, twelve states one after the other; in each the selected rows of all four tables, the counter, every
    histogram and the scatter sources equal the rule."""
    state = _node_page(node_session, tmp_path, physics_300)
    ref = _reference(physics_300)
    states = [
        ({"selection_coll_vz": (-3.0, 8.0), "selection_trk_pt": (0.2, 5.0)}, []),
        ({"selection_coll_vz": (-3.0, 8.0), "selection_trk_pt": (0.2, 5.0)}, ["Tracks"]),
        ({"selection_pos_pt": (0.5, 10.0), "selection_neg_eta": (-0.5, 0.5), "selection_v0_mass": (0.4, 1.2)}, []),
        ({"selection_pos_pt": (0.5, 10.0), "selection_neg_eta": (-0.5, 0.5), "selection_v0_mass": (0.4, 1.2)}, ["Neg"]),
        ({"selection_pos_pt": (0.5, 10.0), "selection_neg_eta": (-0.5, 0.5), "selection_v0_mass": (0.4, 1.2)}, ["selection_v0_mass"]),
        ({"selection_casc_mass": (1.2, 1.8), "selection_casc_pt": (0.5, 10.0)}, []),
        ({"selection_casc_deltaMass": (-0.03, 0.03), "selection_v0_deltaMass": (-0.02, 0.02)}, []),
        ({"selection_casc_deltaMass": (-0.03, 0.03), "selection_v0_deltaMass": (-0.02, 0.02)}, ["V0s"]),
        ({"selection_trk_eta": (-0.6, 0.6), "selection_pos_eta": (-0.3, 1.5), "selection_coll_vz": (-30.0, 0.0)}, []),
        ({"selection_trk_eta": (-0.6, 0.6), "selection_pos_eta": (-0.3, 1.5), "selection_coll_vz": (-30.0, 0.0)}, ["Collisions", "Pos"]),
        ({name: ((start + end) / 2 - (end - start) / 4, (start + end) / 2 + (end - start) / 4)
          for name, (start, end) in {f"selection_{p}{c}": (s, e) for p, sl in dash.SELECTIONS.values() for c, _, s, e, _ in sl}.items()}, []),
        ({}, list(TABS)),
    ]
    seen = set()
    for cuts, off in states:
        want = _assert_state(state(cuts, off), physics_300, ref, cuts, off)
        seen.add(tuple(int(want[t].sum()) for t in ROWS))
    assert len(seen) >= 10                                                             # the states really differ


@pytest.mark.node_joins
def test_node_the_five_aliasdataframe_queries_by_the_sliders_of_the_page(node_session, tmp_path):
    """27,500 collisions (302,883 tracks, 68,207 V0s, 21,159 cascades). The cut of each query is set with the sliders
    of the page; the rows the page then selects give the entries, means, standard deviations and correlations that
    AliasDataFrame printed for the same query on alma2."""
    raw = gen.generate_toy_mc_events(n_collisions=27500, seed=42)
    c, t, v, ca = (raw[k] for k in ("collisions", "tracks", "v0s", "cascades"))
    state = _node_page(node_session, tmp_path, raw)
    rows = lambda cuts: {k: np.array(x, dtype=np.int64) for k, x in state(cuts, histograms=False, scatter=False)["selected"].items()}
    one = lambda x: {"n": len(x), "mean": x.mean(), "std": x.std()}
    two = lambda x, y: {"n": len(x), "mean_x": x.mean(), "mean_y": y.mean(), "corr": np.corrcoef(x, y)[0, 1]}
    pt = t["pt"].to_numpy()
    by_vz = rows({"selection_coll_vz": (-5.0, 5.0)})
    by_mass = rows({"selection_v0_deltaMass": (-0.02, 0.02)})["V0J"]
    by_tgl = rows({"selection_v0_tgl": (-1.0, 1.0)})["V0J"]
    pos, neg = v["posTrackIdx"].to_numpy(), v["negTrackIdx"].to_numpy()
    got = [one(pt[by_vz["TrkJ"]]), two(pt[neg[by_mass]], pt[pos[by_mass]]), one(v["mass"].to_numpy()[by_vz["V0J"]]),
           one(ca["mass"].to_numpy()[by_vz["CascJ"]]),
           two(1 / v["pt"].to_numpy()[by_tgl], (pt[pos] / v["pt"].to_numpy())[by_tgl])]
    for (query, want), result in zip(ADF.items(), got):
        assert result["n"] == want["n"], (query, result, want)
        for key in want:
            assert math.isclose(result[key], want[key], rel_tol=1e-9), (query, key, result[key], want[key])


# ============================================================================================ the real browser
@pytest.mark.browser_a4
@pytest.mark.backend("browser")
@pytest.mark.layer("integration")
def test_browser_the_mouse_and_the_keyboard_move_the_selection(headless_session, tmp_path, physics_300):
    """The page as a standalone HTML file in Chromium. A person's actions, not a script setting values: the Collisions
    switch is clicked with the mouse, the lower handle of the collision vz slider is moved with the arrow keys, and
    the V0s tab is opened by a click. After each action the rows the page selects, its counter, its histograms and
    its scatter sources equal the rule; the page reports no error."""
    from bokeh.io import save
    from bokeh.resources import INLINE
    html = tmp_path / "toy_mc_joins_dashboard_selections.html"
    save(dash.toy_mc_joins_dashboard(physics_300), filename=str(html), resources=INLINE, title="ToyMC through joins")
    ref = _reference(physics_300)
    ctx, page, errors = _open(headless_session, html)
    try:
        page.set_viewport_size({"width": 1600, "height": 1400})
        page.wait_for_timeout(1500)
        read = lambda: page.evaluate(READ, _read_arg())
        out = read()
        _assert_state(out, physics_300, ref, counter=False)
        assert "after the first change" in out["counter"]                     # nobody touched a control yet
        # 1. the lower handle of the collision vz slider: 50 steps of 0.5 cm to the right with the keyboard
        handle = page.locator(".noUi-handle-lower").first
        handle.scroll_into_view_if_needed()
        handle.click()
        for _ in range(50):
            page.keyboard.press("ArrowRight")
        page.wait_for_timeout(800)
        cuts = {"selection_coll_vz": (-5.0, 30.0)}
        out = read()
        moved = _assert_state(out, physics_300, ref, cuts)
        assert 0 < moved["Collisions"].sum() < 300 and 0 < moved["TrkJ"].sum() < len(moved["TrkJ"])
        # 2. the switch of the Collisions table, with the mouse
        page.get_by_text("Selection on Collisions: ON").click()
        page.wait_for_timeout(800)
        out = read()
        switched = _assert_state(out, physics_300, ref, cuts, off=["Collisions"])
        assert switched["TrkJ"].all() and out["labels"]["selection_switch_Collisions"] == "Selection on Collisions: OFF"
        page.get_by_text("Selection on Collisions: OFF").click()
        page.wait_for_timeout(800)
        _assert_state(read(), physics_300, ref, cuts)
        # 3. the V0s tab of the selection area is opened by a click; its sliders are then on the screen
        page.get_by_text("V0s", exact=True).last.click()
        page.wait_for_timeout(800)
        assert page.get_by_text("V0s: V0 pz/pt").first.is_visible()
    finally:
        ctx.close()
    assert not errors, errors


# ============================================================================================ no page: the checks themselves
@pytest.mark.node_joins
def test_the_rule_reference_is_not_trivial_and_agrees_with_the_generator(physics_300):
    """Before the reference is used on the page: with a cut on collisions it equals the generator's own selection, a
    switch that is OFF removes the cut, and a row without a parent follows the rule."""
    want = _selected(physics_300, {"selection_coll_vz": (-5.0, 5.0)})
    theirs = gen.select_collision_descendants(physics_300, np.flatnonzero(want["Collisions"]).tolist())
    assert all(np.flatnonzero(want[target]).tolist() == theirs[table].tolist() for target, table in ROWS.items())
    assert all(m.all() for m in _selected(physics_300, {"selection_coll_vz": (-5.0, 5.0)}, off=["Collisions"]).values())
    assert all(m.all() for m in _selected(physics_300, {"selection_coll_vz": (-5.0, 5.0)}, off=["selection_coll_vz"]).values())
    raw = gen.inject_orphan_test_rows(gen.generate_toy_mc_events(n_collisions=10, seed=42))
    assert not _selected(raw)["TrkJ"][-1] and _selected(raw, off=["Collisions"])["TrkJ"][-1]
    assert _take([1.0, 2.0], [1, -1]).tolist()[0] == 2.0


@pytest.mark.node_joins
def test_the_description_of_the_selections_is_complete():
    """Every slider of the tests exists in the description and the other way round; every tab acts on its own table
    or join; nothing acts upwards (the Collisions table is cut only by the Collisions tab)."""
    described = {f"selection_{prefix}{col}": (tab, col) for tab, (prefix, sliders) in dash.SELECTIONS.items() for col, *_ in sliders}
    assert described == SLIDERS and tuple(dash.SELECTIONS) == TABS == tuple(dash.SELECTION_TARGETS)
    assert set(dash.TARGETS) == set(ROWS)
    assert [tab for tab, targets in dash.SELECTION_TARGETS.items() if "Collisions" in targets] == ["Collisions"]
    assert [tab for tab, targets in dash.SELECTION_TARGETS.items() if "TrkJ" in targets] == ["Collisions", "Tracks"]
    assert all("CascJ" in targets for targets in dash.SELECTION_TARGETS.values())      # every cut reaches the cascades
    for prefix, sliders in dash.SELECTIONS.values():
        for _, _, start, end, step in sliders:
            assert start < end and step > 0 and math.isclose((end - start) / step, round((end - start) / step), abs_tol=1e-9)
