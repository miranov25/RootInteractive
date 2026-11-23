"""PHASE_5 Group 7, dashboard D3: twenty invariance and oracle tests on the live ToyMC joins page (Claude21.RI,
2026-11-23).

In plain words:
- **The page:** the ToyMC joins dashboard (Tools/generators/toy_mc_joins_dashboard.py, unchanged) on the generator's
  9,200 collisions (seed 42): 101,260 tracks, 22,763 V0s and 7,151 cascades, plus the generator's own rows without a
  parent (a track without a collision, a V0 without its positive track, three cascades: without a V0, without a
  bachelor, and with a V0 that misses a track). Built by dashboards/invariance_dashboards.py, which you can run
  yourself: `... toymc --collisions 9200 --output page.html`.
- **Its figures read their parents through joins by row number:** TrkJ (a track and its collision), V0J (a V0, its
  positive and negative track, Pos and Neg, and the collision of each), CascJ (a cascade, everything of its V0, and its
  bachelor track with its collision). Under the figures, one selection tab per table (Collisions, Tracks, Pos, Neg,
  V0s, Cascades) with range sliders and on/off switches; the page keeps one mask per table or join (selection_mask_*).
- **What you can check by eye:** the page's tab "Consistency": the mass of every V0 and cascade computed in the browser
  from the joined tracks against its stored mass (on the diagonal), and the differences (at 0), for any selection.
- **One page, twenty tests,** as for D1 and D2: fixed states, after each state the page against numpy and pandas: the
  four masks row by row against the rule written level by level with plain numpy indexing (`_selected` of the
  Batch 4.c tests, checked there against the generator's own selection of descendants), every histogram bin by bin
  with its edges bit for bit, the profile cell by cell, the joined columns against plain numpy indexing
  (`_reference` of the Batch 4.c tests), the counter of selected rows, and what the scatter plots draw. Each test ends
  with every control back and the page exactly as at load.
- **The rule of the page** (the Architect, 2026-11-16): the mask of a table is the AND of the cuts in its tab, and it
  can be switched ON and OFF; a row is shown only if its parents are selected; the Tracks tab cuts every track, the Pos
  and Neg tabs only the positive or negative track of a V0; a row without a parent is hidden while a mask of that
  parent is ON; nothing acts upwards; a slider at the end of its scale is open there.
- **Real mouse and keys:** J05 (two table switches clicked), J16 (figure tabs clicked), J17 (slider handles moved with
  the arrow keys, a switch clicked), J18 (a query typed, Enter).
- **A defect found, a strict expected failure with its exact result (TQ_D7 below), in J18, on its own small page
  (300 collisions, with two text queries on joins):** a text query on a join does not see the join's columns.

Words used: the **baseline** is the state at load; a **target** is a table or join that has a mask (Collisions, TrkJ,
V0J, CascJ); a **precondition** is a property of the data a test relies on, checked first.

Switches: RI_INVARIANCE_COLLISIONS (default 9200); RI_INVARIANCE_FRESH_PAGE=1 (the page opened again after every
test). Target: browser_invariance_dashboards, group `invariance_toymc` (one worker).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.dashboards import invariance_dashboards as pages
from RootInteractive.InteractiveDrawing.bokeh.headless.dashboards import invariance_driver as drv
from RootInteractive.InteractiveDrawing.bokeh.headless.dashboards import invariance_oracle as O
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import histograms as H
from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4c_toymc_joins_dashboard import (
    KEYS, MASS_KEYS, MASS_TOLERANCE, NOISE_HISTOGRAMS, _reference)
from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_batch4c_toymc_joins_dashboard_selections import (
    SLIDERS, TABS, _selected, dash, gen)

COLLISIONS = int(os.environ.get("RI_INVARIANCE_COLLISIONS", str(pages.TOYMC_COLLISIONS)))
# target -> the table its figures read (the page builder names it "<target>_full")
TARGETS = {"Collisions": "Collisions_full", "TrkJ": "TrkJ_full", "V0J": "V0J_full", "CascJ": "CascJ_full"}
MASKS = list(TARGETS.items())
COUNTED = {"Collisions": "collisions", "TrkJ": "tracks", "V0J": "V0s", "CascJ": "cascades"}   # the counter's words
SWITCH = {name: name.replace("selection_", "selection_switch_") for name in SLIDERS}         # a slider's own switch
TABLE_SWITCH = {tab: "selection_switch_" + tab for tab in TABS}
SCALE = {f"selection_{prefix}{col}": (start, end) for prefix, sl in dash.SELECTIONS.values() for col, _, start, end, _ in sl}
FIGURE_TABS = ["Collisions", "Tracks", "V0s", "Cascades", "Consistency"]                  # checked on the page
CONTROLS = ([("slider", n) for n in SLIDERS] + [("switch", SWITCH[n]) for n in SLIDERS]
            + [("switch", TABLE_SWITCH[t]) for t in TABS] + [("tabs", "Collisions"), ("seltabs", "selection_tabs")])
PROBES = [["hCollVz", "bin_count", ["HistogramCDS"]], ["hCascMass", "bin_count", ["HistogramCDS"]]]
HISTS = list(dash.HISTOGRAMS)
PROFILE = dash.PROFILE
PROF = ["entries", "mean", "std", "quantile_0"]
REL = 1e-12
# the joined columns that are a plain gather or one basic operation of gathered values: equal bit for bit
EXACT_KEYS = {j: [k for k in keys if k not in MASS_KEYS and k != "trk_tgl"] for j, keys in KEYS.items()}
FINGERPRINT = {"mask": False, "masks": [list(m) for m in MASKS], "hist1": [[h, []] for h in HISTS],
               "hist": [PROFILE["name"]], "prof": [[PROFILE["name"] + "_1", PROF]]}

# A defect found by these tests (new in Group 7), a strict expected failure until it is fixed:
TQ_D7 = ("TQ-D7 (new, Group 7): a text query on a join does not see the join's columns. On a ToyMC page with a text "
         "query whose source is the join TrkJ, the query 'return trk_pt > 1 && coll_vz > 0' turns its field red with "
         "'error while computing: ReferenceError: trk_pt is not defined; the previous result is kept', and the "
         "histograms of the join stay unselected (all tracks counted). The query is evaluated on a table that does "
         "not hold the join's columns")


def browser(fn):
    """A test of this page: the target marker, and the group that keeps the twenty tests on one worker."""
    for mark in (pytest.mark.browser_invariance_dashboards, pytest.mark.backend("browser"), pytest.mark.layer("invariance"),
                 pytest.mark.xdist_group("invariance_toymc")):
        fn = mark(fn)
    return fn


def open_ended(cuts: dict) -> dict:
    """The cut a slider value means: a value at the end of the slider's scale is open there."""
    return {name: (-np.inf if lo <= SCALE[name][0] else lo, np.inf if hi >= SCALE[name][1] else hi)
            for name, (lo, hi) in cuts.items()}


class D3:
    """The shared page with the current state of its controls, and what the page must show in that state."""

    def __init__(self, page: drv.Page, raw: dict):
        self.page, self.raw = page, raw
        self.ref = _reference(raw)                                     # every joined column, by plain numpy indexing
        c = raw["collisions"]
        self.columns = {"Collisions": {"coll_vz": c["vz"].to_numpy(float), "coll_nTracks": c["nTracks"].to_numpy(float)},
                        **{j: {k: np.asarray(v, float) for k, v in cols.items()} for j, cols in self.ref.items()}}
        # the rows hidden at the baseline: exactly the rows without a parent (or with one missing further up)
        self.orphans = {t: ~m for t, m in _selected(raw).items()}
        self.reset()

    def reset(self):
        self.cuts, self.off = {}, set()

    @contextmanager
    def case(self, name: str, allowed_errors=()):
        try:
            with self.page.case(name, allowed_errors):
                yield self
        finally:
            self.reset()

    # -- states
    def set(self, label: str, cuts=None, full=(), off=(), on=()) -> dict:
        """Cuts {slider: (lo, hi)}, sliders back to their full scale, switches OFF or ON (a tab name: the table's
        switch; a slider name: the slider's own switch). The expected state follows."""
        changes = []
        for name, (lo, hi) in (cuts or {}).items():
            changes.append(["slider", name, [float(lo), float(hi)]])
            self.cuts[name] = (float(lo), float(hi))
        for name in full:
            changes.append(["slider", name, list(SCALE[name])])
            self.cuts.pop(name, None)
        for name, active in [(n, False) for n in off] + [(n, True) for n in on]:
            changes.append(["switch", TABLE_SWITCH.get(name) or SWITCH[name], active])
            (self.off.discard if active else self.off.add)(name)
        return self.page.apply(label, changes)

    def read_back(self):
        """After a real mouse or key action: the expected state takes the values the controls show."""
        v = self.page.values()
        self.cuts = {n: tuple(v[("slider", n)]) for n in SLIDERS if tuple(v[("slider", n)]) != SCALE[n]}
        self.off = {t for t in TABS if not v[("switch", TABLE_SWITCH[t])]} | {n for n in SLIDERS if not v[("switch", SWITCH[n])]}

    # -- what the page must show
    def want(self) -> dict:
        key = (tuple(sorted(self.cuts.items())), tuple(sorted(self.off)))
        if getattr(self, "_want_key", None) != key:
            self._want_key, self._want = key, _selected(self.raw, open_ended(self.cuts), tuple(self.off))
        return self._want

    def hist(self, name: str, want=None) -> dict:
        column, source, nbins, lo, hi = dash.HISTOGRAMS[name][:5]
        mask = (want or self.want())[source]
        return {"edges": [H.canonical_edges(nbins, lo, hi)],
                "count": H.counts([self.columns[source][column]], [nbins], [(lo, hi)], mask=mask)}

    def prof(self, want=None):
        cols = [self.columns[PROFILE["source"]][v] for v in PROFILE["variables"]]
        mask = (want or self.want())[PROFILE["source"]]
        p = O.profile(cols, PROFILE["nbins"], mask, PROFILE["axis"][0], PROFILE["quantiles"], ranges=PROFILE["range"])
        h = O.histogram(cols, PROFILE["nbins"], mask, ranges=PROFILE["range"])
        return p, h

    # -- reading and checking
    def read(self, hists=HISTS, prof=False, columns=(), drawn=()) -> dict:
        return self.page.read(mask=False, masks=MASKS, hist1=[(h, []) for h in hists], columns=columns, drawn=drawn,
                              prof=[(PROFILE["name"] + "_1", PROF)] if prof else (), hist=[PROFILE["name"]] if prof else ())

    def counter(self):
        """The four counts the page's counter shows; None while it shows its text at load (it is written at the first
        change of a control)."""
        text = self.page.pg.evaluate("() => window.__g7.one(m => m.type === 'Div' && m.name === 'selection_counter', "
                                     "'the counter').text")
        if "after the first change" in text:
            return None
        return [int(a) for a in re.findall(r"(\d+) of \d+", text)]

    def check(self, r: dict, where: str, hists=HISTS, prof=False, counter=True) -> dict:
        want = self.want()
        for target in TARGETS:
            drv.same_rows(r["mask:" + target], want[target], f"{where}, the selected rows of {target}")
            assert r[f"mask:{target}:padding"] == 0, f"{where}: rows past the end of {target} are selected"
        if counter:
            got = self.counter()
            if got is None:                           # not written yet: only right while the page is at its baseline
                assert not self.cuts and not self.off, f"{where}: the counter still shows its text at load"
            else:
                assert got == [int(want[t].sum()) for t in TARGETS], (f"{where}: the counter shows {got}, the rule "
                                                                        f"{[int(want[t].sum()) for t in TARGETS]}")
        for name in hists:
            got, exp = r["hist:" + name], self.hist(name, want)
            if name in NOISE_HISTOGRAMS:              # rounding noise around 0: compared by the entries in the two middle bins
                n = len(exp["count"]) // 2
                drv.same_values(got["edges"][0], exp["edges"][0], f"{where}, histogram {name}, the bin edges")
                assert got["count"].sum() == got["count"][n - 1:n + 1].sum() == exp["count"].sum(), \
                    f"{where}, histogram {name}: {got['count'].sum()} entries, {exp['count'].sum()} expected, all in the two middle bins"
            else:
                drv.same_histogram(got, exp, f"{where}, histogram {name}")
        if prof:
            p, h = self.prof(want)
            got = r["prof:" + PROFILE["name"] + "_1"]
            drv.same_values(got["entries"], p["entries"], f"{where}, the profile, entries")
            for c, e in (("mean", "mean"), ("std", "std"), ("quantile_0", "quantile_0")):
                drv.scaled_close(got[c], p[e], f"{where}, the profile, {c}", REL)
            drv.same_histogram(r["hist:" + PROFILE["name"]], h, f"{where}, the 3D histogram of the profile")
        return want

    def check_columns(self, r: dict, where: str, rows=None):
        """The joined columns read from the page against plain numpy indexing; `rows`: {join: rows to compare}."""
        for join, keys in KEYS.items():
            cols = r.get("col:" + TARGETS[join])
            if cols is None:
                continue
            for k in keys:
                got, exp = cols[k], np.asarray(self.ref[join][k], float)
                if rows is not None:
                    got, exp = got[rows[join]], exp[rows[join]]
                label = f"{where}, {join}.{k}"
                if k in MASS_KEYS:
                    drv.same_values(got, exp, label, atol=MASS_TOLERANCE)
                elif k == "trk_tgl":                  # sinh in the browser and in numpy: the last bit may differ
                    drv.scaled_close(got, exp, label, REL)
                else:
                    drv.same_values(got, exp, label)

    def check_drawn(self, r: dict, target: str, columns, where: str):
        """What the scatter plots of a target draw: rows of the selection only, as many as the page draws at most,
        each with the values of its row."""
        d = r["drawn:" + target]
        sel = np.flatnonzero(self.want()[target])
        assert d["__watched"], f"{where}: the drawn table of {target} is not watched (is its tab shown?)"
        rows = np.arange(len(self.want()[target])) if d["__rows"] is None else d["__rows"].astype(np.int64)
        limit = d["__nPoints"]
        n = len(sel) if limit < 0 or limit >= len(self.want()[target]) else min(limit, len(sel))
        assert len(rows) == n, f"{where}: {target} draws {len(rows)} rows, {n} expected ({len(sel)} selected, at most {limit})"
        assert np.isin(rows, sel).all(), f"{where}: {target} draws rows outside the selection: {rows[~np.isin(rows, sel)][:8].tolist()}"
        for c in columns:
            drv.same_values(d[c], np.asarray(self.ref[target][c], float)[rows], f"{where}, {target} drawn {c}",
                            atol=MASS_TOLERANCE if c in MASS_KEYS else 0.0)


def _build(out: Path, collisions: int, text_query: bool = False):
    """The page in a child process (as you run it by hand), and meanwhile the same tables here."""
    name = "page_text_query.html" if text_query else "page.html"
    cmd = [sys.executable, "-m", "RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.invariance_dashboards",
           "toymc", "--collisions", str(collisions), "--output", str(out / name)] + (["--text-query"] if text_query else [])
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(out))
    raw = pages.toymc_tables(collisions)
    _, err = child.communicate(timeout=900)
    if child.returncode != 0:
        pytest.fail(f"the page could not be built ({child.returncode}):\n{err[-3000:]}", pytrace=False)
    manifest = json.loads((out / (name + ".manifest.json")).read_text())
    assert manifest["table_sha256"] == pages.toymc_sha256(raw), "the page was built from other tables"
    manifest["build_process_s"] = round(time.perf_counter() - t0, 2)
    return out / name, raw, manifest


@pytest.fixture(scope="module")
def d3(headless_session, headless_out):
    out = headless_out / "invariance" / "toymc"
    out.mkdir(parents=True, exist_ok=True)
    html, raw, manifest = _build(out, COLLISIONS)
    page = drv.Page(headless_session, html, {}, CONTROLS, fingerprint_spec=FINGERPRINT, probes=PROBES, out_dir=out)
    titles = page.pg.evaluate("() => window.__g7.widget('tabs', 'Collisions').tabs.map(p => p.title)")
    assert titles == FIGURE_TABS, f"the figure tabs of the page: {titles}"
    page.pg.screenshot(path=str(out / "at_load.png"))
    d = D3(page, raw)
    d.manifest, d.out = manifest, out
    d.at_load = d.read(prof=True)
    yield d
    summary = page.write_timings(out / "timings.json")
    page.apply("the tab Consistency, for the picture", [["tabs", "Collisions", FIGURE_TABS.index("Consistency")]])
    page.pg.screenshot(path=str(out / "at_the_end_tab_Consistency.png"))
    print("\nD3 timing:", json.dumps({k: summary[k] for k in ("load_s", "states", "page_ms_median", "page_ms_p90",
                                                               "read_s_total", "tests_s_total")}))
    page.close()


# ================================================================== the twenty tests
@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("FILTER.range")
def test_j01_a_collision_cut_acts_through_the_joins(d3):
    """J01. Collision vz in [-10, 10], [-2, 2.5], then full: the masks of the collisions, tracks, V0s and cascades
    equal the rule row by row, and also the generator's own selection of the descendants of the selected collisions;
    every histogram equals numpy; the counter shows the four counts."""
    with d3.case("J01"):
        for lo, hi in [(-10.0, 10.0), (-2.0, 2.5), (-30.0, 30.0)]:
            where = f"J01 vz [{lo}, {hi}]"
            d3.set(where, cuts={"selection_coll_vz": (lo, hi)})
            want = d3.check(d3.read(), where)
            theirs = gen.select_collision_descendants(d3.raw, np.flatnonzero(want["Collisions"]).tolist())
            for target, table in (("TrkJ", "tracks"), ("V0J", "v0s"), ("CascJ", "cascades")):
                assert np.flatnonzero(want[target]).tolist() == list(theirs[table]), f"{where}: {target} against the generator"
                if hi - lo < 10:                      # the narrow cut is a real cut on every table
                    assert 0.1 * len(want[target]) < want[target].sum() < 0.6 * len(want[target]), (where, target)


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("FILTER.range")
def test_j02_a_track_cut_acts_on_both_tracks_of_a_v0_and_on_the_bachelor(d3):
    """J02. Track pt from 0.5, then in [0.3, 1.2], then pt from 0.2 with eta in [-0.5, 0.8]: the rule; every selected
    V0 has both tracks in the cut, every selected cascade its bachelor and its V0's tracks."""
    ref = d3.ref
    with d3.case("J02"):
        for cuts in [{"selection_trk_pt": (0.5, 10.0)}, {"selection_trk_pt": (0.3, 1.2)},
                     {"selection_trk_pt": (0.2, 10.0), "selection_trk_eta": (-0.5, 0.8)}]:
            where = f"J02 {cuts}"
            d3.set(where, cuts=cuts, full=[n for n in ("selection_trk_pt", "selection_trk_eta") if n not in cuts])
            want = d3.check(d3.read(), where)
            lo, hi = open_ended(cuts)["selection_trk_pt"]
            for join, keys in (("V0J", ("pos_pt", "neg_pt")), ("CascJ", ("pos_pt", "neg_pt", "bach_pt"))):
                for k in keys:
                    v = np.asarray(ref[join][k])[want[join]]
                    assert ((v >= lo) & (v <= hi)).all(), f"{where}: a selected row of {join} has {k} outside the cut"
            assert want["Collisions"].all() and 0 < want["V0J"].sum() < len(want["V0J"])


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("FILTER.range")
def test_j03_pos_and_neg_are_separate_and_together_equal_the_tracks_cut_for_v0s(d3):
    """J03. Pos.pt from 1 alone, Neg.pt from 1 alone, both, and Tracks pt from 1: each acts only on its own track of a
    V0 (the tracks and collisions stay whole); Pos and Neg together select exactly the V0s that the Tracks cut
    selects (the Tracks cut also acts on the bachelors, so the cascades may differ)."""
    with d3.case("J03"):
        got = {}
        for label, cuts, full in [("pos", {"selection_pos_pt": (1.0, 10.0)}, []),
                                  ("neg", {"selection_neg_pt": (1.0, 10.0)}, ["selection_pos_pt"]),
                                  ("both", {"selection_pos_pt": (1.0, 10.0), "selection_neg_pt": (1.0, 10.0)}, []),
                                  ("tracks", {"selection_trk_pt": (1.0, 10.0)}, ["selection_pos_pt", "selection_neg_pt"])]:
            where = f"J03 {label}"
            d3.set(where, cuts=cuts, full=full)
            r = d3.read()
            got[label] = d3.check(r, where)
            if label != "tracks":
                assert got[label]["TrkJ"].sum() == len(got[label]["TrkJ"]) - 1 and got[label]["Collisions"].all(), where
        assert not np.array_equal(got["pos"]["V0J"], got["neg"]["V0J"])
        drv.same_rows(got["both"]["V0J"], got["pos"]["V0J"] & got["neg"]["V0J"], "J03 Pos and Neg = each of them")
        drv.same_rows(got["both"]["V0J"], got["tracks"]["V0J"], "J03 Pos and Neg together against the Tracks cut, V0s")


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("FILTER.intersection")
def test_j04_the_order_of_cuts_does_not_matter(d3):
    """J04. A collision cut then a track cut, against the track cut then the collision cut: every mask, histogram and
    the profile give exactly the same arrays; removing them in the other order gives the baseline (checked at the
    end of the test)."""
    vz, pt = {"selection_coll_vz": (-5.0, 5.0)}, {"selection_trk_pt": (0.4, 10.0)}
    with d3.case("J04"):
        d3.set("J04 vz first", cuts=vz)
        d3.set("J04 then pt", cuts=pt)
        a = d3.page.fingerprint()
        d3.check(d3.read(prof=True), "J04 vz then pt", prof=True)
        d3.set("J04 remove vz first", full=list(vz))
        d3.set("J04 remove pt", full=list(pt))
        d3.set("J04 pt first", cuts=pt)
        d3.set("J04 then vz", cuts=vz)
        b = d3.page.fingerprint()
        assert a == b, f"J04: the arrays differ by the order of the cuts: {[k for k in a if a[k] != b.get(k)]}"
        d3.set("J04 remove pt first", full=list(pt))
        d3.set("J04 then vz", full=list(vz))


ALL_TABS_CUTS = {"selection_coll_vz": (-8.0, 8.0), "selection_trk_pt": (0.2, 10.0), "selection_pos_eta": (-1.0, 1.0),
                 "selection_neg_eta": (-1.0, 1.0), "selection_v0_tgl": (-1.0, 1.0), "selection_casc_pt": (0.5, 10.0)}


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("FILTER.intersection")
def test_j05_a_table_switched_off_and_on(d3):
    """J05. With a cut in every tab, each table's switch OFF: its cuts do nothing (its sliders keep their values);
    ON again: every mask exactly as before. Tracks and Cascades are switched with the real mouse, the others on the
    models."""
    with d3.case("J05"):
        d3.set("J05 a cut in every tab", cuts=ALL_TABS_CUTS)
        before = d3.check(d3.read(), "J05 all ON")
        for tab in TABS:
            where = f"J05 {tab} OFF"
            if tab in ("Tracks", "Cascades"):
                d3.page.click_tab(tab, which="last")
                d3.page.pg.get_by_text(f"Selection on {tab}: ON", exact=True).click()
                d3.page.waited(where + " (real click)")
                d3.read_back()
                assert tab in d3.off, f"{where}: the click did not switch the table off"
            else:
                d3.set(where, off=[tab])
            d3.check(d3.read(), where)
            label = d3.page.pg.evaluate("n => window.__g7.widget('switch', n).label", TABLE_SWITCH[tab])
            assert label == f"Selection on {tab}: OFF", (where, label)
            if tab in ("Tracks", "Cascades"):
                d3.page.pg.get_by_text(f"Selection on {tab}: OFF", exact=True).click()
                d3.page.waited(f"J05 {tab} ON (real click)")
                d3.read_back()
                assert tab not in d3.off
            else:
                d3.set(f"J05 {tab} ON", on=[tab])
            r = d3.read(hists=())
            for target in TARGETS:
                drv.same_rows(r["mask:" + target], before[target], f"J05 {tab} ON again, {target}")
        d3.page.click_tab("Collisions", which="last")


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("FILTER.range")
def test_j06_a_sliders_own_switch_off_equals_the_slider_at_full_range(d3):
    """J06. Every one of the 14 sliders cut to the middle half of its scale with its own switch OFF: every row that has
    its parents is selected as at the baseline (the slider at full range); the rule for all rows (a row without a
    parent is shown when no cut of its parent's table is ON: here when the only slider of Collisions is OFF).
    Switched ON: the rule with that cut."""
    base = d3.at_load
    with d3.case("J06"):
        for name in SLIDERS:
            start, end = SCALE[name]
            q = (end - start) / 4
            d3.set(f"J06 {name} cut, its switch OFF", cuts={name: (start + q, end - q)}, off=[name])
            r = d3.read(hists=())
            d3.check(r, f"J06 {name} OFF", hists=())
            for target in TARGETS:
                keep = ~d3.orphans[target]
                drv.same_rows(r["mask:" + target][keep], base["mask:" + target][keep], f"J06 {name} OFF, {target}")
            d3.set(f"J06 {name} switch ON", on=[name])
            want = d3.check(d3.read(hists=()), f"J06 {name} ON", hists=())
            assert any(want[t].sum() < base["mask:" + t].sum() for t in TARGETS), f"J06 {name}: the cut selects nothing less"
            d3.set(f"J06 {name} back", full=[name])


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("HIST.histogram_1d")
def test_j07_a_partition_of_the_collisions_adds_up_through_the_joins(d3):
    """J07. vz up to 0.25, then from 0.25 (no collision has vz exactly 0.25, checked): for every table the two masks
    are disjoint and together the baseline, and every histogram of the two halves adds up to the baseline's, bin by
    bin."""
    s = 0.25
    assert not (d3.raw["collisions"]["vz"].to_numpy() == s).any(), "precondition: no collision at the split"
    base = d3.at_load
    with d3.case("J07"):
        d3.set("J07 vz up to 0.25", cuts={"selection_coll_vz": (-30.0, s)})
        low = d3.read()
        d3.check(low, "J07 low half")
        d3.set("J07 vz from 0.25", cuts={"selection_coll_vz": (s, 30.0)})
        high = d3.read()
        d3.check(high, "J07 high half")
        for target in TARGETS:
            a, b = low["mask:" + target], high["mask:" + target]
            assert not (a & b).any(), f"J07 {target}: a row in both halves"
            drv.same_rows(a | b, base["mask:" + target], f"J07 {target}: the two halves against the baseline")
            assert 0 < a.sum() < len(a) and 0 < b.sum() < len(b), f"J07 {target}: a half is empty"
        for name in HISTS:
            drv.same_values(low["hist:" + name]["count"] + high["hist:" + name]["count"], base["hist:" + name]["count"],
                            f"J07 histogram {name}: low + high against the baseline")


@browser
@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("DISPLAY.downsampler")
def test_j08_cuts_do_not_change_joined_values_and_the_plots_draw_the_selection(d3):
    """J08. Under two cuts, every joined column of TrkJ, V0J and CascJ equals plain numpy indexing of the four tables
    (bit for bit; sinh of eta to 1e-12; the masses computed in the browser to 1e-9 GeV/c2); with the V0s tab shown,
    its scatter plots draw only selected V0s, each with its own values."""
    cols = [(TARGETS[j], KEYS[j]) for j in KEYS]
    with d3.case("J08"):
        for label, cuts in [("vz and pt", {"selection_coll_vz": (-5.0, 5.0), "selection_trk_pt": (0.5, 10.0)}),
                            ("V0 mass window", {"selection_v0_deltaMass": (-0.02, 0.02)})]:
            where = f"J08 {label}"
            d3.set(where, cuts=cuts, full=[n for n in d3.cuts if n not in cuts])
            d3.page.apply(where + ", the tab V0s", [["tabs", "Collisions", FIGURE_TABS.index("V0s")]])
            r = d3.read(hists=(), columns=cols, drawn=[("V0J", ["pos_pt", "neg_pt", "v0_mass"])])
            d3.check(r, where, hists=())
            d3.check_columns(r, where)
            d3.check_drawn(r, "V0J", ["pos_pt", "neg_pt", "v0_mass"], where)


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("ALIAS.context")
@pytest.mark.feature("DISPLAY.downsampler")
def test_j09_the_masses_from_the_joined_tracks_agree_under_cuts(d3):
    """J09. The tab Consistency under three cuts: for every selected V0 and cascade, the mass computed in the browser
    from the joined tracks equals the stored mass to 1e-9 GeV/c2, and the two tracks' collisions have the same vz
    exactly; the difference histograms hold exactly the selected rows; the plots of the tab draw selected rows."""
    cols = [("V0J_full", ["v0_massDiff", "v0_collisionVzDiff"]), ("CascJ_full", ["casc_massDiff", "casc_collisionVzDiff"])]
    names = ["hV0MassDiff", "hCascMassDiff", "hV0CollVzDiff", "hCascCollVzDiff"]
    with d3.case("J09"):
        d3.page.apply("J09 the tab Consistency", [["tabs", "Collisions", FIGURE_TABS.index("Consistency")]])
        for label, cuts in [("vz", {"selection_coll_vz": (-5.0, 5.0)}),
                            ("V0 mass and cascade pt", {"selection_v0_mass": (0.4, 1.2), "selection_casc_pt": (0.5, 10.0)}),
                            ("track eta", {"selection_trk_eta": (-0.8, 0.8)})]:
            where = f"J09 {label}"
            d3.set(where, cuts=cuts, full=[n for n in d3.cuts if n not in cuts])
            r = d3.read(hists=names, columns=cols, drawn=[("V0J", ["v0_mass", "v0_massFromTracks"]),
                                                          ("CascJ", ["casc_mass", "casc_massFromV0AndBachelor"])])
            want = d3.check(r, where, hists=names)
            for table, diff, vzdiff, target in (("V0J_full", "v0_massDiff", "v0_collisionVzDiff", "V0J"),
                                                ("CascJ_full", "casc_massDiff", "casc_collisionVzDiff", "CascJ")):
                m = r["col:" + table][diff][want[target]]
                assert np.isfinite(m).all() and (np.abs(m) <= MASS_TOLERANCE).all(), f"{where}: {diff} beyond 1e-9 GeV/c2"
                drv.every_value(r["col:" + table][vzdiff][want[target]], 0.0, f"{where}, {vzdiff}")
            d3.check_drawn(r, "V0J", ["v0_mass", "v0_massFromTracks"], where)
            d3.check_drawn(r, "CascJ", ["casc_mass", "casc_massFromV0AndBachelor"], where)


ORPHAN_STATES = [("all ON", []), ("Collisions OFF", ["Collisions"]),
                 ("the parents of V0s OFF", ["Collisions", "Tracks", "Pos", "Neg", "V0s"]), ("all OFF", list(TABS))]


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("JOIN.cdsjoin.outer")
def test_j10_rows_without_a_parent_are_hidden_while_a_mask_of_the_parent_is_on(d3):
    """J10. The generator's rows without a parent (the last track, the last V0, the last three cascades): with every
    mask ON they are hidden; with the Collisions mask OFF the track is shown; with the masks of its parents (and its
    own, its mass is unknown) OFF the V0 is shown; with all OFF every row is shown."""
    t, v = len(d3.raw["tracks"]) - 1, len(d3.raw["v0s"]) - 1
    cascades = list(range(len(d3.raw["cascades"]) - 3, len(d3.raw["cascades"])))
    assert d3.raw["tracks"]["collisionIdx"].iloc[t] == -1 and d3.raw["v0s"]["posTrackIdx"].iloc[v] == -1, "precondition"
    with d3.case("J10"):
        seen = {}
        for label, off in ORPHAN_STATES:
            where = f"J10 {label}"
            d3.set(where, on=[x for x in TABS if x not in off and x in d3.off], off=[x for x in off if x not in d3.off])
            seen[label] = d3.check(d3.read(), where)
        assert not seen["all ON"]["TrkJ"][t] and not seen["all ON"]["V0J"][v] and not seen["all ON"]["CascJ"][cascades].any()
        assert seen["Collisions OFF"]["TrkJ"][t] and not seen["Collisions OFF"]["V0J"][v]
        assert seen["the parents of V0s OFF"]["V0J"][v]
        assert all(m.all() for m in seen["all OFF"].values())


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("FILTER.intersection")
def test_j11_nothing_acts_upwards(d3):
    """J11. Cuts on tracks, on V0s, on cascades, then all three: a cut never removes a row of a table above its own
    (the collisions stay whole under a track cut; the tracks under a V0 cut; the V0s under a cascade cut)."""
    base = d3.at_load
    trk, v0, casc = ({"selection_trk_pt": (0.6, 10.0)}, {"selection_v0_tgl": (-0.7, 0.7)},
                     {"selection_casc_deltaMass": (-0.02, 0.02)})
    with d3.case("J11"):
        for label, cuts, untouched in [("tracks", trk, ["Collisions"]), ("V0s", v0, ["Collisions", "TrkJ"]),
                                       ("cascades", casc, ["Collisions", "TrkJ", "V0J"]),
                                       ("all three", {**trk, **v0, **casc}, ["Collisions"])]:
            where = f"J11 {label}"
            d3.set(where, cuts=cuts, full=[n for n in d3.cuts if n not in cuts])
            r = d3.read(hists=())
            want = d3.check(r, where, hists=())
            for target in untouched:
                drv.same_rows(r["mask:" + target], base["mask:" + target], f"{where}: {target} stays as at load")
            assert any(want[x].sum() < base["mask:" + x].sum() for x in TARGETS if x not in untouched), where


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("HIST.histogram_1d")
def test_j12_every_histogram_of_the_joins_equals_numpy_under_combined_cuts(d3):
    """J12. Three states with cuts in several tabs at once: all twenty histograms (collisions, tracks, V0s, cascades,
    the consistency differences) equal numpy bin by bin with the same edges, and the counter shows the four counts."""
    states = [{"selection_coll_vz": (-6.0, 9.0), "selection_pos_pt": (0.3, 10.0), "selection_v0_tgl": (-1.2, 1.2)},
              {"selection_trk_eta": (-1.0, 0.4), "selection_casc_mass": (1.25, 1.75), "selection_neg_pt": (0.25, 3.0)},
              {"selection_v0_pt": (0.8, 6.0), "selection_casc_deltaMass": (-0.05, 0.05), "selection_coll_vz": (-30.0, 2.0)}]
    with d3.case("J12"):
        for k, cuts in enumerate(states):
            where = f"J12 state {k + 1}"
            d3.set(where, cuts=cuts, full=[n for n in d3.cuts if n not in cuts])
            want = d3.check(d3.read(), where)
            assert all(0 < want[t].sum() < len(want[t]) for t in ("TrkJ", "V0J", "CascJ")), where


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("FILTER.range")
def test_j13_a_cascade_follows_the_cut_on_its_v0(d3):
    """J13. The V0 mass window, the Lambda mass region, and V0 pz/pt: every selected cascade has its V0 selected; the
    cascades and the histogram of their V0's mass equal numpy; the tracks and collisions stay whole."""
    v0_idx = d3.raw["cascades"]["v0Idx"].to_numpy()
    with d3.case("J13"):
        for cuts in [{"selection_v0_deltaMass": (-0.02, 0.02)}, {"selection_v0_mass": (1.1, 1.13)},
                     {"selection_v0_tgl": (-0.5, 0.5)}]:
            where = f"J13 {cuts}"
            d3.set(where, cuts=cuts, full=[n for n in d3.cuts if n not in cuts])
            want = d3.check(d3.read(hists=["hCascV0Mass", "hCascMass", "hV0Mass"]), where,
                            hists=["hCascV0Mass", "hCascMass", "hV0Mass"])
            sel = np.flatnonzero(want["CascJ"])
            assert (v0_idx[sel] >= 0).all() and want["V0J"][v0_idx[sel]].all(), f"{where}: a cascade whose V0 is not selected"
            assert 0 < len(sel) < len(want["CascJ"]) and want["Collisions"].all() and want["TrkJ"].sum() == len(want["TrkJ"]) - 1


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("FILTER.intersection")
def test_j14_four_tables_cut_at_once_in_sixteen_states(d3):
    """J14. Collisions, Tracks, V0s and Cascades each full or cut, all 16 combinations, one change per state (Gray
    code): the four masks and the counter equal the rule in every state; the histograms in two of them."""
    cut = {"selection_coll_vz": (-4.0, 6.0), "selection_trk_pt": (0.3, 10.0), "selection_v0_deltaMass": (-0.03, 0.03),
           "selection_casc_pt": (0.6, 10.0)}
    names = list(cut)
    with d3.case("J14"):
        seen = set()
        for k in range(16):
            g = k ^ (k >> 1)
            chosen = {names[b]: cut[names[b]] for b in range(4) if g >> b & 1}
            where = f"J14 state {k} ({sorted(n.split('_', 1)[1] for n in chosen)})"
            d3.set(where, cuts=chosen, full=[n for n in d3.cuts if n not in chosen])
            hists = HISTS if k in (7, 15) else ()
            want = d3.check(d3.read(hists=hists), where, hists=hists)
            seen.add(tuple(int(want[t].sum()) for t in TARGETS))
        assert len(seen) >= 12, f"J14: the states are not different enough ({len(seen)})"


@browser
@pytest.mark.feature("JOIN.cdsjoin.basic")
@pytest.mark.feature("JOIN.recursive")
def test_j15_rows_whose_tracks_have_row_numbers_past_65536(d3):
    """J15. The V0s and cascades whose tracks have row numbers above 65,536 (they exist: 101,260 tracks; checked):
    their joined columns equal plain numpy indexing, and under a track cut their selection equals the rule."""
    v, ca = d3.raw["v0s"], d3.raw["cascades"]
    high_v0 = (v["posTrackIdx"].to_numpy() > 65536) | (v["negTrackIdx"].to_numpy() > 65536)
    high_casc = (ca["bachelorTrackIdx"].to_numpy() > 65536)
    assert high_v0.sum() > 1000 and high_casc.sum() > 500, f"precondition: {high_v0.sum()} V0s, {high_casc.sum()} cascades"
    rows = {"TrkJ": np.arange(65536, len(d3.raw["tracks"])), "V0J": np.flatnonzero(high_v0), "CascJ": np.flatnonzero(high_casc)}
    with d3.case("J15"):
        d3.set("J15 track pt from 0.6", cuts={"selection_trk_pt": (0.6, 10.0)})
        r = d3.read(hists=(), columns=[(TARGETS[j], KEYS[j]) for j in KEYS])
        want = d3.check(r, "J15", hists=())
        d3.check_columns(r, "J15 rows past 65,536", rows=rows)
        for target, sel in rows.items():
            drv.same_rows(r["mask:" + target][sel], want[target][sel], f"J15 {target} rows past 65,536")
            assert 0 < want[target][sel].sum() < len(sel), f"J15 {target}: the cut does not act on these rows"


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("DISPLAY.lazy_tabs")
@pytest.mark.feature("DISPLAY.downsampler")
def test_j16_a_hidden_tab_shows_the_current_selection_when_opened(d3):
    """J16. Cuts changed while the figure tab Cascades is hidden; then the tab is opened with a real click: its plots
    draw exactly the selected cascades and its histograms equal numpy; hidden again, another cut, opened again: the
    new selection."""
    names = [n for n, h in dash.HISTOGRAMS.items() if h[1] == "CascJ"]
    with d3.case("J16"):
        for k, cuts in enumerate([{"selection_casc_mass": (1.2, 1.8), "selection_v0_deltaMass": (-0.02, 0.02)},
                                  {"selection_casc_pt": (1.0, 10.0)}]):
            where = f"J16 step {k + 1}"
            d3.page.click_tab("Collisions", which="first")
            d3.page.waited(where + ": the tab Collisions (real click)")
            d3.set(where + ": cuts while Cascades is hidden", cuts=cuts)
            d3.page.click_tab("Cascades", which="first")
            d3.page.waited(where + ": the tab Cascades (real click)")
            r = d3.read(hists=names, drawn=[("CascJ", ["casc_mass", "bach_pt", "v0_mass"])])
            d3.check(r, where, hists=names)
            d3.check_drawn(r, "CascJ", ["casc_mass", "bach_pt", "v0_mass"], where)
        d3.page.click_tab("Collisions", which="first")
        d3.page.waited("J16 back to the tab Collisions")


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("FILTER.range")
def test_j17_the_mouse_and_the_keyboard_move_the_selection(d3):
    """J17. A person's actions: the lower handle of the collision vz slider moved 20 steps right with the arrow keys,
    the upper handle of the track pt slider 30 steps left, the Collisions switch clicked off and on with the mouse.
    After each action the controls' values are read back, and the masks, the histograms and the counter equal the
    rule for them."""
    with d3.case("J17"):
        d3.page.click_tab("Collisions", which="last")
        d3.page.keys("slider", "selection_coll_vz", "lower", "ArrowRight", 20)
        d3.page.waited("J17 vz lower handle (keys)")
        d3.read_back()
        assert d3.cuts.get("selection_coll_vz", (None,))[0] == -20.0, d3.cuts
        d3.check(d3.read(), "J17 vz moved")
        d3.page.click_tab("Tracks", which="last")
        d3.page.keys("slider", "selection_trk_pt", "upper", "ArrowLeft", 30)
        d3.page.waited("J17 pt upper handle (keys)")
        d3.read_back()
        assert abs(d3.cuts["selection_trk_pt"][1] - 7.0) < 1e-9, d3.cuts
        d3.check(d3.read(), "J17 pt moved")
        d3.page.click_tab("Collisions", which="last")
        for state in ("OFF", "ON"):
            d3.page.pg.get_by_text("Selection on Collisions: " + ("ON" if state == "OFF" else "OFF"), exact=True).click()
            d3.page.waited(f"J17 Collisions {state} (click)")
            d3.read_back()
            assert ("Collisions" in d3.off) == (state == "OFF"), d3.off
            d3.check(d3.read(), f"J17 Collisions {state}")


@pytest.fixture(scope="module")
def d3_query(headless_session, headless_out):
    """J18's own small page: 300 collisions, the figures, and a text query on TrkJ and one on V0J."""
    out = headless_out / "invariance" / "toymc_text_query"
    out.mkdir(parents=True, exist_ok=True)
    html, raw, manifest = _build(out, pages.TOYMC_QUERY_COLLISIONS, text_query=True)
    page = drv.Page(headless_session, html, {}, [], fingerprint_spec={}, probes=[["hTrkPt", "bin_count", ["HistogramCDS"]]],
                    out_dir=out)
    yield page, raw
    page.pg.screenshot(path=str(out / "after_J18.png"), full_page=True)
    page.close()


@browser
@pytest.mark.xfail(strict=True, raises=drv.KnownDefect, reason=TQ_D7)
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.feature("JOIN.recursive")
def test_j18_a_text_query_on_a_join_selects_its_rows(d3_query):
    """J18. On its own page: the query 'return trk_pt > 1 && coll_vz > 0' typed into the text query of the join TrkJ
    (real keys, Enter): the histogram of the tracks' pt counts exactly the tracks numpy selects, and the field is not
    red. Today: TQ_D7."""
    page, raw = d3_query
    t, c = raw["tracks"], raw["collisions"]
    idx = t["collisionIdx"].to_numpy()
    vz = np.where(idx >= 0, c["vz"].to_numpy()[np.maximum(idx, 0)], np.nan)
    pt = t["pt"].to_numpy(float)
    keep = (pt > 1) & (vz > 0)
    _, _, nbins, lo, hi = dash.HISTOGRAMS["hTrkPt"][:5]
    expected = H.counts([pt], [nbins], [(lo, hi)], mask=keep)
    with page.case("J18", allowed_errors=[r"ReferenceError: \w+ is not defined"]):
        field = page.pg.locator("textarea").first
        field.click()
        field.fill("return trk_pt > 1 && coll_vz > 0")
        field.press("Enter")
        page.waited("J18 the query typed (real keys)")
        got = page.read(mask=False, hist1=[("hTrkPt", [])])["hist:hTrkPt"]["count"]
        fields = page.pg.evaluate("() => window.__g7.all().filter(m => m.type === 'TextAreaInput').map(m => "
                                  "[m.value, m.title, [...m.css_classes]])")
        mine = [f for f in fields if f[0].startswith("return trk_pt")]
        assert len(mine) == 1, fields
        drv.known_defect("ReferenceError: trk_pt is not defined" in mine[0][1] and "ri-problem" in mine[0][2]
                         and got.sum() == len(t), TQ_D7 + f" (seen: field title {mine[0][1]!r}, {int(got.sum())} tracks counted)")
        assert "ri-problem" not in mine[0][2], f"J18: the field is red: {mine[0][1]!r}"
        drv.same_values(got, expected, "J18 hTrkPt under the query on TrkJ")


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("HIST.profile_unbinned")
def test_j19_the_profile_through_the_joins_equals_numpy(d3):
    """J19. The profile of Pos.pt/pt against 1/pt per particle type (V0J: entries, mean, standard deviation and median
    per cell, unbinned) and its 3D histogram, at the baseline and under two cuts: equal to numpy (entries and counts
    exactly, the rest to 1e-12)."""
    with d3.case("J19"):
        d3.check(d3.at_load, "J19 baseline", hists=(), prof=True, counter=False)
        for label, cuts in [("vz and eta", {"selection_coll_vz": (-5.0, 5.0), "selection_trk_eta": (-0.8, 0.8)}),
                            ("V0 pt and Pos pt", {"selection_v0_pt": (0.5, 5.0), "selection_pos_pt": (0.3, 10.0)})]:
            where = f"J19 {label}"
            d3.set(where, cuts=cuts, full=[n for n in d3.cuts if n not in cuts])
            d3.check(d3.read(hists=(), prof=True), where, hists=(), prof=True)


@browser
@pytest.mark.feature("JOIN.recursive")
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("HIST.profile_unbinned")
def test_j20_the_whole_sequence_leaves_no_trace(d3):
    """J20. After J01 to J19 the page gives exactly the arrays it gave at load (masks, histograms, profile), and these
    equal numpy at the baseline: every row but the rows without a parent selected."""
    with d3.case("J20"):
        fp = d3.page.fingerprint()
        dirty = [k for k in d3.page.fingerprint_at_load if fp.get(k) != d3.page.fingerprint_at_load[k]]
        assert not dirty, f"J20: arrays differ from those at load: {dirty}"
        r = d3.read(prof=True)
        want = d3.check(r, "J20 baseline", prof=True)
        assert [int(want[t].sum()) for t in TARGETS] == [len(want["Collisions"]), len(want["TrkJ"]) - 1,
                                                          len(want["V0J"]) - 1, len(want["CascJ"]) - 3]
    if d3.page.notes:
        pytest.fail("problems left by earlier tests that failed: " + "; ".join(d3.page.notes), pytrace=False)


# ================================================================== without a browser
@pytest.mark.headless_unit
def test_the_toymc_tables_and_the_fixed_range_oracle():
    """The tables of D3 at 300 collisions: the same checksum twice, the rows without a parent at the end; the rule
    reference equals the generator's selection of descendants; the fixed-range oracle on a hand-checked case: the
    values 0, 0.5, 1 in [0, 1] with 2 bins give 1 and 2 (the maximum in the last bin), a NaN and a value outside are
    not counted, and the profile's median of 1, 2, 10 in one cell is 2."""
    raw = pages.toymc_tables(300)
    assert pages.toymc_sha256(raw) == pages.toymc_sha256(pages.toymc_tables(300))
    assert raw["tracks"]["collisionIdx"].iloc[-1] == -1 and raw["v0s"]["posTrackIdx"].iloc[-1] == -1
    assert raw["cascades"]["origin"].iloc[-3:].tolist() == ["missing_v0", "missing_bachelor", "missing_v0_daughter"]
    want = _selected(raw, {"selection_coll_vz": (-5.0, 5.0)})
    theirs = gen.select_collision_descendants(raw, np.flatnonzero(want["Collisions"]).tolist())
    for target, table in (("TrkJ", "tracks"), ("V0J", "v0s"), ("CascJ", "cascades")):
        assert np.flatnonzero(want[target]).tolist() == list(theirs[table]), target
    h = O.histogram([np.array([0.0, 0.5, 1.0, np.nan, 2.0])], [2], np.ones(5, bool), ranges=[(0.0, 1.0)])
    assert h["count"].tolist() == [1.0, 2.0] and h["edges"][0].tolist() == [0.0, 0.5, 1.0]
    p = O.profile([np.zeros(3), np.array([1.0, 2.0, 10.0])], [1, 1], np.ones(3, bool), 1, [0.5], ranges=[(0, 1), (0, 20)])
    assert p["entries"].tolist() == [3.0] and p["quantile_0"][0] == 2.0
    assert open_ended({"selection_coll_vz": (-30.0, 5.0)}) == {"selection_coll_vz": (-np.inf, 5.0)}
