"""Three linked tables on one real page, V0 -> track -> event — PHASE_4_5 stage A4. Oracle and invariant tests only.

Plain words. A V0 candidate points to its event and to its two daughter tracks by row number (`event_idx`, `pos_idx`,
`neg_idx`). The page shows the V0 table and selects rows with four filters: a VALIDITY cut (every row number points to
an existing row), the event (a multi-select on `event_idx`), and the two daughters' transverse momenta, looked up in the
track table (`tracks.pt[pos_idx]`, `tracks.pt[neg_idx]`, range widgets). Each V0's event centrality is looked up
in the event table (`events.centrality[event_idx]`) and drawn against the mass. It shows histograms of the V0 mass (plain,
weighted by wA, by wB, and by a weight parameter switched between wA and wB), a histogram of an alias that reads a
looked-up value (`mass + 0.1 * tracks.pt[pos_idx]`), a 2-D histogram of mass and positive-daughter pt, and its profile
(entries, mean, std, isOK per mass bin).

THE REFERENCE is GPT24.RI's shared oracle and fixture v0.2 (A4 proposal v0.2, reviewed by GPT25, GPT26, GPT27, GPT28 and
GPT29), installed byte-identical: oracles/a4_shared_oracle_v0_2.py and oracles/a4_shared_fixture_v0_2.json (SHA256
checked below). Its comparator `assert_snapshot_matches` decides every comparison, so the reference is not the
coder's. Its six cases (events x minimum daughter pt) are set on the page through the real widgets and their own
callbacks; the page's numbers are read from the production models (HistogramCDS, HistoNdCDS, HistoNdProfile, the
filters, CDSAlias), never recomputed.

Page design (test-side choices, stated so a reviewer can check them):
  * the validity cut is declared on the page from plain columns, so the reference numbers do not depend on what a
    lookup returns for a missing row (index -1 or 99) — that is open Architect question A4-F3. The fixture's
    same-event rule decides no row of the 12 (checked below);
  * the looked-up track and event columns are shown in their own figures, so they are sent to the page (known defect
    A-D4: a lookup alone does not send its column; covered by its own tests in test_headless_node_aliases.py);
  * only the first filter (validity) excludes rows at load, so known defect B1 (later filters keep the rows they
    excluded at load) cannot change a selection here; checked below with B1's own selection rule, and the filter
    order is read from the page.
Known defect found again (no new id): B2 — at load every 1-D histogram counts all rows, the selection is ignored
(HistogramCDS.ts `initialize`); the 2-D histogram and the profile follow the selection from load on. Exact signature,
strict partner and control below.

Source: GPT24.RI's A4 v0.2 packages (Team N: oracle, fixture; Team B: real-event helper, here corrected for Bokeh
3.3.2: the figure is `.bk-Figure`, a RootInteractive plot draws from a DownsamplerCDS whose `.data` is empty, and the
page is settled through its probes after an action). Integrated and the real-page tests written by Claude20.RI.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import HELPER_JS
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import a4_shared_oracle_v0_2 as oracle
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles.filters import b1_selection, intersection

HERE = Path(__file__).resolve().parent
FIXTURE_PATH = HERE / "oracles" / "a4_shared_fixture_v0_2.json"
FIXTURE_SHA256 = "2c2fb0de9f2760b9494722ca7e67096f72617f6172ab10b2dc2b7e3a4dbbb3e8"   # GPT24.RI v0.2, reviewed
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]
T = FIXTURE["tables"]


class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


STRICT = dict(strict=True, raises=KnownDefect)


# ===================================================================================== the page
def _columns(rows, keys):
    return {k: np.asarray([r[k] for r in rows]) for k in keys}


V0 = pd.DataFrame(_columns(T["v0"], ["v0_id", "event_idx", "pos_idx", "neg_idx", "mass", "wA", "wB"]))
TRACKS = _columns(T["tracks"], ["track_id", "event_id", "pt"])
EVENTS = _columns(T["events"], ["event_id", "centrality"])
N, NT, NE = len(V0), len(T["tracks"]), len(T["events"])
PT_END = 3.0                                    # the daughter-pt widgets' upper limit; every pt is below it
VALID = (f"(pos_idx >= 0) & (pos_idx < {NT}) & (neg_idx >= 0) & (neg_idx < {NT}) & "
         f"(event_idx >= 0) & (event_idx < {NE})")
EDGES = FIXTURE["bins"]
HISTOGRAMS_1D = {"hm": "mass_counts", "hA": "mass_wA", "hB": "mass_wB", "ha": "alias_counts"}   # + hW (parameter)
FILTER_ORDER = ["valid", "event_idx.factor()", "pospt", "negpt"]     # the widget order: validity first (B1, above)


def _from_array(v0=None):
    """The page through the user path, bokehDrawSA.fromArray; the caller suppresses `show`."""
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    m, a = EDGES["mass"], EDGES["alias"]
    hist = dict(variables=["mass"], nbins=len(m) - 1, range=[m[0], m[-1]])
    figures = [
        [["v0_id"], ["mass"]],                                           # the V0 table itself
        [["evcent"], ["mass"]],                                          # V0 mass against its event's centrality
        [["bin_center"], ["bin_count"], {"source": "hm"}],
        [["bin_center"], ["bin_count"], {"source": "hA"}],
        [["bin_center"], ["bin_count"], {"source": "hB"}],
        [["bin_center"], ["bin_count"], {"source": "hW"}],
        [["bin_center"], ["bin_count"], {"source": "ha"}],
        [["bin_center_0"], ["mean"], {"source": "h2_1"}],
        [["tracks.pt"], ["tracks.event_id"]],                            # sends the looked-up columns (A-D4)
        [["events.event_id"], ["events.centrality"]],
    ]
    widgets = [
        ["range", ["valid", 0.5, 1.5]],
        ["multiSelect", ["event_idx"]],
        ["range", ["pospt", 0.0, PT_END]],
        ["range", ["negpt", 0.0, PT_END]],
        ["select", ["wsel"]],
    ]
    return sa.bokehDrawSA.fromArray(
        V0 if v0 is None else v0, None, figures, widgets,
        sourceArray=[
            {"name": "tracks", "data": TRACKS},
            {"name": "events", "data": EVENTS},
            {"name": "hm", **hist},
            {"name": "hA", **hist, "weights": "wA"},
            {"name": "hB", **hist, "weights": "wB"},
            {"name": "hW", **hist, "weights": "wsel"},
            {"name": "ha", "variables": ["mpt"], "nbins": len(a) - 1, "range": [a[0], a[-1]]},
            {"name": "h2", "variables": ["mass", "pospt"], "nbins": [len(m) - 1, len(EDGES["pospt"]) - 1],
             "range": [[m[0], m[-1]], [EDGES["pospt"][0], EDGES["pospt"][-1]]]},
        ],
        aliasArray=[
            ("pospt", "tracks.pt[pos_idx]"),
            ("negpt", "tracks.pt[neg_idx]"),
            ("mpt", "mass + 0.1 * tracks.pt[pos_idx]"),
            ("valid", VALID),
            ("evcent", "events.centrality[event_idx]"),
        ],
        parameterArray=[{"name": "wsel", "value": "wA", "options": ["wA", "wB"]}],
        layout=[[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]], widgetLayout=[list(range(len(widgets)))],
        arrayCompression=[(".*", [("code", 0), ("zip", 0), ("base64", 0)])],
    )


def build_document(tmp_path: Path, name="a4_document.json", v0=None) -> Path:
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        root = _from_array(v0).pAll
    finally:
        sa.show = shown
    doc = Document()
    doc.add_root(root)
    out = tmp_path / name
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


def build_html(tmp_path: Path) -> Path:
    """The same page as a standalone HTML file for Chromium (bokeh `save`, inline resources; no browser is opened)."""
    from bokeh.io import save
    from bokeh.resources import INLINE
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        root = _from_array().pAll
    finally:
        sa.show = shown
    out = tmp_path / "a4_page.html"
    save(root, filename=str(out), resources=INLINE, title="A4 V0-track-event")
    return out


# ===================================================================================== reading and acting on the page
PROBES = [[n, "bin_count", ["HistogramCDS"]] for n in ("hm", "hA", "hB", "hW", "ha")] + \
         [["h2", "bin_count", ["HistoNdCDS"]], ["h2_1", "mean", ["HistoNdProfile"]]]

A4_JS = """async (a) => {
  const H = window.__riH;
  const all = () => H.models();
  const msg = e => (e && typeof e === 'object' && 'message' in e) ? String(e.message) : String(e);
  const one = (pred, what) => { const m = all().filter(pred); if (m.length !== 1) throw new Error(what + ': ' + m.length); return m[0]; };
  const hist = n => one(m => m.name === n && (m.type === 'HistogramCDS' || m.type === 'HistoNdCDS'), 'histogram ' + n);
  const widget = t => one(m => m.title === t && 'value' in m && m.js_property_callbacks, 'widget titled ' + t);
  const N = a.n;
  const snap = () => {
    const names = ['hm', 'hA', 'hB', 'hW', 'ha', 'h2'];
    const masks = names.map(n => {
      const f = hist(n).filter;
      if (f == null) throw new Error('histogram ' + n + ' has no filter');
      const v = f.v_compute();
      if (v == null || v.length < N) throw new Error('filter vector shorter than the table: ' + (v == null ? null : v.length));
      return Array.from(v, x => x ? 1 : 0).slice(0, N);       // padded beyond the table: the first N entries (A3)
    });
    if (!masks.every(m => JSON.stringify(m) === JSON.stringify(masks[0])))
      throw new Error('the histograms disagree on the selected rows: ' + JSON.stringify(masks));
    const mask = masks[0];
    const ids = Array.from(one(m => m.type === 'CDSAlias' && m.name === 'cdsFull', 'cdsFull').get_column('v0_id'));
    const bins = n => Array.from(hist(n).get_column('bin_count'));
    const prof = one(m => m.type === 'HistoNdProfile' && m.name === 'h2_1', 'profile h2_1');
    const col = c => Array.from(prof.get_column(c));
    const lif = hist('h2').filter;
    return {selected_mask: mask, selected_row_ids: ids.filter((_, i) => mask[i]),
            mass_counts: bins('hm'), mass_wA: bins('hA'), mass_wB: bins('hB'), mass_w: bins('hW'),
            alias_counts: bins('ha'), mass_pospt_2d: bins('h2'),
            profile_entries: col('entries'), profile_mean: col('mean'), profile_std: col('std'),
            profile_isOK: col('isOK').map(Boolean),
            event_centrality: Array.from(one(m => m.type === 'CDSAlias' && m.name === 'cdsFull', 'cdsFull')
                                         .get_column('evcent')).filter((_, i) => mask[i]),
            weight: widget('wsel').value, filter_order: (lif.filters || []).map(f => f.field)};
  };
  const set = async (title, value, mute) => {
    const w = widget(title);
    const cbs = (w.js_property_callbacks || {})['change:value'] || [];
    const pending = [];
    for (const cb of cbs) {                       // wait for the widget's own callbacks (asynchronous in Bokeh 3.3.2)
      const orig = cb.execute;
      cb.execute = mute ? function () { const p = Promise.resolve(); pending.push(p); return p; }
                        : function (...x) { const p = orig.apply(this, x); pending.push(p); return p; };
    }
    try { w.value = value; } finally { for (const cb of cbs) delete cb.execute; }
    const settled = await Promise.allSettled(pending);
    const done = await H.settle(a.probes, 50);
    const errors = settled.filter(x => x.status === 'rejected').map(x => msg(x.reason));
    if (!done.settled || errors.length) throw new Error('widget ' + title + ': ' + JSON.stringify({done, errors}));
    return {set: title, callbacks: settled.length};
  };
  const out = [];
  for (const s of a.steps) out.push(s.op === 'snap' ? snap() : await set(s.widget, s.value, !!s.mute));
  return out;
}"""


def snap():
    return {"op": "snap"}


def set_(widget, value, mute=False):
    return {"op": "set", "widget": widget, "value": value, "mute": mute}


def case_steps(name, mute=()):
    """The widget values of a reference case: its events in the multi-select, its minimum pt on both daughters."""
    sel = CASES[name]["selection"]
    low = float(sel["min_pt"])
    return [set_("event_idx", [str(e) for e in sel["event_indices"]], "event_idx" in mute),
            set_("pospt", [low, PT_END], "pospt" in mute), set_("negpt", [low, PT_END], "negpt" in mute)]


def _node_page(node_session, tmp_path, name="a4_document.json", v0=None):
    report = ScenarioResult(id="a4/v0_track_event [node]", outdir=str(tmp_path))
    page = node_session.new_page(report)
    page.goto(build_document(tmp_path, name, v0))
    page.wait_for_function(page.load_condition)
    settled = page.evaluate("p => window.__riH.settle(p, 50)", PROBES)
    assert settled.get("settled"), settled
    return page, report


def run_steps(page, steps):
    out = page.evaluate(A4_JS, {"n": N, "probes": PROBES, "steps": steps})
    assert isinstance(out, list) and len(out) == len(steps), out
    return out


def _float(values):
    return [np.nan if v is None else float(v) for v in values]


def centralities(row_ids):
    """numpy: the centrality of each V0's event, looked up in the event table (an addition to GPT24's reference)."""
    return [float(EVENTS["centrality"][V0["event_idx"][i]]) for i in row_ids]


def assert_case(snapshot: dict, name: str, want=None):
    """GPT24's comparator on every field of the reference; the live weight histogram against the current weight;
    the looked-up event centrality of the selected rows against numpy."""
    want = CASES[name]["expected"] if want is None else want
    oracle.assert_snapshot_matches(snapshot, want)
    assert list(map(float, snapshot["event_centrality"])) == centralities(want["selected_row_ids"]), \
        ("event centrality looked up from the event table", snapshot["event_centrality"])
    weight = snapshot["weight"]
    assert weight in ("wA", "wB"), weight
    assert list(snapshot["mass_w"]) == list(want["mass_" + weight]), ("hW (weight parameter = %s)" % weight,
                                                                      snapshot["mass_w"], want["mass_" + weight])


def b2_prediction():
    """What a 1-D histogram shows at load if it ignores the selection: numpy over ALL 12 rows, nothing selected."""
    mass = V0["mass"].to_numpy(dtype=float)
    mpt = mass + 0.1 * np.asarray([TRACKS["pt"][p] for p in V0["pos_idx"]])   # pos_idx is valid in every row

    def every_row(values, edges, weights=None):
        w = None if weights is None else V0[weights].to_numpy(dtype=float)
        return np.histogram(values, bins=edges, weights=w)[0].tolist()
    return {"mass_counts": every_row(mass, EDGES["mass"]),
            "mass_wA": every_row(mass, EDGES["mass"], "wA"),
            "mass_wB": every_row(mass, EDGES["mass"], "wB"),
            "mass_w": every_row(mass, EDGES["mass"], "wA"),                # the weight parameter starts at wA
            "alias_counts": every_row(mpt, EDGES["alias"])}


# ===================================================================================== Python: the reference and the page design
def _unit(test):
    return pytest.mark.headless_unit(pytest.mark.backend("python")(pytest.mark.layer("unit")(test)))


# GPT24's own checks of the reference (unchanged; collected here under the fast Python target)
test_reference_hand_audited_rows_and_weights = _unit(oracle.test_known_hand_audited_selected_rows_and_weights)
test_reference_conservation_and_orphans = _unit(oracle.test_conservation_and_orphans)
test_reference_row_permutation = _unit(oracle.test_row_permutation_preserves_population_and_bins)
test_reference_shared_daughters_not_deduplicated = _unit(oracle.test_shared_daughters_not_deduplicated)
test_reference_detects_wrong_weight_or_relation = _unit(oracle.test_falsification_wrong_weight_or_relation_is_detected)
test_reference_nonempty_high_pt_and_empty_slice = _unit(oracle.test_nonempty_high_pt_and_empty_profile_slice)
test_reference_gather_vs_selection_probe = _unit(oracle.test_cross_event_and_missing_daughter_gather_vs_selection)
test_reference_comparator_rejects_profile_mutations = _unit(oracle.test_same_production_comparator_rejects_profile_mutations)
test_reference_fixture_round_trip = _unit(oracle.test_fixture_json_has_complete_case_contract)


@_unit
def test_installed_fixture_is_the_reviewed_one_and_regenerates_from_the_reference():
    raw = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    assert FIXTURE == json.loads(json.dumps(oracle.build_fixture())), "the fixture is not what the oracle builds"
    assert sorted(CASES) == sorted(oracle.CASES)


def _filter_masks(name, fill=np.nan):
    """numpy masks of the page's four filters for a reference case, in the page's filter order. `fill` is the value a
    lookup of a missing row would give (open question A4-F3); the page today gives no number (see the CRR)."""
    sel = CASES[name]["selection"]
    pt = np.asarray(TRACKS["pt"], dtype=float)
    look = lambda idx: np.asarray([pt[i] if 0 <= i < NT else fill for i in idx])   # noqa: E731
    valid = V0.eval(VALID).to_numpy(dtype=bool)
    events = V0["event_idx"].isin(sel["event_indices"]).to_numpy()
    low = float(sel["min_pt"])
    with np.errstate(invalid="ignore"):
        pos = (look(V0["pos_idx"]) >= low) & (look(V0["pos_idx"]) <= PT_END)
        neg = (look(V0["neg_idx"]) >= low) & (look(V0["neg_idx"]) <= PT_END)
    return [valid, events, pos, neg]


@_unit
def test_page_design_the_filters_select_exactly_the_reference_rows():
    """The page's four filters, as numpy, select each case's reference rows; the validity cut equals the fixture's
    policy row by row, and the fixture's same-event rule decides none of the 12 rows."""
    valid = V0.eval(VALID).to_numpy(dtype=bool)
    assert valid.tolist() == [oracle.valid_row(r) for r in T["v0"]]
    in_range = [0 <= r["event_idx"] < NE and 0 <= r["pos_idx"] < NT and 0 <= r["neg_idx"] < NT for r in T["v0"]]
    assert in_range == [oracle.valid_row(r) for r in T["v0"]], "the same-event rule would decide a row"
    for name in CASES:
        got = intersection(_filter_masks(name), N).astype(int).tolist()
        assert got == CASES[name]["expected"]["selected_mask"], name
    # the validity cut makes the result independent of what a lookup of a missing row gives: with a fill of 0 the
    # reference rows are still selected; control: WITHOUT the validity cut a fill of 0 selects the orphans 9 and 11
    for name in CASES:
        assert intersection(_filter_masks(name, fill=0.0), N).astype(int).tolist() == \
            CASES[name]["expected"]["selected_mask"], name
    without = intersection(_filter_masks("all_valid", fill=0.0)[1:], N)
    assert without[[9, 11]].all() and not without[10], without


@_unit
def test_page_design_known_defect_b1_cannot_change_a_selection_here():
    """B1's own selection rule (oracles/filters.py) equals the plain intersection for every case: only the first
    filter (validity) excludes rows at load. Control: with the validity cut LAST, B1 would change a selection."""
    valid, _, pos, neg = _filter_masks("all_valid")    # the range widgets start at [0, 3], as in "all_valid"
    at_load = [valid, np.ones(N, bool), pos, neg]      # the multi-select starts with every option, -1 included
    for name in CASES:
        now = _filter_masks(name)
        assert np.array_equal(b1_selection(now, at_load), intersection(now, N)), name
    # control: validity last
    reorder = lambda ms: ms[1:] + ms[:1]               # noqa: E731
    changed = [n for n in CASES if not np.array_equal(b1_selection(reorder(_filter_masks(n)), reorder(at_load)),
                                                      intersection(_filter_masks(n), N))]
    assert changed, "the control should show that the filter order matters for B1"


@_unit
def test_b2_prediction_differs_from_the_reference_at_load():
    """B2's prediction (every row counted) is distinguishable from the correct load state in every 1-D histogram."""
    want, pred = CASES["all_valid"]["expected"], b2_prediction()
    for key in ("mass_counts", "mass_wA", "mass_wB", "alias_counts"):
        assert pred[key] != want[key], key


# ===================================================================================== Node: the real page
SEQUENCE = ["events_0_2_4", "events_0_2_4_pt_ge_1", "events_2_pt_ge_1_5", "all_pt_ge_2", "empty_events",
            "all_valid", "events_0_2_4_pt_ge_1"]
PERMUTATION = [8, 5, 2, 11, 0, 4, 9, 7, 10, 1, 3, 6]      # GPT24's row permutation (its oracle self-test)


@pytest.mark.node_a4
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("JOIN.multi_table_chain")
@pytest.mark.feature("DSL.gather_operation")
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.parametrize("order", ["fixture", "permuted"])
def test_node_every_state_after_load_equals_the_reference(node_session, tmp_path, order):
    """Full -> event cut -> event and daughter cut -> nonempty high-pt state with an empty profile slice -> both
    daughters >= 2 -> empty -> restored to full -> the same cut again (no stale cache). Every field of every state.
    `permuted`: the V0 rows in GPT24's permuted order; the reference is GPT24's oracle on the same permuted rows
    (the selection by position follows the rows; the histograms and the profile do not change)."""
    rows = T["v0"] if order == "fixture" else [T["v0"][i] for i in PERMUTATION]
    v0 = None if order == "fixture" else V0.iloc[PERMUTATION].reset_index(drop=True)
    page, report = _node_page(node_session, tmp_path, v0=v0)
    first = run_steps(page, [snap()])[0]
    assert first["filter_order"] == FILTER_ORDER, first["filter_order"]
    for name in SEQUENCE:
        want = json.loads(json.dumps(oracle.reference(CASES[name]["selection"], v0=rows)))
        if order == "fixture":
            assert want == CASES[name]["expected"], name           # the installed fixture is the oracle's output
        out = run_steps(page, case_steps(name) + [snap()])
        assert_case(out[-1], name, want)
    assert not report.page_errors and not report.console_errors, (report.page_errors, report.console_errors)


@pytest.mark.node_a4
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("JOIN.multi_table_chain")
@pytest.mark.feature("HIST.histogram_1d")
def test_node_live_weight_switch_follows_the_parameter(node_session, tmp_path):
    """The weight parameter switched by its real Select widget, wA -> wB -> wA, with selection changes in between:
    the parameter histogram always equals the reference histogram of the current weight; the fixed ones never move."""
    page, report = _node_page(node_session, tmp_path)
    plan = [("events_0_2_4_pt_ge_1", None), ("events_0_2_4_pt_ge_1", "wB"), ("events_2_pt_ge_1_5", None),
            ("all_valid", "wA"), ("events_0_2_4", "wB"), ("empty_events", "wA")]
    current = "wA"
    for name, weight in plan:
        steps = case_steps(name) + ([set_("wsel", weight)] if weight else [])
        snapshot = run_steps(page, steps + [snap()])[-1]
        current = weight or current
        assert snapshot["weight"] == current, (snapshot["weight"], current)
        assert_case(snapshot, name)
    assert not report.page_errors and not report.console_errors, (report.page_errors, report.console_errors)


def _load_state(node_session, tmp_path):
    page, report = _node_page(node_session, tmp_path)
    snapshot = run_steps(page, [snap()])[0]
    assert not report.page_errors and not report.console_errors, (report.page_errors, report.console_errors)
    return snapshot


def _is_b2(snapshot) -> bool:
    """B2's exact signature at load: every 1-D histogram equals ALL rows; the selection, the 2-D histogram and the
    profile equal the reference."""
    want, pred = CASES["all_valid"]["expected"], b2_prediction()
    one_d = all(list(snapshot[k]) == pred[k] for k in pred)
    rest = ["selected_mask", "selected_row_ids", "mass_pospt_2d", "profile_entries", "profile_mean", "profile_std",
            "profile_isOK"]
    try:
        oracle.assert_snapshot_matches(snapshot, want, keys=rest)
    except AssertionError:
        return False
    return one_d


@pytest.mark.node_a4
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("JOIN.multi_table_chain")
@pytest.mark.feature("HIST.histogram_1d")
def test_signature_b2_at_load_the_1d_histograms_count_every_row(node_session, tmp_path):
    snapshot = _load_state(node_session, tmp_path)
    assert _is_b2(snapshot), {k: snapshot[k] for k in ("selected_mask", "mass_counts", "mass_wA", "alias_counts")}


@pytest.mark.node_a4
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("JOIN.multi_table_chain")
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.xfail(**STRICT, reason="B2: a 1-D histogram ignores the selection at page load (HistogramCDS.ts initialize)")
def test_strict_the_load_state_equals_the_reference(node_session, tmp_path):
    snapshot = _load_state(node_session, tmp_path)
    known_defect(_is_b2(snapshot), "B2: at load the 1-D histograms count every row")
    assert_case(snapshot, "all_valid")


@pytest.mark.node_a4
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("JOIN.multi_table_chain")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.profile")
def test_control_b2_the_2d_histogram_and_the_profile_follow_the_selection_from_load(node_session, tmp_path):
    """B2's control: at load the selection, the 2-D histogram and the profile equal the reference (B2 is 1-D only)."""
    snapshot = _load_state(node_session, tmp_path)
    oracle.assert_snapshot_matches(snapshot, CASES["all_valid"]["expected"],
                                   keys=["selected_mask", "selected_row_ids", "mass_pospt_2d", "profile_entries",
                                         "profile_mean", "profile_std", "profile_isOK"])


@pytest.mark.node_a4
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
@pytest.mark.feature("JOIN.multi_table_chain")
@pytest.mark.parametrize("muted", ["pospt", "event_idx", "wsel"])
def test_control_a_muted_widget_is_caught_by_the_comparator(node_session, tmp_path, muted):
    """The checks can fail through the real page: the widget is moved but its own callbacks are muted (this page
    only); the reference comparator must then reject the state at the field that widget drives."""
    page, _ = _node_page(node_session, tmp_path)
    start = "events_0_2_4" if muted == "wsel" else "all_valid"
    run_steps(page, case_steps(start))
    if muted == "wsel":
        snapshot = run_steps(page, [set_("wsel", "wB", mute=True), snap()])[-1]
        assert snapshot["weight"] == "wB"                             # the widget moved ...
        assert snapshot["mass_w"] == CASES["events_0_2_4"]["expected"]["mass_wA"]     # ... the histogram did not
        with pytest.raises(AssertionError, match="hW"):
            assert_case(snapshot, "events_0_2_4")
    else:
        snapshot = run_steps(page, case_steps("events_0_2_4_pt_ge_1", mute=(muted,)) + [snap()])[-1]
        with pytest.raises(AssertionError):
            oracle.assert_snapshot_matches(snapshot, CASES["events_0_2_4_pt_ge_1"]["expected"], keys=["selected_mask"])
        oracle.assert_snapshot_matches(snapshot, {"selected_mask": intersection(
            [m for m, f in zip(_filter_masks("events_0_2_4_pt_ge_1"), FILTER_ORDER) if not f.startswith(muted)] +
            [m for m, f in zip(_filter_masks(start), FILTER_ORDER) if f.startswith(muted)], N
        ).astype(int).tolist()}, keys=["selected_mask"])                # exactly the muted widget's old selection


# ===================================================================================== Chromium: real input on the same page
def _open_browser_page(headless_session, html: Path):
    """The pinned local copies of the page's external scripts, every other request blocked (engine/runner.py
    `_browser_run`; as in test_headless_widgets.py)."""
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
    page.add_init_script(HELPER_JS)
    page.goto(html.resolve().as_uri())
    page.wait_for_function("() => window.Bokeh && Bokeh.documents.length > 0 && window.__riH.models().length > 0",
                           timeout=60_000)
    done = page.evaluate("p => window.__riH.settle(p, 200)", PROBES)
    assert done.get("settled"), done
    return ctx, page, errors


RANGE_JS = """(title) => { const s = window.__riH.slider(title); return s ? Array.from(s.value) : null; }"""


def real_range_keys(page, title, handle, key, presses):
    """Team B's real-event helper (GPT24.RI v0.2), corrected for Bokeh 3.3.2: real key presses on the noUiSlider
    handle of the uniquely titled RangeSlider; the model is only read; after the presses the page is settled through
    its probes, so the asynchronous callbacks have finished."""
    widgets = page.locator(".bk-RangeSlider").filter(has_text=title)
    assert widgets.count() == 1, f"expected one RangeSlider titled {title!r}: {widgets.count()}"
    handles = widgets.locator('.noUi-handle[role="slider"]')
    assert handles.count() == 2, f"expected two keyboard handles on {title!r}"
    target = handles.nth(handle)
    target.focus()
    for _ in range(presses):
        before = page.evaluate(RANGE_JS, title)
        target.press(key)
        page.wait_for_function("([t, b]) => JSON.stringify(Array.from(window.__riH.slider(t).value)) !== JSON.stringify(b)",
                               arg=[title, before], timeout=5_000)
    done = page.evaluate("p => window.__riH.settle(p, 200)", PROBES)
    assert done.get("settled"), done
    return page.evaluate(RANGE_JS, title)


def real_select(page, multiple: bool, values):
    """A real selection in the page's only <select multiple> (event multi-select) or only single <select> (weight)."""
    sel = page.locator("select[multiple]" if multiple else "select:not([multiple])")
    assert sel.count() == 1, sel.count()
    sel.select_option(values)
    done = page.evaluate("p => window.__riH.settle(p, 200)", PROBES)
    assert done.get("settled"), done


DRAWN_JS = """() => {
  const rs = window.__riH.models().filter(m => m.type === 'GlyphRenderer');
  const isV0 = r => r.data_source.type === 'DownsamplerCDS' && r.data_source.source && r.data_source.source.name === 'cdsFull';
  const v0 = rs.filter(isV0);
  // the views that paint the V0 table: walk the views from Bokeh's root views (Bokeh 3.3.2 internals)
  const idx = window.Bokeh.index, seen = new Set(), views = new Map();
  const visit = v => {
    if (!v || typeof v !== 'object' || seen.has(v)) return; seen.add(v);
    if (v0.includes(v.model)) views.set(v.model, [...(views.get(v.model) || []), v]);
    for (const k of ['child_views', 'renderer_views']) {
      const c = v[k]; if (!c) continue;
      for (const x of (c instanceof Map ? c.values() : Object.values(c))) visit(x);
    }
  };
  for (const r of (idx instanceof Map ? idx.values() : Object.values(idx))) visit(r);
  return {renderers: rs.length, with_rows: rs.filter(r => r.data_source.get_length() > 0).length,
          v0: v0.map(r => {
            const ds = r.data_source, n = ds.get_length(), vs = views.get(r) || [];
            const x = r.glyph.x.field;
            return {x, views: vs.length, source_rows: n, source_x: Array.from(ds.get_column(x)).slice(0, n),
                    painted_rows: vs.length === 1 ? vs[0].glyph.data_size : null,
                    painted_x: vs.length === 1 ? Array.from(vs[0].glyph._x) : null};
          })};
}"""


def assert_drawn(page, row_ids):
    """Team B's rendering check, corrected and made exact: a visible figure canvas (`.bk-Figure` in Bokeh 3.3.2),
    renderers with rows (counted with get_length(): a RootInteractive plot draws from a DownsamplerCDS whose `.data` is
    empty), and the two V0 scatters (x = v0_id, x = looked-up event centrality) paint exactly the selected V0 rows,
    read from their glyph views. The downsampler's column can be longer than its length after the selection shrinks
    (old entries kept at the end; Bokeh paints get_length() rows), so only its first get_length() entries are read."""
    canvases = page.locator(".bk-Figure canvas")
    assert canvases.count() > 0 and canvases.first.is_visible(), "no visible figure canvas"
    info = page.evaluate(DRAWN_JS)
    assert info["renderers"] > 0 and info["with_rows"] > 0, info
    want = {"v0_id": sorted(map(float, row_ids)), "evcent": sorted(centralities(row_ids))}
    assert sorted(r["x"] for r in info["v0"]) == sorted(want), ("V0 scatters", info["v0"])
    for r in info["v0"]:
        assert r["views"] == 1, ("views painting the V0 scatter", r)
        assert sorted(map(float, r["source_x"])) == want[r["x"]], ("V0 scatter source", r, want[r["x"]])
        assert r["painted_rows"] == len(row_ids) and sorted(map(float, r["painted_x"])) == want[r["x"]], \
            ("painted V0 points", r, want[r["x"]])


@pytest.mark.browser_a4
@pytest.mark.backend("browser")
@pytest.mark.layer("integration")
@pytest.mark.feature("JOIN.multi_table_chain")
def test_browser_real_input_equals_the_reference_and_node(headless_session, node_session, tmp_path):
    """Chromium, real input only: select events 0, 2, 4 in the multi-select; raise both daughter-pt lower limits with
    the keyboard to 1.0; switch the weight to wB; restore everything. After each action every field equals the
    reference, the V0 scatter draws exactly the selected rows, and the same widget values in Node give the same
    numbers (browser = Node)."""
    html = build_html(tmp_path)
    ctx, page, errors = _open_browser_page(headless_session, html)
    try:
        def state():
            return page.evaluate(A4_JS, {"n": N, "probes": PROBES, "steps": [snap()]})[0]
        node, _ = _node_page(node_session, tmp_path, "a4_document_parity.json")

        def parity(snapshot, steps):
            twin = run_steps(node, steps + [snap()])[-1]
            for k in snapshot:
                assert np.array_equal(np.asarray(_float(snapshot[k]) if "profile_m" in k or "profile_s" in k else snapshot[k]),
                                      np.asarray(_float(twin[k]) if "profile_m" in k or "profile_s" in k else twin[k]),
                                      equal_nan=("profile_m" in k or "profile_s" in k)), (k, snapshot[k], twin[k])

        real_select(page, True, ["0", "2", "4"])
        s = state()
        assert_case(s, "events_0_2_4")
        assert_drawn(page, CASES["events_0_2_4"]["expected"]["selected_row_ids"])
        with pytest.raises(AssertionError, match="V0 scatter source"):           # control: the check can fail
            assert_drawn(page, CASES["all_valid"]["expected"]["selected_row_ids"])
        parity(s, [set_("event_idx", ["0", "2", "4"])])

        low_pos = real_range_keys(page, "pospt", 0, "ArrowRight", 10)
        low_neg = real_range_keys(page, "negpt", 0, "ArrowRight", 10)
        for low in (low_pos[0], low_neg[0]):
            assert 0.9 < low < 1.1, low              # no daughter pt lies in (0.9, 1.1): the case's min_pt = 1.0
        s = state()
        assert_case(s, "events_0_2_4_pt_ge_1")
        assert_drawn(page, CASES["events_0_2_4_pt_ge_1"]["expected"]["selected_row_ids"])
        parity(s, [set_("pospt", low_pos), set_("negpt", low_neg)])

        real_select(page, False, "wB")
        s = state()
        assert s["weight"] == "wB"
        assert_case(s, "events_0_2_4_pt_ge_1")
        parity(s, [set_("wsel", "wB")])

        real_select(page, True, ["0", "1", "2", "3", "4"])
        back_pos = real_range_keys(page, "pospt", 0, "ArrowLeft", 10)
        back_neg = real_range_keys(page, "negpt", 0, "ArrowLeft", 10)
        assert back_pos[0] == 0.0 and back_neg[0] == 0.0, (back_pos, back_neg)   # the widget's start clamps
        s = state()
        assert_case(s, "all_valid")
        assert_drawn(page, CASES["all_valid"]["expected"]["selected_row_ids"])
        parity(s, [set_("event_idx", ["0", "1", "2", "3", "4"]), set_("pospt", back_pos), set_("negpt", back_neg)])
        assert not errors, errors
    finally:
        ctx.close()
