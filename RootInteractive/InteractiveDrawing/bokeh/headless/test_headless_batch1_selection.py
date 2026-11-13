"""PHASE_5 Batch 1 — selections on ADF-shaped pages, in Node. Tests only (Batch 1 CRR v0.1, Claude20.RI, 2026-11-13).

Plain words. The ADF/dfdraw compiler (PHASE_3, on hold) will turn `adf.draw(...)` calls into RootInteractive pages whose
widgets select rows. Batch 1 of the PHASE_5 bug fixes repairs that selection path. These tests come before the fix
(gate G5a of the audit errata v0.3a): they state what such a page must show, compare it with numpy and pandas, and show
every Batch 1 defect RED on unfixed master with its exact signature. No production code is changed.

FIX COMMIT (Batch 1 fix CRR v0.1, Claude20.RI, 2026-11-13). The fix removes B1, B2, WT-D6, DS-D1, WG-D2, WG-D1 and
H01-D1. The strict marks are off and the signature tests are removed: every page must equal the oracle. The defect
model stays, as the record of master f1cbe0f: a page that equals the model of any set of the old defects is now a
REGRESSION, an ordinary failure that names the defect that came back.

Defects (register PHASE_5_0_Defect_Register_v0_1, master f1cbe0f):
  B1     with several range filters on a table, a later filter keeps the rows it excluded at load
         (LazyIntersectionFilter.ts 191, 193, 208, 210: `||=` skips from_bits_8 for the later filters);
  B2     a 1-D histogram ignores the selection at load (HistogramCDS.ts 44-50: no change_selection at initialize);
  WT-D6  a range widget at its full automatic range still cuts rows of a compressed column: the slider's ends are the
         raw minimum and maximum (bokehTools.py makeSliderParameters), the filter compares the decoded, rounded values
         (RangeFilter.ts 81, 115). Here an independent test; in PHASE_4 it was only part of the shared WR1 test;
  DS-D1  after the selection shrinks, the scatter plot's downsampler keeps old entries at the end of its columns;
  WG-D2  a toggleable range, once enabled, keeps the selection of the range it had when it was enabled: later moves
         change its filter's range but not the selected rows (measured here; PHASE_4 saw "selects every row" because
         its range was enabled at its full limits).
  WG-D1 (a toggleable range alone on its table breaks the page) and H01-D1 (a slider moved back to its end with the
  arrow keys stops 2e-16 short) keep their PHASE_4 tests (test_headless_widgets.py, test_headless_oracle_dashboards.py).
  The fix of WT-D6 also fixes H01-D1: a widget value within 1e-9 of its range of an end is that end.

Selection states (audit errata v0.3a, section 3), each checked against numpy row masks:
  S1  default at load: every widget at its full automatic range keeps all exported rows (the contract's D12 proposal);
  S2  a widget narrower at load: its limits are a static cut (PHASE_3_0b rule W3);
  S3  later narrowing by widget moves;
  S4  restore: the widgets back at their limits.

THE PAGES are built by RootInteractive's own builder (bokehDrawSA.fromArray), as a compiled page would be: one table of
600 rows (columns row, A, B, Y), a 1-D histogram of A (HistogramCDS), a 2-D histogram of A and Y with its unbinned
profile of Y per A bin (HistoNdCDS, HistoNdProfile), and a scatter plot (DownsamplerCDS). Two codings: lossless, and
relative 16 bits on the float columns (the `row` column stays lossless, as the relation exporter keeps indices).

THE ORACLE is numpy over the values the page receives: the decoded values, rebuilt by dashboards/oracle_weightsref.round16
(equal to RootInteractive's own rounding, test_headless_oracle_weightsref.py, DEC-11). Bins: oracles/histograms.py
(rulebook RB-2, reviewed); filters: oracles/filters.py (RB-1). The intended rule: a widget at its full automatic range
keeps every exported row; otherwise a row is kept when lo <= value <= hi. Every cut used here is checked to be
unambiguous: no row lies on different sides of it with its raw and its decoded value.

THE DEFECT MODEL predicts the page with a chosen set of Batch 1 defects present (B1 by oracles/filters.b1_selection,
the measured rule of PHASE_4). A strict test passes when the page equals the oracle at every step. It raises
KnownDefect only when the page equals the model with ALL Batch 1 defects at every step and output, and the defects the
model blames (those whose removal changes the prediction) are exactly the declared ones. Anything else is an ordinary
failure (fail-closed, gate G1). After the Batch 1 fix every strict test must pass; an XPASS fails the run.
(Fix commit: the marks are off; see above.)

THE T1 ACCEPTANCE PAGE is PHASE_3_0b's hand-compiled T1 call
    refs.draw("p/MC.p:r", type="profile", selection="$w(isMonopole)$ & $w(tgl)$", group_by="MC.p",
              group_by_bins="$w(nbZ)$", bins=20, range=(0, 50))
(Claude10's worked example, PHASE_3_1 v0.3 review) on a generated table with the Kine example's columns: 12 000
references, 400 MC tracks plus a NaN fill row for the 2 % orphans. Its two cuts are narrower at load (S2), so B1 makes
it wrong today, as it made the original Kine page wrong (45 777 entries instead of 13 888). Compared with pandas.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.oracle_weightsref import round16
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import histograms as ho
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles.filters import b1_selection, range_mask


def regression(condition: bool, message: str) -> None:
    """Batch 1 fix: a page with the exact signature of a fixed defect is a regression - an ordinary failure."""
    if condition:
        raise AssertionError("REGRESSION, a Batch 1 defect is back: " + message)


ALL_DEFECTS = ("B1", "B2", "WT-D6", "DS-D1", "WG-D2")


def _node(test):
    return pytest.mark.node_widgets(pytest.mark.backend("node")(pytest.mark.layer("integration")(test)))


def _unit(test):
    return pytest.mark.headless_unit(pytest.mark.backend("python")(pytest.mark.layer("unit")(test)))


# ===================================================================================== the table
SEED = 20261113
N = 600                         # not a multiple of 32: B1's chain starts at the first filter (filters.b1_first_changed)
A_LOW_ROW, A_HIGH_ROW = 17, 311
LOSSLESS = [("code", 0), ("zip", 0), ("base64", 0)]
CODINGS = {"lossless": [(".*", LOSSLESS)],
           "relative16": [("^(A|B|Y)$", [("relative", 16), "code", "zip"]), (".*", LOSSLESS)]}


def table() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    a = rng.uniform(0.5, 9.5, N)
    b = rng.uniform(-4.5, 4.5, N)
    y = np.clip(rng.normal(0.0, 1.0, N), -3.9, 3.9)
    a[A_LOW_ROW] = 0.25 * (1.0 + 0.3 / 32768)     # relative 16 bits rounds it DOWN to 0.25, below the raw minimum
    a[A_HIGH_ROW] = 9.9999                        # ... rounds it UP to 10.0, above the raw maximum
    return pd.DataFrame({"row": np.arange(N, dtype=np.int64), "A": a, "B": b, "Y": y})


def decoded(coding: str) -> dict:
    """The values the page receives, per column."""
    t = table()
    if coding == "lossless":
        return {c: t[c].to_numpy(dtype=np.float64) for c in ("A", "B", "Y")}
    return {c: round16(t[c].to_numpy()) for c in ("A", "B", "Y")}


H1 = {"name": "h1", "variables": ["A"], "nbins": 20, "range": [0.0, 10.0]}
H2 = {"name": "h2", "variables": ["A", "Y"], "nbins": [10, 8], "range": [[0.0, 10.0], [-4.0, 4.0]],
      "unbinned_projections": True}

# The widgets of each page. "auto": limits from the data (state S1); "narrow": limits are static cuts (state S2).
PAGES = {
    "auto": [["range", ["A"]], ["range", ["B"]]],
    "narrow": [["range", ["A", 2.0, 8.0, 0.1]], ["range", ["B", -3.0, 3.0, 0.1]]],
    "toggle": [["range", ["A"]], ["range", ["B", -3.0, 3.0, 0.1], {"toggleable": True}]],
    "one_auto": [["range", ["A"]]],
    "one_narrow": [["range", ["A", 2.0, 8.0, 0.1]]],
}
AUTO = {"auto": {"A", "B"}, "narrow": set(), "toggle": {"A"}, "one_auto": {"A"}, "one_narrow": set()}
TOGGLEABLE = {"toggle": {"B"}}
FULL = "full"                   # a move back to the widget's own limits (state S4)
STEPS = {
    "auto": [("load",), ("set", "A", [2.0, 8.0]), ("set", "B", [-2.0, 3.0]), ("set", "A", FULL), ("set", "B", FULL)],
    "narrow": [("load",), ("set", "A", [3.0, 7.0]), ("set", "B", [-1.5, 2.5]), ("set", "A", FULL), ("set", "B", FULL)],
    "toggle": [("load",), ("enable", "B"), ("set", "B", [-2.0, 2.5]), ("set", "A", [2.0, 8.0]), ("set", "A", FULL)],
    "one_auto": [("load",), ("set", "A", [2.0, 8.0]), ("set", "A", FULL)],
    "one_narrow": [("load",), ("set", "A", [3.0, 7.0]), ("set", "A", FULL)],
}


def _widget_names(page_kind):
    return [w[1][0] for w in PAGES[page_kind]]


def limits(page_kind) -> dict:
    """Each widget's limits as RootInteractive's builder sets them: the raw minimum and maximum for an automatic range
    (makeSliderParameters, type 'auto'), the declared cut otherwise."""
    t = table()
    out = {}
    for w in PAGES[page_kind]:
        name = w[1][0]
        out[name] = (float(np.nanmin(t[name])), float(np.nanmax(t[name]))) if len(w[1]) == 1 else (w[1][1], w[1][2])
    return out


def widget_history(page_kind) -> list:
    """Per step: {widget: (lo, hi, active, range when enabled)}; a toggleable range starts disabled (its default)."""
    lim = limits(page_kind)
    state = {n: (lim[n][0], lim[n][1], n not in TOGGLEABLE.get(page_kind, set()), lim[n]) for n in _widget_names(page_kind)}
    out = []
    for step in STEPS[page_kind]:
        if step[0] == "set":
            lo, hi = lim[step[1]] if step[2] == FULL else step[2]
            state[step[1]] = (lo, hi, state[step[1]][2], state[step[1]][3])
        elif step[0] == "enable":
            lo, hi = state[step[1]][:2]
            state[step[1]] = (lo, hi, True, (lo, hi))
        out.append(dict(state))
    return out


# ===================================================================================== oracle and defect model
def _filter_masks(page_kind, coding, state, defects, raw) -> list:
    """The mask of each widget's filter at one step, in widget order: intended (no defect), or as RootInteractive
    computes it with the given defects."""
    dec = decoded(coding)
    lim = limits(page_kind)
    out = []
    for name in _widget_names(page_kind):
        lo, hi, active, enabled_at = state[name]
        decoded_mask = range_mask(dec[name], lo, hi)
        if lo != lim[name][0] or hi != lim[name][1] or name not in AUTO[page_kind]:
            raw_mask = range_mask(raw[name], lo, hi)
            if not np.array_equal(raw_mask, decoded_mask):            # the cut must be unambiguous (see the header)
                raise AssertionError(f"ambiguous cut {name} in [{lo}, {hi}]: rows {np.flatnonzero(raw_mask != decoded_mask)}")
        if not active:
            out.append(np.ones(N, dtype=bool))
        elif name in TOGGLEABLE.get(page_kind, set()) and "WG-D2" in defects:
            out.append(range_mask(dec[name], *enabled_at))             # WG-D2: the range it had when enabled
        elif name in AUTO[page_kind] and (lo, hi) == lim[name] and "WT-D6" not in defects:
            out.append(np.isfinite(dec[name]))                         # intended: the full range keeps every row
        else:
            out.append(decoded_mask)                                   # RangeFilter: lo <= decoded <= hi
    return out


def expected(page_kind, coding, defects=()) -> list:
    """Per step the page's outputs: intended with defects=(), else as predicted with those defects present."""
    defects = set(defects)
    t = table()
    raw = {c: t[c].to_numpy(dtype=np.float64) for c in ("A", "B", "Y")}
    dec = decoded(coding)
    history = widget_history(page_kind)
    load_masks = _filter_masks(page_kind, coding, history[0], defects, raw)
    shown = []                                                         # the scatter plot's columns (DS-D1)
    out = []
    for k, state in enumerate(history):
        now = _filter_masks(page_kind, coding, state, defects, raw)
        sel = b1_selection(now, load_masks) if "B1" in defects else np.logical_and.reduce(now)
        h1_rows = np.ones(N, dtype=bool) if ("B2" in defects and k == 0) else sel
        rows = np.flatnonzero(sel).tolist()
        if "DS-D1" in defects and len(shown) > len(rows):
            shown = rows + shown[len(rows):]                           # refills the first rows, keeps the old tail
        else:
            shown = rows
        out.append({"mask": sel.astype(int).tolist(),
                    "h1": ho.counts([dec["A"]], [H1["nbins"]], [H1["range"]], mask=h1_rows).tolist(),
                    "h2": ho.counts([dec["A"], dec["Y"]], H2["nbins"], H2["range"], mask=sel).tolist(),
                    **_profile(dec, sel),
                    "ds_length": len(rows), "ds_rows": list(shown)})
    return out


def _profile(dec, sel) -> dict:
    """The unbinned profile of Y per A bin over the selected rows inside both ranges (entries, mean, population std)."""
    idx = ho.bin_index_nd([dec["A"], dec["Y"]], H2["nbins"], H2["range"])
    keep = sel & (idx >= 0)
    a_bin = idx[keep] % H2["nbins"][0]
    y = dec["Y"][keep]
    entries, mean, std = [], [], []
    for b in range(H2["nbins"][0]):
        v = y[a_bin == b]
        entries.append(float(len(v)))
        mean.append(float(np.mean(v)) if len(v) else None)
        std.append(float(np.std(v)) if len(v) else None)
    return {"entries": entries, "mean": mean, "std": std}


def explanations(observed, page_kind, coding) -> list:
    """The smallest sets of Batch 1 defects whose model equals the page at every step and output. [[]] means the oracle
    itself; [] means no set of known defects explains the page."""
    for size in range(len(ALL_DEFECTS) + 1):
        found = [sorted(c) for c in itertools.combinations(ALL_DEFECTS, size)
                 if _same_run(observed, expected(page_kind, coding, c))]
        if found:
            return found
    return []


def _same(a: dict, b: dict) -> bool:
    return not _differences(a, b)


def _same_run(xs, ys) -> bool:
    return len(xs) == len(ys) and all(_same(x, y) for x, y in zip(xs, ys))


def _differences(got: dict, want: dict) -> list:
    """The outputs in which a page state differs from an expected one; means and std within 1e-12 (relative), every
    count, mask and row exactly."""
    out = []
    for key in ("mask", "h1", "h2", "entries", "ds_length", "ds_rows"):
        if got[key] != want[key]:
            out.append(key)
    for key in ("mean", "std"):
        for g, w, n in zip(got[key], want[key], want["entries"]):
            if n == 0:
                continue
            if g is None or w is None or not np.isclose(g, w, rtol=1e-12, atol=1e-12):
                out.append(key)
                break
    return out


def classify(observed, page_kind, coding) -> tuple:
    """PASS: the page equals the oracle at every step and output. KNOWN: it equals the model of a set of Batch 1
    defects at every step and output; the detail is the smallest such sets. UNEXPLAINED: anything else; the detail is
    the first step that differs from the oracle and from the full model."""
    intended = expected(page_kind, coding)
    if _same_run(observed, intended):
        return "PASS", None
    sets = explanations(observed, page_kind, coding)
    if sets:
        return "KNOWN", sets
    model = expected(page_kind, coding, ALL_DEFECTS)
    for k, (o, i, m) in enumerate(zip(observed, intended, model)):
        if not (_same(o, i) or _same(o, m)):
            return "UNEXPLAINED", {"step": k, "action": STEPS[page_kind][k], "selected_rows": sum(o["mask"]),
                                   "differs_from_oracle": _differences(o, i), "differs_from_full_model": _differences(o, m)}
    return "UNEXPLAINED", {"lengths": [len(observed), len(intended)]}


# The defects that explained each page on master f1cbe0f (the smallest set whose model equals the page). Kept after the
# fix as the record of master and as the model's own check (test_the_model_blames_the_declared_defects); the fixed
# page must PASS, and a page equal to any of these models is a regression.
EXPECTED_BLAME = {
    ("auto", "lossless"): ["DS-D1"],
    ("auto", "relative16"): ["B1", "B2", "DS-D1", "WT-D6"],
    ("narrow", "lossless"): ["B1", "B2", "DS-D1"],
    ("narrow", "relative16"): ["B1", "B2", "DS-D1"],
    ("toggle", "lossless"): ["DS-D1", "WG-D2"],
    ("toggle", "relative16"): ["B2", "DS-D1", "WG-D2", "WT-D6"],
}


# ===================================================================================== the page in Node
def build_document(tmp_path: Path, page_kind: str, coding: str, mute=None, selection_table=False) -> Path:
    """The page through bokehDrawSA.fromArray (show suppressed) as a Bokeh document. `mute`: a widget whose link to
    its filter is removed - only for the live falsifier. `selection_table`: also the selection table of the dashboard
    template (bokehTools selectionTable)."""
    from bokeh.document import Document
    from bokeh.models import RangeSlider
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    widgets = PAGES[page_kind]
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        root = sa.bokehDrawSA.fromArray(
            table(), None,
            [[["bin_center"], ["bin_count"], {"source": "h1"}],
             [["bin_center_0"], ["mean"], {"source": "h2_1"}],
             [["A"], ["Y"]],
             [["row"], ["A"]]]                                         # sends `row`: the scatter's row identity
            + ([["selectionTable", {"name": "selection"}]] if selection_table else []),
            widgets, sourceArray=[dict(H1), dict(H2)], aliasArray=[], parameterArray=[],
            layout=[[0, 1, 2, 3] + ([4] if selection_table else [])], widgetLayout=[list(range(len(widgets)))],
            arrayCompression=CODINGS[coding]).pAll
    finally:
        sa.show = shown
    doc = Document()
    doc.add_root(root)
    if mute is not None:
        sliders = [s for s in doc.select({"type": RangeSlider}) if s.title == mute]
        assert len(sliders) == 1 and any(sliders[0].js_property_callbacks.values()), "no link to break: test invalid"
        sliders[0].js_property_callbacks = {}
    out = tmp_path / f"batch1_{page_kind}_{coding}{'_muted' if mute else ''}{'_table' if selection_table else ''}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


PROBES = [["h1", "bin_count", ["HistogramCDS"]], ["h2", "bin_count", ["HistoNdCDS"]],
          ["h2_1", "mean", ["HistoNdProfile"]]]

READ_JS = """(n) => {
  const H = window.__riH;
  const one = (name, types) => { const m = H.model(name, types); if (!m) throw new Error(H.why(name, types)); return m; };
  const h1 = one('h1', ['HistogramCDS']), h2 = one('h2', ['HistoNdCDS']), prof = one('h2_1', ['HistoNdProfile']);
  const ds = one('default cds', ['DownsamplerCDS']);
  if (h1.filter == null || h1.filter !== h2.filter) throw new Error('the histograms do not share one table filter');
  const v = h2.filter.v_compute();
  if (v == null || v.length < n) throw new Error('filter vector shorter than the table: ' + (v == null ? null : v.length));
  const num = c => Array.from(c, x => Number.isFinite(x) ? x : null);
  return {mask: Array.from(v, x => x ? 1 : 0).slice(0, n),
          h1: Array.from(h1.get_column('bin_count')), h2: Array.from(h2.get_column('bin_count')),
          entries: Array.from(prof.get_column('entries')), mean: num(prof.get_column('mean')),
          std: num(prof.get_column('std')),
          ds_length: ds.get_length(), ds_rows: Array.from(ds.get_column('row')),
          filters: (h2.filter.filters || []).map(f => ({field: f.field, active: Boolean(f.active)})),
          limits: Object.fromEntries(H.models().filter(m => m.type === 'RangeSlider').map(s => [s.title, [s.start, s.end]]))};
}"""

ACT_JS = """async ([op, name, value, probes]) => {
  const H = window.__riH;
  if (op === 'set') {
    const s = H.slider(name);
    if (!s) throw new Error('range widget ' + name + ' found ' + H.sliders(name).length + ' times');
    s.value = value === 'full' ? [s.start, s.end] : value;
  } else if (op === 'enable') {
    const t = H.models().filter(m => m.type === 'Toggle' && m.label === 'disable');
    if (t.length !== 1) throw new Error('toggle of ' + name + ' found ' + t.length + ' times');
    t[0].active = false;
  } else throw new Error('unknown action ' + op);
  return await H.settle(probes, 50);
}"""


def open_page(node_session, tmp_path, page_kind, coding, mute=None):
    report = ScenarioResult(id=f"batch1/{page_kind}/{coding} [node]", outdir=str(tmp_path))
    page = node_session.new_page(report)
    page.goto(build_document(tmp_path, page_kind, coding, mute))
    page.wait_for_function(page.load_condition)
    done = page.evaluate("p => window.__riH.settle(p, 50)", PROBES)
    assert done.get("settled"), done
    return page, report


def run_page(node_session, tmp_path, page_kind, coding, mute=None) -> list:
    """Every step of the page's script; the page's outputs after each. Checks on the way: the filters are in widget
    order, the limits are those of the builder, and the page logs no error."""
    page, report = open_page(node_session, tmp_path, page_kind, coding, mute)
    out = []
    for step in STEPS[page_kind]:
        if step[0] != "load":
            done = page.evaluate(ACT_JS, [step[0], step[1], step[2] if len(step) > 2 else None, PROBES])
            assert done.get("settled"), (step, done)
        state = page.evaluate(READ_JS, N)
        assert [f["field"] for f in state.pop("filters")] == _widget_names(page_kind), "filter order"
        lim = state.pop("limits")
        assert {k: tuple(v) for k, v in lim.items()} == limits(page_kind), ("widget limits", lim)
        out.append(state)
    assert not report.page_errors and not report.console_errors, (report.page_errors, report.console_errors)
    return out


# ===================================================================================== Python: oracle, model, falsifiers
@_unit
def test_the_table_has_the_rows_the_tests_need():
    """Preconditions: the two A rows rounded outside the raw range by relative 16 bits, nothing else outside; every
    page's cuts unambiguous (checked inside the oracle); S1 keeps all rows, S2 fewer."""
    t = table()
    a = t["A"].to_numpy()
    lost = np.flatnonzero((round16(a) < a.min()) | (round16(a) > a.max()))
    assert lost.tolist() == [A_LOW_ROW, A_HIGH_ROW], lost
    for kind in PAGES:
        for coding in CODINGS:
            runs = [r for r in expected(kind, coding)]                 # every cut is checked to be unambiguous here
            at_load = sum(runs[0]["mask"])
            assert at_load == N if AUTO[kind] == set(_widget_names(kind)) or kind == "toggle" else at_load < N, kind
            assert len({sum(r["mask"]) for r in runs}) > 1, ("the moves must change the selection", kind)


@_unit
@pytest.mark.parametrize("case", sorted(EXPECTED_BLAME), ids=["-".join(c) for c in sorted(EXPECTED_BLAME)])
def test_the_model_blames_the_declared_defects(case):
    """The declared set of each strict page test is the one smallest set that explains the full model's prediction,
    and the prediction differs from the oracle."""
    full = expected(*case, ALL_DEFECTS)
    assert explanations(full, *case) == [EXPECTED_BLAME[case]]
    assert not _same_run(full, expected(*case))


def _mutated(run, step, fn):
    out = [dict(s) for s in run]
    out[step] = fn(dict(out[step]))
    return out


def _flip_row(s, row):
    m = list(s["mask"]); m[row] = 1 - m[row]; s["mask"] = m
    return s


MUTANTS = {
    # one row moved across a cut edge at step 1 (the first selected row dropped)
    "row_across_a_cut": lambda run: _mutated(run, 1, lambda s: _flip_row(s, int(np.flatnonzero(s["mask"])[0]))),
    # the row with the largest A dropped at load
    "largest_row_dropped": lambda run: _mutated(run, 0, lambda s: _flip_row(s, A_HIGH_ROW)),
    # one stale scatter row kept
    "stale_scatter_row": lambda run: _mutated(run, 2, lambda s: {**s, "ds_rows": s["ds_rows"] + [s["ds_rows"][0]]}),
    # the profile's std reported as its mean
    "std_for_mean": lambda run: _mutated(run, 1, lambda s: {**s, "mean": s["std"]}),
    # one 1-D histogram count moved to the neighbouring bin
    "h1_count_moved": lambda run: _mutated(run, 3, lambda s: {**s, "h1": [s["h1"][0] - 1, s["h1"][1] + 1] + s["h1"][2:]}),
}


@_unit
@pytest.mark.parametrize("name", sorted(MUTANTS))
@pytest.mark.parametrize("case", [("narrow", "lossless"), ("auto", "relative16")], ids=["narrow-lossless", "auto-relative16"])
def test_the_comparator_rejects_every_mutant(name, case):
    """Gate G4: each deliberate fault, applied to the oracle's own run and to the full model's run, is neither PASS nor
    KNOWN - so a strict test can not turn it into green or into an expected failure."""
    intended, model = expected(*case), expected(*case, ALL_DEFECTS)
    for base in (intended, model):
        assert classify(MUTANTS[name](base), *case)[0] == "UNEXPLAINED", (name, case)
    assert classify(intended, *case) == ("PASS", None)
    assert classify(model, *case) == ("KNOWN", [EXPECTED_BLAME[case]])


# ----------------------------------------------------------------------- the capability matrix (A4 review F2 and F5)
def _generator():
    import importlib.util
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location("ri_matrix_generator_batch1", root / "scripts" / "generate_capability_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@_unit
def test_matrix_detail_shows_every_parametrized_case_and_the_worst_outcome():
    """A4 F5: the detail list of the matrix found no result for a parametrized test and showed it as unknown. It now
    uses the status's exact matching: the worst outcome of all cases, with their number."""
    g = _generator()
    outcomes = {"RootInteractive/x/test_a.py::test_p[1]": "passed", "RootInteractive/x/test_a.py::test_p[2]": "xfailed",
                "RootInteractive/x/test_a.py::test_one": "passed", "RootInteractive/x/test_a.py::test_p_other": "failed"}
    assert g.detail_outcome("RootInteractive/x/test_a.py::test_p", outcomes) == ("xfailed", 2)
    assert g.detail_outcome("RootInteractive/x/test_a.py::test_one", outcomes) == ("passed", 1)
    assert g.detail_outcome("RootInteractive/x/test_a.py::test_none", outcomes) == (None, 0)
    text, _ = g.generate_matrix({"F.x": {"name": "x", "priority": "P1", "backends": ["node"], "layer": "unit"}},
                                outcomes, {"F.x": ["RootInteractive/x/test_a.py::test_p"]})
    assert "⚠️ `RootInteractive/x/test_a.py::test_p` (2 cases)" in text, text[-800:]


@_unit
def test_filter_intersection_is_claimed_only_by_tests_that_face_b1():
    """A4 F2: the A4 page avoids B1 by its filter order, so it no longer claims FILTER.intersection; every test that
    does runs in Node, and the taxonomy says so."""
    from RootInteractive.InteractiveDrawing.bokeh.feature_taxonomy import FEATURE_TAXONOMY
    g = _generator()
    tests = g.parse_feature_markers_from_files("RootInteractive/InteractiveDrawing/bokeh/headless/test_*.py")["FILTER.intersection"]
    assert not [t for t in tests if "a4_multitable" in t], tests
    assert [t for t in tests if "batch1_selection" in t], tests
    assert FEATURE_TAXONOMY["FILTER.intersection"]["backends"] == ["node"]


# ===================================================================================== Node: the ADF-shaped page
@_node
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("DISPLAY.downsampler")
@pytest.mark.parametrize("coding", sorted(CODINGS))
@pytest.mark.parametrize("page_kind", ["auto", "narrow", "toggle"])
def test_strict_adf_page_follows_every_selection_state(node_session, tmp_path, page_kind, coding):
    """W01-W03: every output of the page - the selected rows, the 1-D and 2-D histograms, the profile and the
    scatter plot's rows - equals numpy at load and after every move and restore."""
    verdict, detail = classify(run_page(node_session, tmp_path, page_kind, coding), page_kind, coding)
    if verdict == "PASS":
        return
    regression(verdict == "KNOWN", f"the page equals the model of {detail} (on master: "
                                   f"{EXPECTED_BLAME[(page_kind, coding)]})")
    raise AssertionError(f"neither the oracle nor any set of Batch 1 defects: {detail}")


@_node
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.profile")
def test_control_lossless_page_at_full_range_selects_and_bins_like_numpy(node_session, tmp_path):
    """Healthy control: on the lossless page whose widgets start at their full range (S1), no Batch 1 selection defect
    applies (B1 and B2 need rows excluded at load, WT-D6 needs rounding). The selected rows, both histograms and the
    profile equal numpy at every step. (Before the fix the scatter plot was left to DS-D1; now it is checked too.)"""
    observed = run_page(node_session, tmp_path, "auto", "lossless")
    for k, (o, i) in enumerate(zip(observed, expected("auto", "lossless"))):
        assert not _differences(o, i), (k, _differences(o, i))


@_node
@pytest.mark.feature("FILTER.intersection")
def test_falsifier_a_muted_widget_is_neither_green_nor_a_known_defect(node_session, tmp_path):
    """Gate G4, live: the B widget's link to its filter is removed on a copy of the page. Its move then changes
    nothing; the strict comparison must call that UNEXPLAINED - not PASS, and not a known defect."""
    observed = run_page(node_session, tmp_path, "auto", "lossless", mute="B")
    verdict, detail = classify(observed, "auto", "lossless")
    assert verdict == "UNEXPLAINED" and detail["step"] == 2, (verdict, detail)


SELECTION_TABLE_JS = """() => {
  const t = window.__riH.models().filter(m => m.type === 'DataTable');
  if (t.length !== 1) throw new Error('selection table found ' + t.length + ' times');
  const d = t[0].source.data;
  return Object.fromEntries(Array.from(d.field, (f, i) => [f, d.value[i]]));
}"""


def _js_number(x: float) -> str:
    """A number as JavaScript prints it (Python's repr, without '.0' for whole numbers)."""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


@_node
@pytest.mark.feature("FILTER.range")
def test_the_selection_table_still_shows_the_widget_values(node_session, tmp_path):
    """The fix of WT-D6 opens an automatic widget's filter at its limits (+-Infinity). The selection table of the
    dashboard template showed the filter's range, which was the widget's value; it now shows the widget's value
    directly - the same text as on master: at load, after a move, and back at the limits."""
    report = ScenarioResult(id="batch1/selection_table [node]", outdir=str(tmp_path))
    page = node_session.new_page(report)
    page.goto(build_document(tmp_path, "auto", "relative16", selection_table=True))
    page.wait_for_function(page.load_condition)
    assert page.evaluate("p => window.__riH.settle(p, 50)", PROBES).get("settled")
    lim = limits("auto")
    assert page.evaluate(SELECTION_TABLE_JS) == {c: f"[{lim[c][0]}, {lim[c][1]}]" for c in "AB"}   # Python's text
    for value, text in (([2.0, 8.0], "[2, 8]"), (FULL, f"[{_js_number(lim['A'][0])}, {_js_number(lim['A'][1])}]")):
        assert page.evaluate(ACT_JS, ["set", "A", value, PROBES]).get("settled")
        shown = page.evaluate(SELECTION_TABLE_JS)
        assert shown == {"A": text, "B": f"[{lim['B'][0]}, {lim['B'][1]}]"}, shown
    assert not report.page_errors and not report.console_errors, (report.page_errors, report.console_errors)


# ===================================================================================== Node: WT-D6 alone
def _load_state(node_session, tmp_path, page_kind, coding):
    page, report = open_page(node_session, tmp_path, page_kind, coding)
    state = page.evaluate(READ_JS, N)
    assert not report.page_errors and not report.console_errors, (report.page_errors, report.console_errors)
    return state


def _is_wt_d6(state) -> bool:
    """WT-D6: at load exactly the two A rows whose decoded value lies outside the raw [min, max] are not selected; the
    2-D histogram follows that selection."""
    dec = decoded("relative16")
    lost = [r for r in range(N) if not state["mask"][r]]
    rest = np.ones(N, dtype=bool)
    rest[[A_LOW_ROW, A_HIGH_ROW]] = False
    return (lost == [A_LOW_ROW, A_HIGH_ROW]
            and state["h2"] == ho.counts([dec["A"], dec["Y"]], H2["nbins"], H2["range"], mask=rest).tolist())


@_node
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_control_wt_d6_the_same_widget_on_a_lossless_column_keeps_every_row(node_session, tmp_path):
    """WT-D6's control: the same page, lossless: every row selected at load, and the 2-D histogram of all rows."""
    state = _load_state(node_session, tmp_path, "one_auto", "lossless")
    dec = decoded("lossless")
    assert state["mask"] == [1] * N
    assert state["h2"] == ho.counts([dec["A"], dec["Y"]], H2["nbins"], H2["range"]).tolist()


@_node
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_strict_wt_d6_a_full_range_widget_keeps_every_exported_row(node_session, tmp_path):
    """One widget on A at its automatic full range, relative 16 bits. On master rows 17 and 311 were lost at load
    (decoded 0.25 below the start 0.2500023, decoded 10.0 above the end 9.9999)."""
    state = _load_state(node_session, tmp_path, "one_auto", "relative16")
    regression(_is_wt_d6(state), "WT-D6: rows 17 and 311 lost at load")
    dec = decoded("relative16")
    assert state["mask"] == [1] * N, [r for r in range(N) if not state["mask"][r]]
    assert state["h2"] == ho.counts([dec["A"], dec["Y"]], H2["nbins"], H2["range"]).tolist()


# ===================================================================================== Node: B2 in the narrow state (S2)
def _is_b2(state, intended_first) -> bool:
    """B2: at load the 1-D histogram counts all 600 rows; the selection and the 2-D histogram follow the cut."""
    dec = decoded("lossless")
    return (state["h1"] == ho.counts([dec["A"]], [H1["nbins"]], [H1["range"]]).tolist() != intended_first["h1"]
            and state["mask"] == intended_first["mask"] and state["h2"] == intended_first["h2"])


@_node
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("FILTER.range")
def test_control_b2_after_a_move_and_the_restore_the_1d_histogram_follows(node_session, tmp_path):
    """B2's control: from the first move on, the 1-D histogram follows the selection, also back at the load state.
    (Before the fix the scatter plot was left to DS-D1; now it is checked too.)"""
    observed = run_page(node_session, tmp_path, "one_narrow", "lossless")
    for k, (o, i) in enumerate(zip(observed, expected("one_narrow", "lossless"))):
        if k:
            assert not _differences(o, i), (k, _differences(o, i))


@_node
@pytest.mark.feature("HIST.histogram_1d")
def test_strict_b2_the_1d_histogram_follows_a_widget_narrower_at_load(node_session, tmp_path):
    """State S2: one widget on A with limits [2, 8]. On master the 1-D histogram counted all 600 rows at load."""
    state = _load_state(node_session, tmp_path, "one_narrow", "lossless")
    first = expected("one_narrow", "lossless")[0]
    regression(_is_b2(state, first), "B2: the 1-D histogram counts all rows at load")
    assert state["h1"] == first["h1"]


# ===================================================================================== Node: B1 in the narrow state (S2)
def _is_b1(state) -> bool:
    """B1: at load the selection is B1's rule (the second filter keeps the rows it excludes), not the intersection."""
    first, model = expected("narrow", "lossless")[0], expected("narrow", "lossless", {"B1"})[0]
    return state["mask"] == model["mask"] != first["mask"] and state["h2"] == model["h2"]


@_node
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_control_b1_one_widget_narrower_at_load_selects_its_rows(node_session, tmp_path):
    """B1's control: with one filter, the selection and the 2-D histogram at load equal numpy."""
    state = _load_state(node_session, tmp_path, "one_narrow", "lossless")
    first = expected("one_narrow", "lossless")[0]
    assert state["mask"] == first["mask"] and state["h2"] == first["h2"]


@_node
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.feature("HIST.histogram_nd")
def test_strict_b1_two_widgets_narrower_at_load_select_their_intersection(node_session, tmp_path):
    """State S2: A in [2, 8] and B in [-3, 3] at load. On master the selection was B1's rule."""
    state = _load_state(node_session, tmp_path, "narrow", "lossless")
    regression(_is_b1(state), "B1: the second filter keeps the rows it excluded at load")
    first = expected("narrow", "lossless")[0]
    assert state["mask"] == first["mask"] and state["h2"] == first["h2"]


# ===================================================================================== Node: the T1 acceptance page
T1_N, T1_NT = 12000, 400
T1_NBZ = ["5", "3", "10"]


def t1_tables():
    """References (r, tgl, p, MC_idx) and MC tracks (p, isMonopole), the last track row the NaN fill row of orphans."""
    rng = np.random.default_rng(SEED + 1)
    tp = np.append(rng.uniform(0.2, 10.0, T1_NT), np.nan)
    mono = np.append((rng.uniform(size=T1_NT) < 0.3).astype(np.float64), np.nan)
    mc = rng.integers(0, T1_NT, T1_N)
    mc[rng.uniform(size=T1_N) < 0.02] = T1_NT                       # 2 % orphans point at the fill row
    p = np.where(mc < T1_NT, tp[np.minimum(mc, T1_NT - 1)], 1.0) * (1.0 + 0.1 * rng.normal(size=T1_N))
    refs = pd.DataFrame({"r": rng.uniform(0.0, 50.0, T1_N), "tgl": rng.uniform(-2.0, 2.0, T1_N),
                         "p": np.abs(p), "MC_idx": mc.astype(np.int64)})
    return refs, {"p": tp, "isMonopole": mono}


def t1_widgets(monopoles_only=True):
    mono = [0.5, 1.0] if monopoles_only else [0.0, 1.0]
    return [["range", ["tgl", -1.0, 1.0], {"name": "tgl"}],                       # $w(tgl)$, narrower than the data
            ["range", ["tracks.isMonopole[MC_idx]", *mono], {"name": "isMonopole"}],  # $w(isMonopole)$
            ["select", ["nbZ"], {"name": "nbZ"}]]                                 # $w(nbZ)$


T1_HIST = {"name": "hRatio", "variables": ["r", "p / tracks.p[MC_idx]", "tracks.p[MC_idx]"],
           "nbins": [20, 100, "nbZ"], "range": [[0, 50], [0, 2], [0, 10]], "quantiles": [0.5], "unbinned_projections": True}
T1_STEPS = [("load",), ("set", "tgl", [-0.5, 0.8]), ("nbz", "3"), ("set", "tgl", FULL), ("nbz", "10")]


def t1_document(tmp_path, monopoles_only=True):
    """Claude10's compiled T1 (PHASE_3_1 v0.3 review), lossless coding as the relation exporter sends indices."""
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    refs, tracks = t1_tables()
    widgets = t1_widgets(monopoles_only)
    shown = sa.show
    sa.show = lambda *a, **k: None
    try:
        root = sa.bokehDrawSA.fromArray(
            refs, None,
            [[["bin_center_0"], ["mean"], {"source": "hRatio_1", "colorZvar": "bin_center_2", "errY": "std/sqrt(entries)"}]],
            widgets, sourceArray=[{"name": "tracks", "data": tracks}, dict(T1_HIST)], aliasArray=[],
            parameterArray=[{"name": "nbZ", "value": 5, "options": [3, 5, 10]}],
            layout=[[0]], widgetLayout=[list(range(len(widgets)))], arrayCompression=CODINGS["lossless"]).pAll
    finally:
        sa.show = shown
    doc = Document()
    doc.add_root(root)
    out = tmp_path / f"batch1_t1_{'mono' if monopoles_only else 'all'}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


def t1_expected(monopoles_only=True, b1=False) -> list:
    """pandas: per step the profile cells (r bin, MC.p bin) -> (entries, mean ratio) over the selected references."""
    refs, tracks = t1_tables()
    mono_now = (0.5, 1.0) if monopoles_only else (0.0, 1.0)
    gathered_mono = tracks["isMonopole"][refs["MC_idx"].to_numpy()]
    mc_p = tracks["p"][refs["MC_idx"].to_numpy()]
    ratio = refs["p"].to_numpy() / mc_p
    tgl = refs["tgl"].to_numpy()
    load = [range_mask(tgl, -1.0, 1.0), range_mask(gathered_mono, *mono_now)]
    tgl_now, nbz = (-1.0, 1.0), 5
    out = []
    for step in T1_STEPS:
        if step[0] == "set":
            tgl_now = (-1.0, 1.0) if step[2] == FULL else tuple(step[2])
        elif step[0] == "nbz":
            nbz = int(step[1])
        now = [range_mask(tgl, *tgl_now), load[1]]
        sel = b1_selection(now, load) if b1 else np.logical_and.reduce(now)
        idx = ho.bin_index_nd([refs["r"].to_numpy(), ratio, mc_p], [20, 100, nbz], [(0, 50), (0, 2), (0, 10)])
        keep = sel & (idx >= 0)
        df = pd.DataFrame({"i0": idx[keep] % 20, "i2": idx[keep] // 2000, "ratio": ratio[keep]})
        g = df.groupby(["i0", "i2"])["ratio"]
        cells = {(int(i0), int(i2)): (float(n), float(m)) for (i0, i2), n, m in zip(g.size().index, g.size(), g.mean())}
        out.append({"nbz": nbz, "cells": cells, "entries": int(keep.sum())})
    return out


T1_READ_JS = """() => {
  const H = window.__riH;
  const p = H.model('hRatio_1', ['HistoNdProfile']);
  if (!p) throw new Error(H.why('hRatio_1', ['HistoNdProfile']));
  const c = n => Array.from(p.get_column(n));
  return {c0: c('bin_center_0'), c2: c('bin_center_2'), entries: c('entries'),
          mean: c('mean').map(x => Number.isFinite(x) ? x : null)};
}"""

T1_ACT_JS = """async ([op, name, value]) => {
  const H = window.__riH;
  if (op === 'set') {
    const s = H.slider(name);
    if (!s) throw new Error('range widget ' + name + ' found ' + H.sliders(name).length + ' times');
    s.value = value === 'full' ? [s.start, s.end] : value;
  } else {
    const w = H.models().filter(m => m.type === 'Select' && m.title === 'nbZ');
    if (w.length !== 1) throw new Error('select nbZ found ' + w.length + ' times');
    w[0].value = value;
  }
  return await H.settle([['hRatio_1', 'mean', ['HistoNdProfile']]], 50);
}"""


def t1_run(node_session, tmp_path, monopoles_only=True) -> list:
    report = ScenarioResult(id=f"batch1/t1 [node]", outdir=str(tmp_path))
    page = node_session.new_page(report)
    page.goto(t1_document(tmp_path, monopoles_only))
    page.wait_for_function(page.load_condition)
    assert page.evaluate("p => window.__riH.settle(p, 50)", [["hRatio_1", "mean", ["HistoNdProfile"]]]).get("settled")
    out = []
    for step in T1_STEPS:
        if step[0] == "set":
            assert page.evaluate(T1_ACT_JS, ["set", step[1], step[2]]).get("settled")
        elif step[0] == "nbz":
            assert page.evaluate(T1_ACT_JS, ["nbz", "nbZ", step[1]]).get("settled")
        out.append(page.evaluate(T1_READ_JS))
    assert not report.page_errors and not report.console_errors, (report.page_errors, report.console_errors)
    return out


def t1_differences(state, want) -> list:
    """Cells where the page's profile differs from pandas: entries exactly, the mean within 1e-12 (relative)."""
    nbz = want["nbz"]
    got = {}
    for c0, c2, n, m in zip(state["c0"], state["c2"], state["entries"], state["mean"]):
        if n:
            got[(int(np.floor(c0 / 2.5)), int(np.floor(c2 / (10.0 / nbz))))] = (float(n), m)
    out = []
    if len(state["c0"]) != 20 * nbz:
        out.append(("cells", len(state["c0"]), 20 * nbz))
    for key in sorted(set(got) | set(want["cells"])):
        g, w = got.get(key), want["cells"].get(key)
        if g is None or w is None or g[0] != w[0] or g[1] is None or not np.isclose(g[1], w[1], rtol=1e-12, atol=1e-12):
            out.append((key, g, w))
    return out


@_unit
def test_t1_pandas_reference_is_not_degenerate():
    """The reference has populated cells, the cuts change it at every step, and B1 would change it at load."""
    good, bad = t1_expected(), t1_expected(b1=True)
    assert all(s["entries"] > 100 for s in good)
    assert len({s["entries"] for s in good}) > 1
    assert bad[0]["entries"] > good[0]["entries"]


@_node
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.feature("HIST.profile")
@pytest.mark.feature("DSL.gather_operation")
def test_strict_t1_acceptance_page_equals_pandas_in_every_state(node_session, tmp_path):
    """The T1 call, both cuts narrower at load: the profile cells (entries and mean ratio) equal pandas at load, after
    the tgl move, after nbZ 5 -> 3, after the restore and after nbZ -> 10."""
    observed = t1_run(node_session, tmp_path)
    diffs = [t1_differences(o, w) for o, w in zip(observed, t1_expected())]
    if not any(diffs):
        return
    b1_diffs = [t1_differences(o, w) for o, w in zip(observed, t1_expected(b1=True))]
    regression(not any(b1_diffs), "B1: the page equals B1's rule at every step")
    raise AssertionError(f"neither pandas nor B1's rule: {[d[:3] for d in diffs]}")


@_node
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.profile")
@pytest.mark.feature("DSL.gather_operation")
def test_control_t1_with_the_monopole_widget_at_full_range_equals_pandas(node_session, tmp_path):
    """T1's control: the monopole widget at [0, 1] excludes only the orphans at load, which the profile skips anyway
    (their ratio is NaN), so B1 cannot change a cell: every state equals pandas."""
    observed = t1_run(node_session, tmp_path, monopoles_only=False)
    for k, (o, w) in enumerate(zip(observed, t1_expected(monopoles_only=False))):
        assert not t1_differences(o, w), (k, t1_differences(o, w)[:3])
