"""PHASE_4_7 stage 3 — the oracle test dashboard WR1: RootInteractive's WeightsRef example dashboard, end to end in
Chromium (Claude20.RI, 2026-11-12).

In plain words. The original function `test_interactiveTemplateWeightsRef` of test_bokehClientHistogram.py builds its
dashboard on the reproducible table (50,000 rows, seed 4711, decision DEC-02). The dashboard compares two weightings of
the same rows, the ANALYSIS weight (`weights`, default A) and the REFERENCE weight (`weightsRef`, default B/(1+C)); its
tab "histoXYWeight" draws, per X bin of A and Y bin of B (100 x 120 bins):
- the weighted 2-D counts of both sides, and their difference (or ratio, or log ratio);
- the weighted mean, median and std of B in each X bin, for both sides, with error bars on each;
- the comparison curves of the mean, median and std.
The test opens the real page in Chromium and drives it with real input, at seven checkpoints:
  C1 open the tab histoXYWeight (mouse click)
  C2 weightsRef := A, the same weight as the analysis (select in the widget tab "Histograms")
  C3 diffFuncWeightsTransform := ratio
  C4 nbinsX := 10 (typed into the spinner)
  C5 nbinsX := 100, ratio := diff, weightsRef := B/(1+C): back to C1
  C6 A range slider: lower handle 8 steps right (keyboard)
  C7 lower handle back 8 steps left: back to C1
At each checkpoint every number is compared with numpy (dashboards/oracle_weightsref.py), in two ways:
- DESIRED, by the Architect's decisions (decisions document v0.3, audit v0.2.1 chapter 3);
- AS SHOWN, the same with the three known defects switched on, each predicted exactly from the data:
    WT-D6   at its end the A slider still cuts: the rounded A of row 44894 lies above the slider's raw end;
    WR-D1   the reference side is computed from bin centres (binned) instead of the values, so equal weights do not
            give equal curves (observation O-1; Q-T-02 read by R2 "proper statistical formula");
    WT-D14  every error bar is drawn as std/sqrt(entries); the proper error of a weighted mean is s/sqrt(N_eff).
A checkpoint is PASS (equal to DESIRED), KNOWN_ISSUE_RED (equal to AS SHOWN, the defects that make the difference
named), or FAIL. Checked: the selected rows (exact set), the histogram ranges, all 12,000 weighted counts of both
sides, both profiles (entries, mean, std, both quantiles), all 12 drawn series (x values, y values, and the screen
coordinates Bokeh computed, anchored to the plot frame and axis ranges), the comparison curves, and the error bars of
the two mean series. The error bars of the median and std series are reported, not judged, until the Architect
confirms their formulas ("R2-errors").

Two lanes (DEC-05): the workflow test is the strict expected failure of the three defects; dashboard_health.json (in
headless_out/dashboards/wr1/) marks each checkpoint and the dashboard RED. Healthy controls: the weighted counts and
the analysis side equal numpy exactly (they have no defect), and live faults injected into the page are caught.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from .dashboards import oracle, release
from .dashboards import oracle_weightsref as W
from .dashboards.compare import _anchored
from .test_headless_oracle_dashboards import ROWS, SEED, DEFAULT_TABLE, KnownDefect, known_defect, _Page, _git_identity

UNIT = pytest.mark.headless_unit
BROWSER = pytest.mark.browser_oracle_dashboards
# The feature mark is written out on each test: scripts/generate_capability_matrix.py reads only the literal form
# @pytest.mark.feature("ID"), so a mark held in a variable left WR1 out of the capability matrix (closing check).
SCENARIO = "WR1"
K_LOW = 8
NBINS = (100, 120)
RTOL_SUM, RTOL_MOMENT, RTOL_ERR = 1e-12, 1e-10, 1e-9
ALL_DEFECTS = tuple(W.DEFECTS)
WR_XFAIL = dict(strict=DEFAULT_TABLE, raises=KnownDefect)
WR_DEFECTS = ("WT-D6, WR-D1, WT-D14 at every checkpoint: the A slider at its end still cuts row 44894; the reference "
              "side is binned; the error bar of a weighted mean is std/sqrt(sum w), not s/sqrt(N_eff)")
JOIN = "histoXYData_1_join"
# (data source, drawn y field) -> what it must equal; x is the X bin centre (profiles) or the 2-D bin's X centre
SERIES = {
    ("histoXYData_1", "mean"): ("analysis", "mean"),
    ("histoXYData_1", "quantile_1"): ("analysis", "q0.5"),
    ("histoXYData_1", "std"): ("analysis", "std"),
    ("histoXYData", "bin_count"): ("bin_count", None),
    ("projXYRef_1", "mean"): ("reference", "mean"),
    ("projXYRef_1", "quantile_1"): ("reference", "q0.5"),
    ("projXYRef_1", "std"): ("reference", "std"),
    ("histoXYData", "histoXYRef"): ("histoXYRef", None),
    (JOIN, "diffFuncWeights(histoXYData_1.mean, projXYRef_1.mean)"): ("comparison", "mean"),
    (JOIN, "diffFuncWeights(histoXYData_1.quantile_1, projXYRef_1.quantile_1)"): ("median_comparison", None),
    (JOIN, "diffFuncWeights(histoXYData_1.std, projXYRef_1.std)"): ("comparison", "std"),
    ("histoXYData", "diffFuncWeights(bin_count, histoXYRef)"): ("comparison", "counts"),
}


def _norm(field: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\s*\)", ")", field or "")).strip()


# ------------------------------------------------------------------ the comparator
def _values(got, want, what: str, rtol: float, atol: float = 1e-12) -> list:
    g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if g.shape != w.shape:
        return [f"{what}: {g.size} values, expected {w.size}"]
    for name, f in (("NaN", np.isnan), ("+inf", np.isposinf), ("-inf", np.isneginf)):
        if not np.array_equal(f(g), f(w)):
            i = int(np.flatnonzero(f(g) != f(w))[0])
            return [f"{what}: {name} at {i} differs (page {g[i]!r}, expected {w[i]!r})"]
    fin = np.isfinite(w)
    bad = ~np.isclose(g[fin], w[fin], rtol=rtol, atol=atol)
    if bad.any():
        i = int(np.flatnonzero(fin)[np.flatnonzero(bad)[0]])
        return [f"{what}: value {i} is {g[i]!r}, expected {w[i]!r} ({int(bad.sum())} values differ)"]
    return []


def _bracket(got, lo, hi, what: str) -> list:
    g, lo, hi = (np.asarray(a, dtype=float) for a in (got, lo, hi))
    if g.shape != lo.shape:
        return [f"{what}: {g.size} values, expected {lo.size}"]
    if not np.array_equal(np.isnan(g), np.isnan(lo)):
        i = int(np.flatnonzero(np.isnan(g) != np.isnan(lo))[0])
        return [f"{what}: NaN at {i} differs (page {g[i]!r}, expected between {lo[i]!r} and {hi[i]!r})"]
    fin = np.isfinite(lo)
    tol = 1e-12 * np.maximum(1.0, np.abs(lo[fin]))
    bad = (g[fin] < lo[fin] - tol) | (g[fin] > hi[fin] + tol)
    if bad.any():
        i = int(np.flatnonzero(fin)[np.flatnonzero(bad)[0]])
        return [f"{what}: value {i} is {g[i]!r}, expected between {lo[i]!r} and {hi[i]!r} ({int(bad.sum())} differ)"]
    return []


def _profile(got: dict, want: dict, side: str) -> list:
    out = _values(got["entries"], want["entries"], f"{side} entries", RTOL_SUM)
    out += _values(got["mean"], want["mean"], f"{side} mean", RTOL_MOMENT)
    out += _values(got["std"], want["std"], f"{side} std", RTOL_MOMENT)
    for q, col in zip(W.QUANTILES, ("quantile_0", "quantile_1")):
        out += _bracket(got[col], want[f"q{q}_lo"], want[f"q{q}_hi"], f"{side} {col} (q = {q})")
    return out


def _drawn(act: dict) -> dict:
    """The drawn scatters by (data source, y field); a key drawn twice or more is kept as a list to be reported."""
    out = {}
    for r in act["drawn"]:
        out.setdefault((r["src"].split("#")[0], _norm(r["y"])), []).append(r)
    return out


def _anchor(r: dict) -> list:
    x, y = np.asarray(r["xv"], dtype=float), np.asarray(r["yv"], dtype=float)
    fin = np.isfinite(x) & np.isfinite(y)
    rr = dict(r, sx=[s for s, f in zip(r["sx"], fin) if f], sy=[s for s, f in zip(r["sy"], fin) if f])
    return _anchored(rr, "x", x[fin]) + _anchored(rr, "y", y[fin])


def wr_compare(exp: dict, act: dict, sections=("state", "selection", "histograms", "analysis", "reference",
                                             "drawn", "comparison", "errors")) -> list:
    """Every difference between the expected response `exp` (oracle_weightsref.expected) and the page `act`."""
    out = []
    if "state" in sections:
        if act["tab"] != "histoXYWeight":
            out.append(f"active tab {act['tab']!r}, expected 'histoXYWeight'")
        for k, v in exp["state"].items():
            if act["state"].get(k) != v:
                out.append(f"widget {k}: page {act['state'].get(k)!r}, expected {v!r}")
        if list(act["A"]) != list(exp["A"]):
            out.append(f"A slider: page {act['A']}, expected {exp['A']}")
    if "selection" in sections:
        sel = np.asarray(act["selected"], dtype=bool)
        if sel.shape != exp["selected_rows"].shape or not np.array_equal(sel, exp["selected_rows"]):
            lost = np.flatnonzero(exp["selected_rows"] & ~sel)[:5].tolist() if sel.shape == exp["selected_rows"].shape else "?"
            out.append(f"selected rows: page {int(sel.sum())}, expected {exp['selected_count']} (missing e.g. {lost})")
    if "histograms" in sections:
        if list(act["nbins"]) != [exp["state"]["nbinsX"], exp["state"]["nbinsY"]]:
            out.append(f"bins {act['nbins']}, expected {[exp['state']['nbinsX'], exp['state']['nbinsY']]}")
        out += _values(act["x_range"], exp["x_range"], "X range (auto range of the selected rows)", 0, 0)
        out += _values(act["y_range"], exp["y_range"], "Y range", 0, 0)
        out += _values(act["bin_count"], exp["bin_count"], "analysis weighted counts bin_count", RTOL_SUM)
        out += _values(act["histoXYRef"], exp["histoXYRef"], "reference weighted counts histoXYRef", RTOL_SUM)
    if "analysis" in sections:
        out += _profile(act["analysis"], exp["analysis"], "analysis (histoXYData_1)")
    if "reference" in sections:
        out += _profile(act["reference"], exp["reference"], "reference (projXYRef_1)")
    drawn = _drawn(act)
    if "drawn" in sections or "comparison" in sections:
        for key, (part, col) in SERIES.items():
            is_cmp = part in ("comparison", "median_comparison")
            if ("comparison" if is_cmp else "drawn") not in sections:
                continue
            if part in ("analysis", "reference") and part not in sections:
                continue
            rs = drawn.get((key[0], _norm(key[1])), [])
            if len(rs) != 1:
                out.append(f"drawn {key}: {len(rs)} renderers, expected exactly one")
                continue
            r, what = rs[0], f"drawn {key[0]} {key[1]}"
            twod = part in ("bin_count", "histoXYRef") or col == "counts"
            out += _values(r["xv"], exp["xy_center_0"] if twod else exp["x_center"], what + " x", 1e-12)
            if part == "median_comparison":
                a = drawn.get(("histoXYData_1", "quantile_1"), [{}])[0].get("yv")
                b = drawn.get(("projXYRef_1", "quantile_1"), [{}])[0].get("yv")
                if a is None or b is None:
                    out.append(f"{what}: the two drawn medians are missing")
                else:
                    out += _values(r["yv"], W.transform(a, b, exp["state"]["transform"]), what, 1e-12)
            elif col in ("q0.5",):
                out += _bracket(r["yv"], exp[part]["q0.5_lo"], exp[part]["q0.5_hi"], what)
            else:
                want = exp[part] if col is None else exp[part][col]
                out += _values(r["yv"], want, what, RTOL_MOMENT)
            out += [f"{what}: {d}" for d in _anchor(r)]
    if "errors" in sections:
        for src, side in (("histoXYData_1", "analysis"), ("projXYRef_1", "reference")):
            mean = drawn.get((src, "mean"), [{}])[0].get("yv")
            quads = [q for q in act["quads"] if q["src"].split("#")[0] == src and mean is not None
                     and len(q["top"]) == len(mean)
                     and np.allclose((np.asarray(q["top"], float) + np.asarray(q["bottom"], float)) / 2,
                                     np.asarray(mean, float), rtol=1e-12, atol=1e-12, equal_nan=True)]
            if len(quads) != 1:
                out.append(f"error bar of the {side} mean: {len(quads)} drawn, expected exactly one")
                continue
            err = (np.asarray(quads[0]["top"], float) - np.asarray(quads[0]["bottom"], float)) / 2
            out += _values(err, exp["errors"][side]["mean"], f"error bar of the {side} mean", RTOL_ERR)
    return out


def _attribute(t, state) -> list:
    """The known defects that change the expected response in this state (each one switched off in turn)."""
    full = W.expected(t, state, ALL_DEFECTS)
    return [d for d in ALL_DEFECTS if _expected_differs(full, W.expected(t, state, [x for x in ALL_DEFECTS if x != d]))]


def _expected_differs(a: dict, b: dict) -> bool:
    if not np.array_equal(a["selected_rows"], b["selected_rows"]):
        return True
    for side in ("analysis", "reference"):
        for k in ("entries", "mean", "std", "q0.5_lo", "q0.35_lo"):
            if not np.allclose(a[side][k], b[side][k], rtol=1e-13, atol=0, equal_nan=True):
                return True
        if not np.allclose(a["errors"][side]["mean"], b["errors"][side]["mean"], rtol=1e-13, atol=0, equal_nan=True):
            return True
    return False


def _detail(t, state) -> dict:
    """Each defect's exact signature in this state (recorded in the health file)."""
    shown, desired = W.expected(t, state, ALL_DEFECTS), W.expected(t, state, ())
    lost = np.flatnonzero(desired["selected_rows"] & ~shown["selected_rows"]).tolist()
    d_ref = np.nanmax(np.abs(shown["reference"]["mean"] - W.expected(t, state, ("WT-D6", "WT-D14"))["reference"]["mean"]))
    no14 = W.expected(t, state, ("WT-D6", "WR-D1"))
    ratio = shown["errors"]["analysis"]["mean"] / no14["errors"]["analysis"]["mean"]
    return {"WT-D6": {"lost_rows": lost},
            "WR-D1": {"reference_mean_binned_minus_unbinned_max": round(float(d_ref), 9)},
            "WT-D14": {"analysis_mean_error_drawn_over_proper_max": round(float(np.nanmax(ratio)), 6)}}


# ------------------------------------------------------------------ the page: capture, snapshot, real input
def _capture(outdir: Path) -> dict:
    cmd = [sys.executable, "-m", "RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.capture",
           W.WR_FUNCTION, str(ROWS), str(SEED), str(outdir)]
    env = {k: v for k, v in os.environ.items() if k not in ("RI_DASHBOARD_ROWS", "RI_DASHBOARD_SEED")}
    root = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    if done.returncode != 0:
        pytest.fail(f"capture failed ({done.returncode}):\n{done.stderr[-3000:]}", pytrace=False)
    manifest = json.loads((outdir / "manifest.json").read_text())
    manifest["seconds"]["capture_process"] = round(time.perf_counter() - t0, 3)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


@pytest.fixture(scope="session")
def wr_page(headless_out):
    out = headless_out / "dashboards" / "wr1"
    out.mkdir(parents=True, exist_ok=True)
    manifest = _capture(out)
    return {"dir": out, "html": out / "page.html", "manifest": manifest}


SNAPSHOT_JS = r"""() => {
  const H = window.__riH, kids = x => !x ? [] : (x instanceof Map ? Array.from(x.values()) : Array.from(x));
  const arr = a => a ? Array.from(a) : null, fld = v => v && typeof v === 'object' ? (v.field ?? null) : null;
  const one = (name, type) => { const m = H.models().filter(x => x.name === name && x.type === type);
    if (m.length !== 1) throw Error('WR1 page: expected one ' + type + ' ' + name + ', found ' + m.length); return m[0]; };
  const val = (type, title) => { const m = H.models().filter(x => x.type === type && x.title === title);
    if (m.length !== 1) throw Error('WR1 page: expected one ' + type + ' ' + title); return m[0].value; };
  const tabs = H.models().filter(m => m.type === 'LazyTabs' && m.tabs.some(t => t.title === 'histoXYWeight'));
  const prof = name => { const m = one(name, 'HistoNdProfile'), r = {};
    for (const c of ['entries', 'mean', 'std', 'quantile_0', 'quantile_1']) r[c] = arr(m.get_column(c)); return r; };
  const h = one('histoXYData', 'HistoNdCDS'), orig = H.model('cdsOrig');
  const n = orig.get_length(), v = H.models().filter(m => m.type === 'LazyIntersectionFilter')[0].v_compute();
  let mask = ''; for (let i = 0; i < n; i++) mask += v[i] ? '1' : '0';
  const drawn = [], quads = [], seen = new Set();
  const walk = view => { if (!view || seen.has(view)) return; seen.add(view);
    if (view.model && view.model.type === 'GlyphRenderer') {
      const c = view.parent?.canvas_view?.el, shown = c ? c.checkVisibility({visibilityProperty: true}) : false;
      if (shown && view.model.visible) { const g = view.model.glyph, gv = view.glyph, plot = view.parent.model;
        const src = view.model.data_source.name;
        if (g.type === 'Scatter') { const bb = view.parent.frame.bbox;
          const label = side => { const a = plot?.[side]; return a && a.length === 1 ? String(a[0].axis_label) : null; };
          drawn.push({src, x: fld(g.x), y: fld(g.y), xv: arr(gv._x), yv: arr(gv._y), sx: arr(gv.sx), sy: arr(gv.sy),
                      frame: {left: bb.left, right: bb.right, top: bb.top, bottom: bb.bottom},
                      x_range: [plot.x_range.start, plot.x_range.end], y_range: [plot.y_range.start, plot.y_range.end],
                      x_scale: plot.x_scale?.type ?? null, y_scale: plot.y_scale?.type ?? null,
                      xlabel: label('below'), ylabel: label('left'), title: plot.title ? String(plot.title.text) : null}); }
        else if (g.type === 'Quad') quads.push({src, top: arr(gv._top), bottom: arr(gv._bottom), left: arr(gv._left)}); } }
    for (const k of kids(view.child_views)) walk(k); for (const k of kids(view.renderer_views)) walk(k); };
  for (const view of (Bokeh.index instanceof Map ? Bokeh.index.values() : Object.values(Bokeh.index))) walk(view);
  return {tab: tabs.length === 1 ? tabs[0].tabs[tabs[0].active].title : null,
          A: Array.from(H.models().filter(m => m.type === 'RangeSlider' && m.title === 'A')[0].value),
          state: {weights: val('Select', 'weights'), weightsRef: val('Select', 'weightsRef'),
                  transform: val('Select', 'diffFuncWeightsTransform'), nbinsX: val('Spinner', 'nbinsX'),
                  nbinsY: val('Spinner', 'nbinsY')},
          mask, nbins: Array.from(h._nbins), x_range: [h._range_min[0], h._range_max[0]],
          y_range: [h._range_min[1], h._range_max[1]],
          bin_count: arr(h.get_column('bin_count')), histoXYRef: arr(h.get_column('histoXYRef')),
          analysis: prof('histoXYData_1'), reference: prof('projXYRef_1'), drawn, quads};
}"""


def _snapshot(pg) -> dict:
    raw = pg.evaluate(SNAPSHOT_JS)
    raw["selected"] = (np.frombuffer(raw.pop("mask").encode(), dtype=np.uint8) == ord("1")).tolist()
    return raw


def _settle(pg, timeout_s: float = 30.0) -> dict:
    """Wait until Bokeh is idle and two reads 150 ms apart are identical (compared as JSON text: NaN == NaN)."""
    deadline = time.monotonic() + timeout_s
    pg.wait_for_function("() => Bokeh.documents[0].is_idle", timeout=int(timeout_s * 1000))
    last = _snapshot(pg)
    while time.monotonic() < deadline:
        pg.wait_for_timeout(150)
        now = _snapshot(pg)
        if json.dumps(now, sort_keys=True) == json.dumps(last, sort_keys=True):
            return now
        last = now
    raise AssertionError("WR1: the page did not settle")


def _control(page: _Page, title: str, tag: str):
    """The widget's own DOM control, by its label, in the widget tab "Histograms" (real input only)."""
    page.tab("Histograms")
    c = page.pg.locator(".bk-input-group").filter(
        has=page.pg.locator("label", has_text=re.compile(rf"^{re.escape(title)}$"))).locator(tag)
    assert c.count() == 1, f"exactly one {tag} for {title}"
    return c


def _select(page: _Page, title: str, value: str):
    _control(page, title, "select").select_option(value)


def _spinner(page: _Page, title: str, value: int):
    c = _control(page, title, "input")
    c.fill(str(value))
    c.press("Enter")


def _states(m: dict):
    """The seven checkpoints: id, name, real-input action text, page state for the oracle."""
    widgets = m["widget_state"]
    a = widgets["A"]
    base = {"widgets": widgets, "weights": widgets["weights"]["value"], "weightsRef": widgets["weightsRef"]["value"],
            "transform": widgets["diffFuncWeightsTransform"]["value"], "nbinsX": NBINS[0], "nbinsY": NBINS[1]}
    lo, _ = oracle.keyboard_range(a, K_LOW, 0)
    moved = dict(base, widgets=oracle.with_a_range(widgets, lo, a["end"]))
    same = dict(base, weightsRef=base["weights"])
    return [("C1", "baseline", "click the tab histoXYWeight", base),
            ("C2", "equal weights", f"select weightsRef = {base['weights']}", same),
            ("C3", "ratio", "select diffFuncWeightsTransform = ratio", dict(same, transform="ratio")),
            ("C4", "10 X bins", "type 10 into nbinsX", dict(same, transform="ratio", nbinsX=10)),
            ("C5", "restored", "type 100 into nbinsX; select diff; select weightsRef = B/(1+C)", base),
            ("C6", "A range cut", f"A slider: lower handle {K_LOW} arrow steps right", moved),
            ("C7", "A range restored", f"A slider: lower handle {K_LOW} arrow steps left", base)]


def _actions(page: _Page, base: dict) -> dict:
    def c5():
        _spinner(page, "nbinsX", NBINS[0])
        _select(page, "diffFuncWeightsTransform", base["transform"])
        _select(page, "weightsRef", base["weightsRef"])

    def keys(key):
        page.tab("Select")
        page.keys(0, key, K_LOW)
    return {"C1": lambda: page.tab("histoXYWeight"),
            "C2": lambda: _select(page, "weightsRef", base["weights"]),
            "C3": lambda: _select(page, "diffFuncWeightsTransform", "ratio"),
            "C4": lambda: _spinner(page, "nbinsX", 10),
            "C5": c5,
            "C6": lambda: keys("ArrowRight"),
            "C7": lambda: keys("ArrowLeft")}


# ------------------------------------------------------------------ Python-only checks (headless_unit)
def _tiny():
    """A hand-checkable table: 8 rows, X bins of A, Y bins of B."""
    t = {"A": np.array([.1, .1, .1, .6, .6, .9, .9, .9]), "B": np.array([.1, .3, .8, .2, .7, .1, .5, .9]),
         "C": np.array([.5, .5, .5, .5, .5, .5, .5, .5]), "D": np.array([.5] * 8)}
    widgets = {c: {"start": float(t[c].min()), "end": float(t[c].max()), "step": .01,
                   "value": [float(t[c].min()), float(t[c].max())]} for c in "ABCD"}
    for c in "CD":                                             # constant columns: a slider over [0, 1]
        widgets[c] = {"start": 0.0, "end": 1.0, "step": .01, "value": [0.0, 1.0]}
    state = {"widgets": widgets, "weights": "A", "weightsRef": "B/(1+C)", "transform": "diff", "nbinsX": 2, "nbinsY": 2}
    return t, state


@UNIT
def test_wr_oracle_on_a_hand_checked_table():
    t, state = _tiny()
    e = W.expected(t, state)
    assert e["selected_count"] == 8
    a = W.round16(t["A"])
    assert e["x_range"] == [a.min(), a.max()]
    # X bin 0 holds the three rows with A = 0.1 (weights 0.1 each), bin 1 the other five
    assert np.allclose(e["analysis"]["entries"], [3 * a[0], a[3:].sum()], rtol=1e-12)
    b = W.round16(t["B"])
    assert np.isclose(e["analysis"]["mean"][0], b[:3].mean(), rtol=1e-12)          # equal weights in bin 0
    w1, y1 = a[3:], b[3:]
    assert np.isclose(e["analysis"]["mean"][1], np.sum(w1 * y1) / w1.sum(), rtol=1e-12)
    assert np.isclose(e["analysis"]["n_eff"][1], w1.sum() ** 2 / np.sum(w1 ** 2), rtol=1e-12)
    assert np.isclose(e["errors"]["analysis"]["mean"][1], e["analysis"]["std"][1] / np.sqrt(e["analysis"]["n_eff"][1]))
    shown = W.expected(t, state, ALL_DEFECTS)
    assert np.isclose(shown["errors"]["analysis"]["mean"][1], shown["analysis"]["std"][1] / np.sqrt(w1.sum()))
    # equal weights on both sides: desired comparison exactly 0; binned reference (WR-D1) is not
    same = dict(state, weightsRef="A")
    d = W.expected(t, same)
    assert np.array_equal(d["comparison"]["mean"], np.zeros(2)) and np.array_equal(d["comparison"]["counts"],
                                                                                   np.zeros(4))
    assert np.abs(W.expected(t, same, ("WR-D1",))["comparison"]["mean"]).max() > 1e-3
    assert np.all(np.isnan(W.transform([0.], [0.], "ratio"))) and W.transform([2.], [1.], "logRatio")[0] == np.log(2)


@UNIT
def test_wr_rounding_rebuild_equals_rootinteractive():
    """DEC-11: the oracle's own relative16 rebuild equals RootInteractive's rounding (the oracle never calls it)."""
    from RootInteractive.Tools.compressArray import roundRelativeBinary
    x = oracle.regenerate_table(2000, 4711)["A"]
    assert np.array_equal(W.round16(x), roundRelativeBinary(x, 16).astype(np.float64))


@UNIT
def test_wr_quantile_bracket_ignores_zero_weights():
    y, w = np.array([1.125, 1.375, 1.875]), np.array([1., 0., 1.])
    yb = np.zeros(3, dtype=int)
    lo, hi = W._quantile_bracket(y, w, yb, np.array([2.]), 2., .5)
    assert (lo, hi) == (1.125, 1.875)
    assert W._quantile_bracket(y, np.array([1., 0., 2.]), yb, np.array([3.]), 3., .5) == (1.875, 1.875)


def _actual_from(exp: dict) -> dict:
    """A page response built from an expected one (for the comparator's own tests)."""
    def series(xv, yv, src, y):
        xs, ys = np.asarray(xv, float), np.asarray(yv, float)
        fin = ys[np.isfinite(ys)]
        ylo, yhi = (float(fin.min()) - 1, float(fin.max()) + 1) if fin.size else (0.0, 1.0)
        return {"src": src + "#f", "x": "bin_center_0", "y": y, "xv": xs.tolist(), "yv": ys.tolist(),
                "sx": (75 + xs * 560).tolist(),
                "sy": (162 - (np.nan_to_num(ys, nan=0, posinf=0, neginf=0) - ylo) / (yhi - ylo) * 135).tolist(),
                "frame": {"left": 75.0, "right": 635.0, "top": 27.0, "bottom": 162.0},
                "x_range": [0.0, 1.0], "y_range": [ylo, yhi], "x_scale": "LinearScale", "y_scale": "LinearScale"}
    prof = lambda p: {"entries": p["entries"].tolist(), "mean": p["mean"].tolist(), "std": p["std"].tolist(),
                      "quantile_0": p["q0.35_lo"].tolist(), "quantile_1": p["q0.5_lo"].tolist()}
    act = {"tab": "histoXYWeight", "A": list(exp["A"]), "state": dict(exp["state"]),
           "selected": exp["selected_rows"].tolist(), "nbins": [exp["state"]["nbinsX"], exp["state"]["nbinsY"]],
           "x_range": list(exp["x_range"]), "y_range": list(exp["y_range"]),
           "bin_count": exp["bin_count"].tolist(), "histoXYRef": exp["histoXYRef"].tolist(),
           "analysis": prof(exp["analysis"]), "reference": prof(exp["reference"]), "drawn": [], "quads": []}
    for (src, y), (part, col) in SERIES.items():
        twod = part in ("bin_count", "histoXYRef") or col == "counts"
        xv = exp["xy_center_0"] if twod else exp["x_center"]
        if part == "median_comparison":
            yv = W.transform(exp["analysis"]["q0.5_lo"], exp["reference"]["q0.5_lo"], exp["state"]["transform"])
        elif col == "q0.5":
            yv = exp[part]["q0.5_lo"]
        else:
            yv = exp[part] if col is None else exp[part][col]
        act["drawn"].append(series(xv, yv, src, y))
    for src, side in (("histoXYData_1", "analysis"), ("projXYRef_1", "reference")):
        m, e = exp[side]["mean"], exp["errors"][side]["mean"]
        act["quads"].append({"src": src + "#f", "top": (m + e).tolist(), "bottom": (m - e).tolist(),
                             "left": exp["x_center"].tolist()})
    return act


@UNIT
def test_wr_comparator_accepts_the_right_answer_and_rejects_wrong_ones():
    t, state = _tiny()
    for defects in ((), ALL_DEFECTS):
        exp = W.expected(t, state, defects)
        assert wr_compare(exp, _actual_from(exp)) == [], wr_compare(exp, _actual_from(exp))
    exp = W.expected(t, state, ALL_DEFECTS)
    mutations = {
        "a weighted count": ("bin_count", lambda a: a.__setitem__("bin_count", [v + 1e-6 for v in a["bin_count"]])),
        "a lost row": ("selected rows", lambda a: a["selected"].__setitem__(0, False)),
        "the analysis mean": ("analysis (histoXYData_1) mean", lambda a: a["analysis"]["mean"].__setitem__(1, 0.123)),
        "the reference median": ("reference (projXYRef_1) quantile_1", lambda a: a["reference"]["quantile_1"]
                                 .__setitem__(1, 5.0)),
        "a drawn mean": ("drawn histoXYData_1 mean", lambda a: a["drawn"][0]["yv"].__setitem__(0, 9.0)),
        "a drawn comparison": ("drawn histoXYData_1_join diffFuncWeights(histoXYData_1.mean",
                               lambda a: a["drawn"][8]["yv"].__setitem__(0, 9.0)),
        "a series not drawn": ("0 renderers", lambda a: a["drawn"].pop(4)),
        "an error bar": ("error bar of the analysis mean", lambda a: a["quads"][0]["top"].__setitem__(1, 7.0)),
        "a point drawn elsewhere": ("px", lambda a: a["drawn"][2]["sx"].__setitem__(0, a["drawn"][2]["sx"][0] + 3)),
        "the weight": ("widget weightsRef", lambda a: a["state"].__setitem__("weightsRef", "A")),
    }
    for name, (words, mutate) in mutations.items():
        act = _actual_from(exp)
        mutate(act)
        found = wr_compare(exp, act)
        assert any(words in f for f in found), (name, found)
    desired = W.expected(t, state)
    assert wr_compare(desired, _actual_from(exp)), "the desired and the as-shown answers must differ (defects)"


@UNIT
def test_wr_known_defects_change_the_answer_on_the_default_table():
    """Non-vacuity: each known defect changes the expected page at C1 on the reproducible table (python only)."""
    if not DEFAULT_TABLE:
        pytest.skip("the signatures are stated for the default table (50,000 rows, seed 4711)")
    t = oracle.regenerate_table(ROWS, SEED)
    a_lo, a_hi = float(t["A"].min()), float(t["A"].max())
    widgets = {c: {"start": float(t[c].min()), "end": float(t[c].max()), "step": (float(t[c].max()) - float(t[c].min())) / 30,
                   "value": [float(t[c].min()), float(t[c].max())]} for c in "ABCD"}
    state = {"widgets": widgets, "weights": "A", "weightsRef": "B/(1+C)", "transform": "diff", "nbinsX": 100,
             "nbinsY": 120}
    assert set(_attribute(t, state)) == set(ALL_DEFECTS)
    det = _detail(t, state)
    assert det["WT-D6"]["lost_rows"] == [44894] and int(np.argmax(t["A"])) == 44894
    assert W.round16(t["A"])[44894] > a_hi > a_lo
    assert det["WR-D1"]["reference_mean_binned_minus_unbinned_max"] > 1e-6
    assert det["WT-D14"]["analysis_mean_error_drawn_over_proper_max"] > 2


# ------------------------------------------------------------------ the page in Chromium (browser_oracle_dashboards)
@BROWSER
@pytest.mark.feature("DASHBOARD.weights_reference")
def test_wr1_page_is_the_original_dashboard_on_the_reproducible_table(wr_page):
    m = wr_page["manifest"]
    assert m["function"] == W.WR_FUNCTION and m["rows"] == ROWS and m["seed"] == SEED and m["opt_in"] is True
    assert m["table_sha256"] == oracle.table_sha256(oracle.regenerate_table(ROWS, SEED))
    w = m["widget_state"]
    assert (w["weights"]["value"], w["weightsRef"]["value"], w["diffFuncWeightsTransform"]["value"]) == \
        ("A", "B/(1+C)", "diff")


@BROWSER
@pytest.mark.feature("DASHBOARD.weights_reference")
@pytest.mark.xfail(**WR_XFAIL, reason=WR_DEFECTS)
def test_wr1_weights_reference_dashboard_follows_weights_transform_bins_and_range_and_restores(wr_page,
                                                                                             headless_session):
    """The whole WR1 workflow, seven checkpoints. It is the strict partner of WT-D6, WR-D1 and WT-D14: KnownDefect is
    raised only after every checkpoint was compared, and only when each one equals the AS-SHOWN oracle exactly and
    differs from DESIRED by exactly these three defects."""
    out, m = wr_page["dir"], wr_page["manifest"]
    t = oracle.regenerate_table(ROWS, SEED)
    assert oracle.table_sha256(t) == m["table_sha256"]
    steps = _states(m)
    desired = {cid: W.expected(t, s) for cid, _, _, s in steps}
    shown = {cid: W.expected(t, s, ALL_DEFECTS) for cid, _, _, s in steps}
    blame = {cid: _attribute(t, s) for cid, _, _, s in steps}
    detail = {cid: _detail(t, s) for cid, _, _, s in steps}
    page = _Page(headless_session, wr_page["html"])
    acts = _actions(page, steps[0][3])
    status, results, seconds, actual = {}, {}, {"first_load": page.load_seconds}, {}
    try:
        for cid, name, text, _ in steps:
            t0 = time.perf_counter()
            acts[cid]()
            actual[cid] = _settle(page.pg)
            seconds[f"action_{cid}_s"] = round(time.perf_counter() - t0, 3)
            want, got = desired[cid], actual[cid]
            d_desired = wr_compare(want, got)
            d_shown = wr_compare(shown[cid], got) if d_desired else []
            status[cid] = "PASS" if not d_desired else "KNOWN_ISSUE_RED" if not d_shown else "FAIL"
            results[cid] = {"name": name, "action": text, "status": status[cid], "differences_from_desired":
                            d_desired[:20], "differences_from_as_shown": d_shown[:20],
                            "known_defects": blame[cid] if status[cid] == "KNOWN_ISSUE_RED" else []}
    finally:
        page.close()
    errors = page.unexpected_errors()
    reported = {cid: _reported_errors(desired[cid], actual[cid]) for cid in actual}
    (out / "comparison.json").write_text(json.dumps(
        {"rows": ROWS, "seed": SEED, "results": results, "seconds": seconds, "errors": page.errors,
         "console_allowed_seen": page.console_allowed(), "known_defect_detail": detail,
         "error_bars_reported_not_judged": reported}, indent=1, default=float))
    health = {"scenario": SCENARIO, "function": W.WR_FUNCTION, "rows": ROWS, "seed": SEED,
              "table_sha256": m["table_sha256"], "page_sha256": m["page_sha256"],
              "code_sha256": release.code_sha256(), "git": _git_identity(),
              "run_id": os.environ.get("RI_RELEASE_RUN_ID") or f"local-{uuid.uuid4().hex[:12]}",
              "actions": status,
              "known_defects": {c: ",".join(sorted(blame[c])) for c, s in status.items() if s == "KNOWN_ISSUE_RED"},
              "known_defect_detail": {c: detail[c] for c, s in status.items() if s == "KNOWN_ISSUE_RED"}}
    health["dashboard"] = "PASS" if all(s == "PASS" for s in status.values()) else "RED"
    (out / "dashboard_health.json").write_text(json.dumps(health, indent=1))

    assert len(status) == len(steps)
    failed = {cid: results[cid] for cid, s in status.items() if s == "FAIL"}
    assert not failed, json.dumps(failed, indent=1)[:6000]
    assert wr_compare(desired["C1"], actual["C2"], sections=("comparison",)), "the weight change did not change the page"
    assert not errors, errors
    known_defect(all(s == "KNOWN_ISSUE_RED" and sorted(blame[c]) == sorted(ALL_DEFECTS) for c, s in status.items())
                 and all(detail[c]["WT-D6"]["lost_rows"] == [44894] for c in status), WR_DEFECTS)


def _reported_errors(exp: dict, act: dict) -> dict:
    """The drawn error bars of the medians and stds against their proper candidates (R2-errors): reported only."""
    out = {}
    for src, side in (("histoXYData_1", "analysis"), ("projXYRef_1", "reference")):
        p = exp[side]
        n_eff = p.get("n_eff")
        if n_eff is None:
            continue
        drawn = {}
        for r in act["drawn"]:
            if r["src"].split("#")[0] == src and r["y"] in ("quantile_1", "std"):
                drawn[r["y"]] = np.asarray(r["yv"], float)
        for col, y in drawn.items():
            q = [qq for qq in act["quads"] if qq["src"].split("#")[0] == src and len(qq["top"]) == len(y)
                 and np.allclose((np.asarray(qq["top"], float) + np.asarray(qq["bottom"], float)) / 2, y,
                                 equal_nan=True)]
            if len(q) == 1:
                err = (np.asarray(q[0]["top"], float) - np.asarray(q[0]["bottom"], float)) / 2
                gauss = p["std"] * (np.sqrt(np.pi / 2) if col == "quantile_1" else 1 / np.sqrt(2)) / np.sqrt(n_eff)
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = err / gauss
                out[f"{side} {col}"] = {"drawn_over_gaussian_candidate_median": float(np.nanmedian(r))}
    return out


@BROWSER
@pytest.mark.feature("DASHBOARD.weights_reference")
def test_wr1_control_weighted_counts_and_analysis_side_equal_numpy(wr_page, headless_session):
    """Healthy control: the parts no known defect touches - both sides' weighted counts, the analysis profile (entries,
    mean, std, quantiles) and their drawn series - equal numpy exactly, on the rows the page selects."""
    m = wr_page["manifest"]
    t = oracle.regenerate_table(ROWS, SEED)
    base = _states(m)[0][3]
    page = _Page(headless_session, wr_page["html"])
    try:
        page.tab("histoXYWeight")
        got = _settle(page.pg)
    finally:
        page.close()
    exp = W.expected(t, base, ("WT-D6",))                   # the page's selection; nothing else switched on
    diffs = wr_compare(exp, got, sections=("state", "selection", "histograms", "analysis", "drawn"))
    assert diffs == [], diffs[:10]
    assert not page.unexpected_errors(), page.unexpected_errors()


_FIND_JS = r"""
const H = window.__riH;
const P = H.models().filter(m => m.type === 'HistoNdProfile' && m.name === 'histoXYData_1');
if (P.length !== 1) throw Error('live fault: expected one histoXYData_1');
const R = H.models().filter(m => m.type === 'GlyphRenderer' && m.data_source.name.startsWith('histoXYData_1#entries >= minEntries and True')
                              && m.glyph.type === 'Scatter' && m.glyph.y?.field === 'mean');
if (R.length !== 1) throw Error('live fault: expected one drawn analysis mean on histoXYWeight, found ' + R.length);
window.__riFault = window.__riFault || {};
"""
LIVE_FAULTS = {   # name: (inject, undo, the words the comparator must report)
    "the analysis side weighted by B instead of A": (
        "window.__riFault.w = P[0].weights; P[0].weights = 'B';", "P[0].weights = window.__riFault.w;",
        "analysis (histoXYData_1) mean"),
    "one drawn mean changed in the plot's own source": (
        "const d = R[0].data_source, get = Object.getPrototypeOf(d).get_column; "
        "d.get_column = function (k) { const c = get.call(this, k); if (k !== 'mean' || !c) return c; "
        "const m = Array.from(c); m[5] += 0.01; return m; }; d.change.emit();",
        "const d = R[0].data_source; delete d.get_column; d.change.emit();",
        "drawn histoXYData_1 mean"),
}


@BROWSER
@pytest.mark.feature("DASHBOARD.weights_reference")
def test_wr1_live_faults_in_the_browser_are_caught(wr_page, headless_session):
    """Faults injected into the live page are reported by the same snapshot and comparator; after each is undone the
    page equals the AS-SHOWN oracle again. Control first: the untouched page equals AS SHOWN."""
    m = wr_page["manifest"]
    t = oracle.regenerate_table(ROWS, SEED)
    shown = W.expected(t, _states(m)[0][3], ALL_DEFECTS)
    found = {}
    page = _Page(headless_session, wr_page["html"])
    try:
        page.tab("histoXYWeight")
        control = wr_compare(shown, _settle(page.pg))
        for name, (inject, undo, _) in LIVE_FAULTS.items():
            page.pg.evaluate("() => {" + _FIND_JS + inject + "}")
            faulty = wr_compare(shown, _settle(page.pg))
            page.pg.evaluate("() => {" + _FIND_JS + undo + "}")
            found[name] = (faulty, wr_compare(shown, _settle(page.pg)))
    finally:
        page.close()
    (wr_page["dir"] / "live_faults.json").write_text(json.dumps({"control": control, "faults": {
        n: {"found": f[:10], "after_undo": u[:10]} for n, (f, u) in found.items()}}, indent=1))
    assert control == [], control[:10]
    for name, (faulty, after) in found.items():
        assert any(LIVE_FAULTS[name][2] in d for d in faulty), (name, faulty[:10])
        assert after == [], (name, "the page did not equal AS SHOWN again after the fault was undone", after[:10])
    assert not page.unexpected_errors(), page.unexpected_errors()
