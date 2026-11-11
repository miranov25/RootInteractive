"""PHASE_4_7 Cycle A — RootInteractive's own example dashboard, end to end in Chromium (H01), Claude20.RI.

In plain words: the original function `testBokehClientHistogramOnlyHisto` of test_bokehClientHistogram.py builds its
dashboard on a reproducible table (the opt-in RI_DASHBOARD_ROWS / RI_DASHBOARD_SEED, default 50,000 rows). The test
opens that real page in Chromium, moves the A range slider with the keyboard, opens and closes the "Cumulative" tab,
and restores everything. At each of the five checkpoints it compares, against numpy computed beforehand from the raw
table only (legacy/oracle.py):
- the slider value read back from the page, the active tab, the selected rows (count and exact set);
- all 20 bins of the histogram `histoA`, its cumulative counts and its CDF;
- what is drawn: the visible plot's glyph values, that its data comes from `histoA`, and that the other tab's plots
  are hidden.
The comparator (legacy/compare.py) is the same one the falsifiers below must see fail.

Evidence in headless_out/legacy/h01/: the page and its manifest, expected_*.json, actual_*.json, comparison.json,
dashboard_health.json (per action: PASS or FAIL) and, with RI_DEMO_VIDEO=1, the browser recording and timeline.json
for the narrated video (legacy/video.py). Targets: browser_legacy_dashboards (page fidelity and the browser run) and
headless_unit (oracle, comparator and seam checks, no browser).
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from .legacy import oracle
from .legacy.compare import compare, differs

ROWS = int(os.environ.get("RI_LEGACY_ROWS", "50000"))
SEED = 4711
# H01-D1 depends on the slider's end value, i.e. on the data: it is reproduced with the default table (50,000 rows,
# seed 4711). With another row count (RI_LEGACY_ROWS, for timing runs) its tests do not claim it.
DEFAULT_TABLE = (ROWS, SEED) == (50000, 4711)
K_LOW, K_HIGH = 8, 8                      # keyboard steps: lower handle right, upper handle left
EXAMPLE = Path(__file__).resolve().parents[1] / "test_bokehClientHistogram.py"
ORIGINAL_LINE = "df = pd.DataFrame(np.random.random_sample(size=(500000, 4)), columns=list('ABCD'))"
UNIT = pytest.mark.headless_unit
BROWSER = pytest.mark.browser_legacy_dashboards


class KnownDefect(AssertionError):
    """Raised only when the page shows the exact signature of a registered defect."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


H01_D1_XFAIL = dict(strict=DEFAULT_TABLE, raises=KnownDefect)
H01_D1 = ("H01-D1: moving the A slider's upper handle back to its end with the arrow keys stops 2e-16 short of the "
          "end; the end is exactly the largest A, so that row stays excluded (49,999 of 50,000 rows)")


# ------------------------------------------------------------------ capture (child process, legacy/capture.py)
def _capture(outdir: Path, rows: int, seed: int, graph_only: bool = False) -> dict:
    cmd = [sys.executable, "-m", "RootInteractive.InteractiveDrawing.bokeh.headless.legacy.capture",
           oracle.H01_FUNCTION, str(rows), str(seed), str(outdir)] + (["--graph-only"] if graph_only else [])
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
def h01_page(headless_out):
    out = headless_out / "legacy" / "h01"
    out.mkdir(parents=True, exist_ok=True)
    manifest = _capture(out, ROWS, SEED)
    return {"dir": out, "html": out / "page.html", "manifest": manifest}


# ------------------------------------------------------------------ Python-only checks (headless_unit)
@UNIT
def test_legacy_seam_keeps_the_original_default_line():
    """The opt-in never touches the default path: the original line is still there, under 'neither variable set'."""
    src = EXAMPLE.read_text()
    assert src.count(ORIGINAL_LINE) == 1, "the original 500,000-row line must appear exactly once"
    block = src[src.index("_riRows, _riSeed"):src.index("initMetadata(df)")]
    assert re.search(r"if _riRows is None and _riSeed is None:\n    " + re.escape(ORIGINAL_LINE), block), block
    assert "raise ValueError" in block                       # only one of the two variables: an error, never a guess


@UNIT
def test_h01_oracle_on_a_hand_checked_table():
    """Eight rows, one per chosen bin; checked by hand: bins of width 0.05."""
    a = np.array([0.01, 0.06, 0.11, 0.16, 0.51, 0.56, 0.91, 0.96])
    t = {"A": a, "B": np.full(8, .5), "C": np.full(8, .5), "D": np.full(8, .5)}
    w = {"A": {"type": "RangeSlider", "value": [0.0, 1.0]}}
    full = oracle.h01_expected(t, w, "Histograms", "hand")
    assert full["selected_count"] == 8
    assert np.flatnonzero(full["bin_count"]).tolist() == [0, 1, 2, 3, 10, 11, 18, 19]
    assert full["cumulative"][-1] == 8 and full["cdf"][-1] == 1.0
    cut = oracle.h01_expected(t, oracle.with_a_range(w, 0.1, 0.6), "Cumulative", "hand")
    assert cut["selected_count"] == 4
    assert np.flatnonzero(cut["bin_count"]).tolist() == [2, 3, 10, 11]
    assert cut["cumulative"][:4] == [0, 0, 1, 2] and cut["cdf"][3] == 0.5 and cut["cdf"][11] == 1.0
    assert cut["visible_renderers"] == [["histoA", "cumulative"], ["histoA", "cdf"]]


def _synthetic():
    t = oracle.regenerate_table(2000, 7)
    w = {"A": {"type": "RangeSlider", "start": float(t["A"].min()), "end": float(t["A"].max()),
               "step": float((t["A"].max() - t["A"].min()) / 30), "value": [float(t["A"].min()), float(t["A"].max())]}}
    base = oracle.h01_expected(t, w, "Histograms", "baseline")
    lo, hi = oracle.keyboard_range(w["A"], K_LOW, K_HIGH)
    changed = oracle.h01_expected(t, oracle.with_a_range(w, lo, hi), "Cumulative", "changed")
    return base, changed


def _actual_from(expected: dict, chain=("HistogramCDS", "histoA")) -> dict:
    """A correct actual response, built from an expected one (only for the falsifiers)."""
    rend = []
    for source, yfield in expected["visible_renderers"]:
        rend.append({"source": source, "yfield": yfield, "chain": [["DownsamplerCDS", source], list(chain)],
                     "y": list(expected[yfield]), "n": len(expected[yfield]), "visible": True})
    for source, yfield in expected["hidden_renderers"]:
        rend.append({"source": source, "yfield": yfield, "chain": [["DownsamplerCDS", source], list(chain)],
                     "y": [], "n": 20, "visible": False})
    return {"widgets": copy.deepcopy(expected["widgets"]), "tab": expected["tab"],
            "selected_count": expected["selected_count"], "selected_sha256": expected["selected_sha256"],
            "edges": list(expected["edges"]), "bin_count": list(expected["bin_count"]),
            "cumulative": list(expected["cumulative"]), "cdf": list(expected["cdf"]), "renderers": rend}


def _mutations(base, changed):
    def m_bin(a):
        a["bin_count"][7] += 1
    def m_redistribute(a):
        i = int(np.flatnonzero(np.asarray(a["bin_count"]) > 0)[0])
        a["bin_count"][i] -= 1
        a["bin_count"][i + 1] += 1
    def m_swapped_row(a):
        a["selected_sha256"] = oracle.mask_sha256(np.r_[True, np.zeros(1999, bool)])    # same count kept below
    def m_stale(a):
        for k in ("selected_count", "selected_sha256", "bin_count", "cumulative", "cdf"):
            a[k] = copy.deepcopy(base[k])
    def m_wrong_source(a):
        for r in a["renderers"]:
            r["chain"] = [["DownsamplerCDS", r["source"]], ["HistogramCDS", "histoB"]]
    def m_view_not_updated(a):
        for r in a["renderers"]:
            if r["visible"]:
                r["y"] = list(base[r["yfield"]]) if r["yfield"] in base else r["y"]
    def m_hidden_plot(a):
        a["renderers"][0]["visible"] = False
    def m_wrong_restore(a):
        a["widgets"]["A"] = [a["widgets"]["A"][0] + 0.03, a["widgets"]["A"][1]]
    def m_cdf(a):
        a["cdf"][-1] *= 1 + 1e-9
    def m_missing(a):
        del a["cumulative"]
    return {"one bin +1": (m_bin, "histoA.bin_count"), "two bins redistributed": (m_redistribute, "histoA.bin_count"),
            "one selected row swapped, same count": (m_swapped_row, "selected rows differ"),
            "stale after the slider change": (m_stale, "selection:"),
            "plot bound to another source": (m_wrong_source, "does not come from HistogramCDS histoA"),
            "plot not redrawn": (m_view_not_updated, "(drawn values)"), "plot hidden": (m_hidden_plot, "not visible"),
            "wrong slider value": (m_wrong_restore, "widget A"), "cdf off by 1e-9": (m_cdf, "histoA.cdf"),
            "field missing": (m_missing, "missing from the actual response")}


@UNIT
def test_h01_comparator_accepts_the_right_answer():
    base, changed = _synthetic()
    assert compare(changed, _actual_from(changed)) == []
    assert compare(base, _actual_from(base)) == []
    assert differs(base, changed)


@UNIT
@pytest.mark.parametrize("name", list(_mutations(*_synthetic()).keys()))
def test_h01_comparator_rejects_a_wrong_answer(name):
    base, changed = _synthetic()
    mutate, words = _mutations(base, changed)[name]
    actual = _actual_from(changed)
    mutate(actual)
    found = compare(changed, actual)
    assert found and any(words in f for f in found), (name, found)


# ------------------------------------------------------------------ the page itself (browser_legacy_dashboards)
@BROWSER
@pytest.mark.feature("DASHBOARD.histogram_only")
def test_h01_page_is_the_original_dashboard_on_the_reproducible_table(h01_page, headless_out):
    """The captured page has exactly the structure of the original 500,000-row page (models, widgets, tabs, plots),
    and its table is the one the oracle regenerates from (rows, seed)."""
    m = h01_page["manifest"]
    assert m["opt_in"] and m["rows"] == ROWS and m["seed"] == SEED
    t = oracle.regenerate_table(ROWS, SEED)
    assert oracle.table_sha256(t) == m["table_sha256"], "the page was not built from the oracle's table"
    original = _capture(headless_out / "legacy" / "h01_original_500k", 0, 0, graph_only=True)
    assert not original["opt_in"] and original["rows"] == 500_000
    assert original["graph"] == m["graph"], "the 50,000-row page differs in structure from the original page"
    assert m["graph"]["tabs"] == [["Histograms", "Cumulative"]]
    assert ["HistogramCDS", "histoA", 20] in m["graph"]["histograms"]
    assert m["widget_state"]["A"]["type"] == "RangeSlider"


# ------------------------------------------------------------------ the browser run
SNAPSHOT_JS = r"""() => {
  const H = window.__riH;
  const slider = H.models().filter(m => m.type === 'RangeSlider' && m.title === 'A');
  const tabs = H.models().filter(m => m.type === 'LazyTabs');
  const hist = H.model('histoA', ['HistogramCDS']);
  const filters = H.models().filter(m => m.type === 'LazyIntersectionFilter');
  const orig = H.model('cdsOrig');
  if (slider.length !== 1 || tabs.length !== 1 || !hist || filters.length !== 1 || !orig)
    throw Error('H01 page: expected one A slider, one LazyTabs, histoA, one filter, cdsOrig');
  const n = orig.get_length(), v = filters[0].v_compute();
  let mask = '';
  for (let i = 0; i < n; i++) mask += v[i] ? '1' : '0';
  const renderers = [], seen = new Set();
  const kids = x => !x ? [] : (x instanceof Map ? Array.from(x.values()) : Array.from(x));
  const walk = view => {
    if (!view || seen.has(view)) return; seen.add(view);
    if (view.model && view.model.type === 'GlyphRenderer' && view.model.glyph.type === 'Scatter') {
      const chain = []; let m = view.model.data_source, guard = 0;
      while (m && guard++ < 10) { chain.push([m.type, m.name]); m = m.source ?? (m.sources ? m.sources[0] : null); }
      const canvas = view.parent?.canvas_view?.el;
      renderers.push({source: view.model.data_source.name, yfield: view.model.glyph.y?.field ?? null, chain,
                      y: view.glyph._y ? Array.from(view.glyph._y) : null, n: view.glyph.data_size,
                      visible: canvas ? canvas.checkVisibility({visibilityProperty: true, opacityProperty: true}) : null});
    }
    for (const c of kids(view.child_views)) walk(c);
    for (const c of kids(view.renderer_views)) walk(c);
  };
  for (const view of (Bokeh.index instanceof Map ? Bokeh.index.values() : Object.values(Bokeh.index))) walk(view);
  const bottom = Array.from(hist.get_column('bin_bottom')), top = Array.from(hist.get_column('bin_top'));
  return {widgets: {A: Array.from(slider[0].value)}, tab: tabs[0].tabs[tabs[0].active].title, mask,
          edges: bottom.concat([top[top.length - 1]]), bin_count: Array.from(hist.get_column('bin_count')),
          cumulative: Array.from(hist.get_column('cumulative')), cdf: Array.from(hist.get_column('cdf')), renderers};
}"""


def _actual(pg) -> dict:
    raw = pg.evaluate(SNAPSHOT_JS)
    mask = np.frombuffer(raw.pop("mask").encode(), dtype=np.uint8) == ord("1")
    raw["selected_count"] = int(mask.sum())
    raw["selected_sha256"] = oracle.mask_sha256(mask)
    raw["renderers"] = [r for r in raw["renderers"] if r["yfield"] in ("bin_count", "cumulative", "cdf")
                        and r["source"] in ("histo1Ds", "histoA")]
    return raw


def _settle(pg, timeout_s: float = 20.0) -> dict:
    """Wait until Bokeh is idle and two reads 150 ms apart are identical; return that read."""
    deadline = time.monotonic() + timeout_s
    pg.wait_for_function("() => Bokeh.documents[0].is_idle", timeout=int(timeout_s * 1000))
    last = _actual(pg)
    while time.monotonic() < deadline:
        pg.wait_for_timeout(150)
        now = _actual(pg)
        if now == last:
            return now
        last = now
    raise AssertionError("H01: the page did not settle")


class _Timeline:
    """Checkpoint times for the narrated video, measured from the start of the browser recording."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.events = []

    def add(self, **event):
        self.events.append(dict(t=round(time.monotonic() - self.t0, 3), **event))


def _overlay(exp: dict, act: dict, diffs: list, status: str) -> list:
    lines = [f"selected rows: expected {exp['selected_count']:,}  actual {act['selected_count']:,}",
             f"histoA bins: expected {sum(exp['bin_count']):,} entries, actual {sum(act['bin_count']):,}; "
             f"{'all 20 bins equal' if not any('bin_count' in d for d in diffs) else 'bins differ'}"]
    if exp["tab"] == "Cumulative":
        lines.append(f"cumulative last bin: expected {exp['cumulative'][-1]:,}  actual {act['cumulative'][-1]:,};  "
                     f"cdf last bin: expected {exp['cdf'][-1]:.3f}  actual {act['cdf'][-1]:.3f}")
    if status == "PASS":
        lines.append("check: PASS")
    elif status == "KNOWN_ISSUE_RED":
        lines.append(f"check: KNOWN ISSUE H01-D1: slider stops {exp['widgets']['A'][1] - act['widgets']['A'][1]:.1e} "
                     f"short of its end; {exp['selected_count'] - act['selected_count']} row excluded")
    else:
        lines.append(f"check: FAIL ({len(diffs)} differences): {diffs[0][:90]}")
    return lines


class _Page:
    """The captured page in a fresh Chromium context: pinned local scripts only, errors recorded, the A slider's two
    keyboard handles and the tabs located (shadow-aware Playwright locators)."""

    def __init__(self, session, html: Path, video_dir: Path = None):
        from .engine import vendor
        from .engine.runner import BROWSER_LOAD_CONDITION, HELPER_JS
        routes = {g.replace("**/", "/"): p for g, p in vendor.routes().items()}
        self.errors = {"page": [], "console": [], "blocked": []}
        args = {"viewport": {"width": 1280, "height": 720}}
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
            args.update(record_video_dir=str(video_dir), record_video_size={"width": 1280, "height": 720})
        self.ctx = session.browser.new_context(**args)
        self.pg = self.ctx.new_page()
        self.timeline = _Timeline()
        self.pg.on("pageerror", lambda e: self.errors["page"].append(str(e)[:300]))
        self.pg.on("console", lambda m: self.errors["console"].append(m.text[:300]) if m.type == "error" else None)

        def route(r):
            url = r.request.url
            if url.startswith("file:"):
                return r.continue_()
            for suffix, path in routes.items():
                if url.split("?")[0].endswith(suffix):
                    return r.fulfill(path=str(path), content_type="text/javascript")
            self.errors["blocked"].append(url)
            return r.abort()
        self.pg.route("**/*", route)
        self.pg.add_init_script(HELPER_JS)
        t0 = time.perf_counter()
        self.pg.goto(html.resolve().as_uri())
        self.pg.wait_for_function(BROWSER_LOAD_CONDITION, timeout=120_000)
        self.load_seconds = round(time.perf_counter() - t0, 3)
        slider = self.pg.locator(".bk-RangeSlider").filter(
            has=self.pg.locator(".bk-slider-title", has_text=re.compile(r"^A:")))
        assert slider.count() == 1, "exactly one A range slider on the page"
        self.handles = slider.locator('.noUi-handle[role="slider"]')
        assert self.handles.count() == 2

    def keys(self, handle: int, key: str, n: int = 1):
        self.handles.nth(handle).focus()
        for _ in range(n):
            self.pg.keyboard.press(key)

    def tab(self, title: str):
        self.pg.locator(".bk-tab").filter(has_text=re.compile(rf"^{title}$")).click()

    def drag_upper_handle_far_right(self):
        box = self.handles.nth(1).bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.pg.mouse.move(x, y)
        self.pg.mouse.down()
        self.pg.mouse.move(x + 900, y, steps=10)
        self.pg.mouse.up()

    def unexpected_errors(self) -> dict:
        from .engine.runner import CONSOLE_ALLOWED
        console = [e for e in self.errors["console"] if not any(re.search(p, e) for p, _ in CONSOLE_ALLOWED)]
        return {k: v for k, v in {"page": self.errors["page"], "console": console,
                                  "blocked": self.errors["blocked"]}.items() if v}

    def close(self):
        self.ctx.close()


def _is_h01_d1(t, widgets, expected: dict, actual: dict, span: float) -> bool:
    """H01-D1, exact: the lower end is exactly the slider's start, the upper end sits 1 to 4 float steps below the
    slider's end; the page is exactly right for the value it shows (selection, all bins, cumulative, cdf, drawn plots);
    the rows it lost are exactly those with A above the shown value, at least one (the largest A)."""
    a = widgets["A"]
    lo, hi = actual["widgets"]["A"]
    if lo != a["start"] or not 0 < a["end"] - hi <= 4 * np.spacing(a["end"]):
        return False
    shown = oracle.h01_expected(t, oracle.with_a_range(widgets, lo, hi), expected["tab"], expected["checkpoint"],
                                check_margin=False)
    if compare(shown, actual, widget_span=span):
        return False
    lost = expected["selected_count"] - actual["selected_count"]
    return lost >= 1 and lost == int(((t["A"] > hi) & (t["A"] <= a["end"])).sum())


def _run_plan(page: _Page, plan, expected, span: float, out: Path = None):
    """Do each action, wait for the page to settle, read it, compare; return (actual, results, seconds)."""
    actual, results, seconds = {}, {}, {"first_load": page.load_seconds}
    for cid, name, action_text, act in plan:
        page.timeline.add(kind="action", id=cid, scene=name, action=action_text)
        t_act = time.perf_counter()
        act()
        actual[cid] = _settle(page.pg)
        seconds[f"settle_{cid}"] = round(time.perf_counter() - t_act, 3)
        results[cid] = compare(expected[cid], actual[cid], widget_span=span)
        if out is not None:
            (out / f"actual_{cid}.json").write_text(json.dumps(actual[cid], indent=1))
    return actual, results, seconds


@BROWSER
@pytest.mark.feature("DASHBOARD.histogram_only")
@pytest.mark.xfail(**H01_D1_XFAIL, reason=H01_D1)
def test_h01_histogram_dashboard_follows_the_A_range_and_the_cumulative_tab_and_restores(h01_page, headless_session):
    """The whole H01 workflow, six checkpoints. It is H01-D1's strict partner: checkpoint C4 expects the full range
    back after the arrow keys; KnownDefect is raised only after every checkpoint was compared and only when C4's (and
    nothing else's) difference is exactly H01-D1."""
    out, m = h01_page["dir"], h01_page["manifest"]
    widgets = m["widget_state"]
    a = widgets["A"]
    span = a["end"] - a["start"]
    lo, hi = oracle.keyboard_range(a, K_LOW, K_HIGH)
    changed = oracle.with_a_range(widgets, lo, hi)

    # 1. every expected response, from the raw table only, before the browser is opened
    t = oracle.regenerate_table(ROWS, SEED)
    assert oracle.table_sha256(t) == m["table_sha256"]
    steps = [("C1", "baseline", "load the original dashboard", widgets, "Histograms"),
             ("C2", "A range changed", f"arrow keys: lower handle {K_LOW} steps right, upper handle {K_HIGH} steps left",
              changed, "Histograms"),
             ("C3", "Cumulative tab", "click the Cumulative tab", changed, "Cumulative"),
             ("C4", "A range restored with the arrow keys", "arrow keys: both handles back to their ends",
              widgets, "Cumulative"),
             ("C5", "upper end with the End key", "End key on the upper handle", widgets, "Cumulative"),
             ("C6", "Histograms tab restored", "click the Histograms tab", widgets, "Histograms")]
    expected = {cid: oracle.h01_expected(t, w, tab, name) for cid, name, _, w, tab in steps}
    for cid, e in expected.items():
        (out / f"expected_{cid}.json").write_text(json.dumps(e, indent=1))
    assert differs(expected["C1"], expected["C2"]), "the planned change would not change the histogram"

    # 2. the real page in Chromium, optionally recorded for the narrated video
    record = os.environ.get("RI_DEMO_VIDEO") == "1"
    page = _Page(headless_session, h01_page["html"], out / "video" if record else None)
    acts = {"C1": lambda: None,
            "C2": lambda: (page.keys(0, "ArrowRight", K_LOW), page.keys(1, "ArrowLeft", K_HIGH)),
            "C3": lambda: page.tab("Cumulative"),
            "C4": lambda: (page.keys(0, "ArrowLeft", K_LOW), page.keys(1, "ArrowRight", K_HIGH)),
            "C5": lambda: page.keys(1, "End"),
            "C6": lambda: page.tab("Histograms")}
    plan = [(cid, name, text, acts[cid]) for cid, name, text, _, _ in steps]
    video = None
    try:
        actual, results, seconds = {}, {}, {"first_load": page.load_seconds}
        for one in plan:
            act_one, res_one, sec_one = _run_plan(page, [one], expected, span, out)
            actual.update(act_one); results.update(res_one); seconds.update({k: v for k, v in sec_one.items()
                                                                              if k != "first_load"})
            cid = one[0]
            status = ("PASS" if not results[cid] else
                      "KNOWN_ISSUE_RED" if _is_h01_d1(t, widgets, expected[cid], actual[cid], span) else "FAIL")
            page.timeline.add(kind="checkpoint", id=cid, scene=one[1], action=one[2], status=status,
                              overlay=_overlay(expected[cid], actual[cid], results[cid], status),
                              expected={k: expected[cid][k] for k in ("selected_count", "tab")},
                              actual={k: actual[cid][k] for k in ("selected_count", "tab")})
            if record:                         # video only: keep the checked state on screen before the next action
                page.pg.wait_for_timeout(800)
        page.timeline.add(kind="end")
    finally:
        page.close()
        if record:
            vids = sorted((out / "video").glob("*.webm"), key=lambda p: p.stat().st_mtime)
            video = vids[-1].name if vids else None
    errors = page.unexpected_errors()

    # 3. classification and evidence (comparison, per-action health, timeline)
    status = {c["id"]: c["status"] for c in page.timeline.events if c["kind"] == "checkpoint"}
    nonvacuous = differs(actual["C1"], actual["C2"])
    comparison = {"rows": ROWS, "seed": SEED, "a_slider": a, "a_changed": [lo, hi], "results": results,
                  "status": status, "non_vacuity": nonvacuous, "errors": page.errors, "seconds": seconds,
                  "page_bytes": m["page_bytes"], "capture_seconds": m["seconds"]}
    (out / "comparison.json").write_text(json.dumps(comparison, indent=1))
    health = {"scenario": "H01", "function": oracle.H01_FUNCTION, "actions": status,
              "known_defects": {c: "H01-D1" for c, s in status.items() if s == "KNOWN_ISSUE_RED"}}
    health["dashboard"] = "PASS" if all(s == "PASS" for s in status.values()) else "RED"
    (out / "dashboard_health.json").write_text(json.dumps(health, indent=1))
    if record:
        (out / "timeline.json").write_text(json.dumps({"scenario": "H01", "video": video, "rows": ROWS, "seed": SEED,
                                                       "events": page.timeline.events, "health": health}, indent=1))

    # 4. the verdict: everything but the exact H01-D1 state at C4 must be right
    assert len(results) == len(plan)
    failed = {cid: results[cid] for cid, s in status.items() if s == "FAIL"}
    assert not failed, json.dumps(failed, indent=1)[:4000]
    assert all(s == "PASS" for c, s in status.items() if c != "C4"), status
    assert nonvacuous, "the A range change did not change the page"
    assert not errors, errors
    known_defect(status["C4"] == "KNOWN_ISSUE_RED", H01_D1)


@BROWSER
@pytest.mark.feature("DASHBOARD.histogram_only")
def test_signature_h01_d1_arrow_keys_back_to_the_end_lose_the_largest_A(h01_page, headless_session):
    """H01-D1, exact, in its smallest form: one arrow step left and one right on the upper handle."""
    if not DEFAULT_TABLE:
        pytest.skip("H01-D1 is reproduced with the default table (50,000 rows, seed 4711) only")
    m = h01_page["manifest"]
    widgets, t = m["widget_state"], oracle.regenerate_table(ROWS, SEED)
    a = widgets["A"]
    full = oracle.h01_expected(t, widgets, "Histograms", "full")
    page = _Page(headless_session, h01_page["html"])
    try:
        page.keys(1, "ArrowLeft")
        _settle(page.pg)
        page.keys(1, "ArrowRight")
        got = _settle(page.pg)
    finally:
        page.close()
    assert _is_h01_d1(t, widgets, full, got, a["end"] - a["start"]), (got["widgets"], got["selected_count"])
    assert full["selected_count"] - got["selected_count"] == 1 and got["widgets"]["A"][1] < a["end"]
    assert not page.unexpected_errors(), page.unexpected_errors()


@BROWSER
@pytest.mark.feature("DASHBOARD.histogram_only")
def test_control_h01_d1_end_key_and_mouse_drag_restore_the_full_range(h01_page, headless_session):
    """H01-D1's control: the same move back with the End key, or by dragging the handle past the end with the mouse,
    reaches the slider's end exactly; every row and every bin is back."""
    m = h01_page["manifest"]
    widgets, t = m["widget_state"], oracle.regenerate_table(ROWS, SEED)
    a = widgets["A"]
    span = a["end"] - a["start"]
    full = oracle.h01_expected(t, widgets, "Histograms", "full")
    page = _Page(headless_session, h01_page["html"])
    try:
        page.keys(1, "ArrowLeft", K_HIGH)
        moved = _settle(page.pg)
        page.keys(1, "End")
        by_key = _settle(page.pg)
        page.keys(1, "ArrowLeft", K_HIGH)
        _settle(page.pg)
        page.drag_upper_handle_far_right()
        by_mouse = _settle(page.pg)
    finally:
        page.close()
    assert moved["selected_count"] < full["selected_count"]
    assert compare(full, by_key, widget_span=span) == [], compare(full, by_key, widget_span=span)
    assert compare(full, by_mouse, widget_span=span) == [], compare(full, by_mouse, widget_span=span)
    assert by_key["widgets"]["A"][1] == a["end"] and by_mouse["widgets"]["A"][1] == a["end"]
    assert not page.unexpected_errors(), page.unexpected_errors()
