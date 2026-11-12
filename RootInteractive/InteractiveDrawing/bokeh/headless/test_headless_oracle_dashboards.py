"""PHASE_4_7 — the oracle test dashboards: RootInteractive's own example dashboard H01, end to end in Chromium
(Claude20.RI; Cycle A, revised 2026-11-12 after GPT24's consolidated review).

In plain words: the original function `testBokehClientHistogramOnlyHisto` of test_bokehClientHistogram.py builds its
dashboard on a reproducible table (the opt-in RI_DASHBOARD_ROWS / RI_DASHBOARD_SEED; here 50,000 rows, seed 4711).
The test opens that real page in Chromium, moves the A range slider with the keyboard, opens and closes the
"Cumulative" tab, and restores everything. At each of the six checkpoints it compares the page with numpy values
computed beforehand from the raw table only (dashboards/oracle.py):
- the slider value read back from the page, the active tab, the selected rows (count and exact set);
- all 20 bins of the histogram `histoA`, its cumulative counts and its CDF;
- what is drawn: every checked plot present exactly once, visible or hidden as its tab requires; for a visible plot
  its data source, x field and x values (bin centres), y values, axis labels and title, and the screen coordinates
  Bokeh computed for each point, anchored to the plot's frame and axis ranges (not a pixel check).
The comparator (dashboards/compare.py) is the same one the falsifiers below must see fail: on a snapshot, and on the
live page in the browser.

Known defect H01-D1: after the A slider's upper handle is moved back to its end with the arrow keys it stops a few
float steps short of the end, so the row with the largest A stays excluded. Reported in two lanes (decision DEC-05):
the workflow test is its strict expected failure, and dashboard_health.json marks C4 KNOWN_ISSUE_RED and the dashboard
RED. The release check (dashboards/release.py, target release_oracle_dashboards) checks that the health file comes from
this run, this test code, this dashboard, table and page, and fails on a RED dashboard unless the Architect has
accepted the bug for release at its exact checkpoint, with its signature (dashboards/waivers.json).

Evidence in headless_out/dashboards/h01/: the page and its manifest, expected_*.json, actual_*.json, comparison.json
(results, cold and warm times, console messages), dashboard_health.json, inventory.json (every widget and plot of the
page, and which ones H01 checks) and, with RI_DEMO_VIDEO=1, the recording and timeline.json for the narrated video
(dashboards/video.py). Targets: browser_oracle_dashboards (the page in Chromium), headless_unit (oracle, comparator,
release check and seam, no browser), release_oracle_dashboards (the release check, run only as its own target).
"""
from __future__ import annotations

import copy
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
from .dashboards.compare import compare, differs

ROWS = int(os.environ.get("RI_DASHBOARD_TEST_ROWS", "50000"))
SEED = 4711
# H01-D1 depends on the slider's end value, i.e. on the data: it is reproduced with the default table (50,000 rows,
# seed 4711). With another row count (RI_DASHBOARD_TEST_ROWS, for timing runs) its tests do not claim it.
DEFAULT_TABLE = (ROWS, SEED) == (50000, 4711)
K_LOW, K_HIGH = 8, 8                      # keyboard steps: lower handle right, upper handle left
EXAMPLE = Path(__file__).resolve().parents[1] / "test_bokehClientHistogram.py"
ORIGINAL_LINE = "df = pd.DataFrame(np.random.random_sample(size=(500000, 4)), columns=list('ABCD'))"
UNIT = pytest.mark.headless_unit
BROWSER = pytest.mark.browser_oracle_dashboards
RELEASE = pytest.mark.release_oracle_dashboards
CHECKED_PLOTS = {("histo1Ds", "bin_count"), ("histoA", "cumulative"), ("histoA", "cdf")}
CHECKED_WIDGETS = {"A"}


class KnownDefect(AssertionError):
    """Raised only when the page shows the exact signature of a registered defect."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


H01_D1_XFAIL = dict(strict=DEFAULT_TABLE, raises=KnownDefect)
H01_D1 = ("H01-D1: moving the A slider's upper handle back to its end with the arrow keys stops 2e-16 short of the "
          "end; the end is exactly the largest A, so that row stays excluded (49,999 of 50,000 rows)")


# ------------------------------------------------------------------ capture (child process, dashboards/capture.py)
def _capture(outdir: Path, rows: int, seed: int, graph_only: bool = False) -> dict:
    cmd = [sys.executable, "-m", "RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.capture",
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
    out = headless_out / "dashboards" / "h01"
    out.mkdir(parents=True, exist_ok=True)
    manifest = _capture(out, ROWS, SEED)
    return {"dir": out, "html": out / "page.html", "manifest": manifest}


# ------------------------------------------------------------------ Python-only checks (headless_unit)
@UNIT
def test_dashboard_seam_keeps_the_original_default_line():
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
    assert np.allclose(full["plots"]["histo1Ds.bin_count"]["x"][:3], [0.025, 0.075, 0.125], rtol=1e-12, atol=0)
    assert full["plots"]["histo1Ds.bin_count"]["xlabel"] == "bin_center"
    cut = oracle.h01_expected(t, oracle.with_a_range(w, 0.1, 0.6), "Cumulative", "hand")
    assert cut["selected_count"] == 4
    assert np.flatnonzero(cut["bin_count"]).tolist() == [2, 3, 10, 11]
    assert cut["cumulative"][:4] == [0, 0, 1, 2] and cut["cdf"][3] == 0.5 and cut["cdf"][11] == 1.0
    assert cut["visible_renderers"] == [["histoA", "cumulative"], ["histoA", "cdf"]]
    assert cut["plots"]["histoA.cdf"]["xlabel"] == "A (cm)" and cut["plots"]["histoA.cdf"]["title"] == "cdf vs A (cm)"


def _synthetic():
    t = oracle.regenerate_table(2000, 7)
    w = {"A": {"type": "RangeSlider", "start": float(t["A"].min()), "end": float(t["A"].max()),
               "step": float((t["A"].max() - t["A"].min()) / 30), "value": [float(t["A"].min()), float(t["A"].max())]}}
    base = oracle.h01_expected(t, w, "Histograms", "baseline")
    lo, hi = oracle.keyboard_range(w["A"], K_LOW, K_HIGH)
    changed = oracle.h01_expected(t, oracle.with_a_range(w, lo, hi), "Cumulative", "changed")
    return base, changed


FRAME = {"left": 75.0, "right": 635.0, "top": 27.0, "bottom": 162.0}


def _ranges(x, y):
    pad = lambda lo, hi: 0.05 * ((hi - lo) or 1.0)
    return ([min(x) - pad(min(x), max(x)), max(x) + pad(min(x), max(x))],
            [min(y) - pad(min(y), max(y)), max(y) + pad(min(y), max(y))])


def _screen(x, y, xr, yr, frame=FRAME):
    """The coordinates the plot's frame and ranges give: x from the left edge to the right, y from the bottom up."""
    w, h = frame["right"] - frame["left"], frame["bottom"] - frame["top"]
    return ([frame["left"] + (v - xr[0]) / (xr[1] - xr[0]) * w for v in x],
            [frame["bottom"] - (v - yr[0]) / (yr[1] - yr[0]) * h for v in y])


def _actual_from(expected: dict, chain=("HistogramCDS", "histoA")) -> dict:
    """A correct actual response, built from an expected one (only for the falsifiers)."""
    rend = []
    for source, yfield in expected["visible_renderers"]:
        spec = expected["plots"][f"{source}.{yfield}"]
        x, y = list(spec["x"]), list(expected[yfield])
        xr, yr = _ranges(x, y)
        sx, sy = _screen(x, y, xr, yr)
        rend.append({"source": source, "yfield": yfield, "xfield": spec["xfield"],
                     "chain": [["DownsamplerCDS", source], list(chain)], "x": x, "y": y, "sx": sx, "sy": sy,
                     "frame": dict(FRAME), "x_range": xr, "y_range": yr, "x_scale": "LinearScale",
                     "y_scale": "LinearScale",
                     "n": len(y), "canvas_visible": True, "model_visible": True, "visible": True,
                     "xlabel": spec["xlabel"], "ylabel": " " + spec["ylabel"], "title": " " + spec["title"]})
    for source, yfield in expected["hidden_renderers"]:
        rend.append({"source": source, "yfield": yfield, "xfield": "bin_center",
                     "chain": [["DownsamplerCDS", source], list(chain)], "x": [], "y": [], "sx": [], "sy": [],
                     "n": 20, "canvas_visible": False, "model_visible": True, "visible": False,
                     "xlabel": None, "ylabel": None, "title": None})
    return {"widgets": copy.deepcopy(expected["widgets"]), "tab": expected["tab"],
            "selected_count": expected["selected_count"], "selected_sha256": expected["selected_sha256"],
            "edges": list(expected["edges"]), "bin_count": list(expected["bin_count"]),
            "cumulative": list(expected["cumulative"]), "cdf": list(expected["cdf"]), "renderers": rend}


def _visible(a):
    return [r for r in a["renderers"] if r["visible"]]


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
        for r in _visible(a):
            r["y"] = list(base[r["yfield"]]) if r["yfield"] in base else r["y"]
    def m_hidden_plot(a):
        a["renderers"][0]["visible"] = False
    def m_wrong_restore(a):
        a["widgets"]["A"] = [a["widgets"]["A"][0] + 0.03, a["widgets"]["A"][1]]
    def m_cdf(a):
        a["cdf"][-1] *= 1 + 1e-9
    def m_missing(a):
        del a["cumulative"]
    # revision 2026-11-12 (GPT24 review G1, G2): hidden plots, x, labels, painting
    def m_hidden_missing(a):
        a["renderers"] = [r for r in a["renderers"] if r["visible"]]
    def m_hidden_twice(a):
        a["renderers"].append(copy.deepcopy([r for r in a["renderers"] if not r["visible"]][0]))
    def m_unexpected_plot(a):
        a["renderers"].append(dict(copy.deepcopy(a["renderers"][0]), yfield="bin_count", source="histoA"))
    def m_x_shifted(a):
        r = _visible(a)[0]
        r["x"] = [v + 0.025 for v in r["x"]]
    def m_x_field(a):
        _visible(a)[0]["xfield"] = "bin_bottom"
    def m_x_label(a):
        _visible(a)[0]["xlabel"] = "B (cm/s)"
    def m_title(a):
        _visible(a)[0]["title"] = "cdf vs B (cm/s)"
    def m_switched_off(a):
        r = _visible(a)[0]
        r["model_visible"] = False
        r["visible"] = False
    def m_paint_stale(a):                       # the coordinates of the old values, in the same frame and ranges
        r = _visible(a)[0]
        r["sy"] = _screen(r["x"], base[r["yfield"]], r["x_range"], r["y_range"])[1]
    def m_paint_upside_down(a):
        r = _visible(a)[0]
        r["sy"] = [FRAME["top"] + FRAME["bottom"] - v for v in r["sy"]]
    # revision 2026-11-12 (GPT25's summary, H-03): moves that keep a straight-line image but leave the frame anchor
    def m_shifted_20px(a):
        r = _visible(a)[0]
        r["sx"] = [v + 20.0 for v in r["sx"]]
    def m_scaled(a):
        r = _visible(a)[0]
        r["sy"] = [FRAME["top"] + 0.9 * (v - FRAME["top"]) for v in r["sy"]]
    def m_range_cuts_points(a):                 # consistent coordinates, but the x range hides half the points
        r = _visible(a)[0]
        r["x_range"] = [0.5, r["x_range"][1]]
        r["sx"] = _screen(r["x"], r["y"], r["x_range"], r["y_range"])[0]
    def m_log_axis(a):
        _visible(a)[0]["y_scale"] = "LogScale"
    return {"one bin +1": (m_bin, "histoA.bin_count"), "two bins redistributed": (m_redistribute, "histoA.bin_count"),
            "one selected row swapped, same count": (m_swapped_row, "selected rows differ"),
            "stale after the slider change": (m_stale, "selection:"),
            "plot bound to another source": (m_wrong_source, "does not come from HistogramCDS histoA"),
            "plot not redrawn": (m_view_not_updated, "(drawn values)"), "plot hidden": (m_hidden_plot, "not visible"),
            "wrong slider value": (m_wrong_restore, "widget A"), "cdf off by 1e-9": (m_cdf, "histoA.cdf"),
            "field missing": (m_missing, "missing from the actual response"),
            "hidden plots missing": (m_hidden_missing, "found 0 glyph renderers on the hidden tab"),
            "hidden plot twice": (m_hidden_twice, "found 2 glyph renderers on the hidden tab"),
            "unexpected plot of a checked source": (m_unexpected_plot, "not expected"),
            "x shifted by half a bin": (m_x_shifted, "(drawn x)"),
            "x drawn from another field": (m_x_field, "draws x from 'bin_bottom'"),
            "wrong x axis label": (m_x_label, "xlabel"), "wrong title": (m_title, "title"),
            "renderer switched off": (m_switched_off, "renderer visible: False"),
            "drawing not updated": (m_paint_stale, "(screen y)"),
            "drawn upside down": (m_paint_upside_down, "(screen y)"),
            "every point shifted by 20 px": (m_shifted_20px, "(screen x)"),
            "drawing scaled by 0.9": (m_scaled, "(screen y)"),
            "axis range cuts points off": (m_range_cuts_points, "outside the plot frame"),
            "logarithmic axis": (m_log_axis, "only LinearScale")}


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


# ------------------------------------------------------------------ the release check (headless_unit)
H01 = release.REQUIRED["H01"]
GREEN = {c: "PASS" for c in H01["actions"]}
RED_C4 = dict(GREEN, C4="KNOWN_ISSUE_RED")
LOST = {"C4": {"lost_rows": [44894], "row_with_largest_A": 44894}}
TODAY = __import__("datetime").date(2026, 11, 12)
WAIVER_C4 = {"scenario": "H01", "checkpoint": "C4", "bug": "H01-D1", "signature": {"lost_rows": [44894]},
             "decision": "DEC-99", "date": "2026-11-12", "by": "Architect", "expires": "2026-12-31"}


def _table_hash():
    return oracle.table_sha256(oracle.regenerate_table(H01["rows"], H01["seed"]))


def _health(statuses: dict, bugs: dict = None, dashboard: str = None, scenario: str = "H01", **identity) -> dict:
    bugs = bugs or {}
    dash = dashboard or ("PASS" if all(s == "PASS" for s in statuses.values()) and statuses else "RED")
    h = {"scenario": scenario, "function": H01["function"], "rows": H01["rows"], "seed": H01["seed"],
         "table_sha256": _TABLE_HASH, "page_sha256": None, "code_sha256": release.code_sha256(), "run_id": "run-1",
         "actions": statuses, "known_defects": bugs,
         "known_defect_detail": {c: LOST[c] for c in bugs if c in LOST}, "dashboard": dash}
    h.update(identity)
    return h


def _write(tmp_path, *healths, manifest=None):
    """Each health file in its own folder, with the manifest.json and page.html a real run leaves next to it."""
    files = []
    for i, h in enumerate(healths):
        d = tmp_path / f"run{i}"
        d.mkdir()
        page = d / "page.html"
        page.write_text("<html>a page</html>")
        h = dict(h)
        if h.get("page_sha256") is None:
            h["page_sha256"] = __import__("hashlib").sha256(page.read_bytes()).hexdigest()
        m = {k: h.get(k) for k in ("function", "rows", "seed", "table_sha256", "page_sha256")}
        m.update(manifest or {})
        (d / "manifest.json").write_text(json.dumps(m))
        (d / "dashboard_health.json").write_text(json.dumps(h))
        files.append(d / "dashboard_health.json")
    return files


_TABLE_HASH = _table_hash()


def _check(files, waivers=(), **kw):
    return release.check(files, list(waivers), run_id=kw.pop("run_id", "run-1"), today=TODAY, **kw)


@UNIT
def test_release_check_passes_a_green_dashboard(tmp_path):
    assert _check(_write(tmp_path, _health(GREEN))) == []


@UNIT
def test_release_check_passes_a_known_bug_only_when_accepted_at_its_checkpoint(tmp_path):
    """Without a waiver H01-D1 fails the release; with the waiver for exactly H01, C4, H01-D1 and its lost row it
    passes, is listed as accepted, and the health file stays RED."""
    files = _write(tmp_path, _health(RED_C4, {"C4": "H01-D1"}))
    assert _check(files) == ["H01 C4: known bug H01-D1 - not accepted for release (no waiver for H01 C4 H01-D1 "
                             "in waivers.json)"]
    accepted = []
    assert _check(files, [WAIVER_C4], accepted=accepted) == []
    assert accepted and "H01 C4: H01-D1 accepted by DEC-99" in accepted[0]
    assert json.loads(files[0].read_text())["dashboard"] == "RED"


CASES = {   # case: (health, waivers, words the problem must contain, extra manifest fields)
    "no file": (None, [], "0 health files", None),
    "two files": ("two", [], "2 health files", None),
    "an action failed": (_health(dict(GREEN, C2="FAIL")), [], "H01 C2: FAIL", None),
    "an action not run": (_health({c: s for c, s in GREEN.items() if c != "C6"}), [], "H01 C6: NOT_RUN", None),
    "unknown status": (_health(dict(GREEN, C3="MAYBE")), [], "unknown statuses", None),
    "dashboard status wrong": (_health(RED_C4, {"C4": "H01-D1"}, dashboard="PASS"), [WAIVER_C4], "dashboard status",
                               None),
    "stale waiver": (_health(GREEN), [WAIVER_C4], "remove or correct the waiver", None),
    "unknown dashboard": ("unknown", [], "does not know", None),
    "unreadable file": ("unreadable", [], "unreadable", None),
    # GPT25's summary H-01: the run must be the one asked for
    "wrong dashboard function": (_health(GREEN, function="testBokehClientHistogram"), [], "function", None),
    "wrong row count": (_health(GREEN, rows=40000), [], "rows 40000", None),
    "wrong seed": (_health(GREEN, seed=7), [], "seed 7", None),
    "wrong table hash": (_health(GREEN, table_sha256="0" * 64), [], "table hash", None),
    "other test code": (_health(GREEN, code_sha256="f" * 64), [], "other test code", None),
    "another run": (_health(GREEN, run_id="run-0"), [], "not by this release run", None),
    "manifest differs": (_health(GREEN), [], "manifest rows", {"rows": 1}),
    "page differs": (_health(GREEN, page_sha256="e" * 64), [], "page.html", None),
    # GPT25's summary H-02: a waiver is bound to its place, signature and date
    "waiver at another checkpoint": (_health(dict(GREEN, C2="KNOWN_ISSUE_RED"), {"C2": "H01-D1"}), [WAIVER_C4],
                                     "H01 C2: known bug H01-D1 - not accepted", None),
    "expired waiver": (_health(RED_C4, {"C4": "H01-D1"}), [dict(WAIVER_C4, expires="2026-11-11")], "expired",
                       None),
    "other signature": (_health(RED_C4, {"C4": "H01-D1"}), [dict(WAIVER_C4, signature={"lost_rows": [7]})],
                        "not the accepted signature", None),
}


@UNIT
@pytest.mark.parametrize("case", list(CASES))
def test_release_check_fails_closed(tmp_path, case):
    health, waivers, words, manifest = CASES[case]
    if health is None:
        files = []
    elif health == "two":
        files = _write(tmp_path, _health(GREEN), _health(GREEN))
    elif health == "unknown":
        files = _write(tmp_path, _health(GREEN), _health(GREEN, scenario="H99"))
    elif health == "unreadable":
        files = _write(tmp_path, _health(GREEN))
        files[0].write_text("{not json")
    else:
        files = _write(tmp_path, health, manifest=manifest)
    found = _check(files, waivers)
    assert found and any(words in f for f in found), (case, found)


@UNIT
def test_release_waivers_file_is_valid_and_names_decisions():
    """The shipped waivers.json loads; every waiver names its exact place, signature, decision, date, author and expiry
    (none yet: open decision E-1)."""
    for w in release.load_waivers():
        assert set(release.WAIVER_KEYS) <= set(w), w


@UNIT
def test_release_waivers_are_validated(tmp_path):
    """A waiver without its checkpoint or expiry, or with a decision that is not a DEC number, does not load."""
    for bad in ({k: v for k, v in WAIVER_C4.items() if k != "checkpoint"}, {k: v for k, v in WAIVER_C4.items()
                                                                              if k != "expires"},
                dict(WAIVER_C4, decision="the Architect said so")):
        f = tmp_path / "w.json"
        f.write_text(json.dumps({"waivers": [bad]}))
        with pytest.raises(ValueError):
            release.load_waivers(f)


# ------------------------------------------------------------------ the page itself (browser_oracle_dashboards)
@BROWSER
@pytest.mark.feature("DASHBOARD.histogram_only")
def test_h01_page_is_the_original_dashboard_on_the_reproducible_table(h01_page, headless_out):
    """The captured page has exactly the structure of the original 500,000-row page (models, widgets, tabs, plots),
    and its table is the one the oracle regenerates from (rows, seed)."""
    m = h01_page["manifest"]
    assert m["opt_in"] and m["rows"] == ROWS and m["seed"] == SEED
    t = oracle.regenerate_table(ROWS, SEED)
    assert oracle.table_sha256(t) == m["table_sha256"], "the page was not built from the oracle's table"
    original = _capture(headless_out / "dashboards" / "h01_original_500k", 0, 0, graph_only=True)
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
  const arr = a => a ? Array.from(a) : null;
  const walk = view => {
    if (!view || seen.has(view)) return; seen.add(view);
    if (view.model && view.model.type === 'GlyphRenderer' && view.model.glyph.type === 'Scatter') {
      const chain = []; let m = view.model.data_source, guard = 0;
      while (m && guard++ < 10) { chain.push([m.type, m.name]); m = m.source ?? (m.sources ? m.sources[0] : null); }
      const canvas = view.parent?.canvas_view?.el, plot = view.parent?.model, g = view.glyph;
      const shown = canvas ? canvas.checkVisibility({visibilityProperty: true, opacityProperty: true}) : null;
      const label = side => { const a = plot?.[side]; return a && a.length === 1 ? String(a[0].axis_label) : null; };
      const bb = view.parent?.frame?.bbox, xr = plot?.x_range, yr = plot?.y_range;
      renderers.push({source: view.model.data_source.name, yfield: view.model.glyph.y?.field ?? null,
                      xfield: view.model.glyph.x?.field ?? null, chain, x: arr(g._x), y: arr(g._y),
                      sx: arr(g.sx), sy: arr(g.sy), n: g.data_size, canvas_visible: shown,
                      frame: bb ? {left: bb.left, right: bb.right, top: bb.top, bottom: bb.bottom} : null,
                      x_range: xr ? [xr.start, xr.end] : null, y_range: yr ? [yr.start, yr.end] : null,
                      x_scale: plot?.x_scale?.type ?? null, y_scale: plot?.y_scale?.type ?? null,
                      model_visible: view.model.visible, visible: shown === true && view.model.visible === true,
                      xlabel: label('below'), ylabel: label('left'), title: plot?.title ? String(plot.title.text) : null});
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

INVENTORY_JS = r"""() => {
  const out = [], seen = new Set();
  const kids = x => !x ? [] : (x instanceof Map ? Array.from(x.values()) : Array.from(x));
  const f = s => s == null ? null : (s.field ?? (s.value !== undefined ? 'value ' + String(s.value) : null));
  const walk = view => {
    if (!view || seen.has(view)) return; seen.add(view);
    if (view.model && view.model.type === 'GlyphRenderer') {
      const g = view.model.glyph, canvas = view.parent?.canvas_view?.el, plot = view.parent?.model;
      out.push({glyph: g.type, source: view.model.data_source.name,
                x: f(g.x) ?? f(g.left), y: f(g.y) ?? f(g.top),
                xlabel: plot?.below?.length ? String(plot.below[0].axis_label) : null,
                ylabel: plot?.left?.length ? String(plot.left[0].axis_label) : null,
                shown: canvas ? canvas.checkVisibility({visibilityProperty: true, opacityProperty: true}) : null});
    }
    for (const c of kids(view.child_views)) walk(c);
    for (const c of kids(view.renderer_views)) walk(c);
  };
  for (const v of (Bokeh.index instanceof Map ? Bokeh.index.values() : Object.values(Bokeh.index))) walk(v);
  const tables = window.__riH.models().filter(m => m.type === 'DataTable')
    .map(m => ({glyph: 'DataTable', source: m.source?.name ?? null, x: null, y: null, xlabel: null, ylabel: null,
                columns: m.columns.map(c => String(c.field)), shown: null}));
  return out.concat(tables);
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


def _inventory(manifest: dict, at_c1: list, at_c3: list) -> dict:
    """Every widget and plot of the page, and which ones H01 checks (the rest: the next H01 delivery)."""
    plots, left = [], list(at_c3)
    for r1 in at_c1:
        key = (r1["source"], r1["y"])
        same = [r for r in left if (r["glyph"], r["source"], r["x"], r["y"]) == (r1["glyph"], r1["source"], r1["x"], r1["y"])]
        r3 = left.pop(left.index(same[0])) if same else {"shown": None}
        tab = "Histograms" if r1["shown"] else "Cumulative" if r3["shown"] else "?"
        checked = r1["glyph"] == "Scatter" and key in CHECKED_PLOTS
        plots.append(dict({k: v for k, v in r1.items() if k != "shown"}, tab=tab, checked_by_h01=checked))
    widgets = [{"type": w[0], "title": w[1], "name": w[2], "checked_by_h01": w[0] == "RangeSlider" and
                w[1] in CHECKED_WIDGETS} for w in manifest["graph"]["widgets"]]
    return {"plots": plots, "widgets": widgets,
            "summary": {"plots": len(plots), "plots_checked": sum(p["checked_by_h01"] for p in plots),
                        "widgets": len(widgets), "widgets_checked": sum(w["checked_by_h01"] for w in widgets)}}


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

    def console_allowed(self) -> dict:
        """The console errors the existing policy allows, with how often each was seen (recorded, never hidden)."""
        from .engine.runner import CONSOLE_ALLOWED
        return {p: sum(1 for e in self.errors["console"] if re.search(p, e)) for p, _ in CONSOLE_ALLOWED}

    def unexpected_errors(self) -> dict:
        from .engine.runner import CONSOLE_ALLOWED
        console = [e for e in self.errors["console"] if not any(re.search(p, e) for p, _ in CONSOLE_ALLOWED)]
        return {k: v for k, v in {"page": self.errors["page"], "console": console,
                                  "blocked": self.errors["blocked"]}.items() if v}

    def close(self):
        self.ctx.close()


def _h01_d1_lost_rows(t, widgets, expected: dict, actual: dict, span: float):
    """H01-D1, exact. Returns the indices of the rows the page lost, or None when the page does not show H01-D1.

    The signature: the lower end is exactly the slider's start; the upper end sits 1 to 4 float steps below the
    slider's end; the page is exactly right for the value it shows (selected rows, every bin, cumulative, cdf, drawn
    plots); the rows it lost are exactly the selected rows with A above the shown value, at least one."""
    a = widgets["A"]
    lo, hi = actual["widgets"]["A"]
    if lo != a["start"] or not 0 < a["end"] - hi <= 4 * np.spacing(a["end"]):
        return None
    shown = oracle.h01_expected(t, oracle.with_a_range(widgets, lo, hi), expected["tab"], expected["checkpoint"],
                                check_margin=False)
    if compare(shown, actual, widget_span=span):
        return None
    full = oracle.selection(t, oracle.with_a_range(widgets, *expected["widgets"]["A"]))
    lost = np.flatnonzero(full & ~oracle.selection(t, oracle.with_a_range(widgets, lo, hi)))
    if len(lost) < 1 or len(lost) != expected["selected_count"] - actual["selected_count"]:
        return None
    return lost.tolist()


def _git_identity() -> dict:
    """The commit the test code came from, and whether the dashboard test files differ from it (evidence only)."""
    root = Path(__file__).resolve().parents[4]
    files = [str(Path(__file__).resolve().parent / f) for f in release.CODE_FILES]
    def git(*a):
        r = subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        return {"head": git("rev-parse", "--short", "HEAD"), "test_files_modified": bool(git("status", "--porcelain",
                                                                                            "--", *files))}
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "test_files_modified": None}


def _is_h01_d1(t, widgets, expected: dict, actual: dict, span: float) -> bool:
    return _h01_d1_lost_rows(t, widgets, expected, actual, span) is not None


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
    video, lost_rows, inventory_reads = None, {}, {}
    try:
        actual, results, seconds = {}, {}, {"first_load": page.load_seconds}
        for one in plan:
            act_one, res_one, sec_one = _run_plan(page, [one], expected, span, out)
            actual.update(act_one); results.update(res_one); seconds.update({k: v for k, v in sec_one.items()
                                                                              if k != "first_load"})
            cid = one[0]
            if cid in ("C1", "C3"):
                inventory_reads[cid] = page.pg.evaluate(INVENTORY_JS)
            lost = _h01_d1_lost_rows(t, widgets, expected[cid], actual[cid], span) if results[cid] else None
            status = "PASS" if not results[cid] else "KNOWN_ISSUE_RED" if lost is not None else "FAIL"
            if lost is not None:
                lost_rows[cid] = lost
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

    # 3. classification and evidence (comparison, per-action health, inventory, timeline)
    status = {c["id"]: c["status"] for c in page.timeline.events if c["kind"] == "checkpoint"}
    nonvacuous = differs(actual["C1"], actual["C2"])
    a_max_row = int(np.argmax(t["A"]))
    timing = {"cold": {"capture_child_process_s": m["seconds"].get("capture_process"),
                       "of_which_build_s": m["seconds"].get("build"), "of_which_html_s": m["seconds"].get("html")},
              "page_first_load_s": page.load_seconds,
              "warm": {k.replace("settle_", "action_") + "_s": v for k, v in seconds.items() if k != "first_load"}}
    comparison = {"rows": ROWS, "seed": SEED, "a_slider": a, "a_changed": [lo, hi], "results": results,
                  "status": status, "non_vacuity": nonvacuous, "errors": page.errors,
                  "console_allowed_seen": page.console_allowed(), "timing": timing, "page_bytes": m["page_bytes"],
                  "h01_d1_lost_rows": lost_rows, "row_with_largest_A": a_max_row}
    (out / "comparison.json").write_text(json.dumps(comparison, indent=1))
    health = {"scenario": "H01", "function": oracle.H01_FUNCTION, "rows": ROWS, "seed": SEED,
              # identity, checked by the release check (GPT25's summary H-01)
              "table_sha256": m["table_sha256"], "page_sha256": m["page_sha256"],
              "code_sha256": release.code_sha256(), "git": _git_identity(),
              "run_id": os.environ.get("RI_RELEASE_RUN_ID") or f"local-{uuid.uuid4().hex[:12]}",
              "actions": status,
              "known_defects": {c: "H01-D1" for c, s in status.items() if s == "KNOWN_ISSUE_RED"},
              "known_defect_detail": {c: {"lost_rows": r, "row_with_largest_A": a_max_row} for c, r in lost_rows.items()}}
    health["dashboard"] = "PASS" if all(s == "PASS" for s in status.values()) else "RED"
    (out / "dashboard_health.json").write_text(json.dumps(health, indent=1))
    if set(inventory_reads) == {"C1", "C3"}:
        (out / "inventory.json").write_text(json.dumps(_inventory(m, inventory_reads["C1"], inventory_reads["C3"]),
                                                       indent=1))
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
    assert set(inventory_reads) == {"C1", "C3"}
    known_defect(status["C4"] == "KNOWN_ISSUE_RED" and lost_rows.get("C4") == [a_max_row], H01_D1)


@BROWSER
@pytest.mark.feature("DASHBOARD.histogram_only")
def test_signature_h01_d1_arrow_keys_back_to_the_end_lose_the_largest_A(h01_page, headless_session):
    """H01-D1, exact, in its smallest form: one arrow step left and one right on the upper handle. The one row lost
    is the row with the largest A."""
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
    lost = _h01_d1_lost_rows(t, widgets, full, got, a["end"] - a["start"])
    assert lost == [int(np.argmax(t["A"]))], (got["widgets"], got["selected_count"], lost)
    assert t["A"][lost[0]] == a["end"] and got["widgets"]["A"][1] < a["end"]
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


# ------------------------------------------------------------------ live faults in the browser (G2, G6)
_FIND_JS = r"""
const H = window.__riH;
const r = H.models().filter(m => m.type === 'GlyphRenderer' && m.data_source.name === 'histo1Ds'
                              && m.glyph.type === 'Scatter' && m.glyph.y?.field === 'bin_count');
if (r.length !== 1) throw Error('live fault: expected one histo1Ds bin_count scatter, found ' + r.length);
const R = r[0];
const P = H.models().filter(m => (m.type === 'Figure' || m.type === 'Plot') && m.renderers.includes(R))[0];
window.__riFault = window.__riFault || {};
"""
LIVE_FAULTS = {
    "x axis range narrowed: half the points cut off": (
        "window.__riFault.xr = [P.x_range.start, P.x_range.end]; P.x_range.start = 0.5;",
        "P.x_range.start = window.__riFault.xr[0]; P.x_range.end = window.__riFault.xr[1];",
        "outside the plot frame"),
    # name: (inject, undo, the words the comparator must report)
    "x drawn from the bins' lower edges": (
        "R.glyph.x = {field: 'bin_bottom'};", "R.glyph.x = {field: 'bin_center'};", "draws x from 'bin_bottom'"),
    "plot switched off": ("R.visible = false;", "R.visible = true;", "renderer visible: False"),
    "x axis relabelled": ("window.__riFault.xl = P.below[0].axis_label; P.below[0].axis_label = 'B (cm/s)';",
                          "P.below[0].axis_label = window.__riFault.xl;", "xlabel 'B (cm/s)'"),
    "one drawn value changed in the plot's own source": (   # the DownsamplerCDS serves its columns by get_column
        "const d = R.data_source, get = Object.getPrototypeOf(d).get_column; "
        "d.get_column = function (k) { const c = get.call(this, k); if (k !== 'bin_count' || !c) return c; "
        "const m = Array.from(c); m[5] += 1; return m; }; d.change.emit();",
        "const d = R.data_source; delete d.get_column; d.change.emit();",
        "(drawn values)"),
}


@BROWSER
@pytest.mark.feature("DASHBOARD.histogram_only")
def test_h01_live_faults_in_the_browser_are_caught(h01_page, headless_session):
    """Faults injected into the live page in Chromium (not into a saved snapshot) are reported by the same snapshot
    and comparator the workflow uses; after each fault is undone the page passes again. Control first: the
    untouched page passes."""
    m = h01_page["manifest"]
    widgets, t = m["widget_state"], oracle.regenerate_table(ROWS, SEED)
    a = widgets["A"]
    span = a["end"] - a["start"]
    full = oracle.h01_expected(t, widgets, "Histograms", "full")
    found, snapshots, control = {}, {}, None
    page = _Page(headless_session, h01_page["html"])
    try:
        control = compare(full, _settle(page.pg), widget_span=span)
        for name, (inject, undo, _) in LIVE_FAULTS.items():
            page.pg.evaluate("() => {" + _FIND_JS + inject + "}")
            snapshots[name] = _settle(page.pg)
            faulty = compare(full, snapshots[name], widget_span=span)
            page.pg.evaluate("() => {" + _FIND_JS + undo + "}")
            found[name] = (faulty, compare(full, _settle(page.pg), widget_span=span))
    finally:
        page.close()
    (h01_page["dir"] / "live_faults.json").write_text(json.dumps(
        {"expected": full, "control": control,
         "faults": {n: {"inject": LIVE_FAULTS[n][0], "found": f, "after_undo": u, "snapshot": snapshots.get(n)}
                    for n, (f, u) in found.items()}}, indent=1))
    assert control == [], control
    for name, (faulty, after) in found.items():
        words = LIVE_FAULTS[name][2]
        assert any(words in d for d in faulty), (name, faulty)
        assert after == [], (name, "the page did not pass again after the fault was undone", after)
    assert not page.unexpected_errors(), page.unexpected_errors()


# ------------------------------------------------------------------ the release check (release_oracle_dashboards)
@RELEASE
def test_release_check_of_the_oracle_dashboards(request, tmp_path):
    """Runs the oracle dashboards in Chromium (a child pytest of the target browser_oracle_dashboards) and passes only
    when that run is green AND every dashboard is healthy or shows only bugs the Architect accepted for release
    (dashboards/waivers.json). Runs only as its own target: in an ordinary run it is skipped."""
    if request.config.getoption("markexpr") != "release_oracle_dashboards":
        pytest.skip("the release check runs only as its own target: pytest -m release_oracle_dashboards")
    out = tmp_path / "release_run"
    run_id = f"release-{uuid.uuid4().hex[:12]}"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTEST_XDIST_WORKER", "PYTEST_ADDOPTS", "RI_HEADLESS_SESSION",
                                                            "RI_HEADLESS_TEST_SESSION_ID", "RI_HEADLESS_CHECK_REQUIRED")}
    env.update(RI_HEADLESS_OUT=str(out), RI_HEADLESS_REQUIRED="1", RI_HEADLESS_TARGET="browser_oracle_dashboards",
               RI_RELEASE_RUN_ID=run_id)
    root = Path(__file__).resolve().parents[4]
    done = subprocess.run([sys.executable, "-m", "pytest", str(Path(__file__).resolve().parent), "-m",
                           "browser_oracle_dashboards", "-p", "no:xdist", "-p", "no:cacheprovider", "-q",
                           "--timeout=900"], cwd=root, env=env, capture_output=True, text=True, timeout=1800)
    (tmp_path / "release_run.log").write_text(done.stdout + done.stderr)
    assert done.returncode == 0, "the oracle dashboards did not pass:\n" + (done.stdout + done.stderr)[-3000:]
    files = sorted(out.rglob("dashboard_health.json"))
    accepted = []
    problems = release.check(files, release.load_waivers(), run_id=run_id, accepted=accepted)
    print(f"release run {run_id}: {len(files)} health file(s); code {release.code_sha256()[:12]}")
    for a in accepted:
        print("ACCEPTED (dashboard stays RED):", a)
    assert not problems, "release check failed:\n" + "\n".join(problems)
