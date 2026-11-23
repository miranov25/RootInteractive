"""PHASE_5 Group 7, dashboard D2: twenty invariance and oracle tests on one live page with weights (Claude21.RI,
2026-11-23).

In plain words:
- **The page:** the dashboard of `test_interactiveTemplateWeights` (`getDefaultVarsDiff` with several weights at once,
  `multiAxis="weights"`, the lists varXMulti and varYMulti with their histograms), on 100,000 rows of exact numbers
  (dashboards/invariance_dashboards.py, which you can run yourself: `... weights --rows 100000 --output page.html`).
  The list "weights" chooses any of None (no weight), W, W1, W2, Wsel, A>8 and A<=8: every histogram then holds one
  series per chosen weight. The weights are multiples of 1/8, so their sums are exact.
- **Its first tab, "Invariances":** four ratios or differences of weighted counts of one histogram of varX, which
  must be 1, 0, 2 and 1 in every bin: W1 against no weight, W2 - 2 W, W2 / W, and (A>8) + (A<=8) against no weight.
- **One page, twenty tests,** as for D1 (test_headless_invariance_template.py): fixed states, after each state the
  page against numpy and pandas (selected rows one by one, every histogram bin by bin with its edges bit for bit,
  profiles cell by cell), and at the end of each test every control back and the page exactly as at load.
- **What is compared for a weighted series:** the sums of the weights per bin (exactly), and the weighted mean and
  standard deviation of a profile (to 1e-12, relative: the page sums in another order). The weighted quantiles are
  not compared: their rule between rows is left to compilation time (decision DEC-30); the unweighted series' are.
- **Real mouse and keys:** W02 and W09 (the list of weights), W15 (slider keys and the list cat), W16 (tab clicks),
  W17 (the list varXMulti).
- **A defect found, a strict expected failure with its exact result (WT_D15 below), in W11 and W17:** a plot in a hidden
  tab whose table changed length throws "Size mismatch" page errors at the next change of its table (choosing a
  weight after the number of bins changed; fewer, then more variables in a list). The arrays are right.

Words used: a **series** is the part of a histogram or profile for one chosen weight (or one chosen variable of a
multi-list); the **baseline** is the state at load; a **precondition** is a property of the data a test relies on,
checked first.

Switches, as for D1: RI_INVARIANCE_ROWS, RI_INVARIANCE_FRESH_PAGE=1. Target: browser_invariance_dashboards, group
`invariance_weights` (one worker).
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

from .dashboards import invariance_dashboards as pages
from .dashboards import invariance_driver as drv
from .dashboards import invariance_oracle as O

ROWS = int(os.environ.get("RI_INVARIANCE_ROWS", str(pages.TEMPLATE_ROWS)))
FULL = "full"
CUTS = [(3.0, 9.0), (0.5, 4.5), (9.5, 18.5)]
REL = 1e-12
WOPTS = ["None" if w is None else w for w in pages.WEIGHTS]
PROF = ["entries", "mean", "std", "quantile_0", "quantile_1"]
KINDS = {**{c: "range" for c in "ABCD"}, **{k: "list" for k in ("cat", "DDC", "weights", "varXMulti", "varYMulti")},
         **{k: "select" for k in ("varX", "varY", "varYNorm", "varZ", "varZNorm")},
         **{k: "spinner" for k in ("nbinsX", "nbinsY", "nbinsZ")},
         **{f"funCustomForm{k}": "text" for k in range(3)}}
CONTROLS = ([(kind, key) for key, kind in KINDS.items()] + [("query", k) for k in range(3)]
            + [("off", k) for k in range(3)] + [("tabs", "Invariances"), ("tabs", "Select")])
PROBES = [["histoInvW", "bin_count", ["HistogramCDS"]]]
# the axes of each histogram family: a parameter (varX ...), the normalized columns, or V, the variable of a multi-list
FAMILIES = {"histoXYData": ["varX", "varY"], "histoXYZData": ["varX", "varY", "varZ"],
            "histoXYNormData": ["varX", "diffN"], "histoXYNormZData": ["varX", "diffN", "varZ"],
            "histoXYZNormData": ["varX", "varY", "diffZN"], "histoNDMultiX": ["V", "varY"],
            "histoNDMultiY": ["varX", "V"], "histo1D": ["varX"], "histo1DMulti": ["V"], "histoInvW": ["varX"]}
NBINS = {"varX": "nbinsX", "varY": "nbinsY", "diffN": "nbinsY", "varZ": "nbinsZ", "diffZN": "nbinsZ"}
NBINS_V = {"histoNDMultiX": "nbinsX", "histoNDMultiY": "nbinsY", "histo1DMulti": "nbinsX"}
WEIGHTED = {"histoXYData", "histoXYZData", "histoXYNormData", "histoXYNormZData", "histoXYZNormData"}


# A defect found by these tests (new in Group 7), a strict expected failure until it is fixed:
WT_D15 = ("WT-D15 (new, Group 7): on a page with a list of several variables (or weights) whose plots sit in a hidden "
          "tab, choosing fewer series and then more again (here varXMulti: A and B, then B, then B and C) throws page "
          "errors 'Size mismatch' (Bokeh's BitSet) from every hidden plot of that list: the plot's selection of rows "
          "still has the old length while its table already has the new one. The arrays are right, and the plots are "
          "right once their tab is shown; but each error stops the other listeners of the same change. The same after "
          "the number of bins changed and then the weight (W11)")


def browser(fn):
    """A test of this page: the target marker, and the group that keeps the twenty tests on one worker."""
    for mark in (pytest.mark.browser_invariance_dashboards, pytest.mark.backend("browser"), pytest.mark.layer("invariance"),
                 pytest.mark.xdist_group("invariance_weights")):
        fn = mark(fn)
    return fn


def parse(name: str):
    """A histogram's family and its argument: histoXYData[W] -> (histoXYData, "W"); histoNDMultiX[A] ->
    (histoNDMultiX, "A"); histo1DMulti_A -> (histo1DMulti, "A"); histo1D -> (histo1D, None)."""
    m = re.fullmatch(r"(\w+?)\[(.+)\]", name)
    if m:
        return m.group(1), m.group(2)
    if name.startswith("histo1DMulti_"):
        return "histo1DMulti", name[len("histo1DMulti_"):]
    return name, None


class D2:
    """The shared page with the current state of its controls, and the expected arrays for that state."""

    def __init__(self, page: drv.Page, table: dict):
        self.page, self.t, self.n = page, table, len(table["A"])
        self.reset()

    def reset(self):
        self.cur = {key: val for (kind, key), val in self.page.at_load.items()
                    if kind in ("range", "list", "select", "spinner", "text")}
        self.full = {c: tuple(self.cur[c]) for c in "ABCD"}
        self.queries = {}

    @contextmanager
    def case(self, name: str, allowed_errors=()):
        try:
            with self.page.case(name, allowed_errors):
                yield self
        finally:
            self.reset()

    def set(self, label: str, **kw) -> dict:
        page_kw = {}
        for key, value in kw.items():
            if key.startswith("query"):
                k = int(key[5:])
                if value is None:
                    self.queries.pop(k, None)
                    page_kw[key] = None
                else:
                    page_kw[key], self.queries[k] = value
                continue
            if value == FULL:
                value = self.full[key]
            page_kw[key] = value
            self.cur[key] = list(value) if isinstance(value, tuple) else value
        return self.page.set(label, **page_kw)

    def read_back(self, *keys):
        """After a real mouse or key action: the expected state takes the values the controls show."""
        v = self.page.values()
        for key in keys:
            self.cur[key] = v[(KINDS[key], key)]

    # -- expected values
    def mask(self) -> np.ndarray:
        key = (tuple(tuple(self.cur[c]) for c in "ABCD"), tuple(self.cur["cat"]), tuple(self.cur["DDC"]),
               tuple(sorted((k, id(f)) for k, f in self.queries.items())))
        if getattr(self, "_mask_key", None) != key:
            self._mask_key = key
            self._mask = O.selection(self.t, {c: self.cur[c] for c in "ABCD"},
                                     {"cat": self.cur["cat"], "DDC": self.cur["DDC"]}, list(self.queries.values()))
        return self._mask

    def var(self, name: str) -> np.ndarray:
        fns = {f"funCustom{k}": (lambda t: np.ones(len(t["A"]))) for k in range(3)}
        return O.variable(self.t, name, {"exponentX": 1.0}, fns)

    def weight(self, w):
        """The weights of the list entry `w`: None (no weight), a column, or an expression."""
        if w in (None, "None"):
            return None
        if w in ("A>8", "A<=8"):
            return ((self.t["A"] > 8) if w == "A>8" else (self.t["A"] <= 8)).astype(np.float64)
        return np.asarray(self.t[w], np.float64)

    def axis(self, a: str, v=None) -> np.ndarray:
        if a == "V":
            return self.var(v)
        if a == "diffN":
            return self.var(self.cur["varY"]) - self.var(self.cur["varYNorm"])
        if a == "diffZN":
            return self.var(self.cur["varZ"]) - self.var(self.cur["varZNorm"])
        return self.var(self.cur[a])

    def geometry(self, family: str, v=None):
        cols = [self.axis(a, v) for a in FAMILIES[family]]
        nbins = [int(self.cur[NBINS_V[family] if a == "V" else NBINS[a]]) for a in FAMILIES[family]]
        return cols, nbins

    def hist(self, name: str, weights=None) -> dict:
        """Expected histogram `name`; a weighted family takes the weight in its name, a 1D one `weights`."""
        family, arg = parse(name)
        cols, nbins = self.geometry(family, arg if family not in WEIGHTED else None)
        w = self.weight(arg) if family in WEIGHTED else (None if weights is None else self.weight(weights))
        return O.histogram(cols, nbins, self.mask(), weights=w)

    def prof(self, name: str) -> list:
        """Expected profile `name` as its series: [(weight, columns)], in the order of the list of weights for the
        template's profiles (histoXYData_1 ...), one unweighted series for a multi-list profile."""
        m = re.fullmatch(r"(histoNDMulti[XY])_(\d)_(.+)", name)
        if m:
            family, v = parse(m.group(3))
            cols, nbins = self.geometry(family, v)
            return [(None, O.profile(cols, nbins, self.mask(), int(m.group(2)), [0.35, 0.5]))]
        family, axis = name.rsplit("_", 1)
        cols, nbins = self.geometry(family)
        out = []
        for w in self.cur["weights"]:
            wv = self.weight(w)
            out.append((w, O.profile(cols, nbins, self.mask(), int(axis), [0.35, 0.5] if wv is None else [], weights=wv)))
        return out

    # -- checks
    def check(self, r: dict, where: str, hists=(), profs=(), hist1=()):
        if "mask" in r:
            drv.same_rows(r["mask"], self.mask(), where + ", the selected rows")
            assert r["mask_padding_selected"] == 0, f"{where}: rows past the end of the table are selected"
        for h in hists:
            drv.same_histogram(r["hist:" + h], self.hist(h), f"{where}, histogram {h}")
        for h, extra in hist1:
            got = r["hist:" + h]
            drv.same_histogram(got, self.hist(h), f"{where}, histogram {h}")
            for c in extra:
                drv.same_values(got[c], self.hist(h, weights=pages.WEIGHT_COUNTS[c])["count"], f"{where}, {h}.{c}")
        for p in profs:
            got = r["prof:" + p]
            series = self.prof(p)
            n = len(got["entries"]) // len(series)
            assert n * len(series) == len(got["entries"]), (where, p, "series of unequal length", len(got["entries"]))
            for k, (w, e) in enumerate(series):
                part = slice(k * n, (k + 1) * n)
                label = f"{where}, profile {p}, series {k} (weight {w})"
                drv.same_values(got["entries"][part], e["entries"], label + ", entries")
                drv.scaled_close(got["mean"][part], e["mean"], label + ", mean", REL)
                drv.scaled_close(got["std"][part], e["std"], label + ", std", REL)
                if self.weight(w) is None:
                    for c in ("quantile_0", "quantile_1"):
                        if c in got and got[c] is not None:
                            drv.scaled_close(got[c][part], e[c], label + ", " + c, REL)

    def series(self, got: dict, k: int, count: int) -> dict:
        """Series k of `count` of a profile read from the page."""
        n = len(got["entries"]) // count
        return {c: v[k * n:(k + 1) * n] for c, v in got.items() if isinstance(v, np.ndarray)}


def _build(out: Path) -> dict:
    cmd = [sys.executable, "-m", "RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.invariance_dashboards",
           "weights", "--rows", str(ROWS), "--output", str(out / "page.html")]
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env, cwd=str(out))
    if done.returncode != 0:
        pytest.fail(f"the page could not be built ({done.returncode}):\n{done.stderr[-3000:]}", pytrace=False)
    manifest = json.loads((out / "page.html.manifest.json").read_text())
    manifest["build_process_s"] = round(time.perf_counter() - t0, 2)
    return manifest


def _checked(name: str) -> bool:
    """The histograms the end of every test compares with the load: all but those of the multi-lists' other
    variables (C, D, cat, A+B, ...: the same code as A and B, 0.8 s more per test)."""
    family, arg = parse(name)
    return not family.startswith(("histoNDMulti", "histo1DMulti")) or arg in ("A", "B")


@pytest.fixture(scope="module")
def d2(headless_session, headless_out):
    out = headless_out / "invariance" / "weights"
    out.mkdir(parents=True, exist_ok=True)
    manifest = _build(out)
    df = pages.weights_table(ROWS)
    assert manifest["table_sha256"] == pages.table_sha256(df), "the page was built from another table"
    table = {c: df[c].to_numpy() for c in df.columns}
    page = drv.Page(headless_session, out / "page.html", KINDS, CONTROLS, fingerprint_spec={}, probes=PROBES, out_dir=out)
    inv = page.pg.evaluate("() => window.__g7.inventory()")
    page.fingerprint_spec = {
        "mask": True, "hist": [n for n in inv["hist"] if _checked(n)],
        "hist1": [[n, c] for n, c in inv["hist1"] if _checked(n)],
        "prof": [[n, ["entries", "mean", "std"] + [f"quantile_{k}" for k in range(q)]] for n, q in inv["prof"]
                 if "histoNDMulti" not in n or n.endswith(("[A]", "[B]"))]}
    page.fingerprint_at_load = page.fingerprint()
    page.pg.screenshot(path=str(out / "at_load_tab_Invariances.png"))
    d = D2(page, table)
    d.manifest, d.out, d.inventory = manifest, out, inv
    yield d
    summary = page.write_timings(out / "timings.json")
    page.pg.screenshot(path=str(out / "at_the_end_tab_Invariances.png"))
    print("\nD2 timing:", json.dumps({k: summary[k] for k in ("load_s", "states", "page_ms_median", "page_ms_p90",
                                                               "read_s_total", "tests_s_total")}))
    page.close()


def _weights_by_list(d2, chosen, where):
    """Choose weights with the real list (select_option), then take the values it shows."""
    d2.page.click_tab("Histograms")
    d2.page.element("list", "weights", "input").select_option(chosen)
    d2.page.waited(where + " (real list)")
    d2.read_back("weights")
    assert d2.cur["weights"] == chosen, (where, d2.cur["weights"])
    d2.page.click_tab("Select")


# ================================================================== the twenty tests
@browser
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.stack")
def test_w01_each_weight_alone_equals_numpy(d2):
    """W01. Each of the seven weights alone (None, W, W1, W2, Wsel, A>8, A<=8): its 2D histogram holds exactly the
    sums of the weights numpy gives, bin by bin, and the stacked table holds that one series."""
    with d2.case("W01"):
        for w in WOPTS:
            where = f"W01 weights [{w}]"
            d2.set(where, weights=[w])
            r = d2.page.read(hist=[f"histoXYData[{w}]"], stack=["histoXYData"], prof=[("histoXYData_1", PROF)])
            d2.check(r, where, hists=[f"histoXYData[{w}]"], profs=["histoXYData_1"])
            s = r["stack:histoXYData"]
            nb = int(d2.cur["nbinsX"]) * int(d2.cur["nbinsY"])
            assert s == {"active": [w], "length": nb}, (where, s)


@browser
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.profile_unbinned")
def test_w02_unit_weights_give_the_counts(d2):
    """W02. The weights None and W1 chosen together with the real list: the W1 histogram equals the unweighted one
    exactly, and so do the entries, means and standard deviations of their two profile series."""
    with d2.case("W02"):
        _weights_by_list(d2, ["None", "W1"], "W02 weights None and W1")
        r = d2.page.read(hist=["histoXYData[None]", "histoXYData[W1]"], prof=[("histoXYData_1", PROF)])
        d2.check(r, "W02", hists=["histoXYData[None]", "histoXYData[W1]"], profs=["histoXYData_1"])
        drv.same_values(r["hist:histoXYData[W1]"]["count"], r["hist:histoXYData[None]"]["count"], "W02 W1 against None")
        a, b = d2.series(r["prof:histoXYData_1"], 0, 2), d2.series(r["prof:histoXYData_1"], 1, 2)
        for c in ("entries", "mean", "std"):
            drv.same_values(b[c], a[c], f"W02 profile {c}: W1 against None, exactly")


@browser
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.profile_unbinned")
def test_w03_doubled_weights_double_the_sums_and_keep_the_mean(d2):
    """W03. W and W2 = 2 W together, all rows and A in [3, 9]: the W2 histogram is exactly twice the W one; in the
    profile the entries double, the mean and the standard deviation stay exactly the same."""
    with d2.case("W03"):
        for cut in (FULL, (3.0, 9.0)):
            where = f"W03 weights W and W2, A {cut}"
            d2.set(where, weights=["W", "W2"], A=cut)
            r = d2.page.read(hist=["histoXYData[W]", "histoXYData[W2]"], prof=[("histoXYData_1", PROF)])
            d2.check(r, where, hists=["histoXYData[W]", "histoXYData[W2]"], profs=["histoXYData_1"])
            drv.same_values(r["hist:histoXYData[W2]"]["count"], 2 * r["hist:histoXYData[W]"]["count"], where + ": W2 = 2 W")
            a, b = d2.series(r["prof:histoXYData_1"], 0, 2), d2.series(r["prof:histoXYData_1"], 1, 2)
            drv.same_values(b["entries"], 2 * a["entries"], where + ": profile entries W2 = 2 W")
            for c in ("mean", "std"):
                drv.same_values(b[c], a[c], f"{where}: profile {c}, W2 against W, exactly")


@browser
@pytest.mark.feature("HIST.histogram_nd")
def test_w04_weighted_2d_and_3d_histograms_equal_numpy_under_cuts(d2):
    """W04. W and W2, three cuts of A: the weighted 2D and 3D histograms equal numpy's weighted histograms bin by bin,
    exactly."""
    with d2.case("W04"):
        d2.set("W04 weights W and W2", weights=["W", "W2"])
        names = ["histoXYData[W]", "histoXYData[W2]", "histoXYZData[W]", "histoXYZData[W2]"]
        for cut in CUTS:
            where = f"W04 A {cut}"
            d2.set(where, A=cut)
            d2.check(d2.page.read(hist=names), where, hists=names)


@browser
@pytest.mark.feature("HIST.histogram_nd")
def test_w05_a_partition_by_weights_adds_up(d2):
    """W05. None, A>8 and A<=8 together, three cuts: the histograms of A>8 and A<=8 add up to the unweighted one, bin by
    bin, exactly, and so do the entries of their profile series; each equals numpy."""
    with d2.case("W05"):
        d2.set("W05 weights None, A>8, A<=8", weights=["None", "A>8", "A<=8"])
        names = ["histoXYData[None]", "histoXYData[A>8]", "histoXYData[A<=8]"]
        for cut in CUTS:
            where = f"W05 A {cut}"
            d2.set(where, A=cut)
            r = d2.page.read(hist=names, prof=[("histoXYData_1", PROF)])
            d2.check(r, where, hists=names, profs=["histoXYData_1"])
            h = [r["hist:" + n]["count"] for n in names]
            drv.same_values(h[1] + h[2], h[0], where + ": H(A>8) + H(A<=8) = H")
            p = [d2.series(r["prof:histoXYData_1"], k, 3)["entries"] for k in range(3)]
            drv.same_values(p[1] + p[2], p[0], where + ": profile entries")


@browser
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("FILTER.text_query")
def test_w06_a_weight_of_0_or_1_equals_a_selection(d2):
    """W06. With varY = B, the weight Wsel (1 where cat > 0, else 0) against no weight with the query
    `return cat > 0`, for A in [0.5, 16] and in [3, 9]: the same bins, exactly (the edges being the same:
    precondition, checked; not with all rows, because the sparse region above A = 16 has only cat 0, so the query
    narrows the range). The query is switched on first with all rows (`return true`), before any cut (WG-D3)."""
    with d2.case("W06"):
        everything = ("return true", lambda t: np.ones(len(t["A"]), bool))
        d2.set("W06 varY = B, query 0 on, return true", varY="B", query0=everything)
        for cut in ((0.5, 16.0), (3.0, 9.0)):
            d2.set(f"W06 weight Wsel, A {cut}", weights=["Wsel"], A=cut, query0=everything)
            by_weight = d2.page.read(hist=["histoXYData[Wsel]"])
            d2.check(by_weight, f"W06 Wsel, A {cut}", hists=["histoXYData[Wsel]"])
            d2.set(f"W06 query cat > 0, A {cut}", weights=["None"], query0=("return cat > 0", lambda t: t["cat"] > 0))
            by_query = d2.page.read(hist=["histoXYData[None]"])
            d2.check(by_query, f"W06 query, A {cut}", hists=["histoXYData[None]"])
            a, b = by_weight["hist:histoXYData[Wsel]"], by_query["hist:histoXYData[None]"]
            for k in range(2):
                drv.same_values(a["edges"][k], b["edges"][k], f"W06 A {cut}: precondition, the edges of axis {k}")
            drv.same_values(a["count"], b["count"], f"W06 A {cut}: weight Wsel = selection cat > 0")


@browser
@pytest.mark.feature("HIST.profile_unbinned")
def test_w07_the_weighted_profile_equals_numpy(d2):
    """W07. The weight W, three cuts, the last with the sparse X bins of 1 to 6 rows: per X bin the entries (the sum of
    the weights) equal numpy exactly, the weighted mean and standard deviation within 1e-12 (relative)."""
    with d2.case("W07"):
        d2.set("W07 weight W, varY = B", weights=["W"], varY="B")
        for cut in (FULL, (3.0, 9.0), (9.5, 18.5)):
            where = f"W07 A {cut}"
            d2.set(where, A=cut)
            r = d2.page.read(hist=["histoXYData[W]"], prof=[("histoXYData_1", PROF)])
            d2.check(r, where, hists=["histoXYData[W]"], profs=["histoXYData_1"])


@browser
@pytest.mark.feature("HIST.stack")
@pytest.mark.feature("HIST.histogram_nd")
def test_w08_several_weights_at_once(d2):
    """W08. None, W and A>8 together, all rows and A in [3, 9]: the stacked table holds the three series in this order,
    each equal to its histogram and to numpy; the profile holds three series, each equal to numpy."""
    with d2.case("W08"):
        for cut in (FULL, (3.0, 9.0)):
            where = f"W08 A {cut}"
            d2.set(where, weights=["None", "W", "A>8"], A=cut)
            names = ["histoXYData[None]", "histoXYData[W]", "histoXYData[A>8]"]
            r = d2.page.read(hist=names, stack=["histoXYData"], prof=[("histoXYData_1", PROF)])
            d2.check(r, where, hists=names, profs=["histoXYData_1"])
            s = r["stack:histoXYData"]
            assert s["active"] == ["None", "W", "A>8"] and s["length"] == 3 * len(r["hist:histoXYData[W]"]["count"]), s
            stacked = d2.page.pg.evaluate("""() => { const s = window.__g7.one(m => m.type === 'CDSStack' && m.name ===
              'histoXYData', 'stack'); return window.__g7.f64(s.get_column('bin_count'), 'b64') }""")
            drv.same_values(drv._f64(stacked), np.concatenate([r["hist:" + n]["count"] for n in names]),
                            where + ": the stacked counts are the three series in order")


@browser
@pytest.mark.feature("HIST.histogram_nd")
def test_w09_changing_the_weight_and_back(d2):
    """W09. None -> W -> W2 -> W1 -> None with the real list: after each change the histogram and the profile equal
    numpy; back at None, exactly the arrays at load (checked by the end of the test)."""
    with d2.case("W09"):
        for w in ("W", "W2", "W1", "None"):
            _weights_by_list(d2, [w], f"W09 weight {w}")
            r = d2.page.read(hist=[f"histoXYData[{w}]"], prof=[("histoXYData_1", PROF)])
            d2.check(r, f"W09 weight {w}", hists=[f"histoXYData[{w}]"], profs=["histoXYData_1"])


@browser
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("ALIAS.cdsalias")
def test_w10_a_plus_b_against_b_plus_a_with_weights(d2):
    """W10. The page's normalization (difference) of A+B against B+A, and of A*B against B*A, with the weights W, W2
    and A>8 together: every weighted sum lies in the middle bin of [-0.5, 0.5], equal to numpy; every profile series
    has mean exactly 0 and standard deviation exactly 0 where it has entries."""
    with d2.case("W10"):
        d2.set("W10 weights W, W2, A>8", weights=["W", "W2", "A>8"])
        names = [f"histoXYNormData[{w}]" for w in ("W", "W2", "A>8")] + [f"histoXYZNormData[{w}]" for w in ("W", "A>8")]
        r = d2.page.read(hist=names, prof=[("histoXYNormData_1", PROF)])
        d2.check(r, "W10", hists=names, profs=["histoXYNormData_1"])
        for k in range(3):
            s = d2.series(r["prof:histoXYNormData_1"], k, 3)
            full = s["entries"] > 0
            assert full.any(), "precondition: some X bins have entries"
            drv.every_value(s["mean"][full], 0.0, f"W10 series {k}: mean")
            drv.every_value(s["std"][full], 0.0, f"W10 series {k}: std")
        e = r["hist:histoXYNormData[W]"]["edges"][1]
        assert e[0] == -0.5 and e[-1] == 0.5, ("W10: the range of A+B - (B+A) must be [-0.5, 0.5]", e[0], e[-1])


@browser
@pytest.mark.xfail(strict=True, raises=drv.KnownDefect, reason=WT_D15)
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("DISPLAY.lazy_tabs")
def test_w11_bins_and_weight_changed_together_and_back(d2):
    """W11. The sequence of the WeightsRef page test, on this page, one control at a time as a user changes them:
    nbinsX 50, then the weight W2; nbinsY 60, then W; then nbinsX 100, nbinsY 120 and no weight (the state at load):
    each state equals numpy, and the end of the test checks that every array is exactly as at load. Today a strict
    expected failure, WT-D15: the weight chosen after the number of bins changed throws "Size mismatch" page errors from
    the hidden plots (the arrays are right)."""
    with d2.case("W11", allowed_errors=[r"^pageerror: Size mismatch$"]):
        n0 = len(d2.page.errors)
        steps = (("nbinsX 50", dict(nbinsX=50)), ("weight W2", dict(weights=["W2"])), ("nbinsY 60", dict(nbinsY=60)),
                 ("weight W", dict(weights=["W"])), ("nbinsX 100", dict(nbinsX=100)), ("nbinsY 120", dict(nbinsY=120)),
                 ("no weight", dict(weights=["None"])))
        for label, kw in steps:
            d2.set("W11 " + label, **kw)
            w = d2.cur["weights"][0]
            r = d2.page.read(hist=[f"histoXYData[{w}]", f"histoXYZData[{w}]"], prof=[("histoXYData_1", PROF)])
            d2.check(r, "W11 " + label, hists=[f"histoXYData[{w}]", f"histoXYZData[{w}]"], profs=["histoXYData_1"])
        errors = d2.page.errors[n0:]
        drv.known_defect(len(errors) > 0 and all(e == "pageerror: Size mismatch" for e in errors),
                         WT_D15 + f"; here {len(errors)} page errors, every array right")
        assert not errors, f"W11: page errors: {errors[:3]}"


@browser
@pytest.mark.feature("HIST.histogram_nd")
def test_w12_finer_and_coarser_bins_with_weights(d2):
    """W12. The weight W; nbinsX 100 (at load) -> 50 -> 25: each equals numpy, and each coarse bin holds exactly the
    sum of its two finer bins (the coarse edges being every second fine edge: precondition)."""
    with d2.case("W12"):
        d2.set("W12 weight W", weights=["W"])
        seen = {}
        for nb in (100, 50, 25):
            d2.set(f"W12 nbinsX {nb}", nbinsX=nb)
            r = d2.page.read(hist=["histoXYData[W]"])
            d2.check(r, f"W12 nbinsX {nb}", hists=["histoXYData[W]"])
            h = seen[nb] = r["hist:histoXYData[W]"]
            fine = seen.get(2 * nb)
            if fine is not None:
                drv.same_values(fine["edges"][0][::2], h["edges"][0], f"W12 precondition: the edges of {nb} and {2 * nb} bins")
                ny = h["nbins"][1]
                f2 = fine["count"].reshape(ny, 2 * nb)
                drv.same_values(f2[:, 0::2] + f2[:, 1::2], h["count"].reshape(ny, nb), f"W12 {nb} bins from {2 * nb}")


@browser
@pytest.mark.feature("HIST.histogram_nd")
def test_w13_the_weighted_3d_histogram_summed_over_z_gives_the_2d_one(d2):
    """W13. The weights W and A>8, three cuts: each weighted 3D histogram summed over Z equals the weighted 2D one, bin
    by bin, exactly (the X and Y edges being the same: precondition)."""
    with d2.case("W13"):
        d2.set("W13 weights W, A>8", weights=["W", "A>8"])
        for cut in CUTS:
            where = f"W13 A {cut}"
            d2.set(where, A=cut)
            names = [f"{f}[{w}]" for f in ("histoXYData", "histoXYZData") for w in ("W", "A>8")]
            r = d2.page.read(hist=names)
            d2.check(r, where, hists=names)
            for w in ("W", "A>8"):
                h3, h2 = r[f"hist:histoXYZData[{w}]"], r[f"hist:histoXYData[{w}]"]
                for a in range(2):
                    drv.same_values(h3["edges"][a], h2["edges"][a], f"{where}: precondition, edges of axis {a}")
                nx, ny, nz = h3["nbins"]
                drv.same_values(h3["count"].reshape(nz, ny * nx).sum(axis=0), h2["count"], f"{where}, {w}: 3D over Z = 2D")


@browser
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.feature("HIST.histogram_nd")
def test_w14_the_selection_grid_with_weights(d2):
    """W14. The weight W; A in {all, [3, 9], [9.5, 18.5]} x cat in {all, 1 and 3} x the query `return B <= 8` off or on
    = 12 states: the selected rows equal pandas row by row, the weighted 2D histogram equals numpy."""
    with d2.case("W14"):
        d2.set("W14 weight W", weights=["W"])
        q = ("return B <= 8", lambda t: t["B"] <= 8)
        for query in (None, q):
            for a in (FULL, (3.0, 9.0), (9.5, 18.5)):
                for cat in (["0", "1", "2", "3"], ["1", "3"]):
                    where = f"W14 A {a}, cat {cat}, query {'on' if query else 'off'}"
                    d2.set(where, A=a, cat=cat, query1=query)
                    d2.check(d2.page.read(hist=["histoXYData[W]"]), where, hists=["histoXYData[W]"])


@browser
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("FILTER.intersection")
def test_w15_real_slider_keys_and_list_with_weights(d2):
    """W15. The weight W2; the A slider moved with the real arrow keys on both handles, then cat 0 and 2 chosen with
    the real list: with the values read back, the rows equal pandas and the weighted histograms numpy."""
    with d2.case("W15"):
        p = d2.page
        d2.set("W15 weight W2", weights=["W2"])
        p.keys("range", "A", "lower", "ArrowRight", 5)
        p.keys("range", "A", "upper", "ArrowLeft", 6)
        p.waited("W15 slider A moved with the arrow keys")
        d2.read_back("A")
        lo, hi = d2.cur["A"]
        assert d2.full["A"][0] < lo < hi < d2.full["A"][1], ("the keys did not move both handles", lo, hi)
        names = ["histoXYData[W2]", "histoXYZData[W2]"]
        d2.check(p.read(hist=names, prof=[("histoXYData_1", PROF)]), f"W15 A [{lo!r}, {hi!r}]", hists=names,
                 profs=["histoXYData_1"])
        p.element("list", "cat", "input").select_option(["0", "2"])
        p.waited("W15 cat 0 and 2 (real list)")
        d2.read_back("cat")
        assert d2.cur["cat"] == ["0", "2"], d2.cur["cat"]
        d2.check(p.read(hist=names, prof=[("histoXYData_1", PROF)]), "W15 and cat 0, 2", hists=names,
                 profs=["histoXYData_1"])


@browser
@pytest.mark.feature("DISPLAY.lazy_tabs")
def test_w16_a_hidden_tab_shows_the_current_weights_when_opened(d2):
    """W16. The tab histoXYZ is hidden while the weight W and a cut of A are chosen; clicked, its drawn profile (the
    plots' own table histoXYZData_1) equals numpy; the weights W and W2 chosen while it is shown: two series, equal to
    numpy; hidden again (tab Invariances), the weight A>8 chosen, shown again: nothing old is left."""
    with d2.case("W16"):
        p = d2.page

        def drawn(where):
            p.waited(where)
            r = p.read(drawn=[("histoXYZData_1", ["entries", "mean"])])
            got = r["drawn:histoXYZData_1"]
            assert got["__watched"] and not got["__needs_update"], (where, "the drawn table is not up to date")
            series = d2.prof("histoXYZData_1")
            drv.same_values(got["entries"], np.concatenate([e["entries"] for _w, e in series]), where + ", drawn entries")
            drv.scaled_close(got["mean"], np.concatenate([e["mean"] for _w, e in series]), where + ", drawn mean", REL)

        d2.set("W16 weight W, A [3, 9] while histoXYZ is hidden", weights=["W"], A=(3.0, 9.0))
        p.click_tab("histoXYZ")
        drawn("W16 histoXYZ clicked")
        d2.set("W16 weights W and W2 while shown", weights=["W", "W2"])
        drawn("W16 after the change while shown")
        p.click_tab("Invariances")
        p.waited("W16 Invariances clicked")
        d2.set("W16 weight A>8 while hidden", weights=["A>8"], A=(0.5, 4.5))
        p.click_tab("histoXYZ")
        drawn("W16 histoXYZ clicked again")
        p.click_tab("Invariances")


@browser
@pytest.mark.xfail(strict=True, raises=drv.KnownDefect, reason=WT_D15)
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("DISPLAY.lazy_tabs")
def test_w17_the_histograms_of_a_variable_list(d2):
    """W17. histo1D (varX) and the stacked histo1DMulti (the list varXMulti, at load A and B): each series equals numpy,
    and histo1DMulti's series A equals histo1D bin by bin. Then, with the real list in the tab Histograms (the plots'
    tabs hidden), varXMulti becomes B alone, then B and C: each state equals numpy, and no page error may appear.
    Today a strict expected failure, WT-D15: choosing two variables after one throws "Size mismatch" page errors from
    the hidden plots of the list (the arrays are right)."""
    with d2.case("W17", allowed_errors=[r"^pageerror: Size mismatch$"]):
        p = d2.page
        r = p.read(hist1=[("histo1D", []), ("histo1DMulti_A", []), ("histo1DMulti_B", [])], stack=["histo1DMulti"])
        d2.check(r, "W17 at load", hist1=[("histo1D", []), ("histo1DMulti_A", []), ("histo1DMulti_B", [])])
        drv.same_histogram(r["hist:histo1DMulti_A"], r["hist:histo1D"], "W17 histo1DMulti A against histo1D")
        assert r["stack:histo1DMulti"]["active"] == ["A", "B"], r["stack:histo1DMulti"]
        d2.set("W17 A [3, 9]", A=(3.0, 9.0))
        p.click_tab("Histograms")
        errors = {}
        for chosen in (["B"], ["B", "C"]):
            n0 = len(p.errors)
            p.element("list", "varXMulti", "input").select_option(chosen)
            p.waited(f"W17 varXMulti {chosen} (real list)")
            d2.read_back("varXMulti")
            assert d2.cur["varXMulti"] == chosen, d2.cur["varXMulti"]
            names = [(f"histo1DMulti_{v}", []) for v in chosen]
            r = p.read(hist1=names, stack=["histo1DMulti"], hist=[f"histoNDMultiX[{v}]" for v in chosen])
            d2.check(r, f"W17 varXMulti {chosen}", hist1=names, hists=[f"histoNDMultiX[{v}]" for v in chosen])
            s = r["stack:histo1DMulti"]
            assert s["active"] == chosen and s["length"] == len(chosen) * int(d2.cur["nbinsX"]), s
            errors[len(chosen)] = p.errors[n0:]
        p.click_tab("Select")
        new = errors[1] + errors[2]
        drv.known_defect(len(new) > 0 and all(e == "pageerror: Size mismatch" for e in new),
                         WT_D15 + f"; here {len(errors[1])} page errors when B alone was chosen, {len(errors[2])} when B "
                         f"and C, every array right")
        assert not new, f"W17: page errors: {new[:3]}"


@browser
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.profile_unbinned")
def test_w18_a_multi_list_histogram_equals_the_templates_own(d2):
    """W18. The histogram of the list varXMulti for A (A against varY) is the template's histoXYData without weight by
    another code path: equal bin by bin, with its profile; the list varYMulti set to A+B and B (varY is A+B at load):
    the histogram for A+B (varX against A+B) equals histoXYData too. Two cuts. (The list keeps two variables: fewer,
    then more, is WT-D15, tested in W17.)"""
    with d2.case("W18"):
        d2.set("W18 varYMulti A+B and B", varYMulti=["A+B", "B"])
        for cut in (FULL, (3.0, 9.0)):
            where = f"W18 A {cut}"
            d2.set(where, A=cut)
            names = ["histoNDMultiX[A]", "histoNDMultiY[A+B]", "histoXYData[None]"]
            r = d2.page.read(hist=names, prof=[("histoNDMultiX_1_histoNDMultiX[A]", PROF), ("histoXYData_1", PROF)])
            d2.check(r, where, hists=names, profs=["histoNDMultiX_1_histoNDMultiX[A]", "histoXYData_1"])
            for n in names[:2]:
                drv.same_histogram(r["hist:" + n], r["hist:histoXYData[None]"], f"{where}: {n} against histoXYData")
            for c in ("entries", "mean", "std", "quantile_0", "quantile_1"):
                drv.same_values(r["prof:histoNDMultiX_1_histoNDMultiX[A]"][c], r["prof:histoXYData_1"][c],
                                f"{where}: profile {c} of the list's histogram against the template's")


@browser
@pytest.mark.feature("HIST.histogram_1d")
def test_w19_the_invariance_tab_holds_under_cuts(d2):
    """W19. The histogram of the tab Invariances under three cuts: its weighted counts equal numpy, and W1 equals the
    counts, W2 is twice W, (A>8) + (A<=8) equals the counts, bin by bin, exactly - what the four plots show."""
    with d2.case("W19"):
        extra = sorted(pages.WEIGHT_COUNTS)
        for cut in CUTS:
            where = f"W19 A {cut}"
            d2.set(where, A=cut)
            r = d2.page.read(hist1=[("histoInvW", extra)])
            d2.check(r, where, hist1=[("histoInvW", extra)])
            h = r["hist:histoInvW"]
            drv.same_values(h["nW1"], h["count"], where + ": W1 = no weight")
            drv.same_values(h["nW2"], 2 * h["nW"], where + ": W2 = 2 W")
            drv.same_values(h["nP"] + h["nNotP"], h["count"], where + ": (A>8) + (A<=8) = no weight")


@browser
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.histogram_1d")
def test_w20_the_whole_sequence_leaves_no_trace(d2):
    """W20. After W01 to W19 the page gives exactly the arrays it gave at load, and these equal numpy at the baseline:
    all rows, the unweighted series, the 1D histograms and the tab Invariances."""
    with d2.case("W20"):
        fp = d2.page.fingerprint()
        dirty = [k for k in d2.page.fingerprint_at_load if fp.get(k) != d2.page.fingerprint_at_load[k]]
        assert not dirty, f"W20: arrays differ from those at load: {dirty}"
        names = [f"{f}[None]" for f in sorted(WEIGHTED)] + ["histoNDMultiX[A]", "histoNDMultiY[B]"]
        extra = sorted(pages.WEIGHT_COUNTS)
        r = d2.page.read(hist=names, hist1=[("histo1D", []), ("histoInvW", extra)],
                         prof=[("histoXYData_1", PROF), ("histoXYNormData_1", PROF)], stack=["histoXYData"])
        assert r["mask"].all(), "W20: at load all rows are selected"
        d2.check(r, "W20 baseline", hists=names, hist1=[("histo1D", []), ("histoInvW", extra)],
                 profs=["histoXYData_1", "histoXYNormData_1"])
        assert r["stack:histoXYData"]["active"] == ["None"], r["stack:histoXYData"]
    if d2.page.notes:
        pytest.fail("problems left by earlier tests that failed: " + "; ".join(d2.page.notes), pytrace=False)


# ================================================================== without a browser
@pytest.mark.headless_unit
def test_the_weights_table_and_the_weighted_oracle():
    """The table of D2: W in eighths from 1 to 2, W1 = 1, W2 = 2 W, Wsel = (cat > 0); and the weighted oracle on a
    hand-checked cell: weights 1, 3 on the values 2, 6: entries 4, mean 5, std sqrt(3); a weight 0 counts for nothing."""
    t = pages.weights_table(4000)
    assert set(np.unique(t.W * 8)) <= set(range(8, 17)) and (t.W1 == 1).all() and (t.W2 == 2 * t.W).all()
    assert ((t.Wsel == 1) == (t.cat > 0)).all()
    x = np.zeros(3)
    y = np.array([2.0, 6.0, 100.0])
    p = O.profile([x, y], [1, 1], np.ones(3, bool), 1, [], weights=np.array([1.0, 3.0, 0.0]))
    assert p["entries"].tolist() == [4.0] and p["mean"][0] == 5.0
    assert np.isclose(p["std"][0], np.sqrt(3.0), rtol=1e-15)
    h = O.histogram([y], [2], np.ones(3, bool), weights=np.array([1.0, 3.0, 0.5]))
    assert h["count"].tolist() == [4.0, 0.5], h["count"]
