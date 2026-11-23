"""PHASE_5 Group 7, dashboard D1: twenty invariance and oracle tests on one live template page (Claude21.RI,
2026-11-22).

In plain words:
- **The page:** the dashboard template without weights (`getDefaultVarsNormAll`), on 100,000 rows of exact numbers
  (dashboards/invariance_dashboards.py, which you can run yourself to build and open the same page). Its control
  "diffFunc" switches live between the difference, the ratio and the log-ratio of varY and varYNorm (the alias diffY)
  and of varZ and varZNorm (diffZ). Its first tab, "Invariances", draws four browser-computed aliases that must be
  0, 0, 0 and 2 in every row.
- **One page, twenty tests:** the page is opened once; the tests run on it one after the other. Every test sets fixed
  states (no random actions), and after each state compares the page with numpy and pandas: the selected rows one by
  one, each histogram bin by bin with the same edges bit for bit, profiles cell by cell. An **invariance** compares two
  definitions that must agree (A+B normalized to B+A is exactly 0 or 1); an **oracle** is the numpy/pandas answer.
- **Each test ends at the state of the page at load:** all controls back; the page must then give exactly its arrays
  at load, or the test fails and the page is opened again (dashboards/invariance_driver.py).
- **Real mouse and keys:** T06 (slider keys), T07 (the list), T16 (the spinner typed), T17 (a query typed, Enter, the
  on/off button), T18 (tab clicks).
- **Timing:** every state is timed (the page's own time to compute, draw and settle, the time to read); the record is
  written to headless_out/invariance/template/timings.json, with the page and its manifest.
- **Exact:** the table is made of multiples of 1/8, none zero, so A+B, B+A, (A+B)-B and (A+A)/A are exact, and the
  normalized values are exactly 0, 1 or 2. Only Math.log in the browser and numpy's log may differ in the last bit:
  there the comparison allows 2e-15, and the histogram of such a column is predicted from the page's own values.

- **Two defects found, each a strict expected failure with its exact result (H_BE2 and WG_D3 below):** T04, the
  log-ratio of A+A against A, whose histogram the page refuses; T09, on a page just opened, the first text query
  switched on after other cuts drops those cuts (the table has a multiple of 32 rows).

Words used: a **state** is one setting of the controls; the **baseline** is the state at load; a **precondition** is a
property of the data a test relies on, checked first so that a data change can never make a test pass for nothing; a
**strict expected failure** fails with exactly the described wrong result until the defect is fixed.

Switches: RI_INVARIANCE_ROWS (default 100000) builds a smaller or larger table; RI_INVARIANCE_FRESH_PAGE=1 opens the
page again after every test, so that each test runs alone (no result may depend on the order).

Target: browser_invariance_dashboards (the three dashboards of Group 7; all tests of one page run in one pytest-xdist
worker, `xdist_group`, with `--dist loadgroup`). The tests without a browser (headless_unit) check the table and that
the comparisons catch wrong answers.
"""
from __future__ import annotations

import json
import os
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
FULL = "full"                                         # a range slider at its full range (as at load)
CUTS = [(3.0, 9.0), (0.5, 4.5), (9.5, 18.5)]          # A ranges; the last one holds the sparse region (A >= 16.5)
LOG_ATOL = 2e-15                                      # Math.log against numpy's log (see the module text)
REL = 1e-12                                           # means, standard deviations and quantiles summed in another order
PROF2 = ["entries", "mean", "std", "quantile_0", "quantile_1"]
KINDS = {**{c: "range" for c in "ABCD"}, "cat": "list", "DDC": "list",
         **{k: "select" for k in ("varX", "varY", "varYNorm", "varZ", "varZNorm", "diffFuncTransform")},
         **{k: "spinner" for k in ("nbinsX", "nbinsY", "nbinsZ", "exponentX")},
         **{f"funCustomForm{k}": "text" for k in range(3)}}
CONTROLS = ([(kind, key) for key, kind in KINDS.items()] + [("query", k) for k in range(3)]
            + [("off", k) for k in range(3)] + [("tabs", "Invariances"), ("tabs", "Select")])
PROBES = [["histoXYData", "bin_count", ["HistoNdCDS"]]]
# the histograms of the page and the columns they are made of (names of the template's parameters and aliases)
HIST_AXES = {"histoXYData": ["varX", "varY"], "histoXYZData": ["varX", "varY", "varZ"],
             "histoXYNormData": ["varX", "diffY"], "histoXYNormZData": ["varX", "diffY", "varZ"],
             "histoXYZNormData": ["varX", "varY", "diffZ"],
             **{"histo_" + n: ["varX", n] for n, _e, _v, _t in pages.INVARIANCES}}
NBINS = {"varX": "nbinsX", "varY": "nbinsY", "diffY": "nbinsY", "varZ": "nbinsZ", "diffZ": "nbinsZ",
         **{n: "nbinsY" for n, _e, _v, _t in pages.INVARIANCES}}


# Defects found by these tests (new in Group 7); each is a strict expected failure until it is fixed:
H_BE2 = ("H-BE2 (new, Group 7; decision needed): the live log-ratio of A+A against A gives values that are equal up to "
         "the last float steps (Math.log), so the normalized histogram's automatic range is a few float steps wide but "
         "not empty; the page refuses it (RangeError: its bin edges are not strictly increasing in binary64) and the "
         "normalized histograms are not drawn. numpy.histogram draws such a range (its 121 edges hold 5 distinct "
         "values). The NumPy rule of L-46 covers two equal ends only")
WG_D3 = ("WG-D3 (new, Group 7): on a page just opened whose table has a multiple of 32 rows (100,000, or the 500,000 of "
         "the example dashboards), the first text query switched on after other cuts drops those cuts: the selection "
         "becomes the query's rows alone, and removing the cuts later loses rows. Cause (LazyIntersectionFilter.ts): "
         "the counts per row are kept in words of 32 rows; a range filter gives ceil(n/32) words, a text query "
         "floor(n/32) + 1, so at that moment the counts grow by one word and resize_counts replaces them with new, empty "
         "ones. Measured: keeping the old counts when they grow gives the right rows")


def browser(fn):
    """A test of this page: the target marker, and the group that keeps the twenty tests on one worker."""
    for mark in (pytest.mark.browser_invariance_dashboards, pytest.mark.backend("browser"), pytest.mark.layer("invariance"),
                 pytest.mark.xdist_group("invariance_template")):
        fn = mark(fn)
    return fn


# ------------------------------------------------------------------ the page, and what it must show
class D1:
    """The shared page with the current state of its controls, and the expected arrays for that state."""

    def __init__(self, page: drv.Page, table: dict):
        self.page, self.t, self.n = page, table, len(table["A"])
        self.reset()

    def reset(self):
        v = {key: val for (kind, key), val in self.page.at_load.items() if kind in ("range", "list", "select",
                                                                                     "spinner", "text")}
        self.cur = dict(v)
        self.full = {c: tuple(v[c]) for c in "ABCD"}
        self.queries = {}                              # query slot -> pandas equivalent, while switched on
        self.functions = {}                            # funCustomK -> numpy equivalent of its text

    @contextmanager
    def case(self, name: str, allowed_errors=()):
        """One test: the page's case (restore and check at the end), and the expected state back to the load state."""
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
            if key.startswith("funCustomForm"):
                text, fn = value
                page_kw[key] = text
                self.functions["funCustom" + key[-1]] = fn
                self.cur[key] = text
                continue
            if value == FULL:
                value = self.full[key]
            page_kw[key] = value
            self.cur[key] = list(value) if isinstance(value, tuple) else value
        return self.page.set(label, **page_kw)

    # -- expected values
    def mask(self) -> np.ndarray:
        key = (tuple(tuple(self.cur[c]) for c in "ABCD"), tuple(self.cur["cat"]), tuple(self.cur["DDC"]),
               tuple(sorted((k, id(f)) for k, f in self.queries.items())))
        if getattr(self, "_mask_key", None) != key:            # computed once per state
            self._mask_key = key
            self._mask = O.selection(self.t, {c: self.cur[c] for c in "ABCD"},
                                     {"cat": self.cur["cat"], "DDC": self.cur["DDC"]}, list(self.queries.values()))
        return self._mask

    def var(self, name: str) -> np.ndarray:
        fns = {f"funCustom{k}": (lambda t: np.ones(len(t["A"]))) for k in range(3)}
        fns.update(self.functions)
        return O.variable(self.t, name, {"exponentX": float(self.cur["exponentX"])}, fns)

    def column(self, name: str, read: dict = None) -> np.ndarray:
        """A histogram axis: a parameter (varX ...), diffY / diffZ, or an alias of the tab Invariances. A log-ratio is
        taken from the page's own column when `read` holds it (checked against numpy before, within LOG_ATOL)."""
        if name in ("diffY", "diffZ"):
            how = self.cur["diffFuncTransform"]
            if how == "logRatio" and read is not None and ("alias:" + name) in read:
                return read["alias:" + name]
            y, n = ("varY", "varYNorm") if name == "diffY" else ("varZ", "varZNorm")
            return O.transform(self.var(self.cur[y]), self.var(self.cur[n]), how)
        for n, expr, _v, _t in pages.INVARIANCES:
            if name == n:
                return O.variable(self.t, expr, {}, {})
        return self.var(self.cur[name])

    def nbins(self, hist: str) -> list:
        return [int(self.cur[NBINS[a]]) for a in HIST_AXES[hist]]

    def hist(self, name: str, read: dict = None, mask=None) -> dict:
        cols = [self.column(a, read) for a in HIST_AXES[name]]
        return O.histogram(cols, self.nbins(name), self.mask() if mask is None else mask)

    def prof(self, name: str, read: dict = None) -> dict:
        h, axis = name.rsplit("_", 1)
        cols = [self.column(a, read) for a in HIST_AXES[h]]
        nq = 1 if h.startswith("histo_inv") else 2
        return O.profile(cols, self.nbins(h), self.mask(), int(axis), [0.35, 0.5][:nq] if nq == 2 else [0.5])

    # -- checks against the expected values
    def check(self, r: dict, where: str, hists=(), profs=(), exact_profiles=False):
        if "mask" in r:
            drv.same_rows(r["mask"], self.mask(), where + ", the selected rows")
            assert r["mask_padding_selected"] == 0, f"{where}: rows past the end of the table are selected"
        for h in hists:
            drv.same_histogram(r["hist:" + h], self.hist(h, r), f"{where}, histogram {h}")
        for p in profs:
            e = self.prof(p, r)
            got = r["prof:" + p]
            drv.same_values(got["entries"], e["entries"], f"{where}, profile {p}, entries")
            for c in [c for c in got if c != "entries" and got[c] is not None]:
                if exact_profiles:
                    drv.same_values(got[c], e[c], f"{where}, profile {p}, {c}")
                else:
                    drv.scaled_close(got[c], e[c], f"{where}, profile {p}, {c}", REL)


def _build(out: Path) -> dict:
    """The page, built by the same command a person runs (a child process; the compile cache is reused)."""
    cmd = [sys.executable, "-m", "RootInteractive.InteractiveDrawing.bokeh.headless.dashboards.invariance_dashboards",
           "template", "--rows", str(ROWS), "--output", str(out / "page.html")]
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


@pytest.fixture(scope="module")
def d1(headless_session, headless_out):
    out = headless_out / "invariance" / "template"
    out.mkdir(parents=True, exist_ok=True)
    manifest = _build(out)
    df = pages.template_table(ROWS)
    assert manifest["table_sha256"] == pages.table_sha256(df), "the page was built from another table"
    table = {c: df[c].to_numpy() for c in df.columns}
    page = drv.Page(headless_session, out / "page.html", KINDS, CONTROLS, fingerprint_spec={}, probes=PROBES, out_dir=out)
    # what the end of every test compares with the load: every histogram and profile of the page (read from the open
    # page), the selection, and the alias columns
    inv = page.pg.evaluate("() => window.__g7.inventory()")
    page.fingerprint_spec = {
        "mask": True, "hist": inv["hist"],
        "prof": [[n, ["entries", "mean", "std"] + [f"quantile_{k}" for k in range(q)]] for n, q in inv["prof"]],
        "alias": ["diffY", "diffZ", "funCustom0", "funCustom1", "funCustom2"] + [n for n, _e, _v, _t in pages.INVARIANCES]}
    page.fingerprint_at_load = page.fingerprint()
    page.pg.screenshot(path=str(out / "at_load_tab_Invariances.png"))
    d = D1(page, table)
    d.manifest, d.out, d.inventory = manifest, out, inv
    yield d
    summary = page.write_timings(out / "timings.json")
    page.pg.screenshot(path=str(out / "at_the_end_tab_Invariances.png"))
    print("\nD1 timing:", json.dumps({k: summary[k] for k in ("load_s", "states", "page_ms_median", "page_ms_p90",
                                                               "read_s_total", "tests_s_total")}))
    page.close()


def _transforms_value(how: str) -> float:
    """What an invariance gives: the ratio 1, the difference and the log-ratio 0."""
    return 1.0 if how == "ratio" else 0.0


# ================================================================== the twenty tests
@browser
@pytest.mark.feature("ALIAS.cdsalias")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.profile_unbinned")
def test_t01_a_plus_b_normalized_to_b_plus_a_is_exactly_0_or_1_in_every_row_bin_and_profile(d1):
    """T01. varY = A+B normalized to varYNorm = B+A, for the difference, the ratio and the log-ratio, with all rows and
    with A in [3, 9]: diffY is exactly 0, 1, 0 in every row; the normalized histogram equals numpy (all rows in the
    middle bin of the range [c - 0.5, c + 0.5]); its profile has mean exactly c and std exactly 0 in every X bin."""
    with d1.case("T01"):
        for how in O.TRANSFORMS:
            for cut in (FULL, (3.0, 9.0)):
                where = f"T01 {how}, A {cut}"
                d1.set(where, varY="A+B", varYNorm="B+A", diffFuncTransform=how, A=cut)
                r = d1.page.read(hist=["histoXYNormData"], prof=[("histoXYNormData_1", PROF2)], alias=["diffY"])
                drv.every_value(r["alias:diffY"], _transforms_value(how), where + ", diffY")
                d1.check(r, where, hists=["histoXYNormData"], profs=["histoXYNormData_1"], exact_profiles=True)


@browser
@pytest.mark.feature("ALIAS.cdsalias")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.profile_unbinned")
def test_t02_a_times_b_normalized_to_b_times_a_is_exactly_0_or_1(d1):
    """T02. varY = A*B normalized to varYNorm = B*A, three modes: exactly 0, 1, 0 in every row, bin and profile.
    The page opens with varZ = A*B and varZNorm = B*A: diffZ is checked the same way, with its histogram."""
    with d1.case("T02"):
        for how in O.TRANSFORMS:
            where = f"T02 {how}"
            d1.set(where, varY="A*B", varYNorm="B*A", diffFuncTransform=how)
            r = d1.page.read(hist=["histoXYNormData", "histoXYZNormData"], prof=[("histoXYNormData_1", PROF2)],
                             alias=["diffY", "diffZ"])
            drv.every_value(r["alias:diffY"], _transforms_value(how), where + ", diffY")
            drv.every_value(r["alias:diffZ"], _transforms_value(how), where + ", diffZ (A*B against B*A)")
            d1.check(r, where, hists=["histoXYNormData", "histoXYZNormData"], profs=["histoXYNormData_1"],
                     exact_profiles=True)


@browser
@pytest.mark.feature("ALIAS.cdsalias")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.profile_unbinned")
def test_t03_a_plus_b_minus_b_normalized_to_a_is_exactly_0_or_1(d1):
    """T03. varY = (A+B)-B normalized to varYNorm = A, difference and ratio, all rows and A in [3, 9]: exactly 0 or 1
    in every row, bin and profile."""
    with d1.case("T03"):
        for how in ("diff", "ratio"):
            for cut in (FULL, (3.0, 9.0)):
                where = f"T03 {how}, A {cut}"
                d1.set(where, varY="(A+B)-B", varYNorm="A", diffFuncTransform=how, A=cut)
                r = d1.page.read(hist=["histoXYNormData"], prof=[("histoXYNormData_1", PROF2)], alias=["diffY"])
                drv.every_value(r["alias:diffY"], _transforms_value(how), where + ", diffY")
                d1.check(r, where, hists=["histoXYNormData"], profs=["histoXYNormData_1"], exact_profiles=True)


@browser
@pytest.mark.xfail(strict=True, raises=drv.KnownDefect, reason=H_BE2)
@pytest.mark.feature("ALIAS.cdsalias")
@pytest.mark.feature("HIST.histogram_nd")
def test_t04_a_variable_against_itself_and_against_its_double(d1):
    """T04. C normalized to C: exactly 0, 1, 0. A+A normalized to A (review correction C1, a value that is not 0 or 1,
    so a transform that returns a constant cannot pass): the difference equals A row by row and the ratio is exactly 2
    in every row, with histograms and profiles equal to numpy; the log-ratio column equals numpy's log(2A) - log(A)
    within 2e-15, and then its histogram is read: today a strict expected failure, H-BE2 (the page refuses the range
    a few float steps wide, with the exact message; the oracle's own edge rule refuses it too)."""
    from playwright.sync_api import Error as PlaywrightError
    from .oracles import histograms as H
    with d1.case("T04"):
        for how in O.TRANSFORMS:
            where = f"T04 C against C, {how}"
            d1.set(where, varY="C", varYNorm="C", diffFuncTransform=how)
            r = d1.page.read(hist=["histoXYNormData"], prof=[("histoXYNormData_1", PROF2)], alias=["diffY"])
            drv.every_value(r["alias:diffY"], _transforms_value(how), where + ", diffY")
            d1.check(r, where, hists=["histoXYNormData"], profs=["histoXYNormData_1"], exact_profiles=True)
        a = d1.t["A"]
        for how in ("diff", "ratio"):
            where = f"T04 A+A against A, {how}"
            d1.set(where, varY="A+A", varYNorm="A", diffFuncTransform=how)
            r = d1.page.read(hist=["histoXYNormData"], prof=[("histoXYNormData_1", PROF2)], alias=["diffY"])
            if how == "ratio":
                drv.every_value(r["alias:diffY"], 2.0, where + ", diffY")
            else:
                drv.same_values(r["alias:diffY"], a, where + ", diffY equals A")
            d1.check(r, where, hists=["histoXYNormData"], profs=["histoXYNormData_1"])
        where = "T04 A+A against A, logRatio"
        d1.set(where, varY="A+A", varYNorm="A", diffFuncTransform="logRatio")
        v = d1.page.read(mask=False, alias=["diffY"])["alias:diffY"]
        drv.same_values(v, np.log(a + a) - np.log(a), where + ", diffY", atol=LOG_ATOL)
        lo, hi, nb = float(v.min()), float(v.max()), int(d1.cur["nbinsY"])
        try:
            r = d1.page.read(hist=["histoXYNormData"], prof=[("histoXYNormData_1", PROF2)], alias=["diffY"])
        except PlaywrightError as exc:
            message = (f'RangeError: Histogram "histoXYNormData", axis 1: the bin edges of the range [{lo!r}, {hi!r}] '
                       f'with {nb} bins are not strictly increasing in binary64')
            try:
                H.canonical_edges(nb, lo, hi)
                refused_by_the_rule = False
            except ValueError:
                refused_by_the_rule = True
            drv.known_defect(lo < hi and refused_by_the_rule and message in str(exc),
                             H_BE2 + f"; here the range [{lo!r}, {hi!r}], {nb} bins")
            raise
        d1.check(r, where, hists=["histoXYNormData"], profs=["histoXYNormData_1"])


@browser
@pytest.mark.feature("ALIAS.cdsalias")
@pytest.mark.feature("HIST.histogram_nd")
def test_t05_the_two_sides_swapped(d1):
    """T05. B normalized to C, then C normalized to B, three modes. Row by row: the difference and the ratio equal
    numpy exactly, the log-ratio within 2e-15; swapped, the difference and the log-ratio change sign exactly, and the
    ratio times its swapped ratio is 1 within two float steps. Histograms and profiles equal numpy."""
    with d1.case("T05"):
        b, c = d1.t["B"], d1.t["C"]
        got = {}
        for y, n in (("B", "C"), ("C", "B")):
            for how in O.TRANSFORMS:
                where = f"T05 {y} against {n}, {how}"
                d1.set(where, varY=y, varYNorm=n, diffFuncTransform=how)
                r = d1.page.read(hist=["histoXYNormData"], prof=[("histoXYNormData_1", PROF2)], alias=["diffY"])
                exp = O.transform(d1.t[y], d1.t[n], how)
                drv.same_values(r["alias:diffY"], exp, where + ", diffY", atol=LOG_ATOL if how == "logRatio" else 0)
                d1.check(r, where, hists=["histoXYNormData"], profs=["histoXYNormData_1"])
                got[(y, n, how)] = r["alias:diffY"]
        for how in O.TRANSFORMS:
            one, two = got[("B", "C", how)], got[("C", "B", how)]
            if how == "ratio":
                drv.same_values(one * two, np.ones_like(one), "T05 ratio B/C times C/B", atol=4.5e-16)
            else:
                drv.same_values(two, -one, f"T05 {how}: swapped is the negative, exactly")
        assert np.any(b != c), "precondition: B and C differ in some rows"


@browser
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("FILTER.text_query")
def test_t06_a_slider_and_the_same_text_query_select_the_same_rows_and_bins(d1):
    """T06. The first query is switched on with `return true` (all rows). Three ranges of A; the first one is set with
    the real arrow keys on the two handles of the slider. Each range, then the text query `return lo <= A && A <= hi`
    with the values read back from the slider (the slider at its full range): the same rows, row by row, the same
    histograms bin by bin, and both equal pandas and numpy."""
    with d1.case("T06"):
        p = d1.page
        everything = ("return true", lambda t: np.ones(len(t["A"]), bool))
        d1.set("T06 query 0 on, return true", query0=everything)
        p.keys("range", "A", "lower", "ArrowRight", 4)
        p.keys("range", "A", "upper", "ArrowLeft", 7)
        p.waited("T06 slider A moved with the arrow keys")
        lo, hi = p.values()[("range", "A")]
        assert d1.full["A"][0] < lo < hi < d1.full["A"][1], ("the keys did not move both handles", lo, hi)
        d1.cur["A"] = [lo, hi]
        hists = ["histoXYData", "histoXYNormData"]
        for k, (lo, hi) in enumerate([(lo, hi), (3.0, 9.0), (9.5, 18.5)]):
            where = f"T06 slider A [{lo!r}, {hi!r}]"
            if k > 0:
                d1.set(where, A=(lo, hi), query0=everything)
            by_slider = p.read(hist=hists)
            d1.check(by_slider, where, hists=hists)
            text = f"return {lo!r} <= A && A <= {hi!r}"
            d1.set(f"T06 query {text}", A=FULL, query0=(text, lambda t, lo=lo, hi=hi: (t["A"] >= lo) & (t["A"] <= hi)))
            by_query = p.read(hist=hists)
            d1.check(by_query, f"T06 query {text}", hists=hists)
            drv.same_rows(by_query["mask"], by_slider["mask"], f"T06 query against slider [{lo!r}, {hi!r}]")
            for h in hists:
                drv.same_histogram(by_query["hist:" + h], by_slider["hist:" + h], f"T06 {h}, query against slider")


@browser
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.feature("FILTER.text_query")
def test_t07_a_category_list_and_the_same_text_query_select_the_same_rows(d1):
    """T07. The first query switched on with `return true`. The list cat: 0 and 1 chosen with the real list
    (select_option), then 2 alone; each against the text query `return cat == 0 || cat == 1` (or `cat == 2`) with the
    list back to all: the same rows and bins, equal to pandas."""
    with d1.case("T07"):
        p = d1.page
        everything = ("return true", lambda t: np.ones(len(t["A"]), bool))
        d1.set("T07 query 0 on, return true", query0=everything)
        cases = [(["0", "1"], "return cat == 0 || cat == 1", lambda t: (t["cat"] == 0) | (t["cat"] == 1)),
                 (["2"], "return cat == 2", lambda t: t["cat"] == 2)]
        for k, (chosen, text, fn) in enumerate(cases):
            where = f"T07 list cat {chosen}"
            if k == 0:
                p.element("list", "cat", "input").select_option(chosen)
                p.waited(where + " (real list)")
                d1.cur["cat"] = list(p.values()[("list", "cat")])
                assert d1.cur["cat"] == chosen, d1.cur["cat"]
                d1.set(where + ", query return true", query0=everything)
            else:
                d1.set(where, cat=chosen, query0=everything)
            by_list = p.read(hist=["histoXYData"])
            d1.check(by_list, where, hists=["histoXYData"])
            d1.set(f"T07 query {text}", cat=["0", "1", "2", "3"], query0=(text, fn))
            by_query = p.read(hist=["histoXYData"])
            d1.check(by_query, f"T07 query {text}", hists=["histoXYData"])
            drv.same_rows(by_query["mask"], by_list["mask"], f"T07 query against the list {chosen}")


@browser
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_nd")
def test_t08_the_selection_grid_and_an_empty_selection(d1):
    """T08. A in {all, [3, 9], [9.5, 18.5]} x cat in {all, 1 and 3} x DDC in {all, A0 and A2} x the query
    `return B <= 8` off or on (in the second query slot) = 24 states: the selected rows equal pandas row by row (the rows
    on the slider ends are kept), the 2D histogram equals numpy bin by bin. Then (review correction C3) an empty
    selection, A in [3.25, 3.4] where no row lies: no row, every count 0, the range [0, 1], every profile cell empty;
    then the recovery to all rows equals numpy."""
    with d1.case("T08"):
        q = ("return B <= 8", lambda t: t["B"] <= 8)
        for query in (None, q):                     # outermost: switching a query on or off costs about 0.6 s
            for a in (FULL, (3.0, 9.0), (9.5, 18.5)):
                for cat in (["0", "1", "2", "3"], ["1", "3"]):
                    for ddc in (["A0", "A1", "A2", "A3"], ["A0", "A2"]):
                        where = f"T08 A {a}, cat {cat}, DDC {ddc}, query {'on' if query else 'off'}"
                        d1.set(where, A=a, cat=cat, DDC=ddc, query1=query)
                        r = d1.page.read(hist=["histoXYData"])
                        d1.check(r, where, hists=["histoXYData"])
        d1.set("T08 empty", A=(3.25, 3.4), cat=["0", "1", "2", "3"], DDC=["A0", "A1", "A2", "A3"], query1=None)
        r = d1.page.read(hist=["histoXYData", "histoXYNormData"], prof=[("histoXYData_1", PROF2)])
        assert not r["mask"].any(), "precondition: no row has A in [3.25, 3.4]"
        d1.check(r, "T08 empty selection", hists=["histoXYData", "histoXYNormData"], profs=["histoXYData_1"])
        assert r["hist:histoXYData"]["edges"][0][[0, -1]].tolist() == [0.0, 1.0]
        d1.set("T08 recovered", A=FULL)
        r = d1.page.read(hist=["histoXYData", "histoXYNormData"], prof=[("histoXYData_1", PROF2)])
        d1.check(r, "T08 after the empty selection", hists=["histoXYData", "histoXYNormData"], profs=["histoXYData_1"])


@browser
@pytest.mark.xfail(strict=True, raises=drv.KnownDefect, reason=WG_D3)
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.feature("FILTER.text_query")
def test_t09_the_order_of_cuts_and_their_removal(d1):
    """T09. On the page opened again (so the result does not depend on the tests before), three cuts, A in [3, 9],
    cat 0 and 2, and the query `return B > 4` (the first query switched on on this page): applied in one order and
    removed in another, then applied in the reverse order: every intermediate state equals pandas row by row; both
    orders end with the same rows; with all removed, the end of the test checks the page at load exactly.
    Today a strict expected failure, WG-D3: with the query, the page selects exactly the query's rows alone."""
    d1.page.reopen()
    with d1.case("T09"):
        query = ("return B > 4", lambda t: t["B"] > 4)
        cuts = {"A": {"A": (3.0, 9.0)}, "cat": {"cat": ["0", "2"]}, "query": {"query2": query}}
        undo = {"A": {"A": FULL}, "cat": {"cat": ["0", "1", "2", "3"]}, "query": {"query2": None}}
        final = {}
        for order, removal in ((["A", "cat", "query"], ["cat", "A", "query"]), (["query", "cat", "A"], ["A", "query", "cat"])):
            for k in order:
                d1.set(f"T09 {order}: add {k}", **cuts[k])
                r = d1.page.read(hist=["histoXYData"])
                if k == "query" and order[0] == "A":
                    alone = np.asarray(query[1](d1.t), bool)
                    drv.known_defect(d1.n % 32 == 0 and np.array_equal(r["mask"], alone)
                                     and not np.array_equal(alone, d1.mask()),
                                     WG_D3 + f"; here {int(r['mask'].sum())} rows selected = the query alone, pandas "
                                     f"{int(d1.mask().sum())} with the three cuts")
                d1.check(r, f"T09 order {order}, after adding {k}", hists=["histoXYData"])
            final[tuple(order)] = r["mask"]
            for k in removal:
                d1.set(f"T09 {order}: remove {k}", **undo[k])
                r = d1.page.read()
                d1.check(r, f"T09 order {order}, after removing {k}")
        a, b = final.values()
        drv.same_rows(a, b, "T09 the two orders")
        assert 0 < a.sum() < d1.n, "precondition: the three cuts select some rows, not all"


@browser
@pytest.mark.feature("FILTER.text_query")
@pytest.mark.feature("HIST.histogram_nd")
def test_t10_a_selection_and_its_complement_add_up_to_the_whole(d1):
    """T10. With varY = B, a slider cut on A and the query P = `return cat < 2`, then its negation
    `return !(cat < 2)`, then `return true`, for three cuts: P and not-P select disjoint rows that together are the
    slider's rows; and, the edges being the same in the three states (precondition, checked), the 2D histograms and the
    profile entries add up bin by bin, exactly. The query is switched on first with all rows (P alone, against pandas),
    before any cut: the defect WG-D3 is T09's."""
    with d1.case("T10"):
        d1.set("T10 varY = B, P alone", varY="B", query0=("return cat < 2", lambda t: t["cat"] < 2))
        d1.check(d1.page.read(hist=["histoXYData"]), "T10 P alone", hists=["histoXYData"])
        for cut in ((3.0, 9.0), (0.5, 4.5), (6.0, 16.0)):
            got = {}
            for name, query in (("P", ("return cat < 2", lambda t: t["cat"] < 2)),
                                ("not P", ("return !(cat < 2)", lambda t: ~(t["cat"] < 2))),
                                ("slider only", ("return true", lambda t: np.ones(len(t["A"]), bool)))):
                where = f"T10 A {cut}, {name}"
                d1.set(where, A=cut, query0=query)
                r = d1.page.read(hist=["histoXYData"], prof=[("histoXYData_1", ["entries"])])
                d1.check(r, where, hists=["histoXYData"])
                got[name] = r
            for a in range(2):
                for name in ("P", "not P"):
                    drv.same_values(got[name]["hist:histoXYData"]["edges"][a], got["slider only"]["hist:histoXYData"]["edges"][a],
                                    f"T10 A {cut}: precondition, the edges of axis {a} are the same with {name}")
            m1, m2, m = got["P"]["mask"], got["not P"]["mask"], got["slider only"]["mask"]
            assert not (m1 & m2).any(), f"T10 A {cut}: P and not P share rows"
            drv.same_rows(m1 | m2, m, f"T10 A {cut}: P or not P")
            drv.same_values(got["P"]["hist:histoXYData"]["count"] + got["not P"]["hist:histoXYData"]["count"],
                            got["slider only"]["hist:histoXYData"]["count"], f"T10 A {cut}: H(P) + H(not P) = H")
            drv.same_values(got["P"]["prof:histoXYData_1"]["entries"] + got["not P"]["prof:histoXYData_1"]["entries"],
                            got["slider only"]["prof:histoXYData_1"]["entries"], f"T10 A {cut}: profile entries")
            assert 0 < m1.sum() < m.sum(), "precondition: P selects some rows of the slider, not all"


@browser
@pytest.mark.feature("HIST.histogram_nd")
def test_t11_finer_and_coarser_bins(d1):
    """T11. nbinsX 100 (at load) -> 50 -> 25 -> 50 -> 100: each state equals numpy; each coarse bin is the sum of its
    two finer bins (the coarse edges being every second fine edge, bit for bit: precondition); back at 50 and 100 the
    bins are exactly those seen before."""
    with d1.case("T11"):
        seen = {}
        for nb in (100, 50, 25, 50, 100):
            where = f"T11 nbinsX {nb}"
            d1.set(where, nbinsX=nb)
            r = d1.page.read(hist=["histoXYData"])
            d1.check(r, where, hists=["histoXYData"])
            h = r["hist:histoXYData"]
            if nb in seen:
                drv.same_histogram(h, seen[nb], f"T11 nbinsX {nb} again")
            seen[nb] = h
            fine = seen.get(nb * 2)
            if fine is not None:
                drv.same_values(fine["edges"][0][::2], h["edges"][0], f"T11 precondition: the edges of {nb} bins are "
                                f"every second edge of {nb * 2}")
                ny = h["nbins"][1]
                f2 = fine["count"].reshape(ny, nb * 2)
                drv.same_values(f2[:, 0::2] + f2[:, 1::2], h["count"].reshape(ny, nb),
                                f"T11 each of {nb} bins = the sum of its two of {nb * 2}")


@browser
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.profile_unbinned")
def test_t12_a_2d_histogram_summed_over_y_gives_the_entries_of_its_profile(d1):
    """T12. Three cuts: histoXYData summed over Y = the entries of its profile histoXYData_1, per X bin; the same for
    the normalized histoXYNormData and for histoXYZData (per X and Z cell); exact. The 2D histogram equals numpy."""
    with d1.case("T12"):
        for cut in CUTS:
            where = f"T12 A {cut}"
            d1.set(where, A=cut)
            r = d1.page.read(hist=["histoXYData", "histoXYNormData", "histoXYZData"],
                             prof=[("histoXYData_1", ["entries"]), ("histoXYNormData_1", ["entries"]),
                                   ("histoXYZData_1", ["entries"])])
            d1.check(r, where, hists=["histoXYData"])
            for h in ("histoXYData", "histoXYNormData"):
                nx, ny = r["hist:" + h]["nbins"]
                drv.same_values(r["hist:" + h]["count"].reshape(ny, nx).sum(axis=0), r["prof:" + h + "_1"]["entries"],
                                f"{where}: {h} summed over Y against its profile's entries")
            nx, ny, nz = r["hist:histoXYZData"]["nbins"]
            drv.same_values(r["hist:histoXYZData"]["count"].reshape(nz, ny, nx).sum(axis=1).reshape(-1),
                            r["prof:histoXYZData_1"]["entries"], f"{where}: histoXYZData summed over Y, per X and Z cell")


@browser
@pytest.mark.feature("HIST.histogram_nd")
def test_t13_a_3d_histogram_summed_over_z_gives_the_2d_histogram(d1):
    """T13. Three cuts: histoXYZData summed over Z = histoXYData, and histoXYNormZData summed over Z =
    histoXYNormData, bin by bin, exact; the X and Y edges of the 3D and 2D histograms are the same (precondition)."""
    with d1.case("T13"):
        for cut in CUTS:
            where = f"T13 A {cut}"
            d1.set(where, A=cut)
            r = d1.page.read(hist=["histoXYData", "histoXYZData", "histoXYNormData", "histoXYNormZData"])
            d1.check(r, where, hists=["histoXYZData", "histoXYNormZData"])
            for h3, h2 in (("histoXYZData", "histoXYData"), ("histoXYNormZData", "histoXYNormData")):
                for a in range(2):
                    drv.same_values(r["hist:" + h3]["edges"][a], r["hist:" + h2]["edges"][a],
                                    f"{where}: precondition, the edges of axis {a} of {h3} and {h2}")
                nx, ny, nz = r["hist:" + h3]["nbins"]
                drv.same_values(r["hist:" + h3]["count"].reshape(nz, ny * nx).sum(axis=0), r["hist:" + h2]["count"],
                                f"{where}: {h3} summed over Z against {h2}")


@browser
@pytest.mark.feature("HIST.profile_unbinned")
def test_t14_the_profile_equals_numpy_with_the_projects_estimators(d1):
    """T14. varY = B with 10 Y bins (so the quantiles interpolate inside a bin), three cuts, the last one with the
    sparse X bins of 1, 2, 3, 4 and 6 rows (review correction C3): per X bin the entries equal numpy exactly; the mean,
    the std, the 0.35 quantile and the median equal numpy with the project's definitions within 1e-12 (relative)."""
    with d1.case("T14"):
        d1.set("T14 varY = B, nbinsY 10", varY="B", nbinsY=10)
        for cut in (FULL, (3.0, 9.0), (9.5, 18.5)):
            where = f"T14 A {cut}"
            d1.set(where, A=cut)
            r = d1.page.read(hist=["histoXYData"], prof=[("histoXYData_1", PROF2)])
            d1.check(r, where, hists=["histoXYData"], profs=["histoXYData_1"])
            if cut == (9.5, 18.5):
                e = set(r["prof:histoXYData_1"]["entries"][r["prof:histoXYData_1"]["entries"] > 0].astype(int).tolist())
                assert {1, 2, 3, 4, 6} <= e, f"precondition: the sparse X bins (1, 2, 3, 4, 6 rows) exist: {sorted(e)}"


@browser
@pytest.mark.feature("DSL.custom_js_func")
@pytest.mark.feature("ALIAS.cdsalias")
def test_t15_a_custom_function_normalized_to_the_same_variable(d1):
    """T15. funCustomForm0 = `return A+B`: the column funCustom0 equals numpy A+B row by row; funCustom0 normalized
    to the variable A+B is exactly 0, 1, 0 in every row, bin and profile."""
    with d1.case("T15"):
        d1.set("T15 funCustomForm0", funCustomForm0=("return A+B", lambda t: t["A"] + t["B"]))
        for how in O.TRANSFORMS:
            where = f"T15 funCustom0 against A+B, {how}"
            d1.set(where, varY="funCustom0", varYNorm="A+B", diffFuncTransform=how)
            r = d1.page.read(hist=["histoXYNormData"], prof=[("histoXYNormData_1", PROF2)],
                             alias=["diffY", "funCustom0"])
            drv.same_values(r["alias:funCustom0"], d1.t["A"] + d1.t["B"], where + ", funCustom0")
            drv.every_value(r["alias:diffY"], _transforms_value(how), where + ", diffY")
            d1.check(r, where, hists=["histoXYNormData"], profs=["histoXYNormData_1"], exact_profiles=True)


@browser
@pytest.mark.feature("ALIAS.parameter")
def test_t16_a_parameter_in_a_variable_is_followed(d1):
    """T16. varY = A+exponentX*B (an alias computed in the browser, with the parameter exponentX), ratio: normalized to
    A+B it is exactly 1; exponentX typed 0 in its spinner (real keys, tab Transform): the ratio equals numpy A/(A+B)
    row by row, and normalized to A exactly 1; exponentX typed 1: normalized to A it equals numpy (A+B)/A, and
    normalized to A+B exactly 1 again. Histograms equal numpy after each change."""
    with d1.case("T16"):
        p = d1.page
        a, b = d1.t["A"], d1.t["B"]
        d1.set("T16 A+exponentX*B against A+B", varY="A+exponentX*B", varYNorm="A+B", diffFuncTransform="ratio")
        r = p.read(hist=["histoXYNormData"], alias=["diffY"])
        drv.every_value(r["alias:diffY"], 1.0, "T16 exponentX 1, against A+B")
        d1.check(r, "T16 exponentX 1", hists=["histoXYNormData"])
        p.click_tab("Transform")
        for value, checks in ((0, (("A+B", a / (a + b)), ("A", 1.0))), (1, (("A", (a + b) / a), ("A+B", 1.0)))):
            p.type_into("spinner", "exponentX", str(value))
            p.waited(f"T16 exponentX typed {value}")
            got = p.values()[("spinner", "exponentX")]
            assert got == value, f"T16: the spinner exponentX shows {got} after typing {value}"
            d1.cur["exponentX"] = value
            for norm, exp in checks:
                where = f"T16 exponentX {value}, against {norm}"
                d1.set(where, varYNorm=norm)
                r = p.read(hist=["histoXYNormData"], alias=["diffY"])
                if isinstance(exp, float):
                    drv.every_value(r["alias:diffY"], exp, where)
                else:
                    drv.same_values(r["alias:diffY"], exp, where)
                d1.check(r, where, hists=["histoXYNormData"])


@browser
@pytest.mark.feature("FILTER.text_query")
def test_t17_a_query_that_cannot_be_built_keeps_the_selection(d1):
    """T17. With real clicks and keys in the tab Custom: the first query switched on with its button, `return B < 8`
    typed and applied with Enter (equal to pandas); `return B <` (cannot be built) applied with Enter: the rows stay
    exactly as before, the field is marked (class ri-problem) and its title shows the reason; `return B > 8`: equal
    to pandas, the mark gone; the button again: all rows."""
    with d1.case("T17", allowed_errors=[r"cannot be built"]):
        p = d1.page
        p.click_tab("Custom")
        p.element("off", 0, "button").click()
        p.waited("T17 query 0 switched on with its button")
        assert p.values()[("off", 0)] is False, "the button did not switch the query on"
        r = p.read()
        d1.check(r, "T17 query 0 on with its text at load (return 1)")
        steps = [("return B < 8", lambda t: t["B"] < 8, False), ("return B <", None, True),
                 ("return B > 8", lambda t: t["B"] > 8, False)]
        before = None
        for text, fn, broken in steps:
            p.type_into("query", 0, text)
            p.waited(f"T17 typed {text!r} and Enter")
            r = p.read()
            state = p.pg.evaluate("""() => { const w = window.__g7.widget('query', 0), v = window.__g7.view('query', 0);
              const l = v.shadow_el.querySelector('label'), s = l ? getComputedStyle(l) : null;
              return {classes: Array.from(w.css_classes), title: w.title, value: w.value,
                      label_shown: !!s && s.display !== 'none', label_colour: s ? s.color : null} }""")
            assert state["value"] == text, state
            if broken:
                drv.same_rows(r["mask"], before, f"T17 {text!r}: the rows of the previous query stay")
                assert "ri-problem" in state["classes"] and "not used" in state["title"], (text, state)
                assert state["label_shown"] and state["label_colour"] == "rgb(211, 47, 47)", (text, state)
            else:
                d1.queries[0] = fn
                d1.check(r, f"T17 {text!r}")
                assert "ri-problem" not in state["classes"], (text, state)
                before = r["mask"]
        p.element("off", 0, "button").click()
        p.waited("T17 query 0 switched off with its button")
        d1.queries.pop(0, None)
        d1.check(p.read(), "T17 query switched off")
        p.click_tab("Select")


@browser
@pytest.mark.feature("DISPLAY.lazy_tabs")
@pytest.mark.feature("DISPLAY.downsampler")
def test_t18_a_hidden_tab_shows_the_current_selection_when_opened(d1):
    """T18. The tab histoXYZ is hidden while the cuts change (A in [3, 9], cat 1 and 2); clicked, its drawn profile
    (the plots' own table histoXYZData_1: entries and mean per X and Z cell) equals numpy for the current cuts; a cut
    changed while it is shown, and again while it is hidden (tab Invariances clicked) and shown again: no old value
    is left."""
    with d1.case("T18"):
        p = d1.page
        cols = ["entries", "mean"]

        def drawn(where):
            p.waited(where)
            r = p.read(drawn=[("histoXYZData_1", cols + ["bin_center_0", "bin_center_2"])])
            got = r["drawn:histoXYZData_1"]
            assert got["__watched"] and not got["__needs_update"], (where, "the drawn table is not up to date")
            e = d1.prof("histoXYZData_1")
            h = d1.hist("histoXYZData")
            nx, nz = len(h["edges"][0]) - 1, len(h["edges"][2]) - 1
            xc = (h["edges"][0][:-1] + h["edges"][0][1:]) / 2
            zc = (h["edges"][2][:-1] + h["edges"][2][1:]) / 2
            drv.same_values(got["bin_center_0"], np.tile(xc, nz), where + ", drawn X centres")
            drv.same_values(got["bin_center_2"], np.repeat(zc, nx), where + ", drawn Z centres")
            drv.same_values(got["entries"], e["entries"], where + ", drawn entries")
            drv.scaled_close(got["mean"], e["mean"], where + ", drawn mean", REL)

        d1.set("T18 cuts while histoXYZ is hidden", A=(3.0, 9.0), cat=["1", "2"])
        p.click_tab("histoXYZ")
        drawn("T18 histoXYZ clicked")
        d1.set("T18 cut while shown", A=(0.5, 4.5))
        drawn("T18 after a cut while shown")
        p.click_tab("Invariances")
        p.waited("T18 Invariances clicked")
        d1.set("T18 cut while hidden again", A=(9.5, 18.5), cat=["0", "3"])
        p.click_tab("histoXYZ")
        drawn("T18 histoXYZ clicked again")
        p.click_tab("Invariances")


@browser
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.feature("HIST.histogram_nd")
def test_t19_every_figure_counts_the_same_selected_rows(d1):
    """T19. Three cuts: every histogram of the page (2D, 3D, normalized, the four of the tab Invariances) counts exactly
    the selected rows; and those with the same X edges have the same count in every X bin."""
    with d1.case("T19"):
        names = sorted(HIST_AXES)
        for cut in CUTS:
            where = f"T19 A {cut}, cat 0 1 3"
            d1.set(where, A=cut, cat=["0", "1", "3"])
            r = d1.page.read(hist=names)
            d1.check(r, where)
            n = int(r["mask"].sum())
            assert n > 0, "precondition: some rows are selected"
            per_x = {}
            for h in names:
                g = r["hist:" + h]
                assert g["count"].sum() == n, f"{where}: {h} counts {g['count'].sum():g} rows, {n} are selected"
                per_x.setdefault(g["edges"][0].tobytes(), []).append((h, g["count"].reshape(-1, g["nbins"][0]).sum(axis=0)))
            assert len(per_x) == 1, f"{where}: the X edges differ between the histograms"
            (first, ref), *rest = next(iter(per_x.values()))
            for h, c in rest:
                drv.same_values(c, ref, f"{where}: {h} per X bin against {first}")


@browser
@pytest.mark.feature("FILTER.intersection")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.feature("HIST.profile_unbinned")
def test_t20_the_whole_sequence_leaves_no_trace(d1):
    """T20. After T01 to T19 the page gives exactly the arrays it gave at load (every histogram, profile, alias and the
    selection), and these equal numpy at the baseline: all rows, every histogram bin by bin, the four invariance
    columns exactly 0, 0, 0 and 2, diffY and diffZ exactly 0."""
    with d1.case("T20"):
        fp = d1.page.fingerprint()
        dirty = [k for k in d1.page.fingerprint_at_load if fp.get(k) != d1.page.fingerprint_at_load[k]]
        assert not dirty, f"T20: arrays differ from those at load: {dirty}"
        names = sorted(HIST_AXES)
        r = d1.page.read(hist=names, prof=[("histoXYData_1", PROF2), ("histoXYNormData_1", PROF2)],
                         alias=["diffY", "diffZ"] + [n for n, _e, _v, _t in pages.INVARIANCES])
        assert r["mask"].all(), "T20: at load all rows are selected"
        d1.check(r, "T20 baseline", hists=names, profs=["histoXYData_1", "histoXYNormData_1"])
        for n, _e, v, _t in pages.INVARIANCES:
            drv.every_value(r["alias:" + n], v, f"T20 alias {n}")
        drv.every_value(r["alias:diffY"], 0.0, "T20 diffY (A+B against B+A)")
        drv.every_value(r["alias:diffZ"], 0.0, "T20 diffZ (A*B against B*A)")
    if d1.page.notes:
        pytest.fail("problems left by earlier tests that failed: " + "; ".join(d1.page.notes), pytrace=False)


# ================================================================== without a browser: the table and the comparisons
@pytest.mark.headless_unit
def test_the_template_table_is_exact_and_has_its_sparse_region():
    """The table of D1: no zero; A+B == B+A, (A+B)-B == A and (A+A)/A == 2 in numpy for every row; the sparse region
    (1, 2, 3, 4, 6 rows at A = 16.5 ... 18.5, all different B); each cat holds every A and every B value (T10)."""
    t = pages.template_table(100_000)
    a, b = t["A"].to_numpy(), t["B"].to_numpy()
    assert (t[list("ABCD")].to_numpy() > 0).all()
    assert ((a + b) - b == a).all() and ((a + a) / a == 2).all() and (a * b == b * a).all()
    counts = t[t.A > 16].groupby("A").size().tolist()
    assert counts == [1, 2, 3, 4, 6], counts
    assert t[t.A > 16].B.is_unique
    for c in range(4):
        sub = t[(t.cat == c) & (t.A <= 16)]
        assert sub.A.nunique() == 32 and sub.B.nunique() == 32, c
    assert pages.table_sha256(t) == pages.table_sha256(pages.template_table(100_000))


@pytest.mark.headless_unit
def test_the_comparisons_catch_wrong_answers():
    """Review correction C8: the comparisons the tests use fail on a wrong mask with the right number of rows, on one
    count moved to the next bin, on an edge one float step off, on the arrays of the previous state (a stale vector),
    and on a profile mean off by more than the tolerance."""
    t = {c: v.to_numpy() for c, v in pages.template_table(4000).items()}
    sel = O.selection(t, {"A": (3.0, 9.0)}, {}, [])
    swapped = sel.copy()
    i, j = np.flatnonzero(sel)[0], np.flatnonzero(~sel)[0]
    swapped[i], swapped[j] = False, True
    with pytest.raises(AssertionError, match="rows differ"):
        drv.same_rows(swapped, sel, "same count, other rows")
    h = O.histogram([t["A"], t["B"]], [100, 120], sel)
    moved = {"edges": h["edges"], "count": h["count"].copy()}
    k = int(np.flatnonzero(moved["count"])[0])
    moved["count"][k] -= 1
    moved["count"][k + 1] += 1
    with pytest.raises(AssertionError, match="counts"):
        drv.same_histogram(moved, h, "a count moved")
    off = {"edges": [h["edges"][0].copy(), h["edges"][1]], "count": h["count"]}
    off["edges"][0][5] = np.nextafter(off["edges"][0][5], np.inf)
    with pytest.raises(AssertionError, match="edges"):
        drv.same_histogram(off, h, "an edge one step off")
    stale = O.histogram([t["A"], t["B"]], [100, 120], O.selection(t, {"A": (0.5, 4.5)}, {}, []))
    with pytest.raises(AssertionError):
        drv.same_histogram(stale, h, "the previous state")
    p = O.profile([t["A"], t["B"]], [100, 120], sel, 1, [0.35, 0.5])
    wrong = p["mean"].copy()
    wrong[np.flatnonzero(np.isfinite(wrong))[0]] *= 1 + 1e-9
    with pytest.raises(AssertionError, match="differ by more"):
        drv.scaled_close(wrong, p["mean"], "a mean off", REL)
    drv.same_histogram(h, O.histogram([t["A"], t["B"]], [100, 120], sel), "the right answer passes")


@pytest.mark.headless_unit
def test_the_profile_oracle_on_a_hand_checked_cell():
    """Eight values in one X bin, two Y bins [1, 5.5) and [5.5, 10] of 4 values each: the entries, mean and std, and
    the page's quantile rule by hand. The 0.35 quantile: 0.35 * 8 = 2.8 is passed in the first sub-bin (4 > 2.8),
    q' = 2.8 / 4 = 0.7, numpy's linear quantile of 1, 2, 3, 4 at 0.7 is 1 + 3 * 0.7 = 3.1. The median: 0.5 * 8 = 4 is
    not passed by the first sub-bin (4 > 4 is false), so the second: q' = (4 - 4) / 4 = 0, its smallest value 7."""
    x = np.zeros(8)
    y = np.array([1.0, 2, 3, 4, 7, 8, 9, 10])
    p = O.profile([x, y], [1, 2], np.ones(8, bool), 1, [0.35, 0.5])
    assert p["entries"].tolist() == [8.0] and p["mean"][0] == 5.5
    assert np.isclose(p["std"][0], np.std(y), rtol=1e-15, atol=0)
    assert np.isclose(p["quantile_0"][0], 3.1, rtol=1e-15) and p["quantile_1"][0] == 7.0
