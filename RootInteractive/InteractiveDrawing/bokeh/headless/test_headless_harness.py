"""Harness tests of the completion conditions (proposal v0.2 §5.3, stages 4.2a/4.2b).

1. A delayed update completes by its stated condition, and the check after it sees the new value.
2. A condition that is never met gives ERROR naming the step — never PASS, never a silent wait.
3. The stability rule alone (one microtask, two equal reads around setTimeout(0)) accepts a stale value when an
   update arrives later (GPT24's counterexample). This is why the engine uses it only for the measured synchronous
   widget path and requires a stated condition (`until=`) for everything else.

Negative controls of the v0.3 fixes (each must FAIL or ERROR; the healthy controls are the scenarios of the suite):
4. A probed model that is not on the page is an error, never a stable page (GPT24 P1-3, Claude3 C-2).
5. `responds` on a value that does not change on the "move" FAILs (a frozen probe cannot pass).
6. A page that the builder rejects leaves a report with ERROR (Claude3 C-2).
7. The load check reads the page: on a page with known defect B1 the page selects more rows than are finite in
   the widget columns, and the check FAILs (Claude10 P1-1; in v0.2 it said PASS there).

Negative controls of stage 4.2b:
8. A loader that never becomes ready, on a step that carries an expected failure, gives ERROR and SKIP — never XFAIL.
9. A probed model that disappears after a JavaScript step is an ERROR, and the report is still written (Claude3 N-1).
10. A console error that is not on the allowed list is an ERROR (console-error policy).
11. An expected failure that does not occur is a strict XPASS and fails the scenario.
12. (4.2b v0.2, GPT24 P1-2) After a step that never completes, every feature whose checks were not executed is
    "Unverified" — not only the scenario's first feature; an error inside the automatic check marks only the affected
    check and its feature.
"""
import pytest

import json
from pathlib import Path

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.expect import Mismatch, expected_failure
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import HELPER_JS, run
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    LOAD_CHECK, Labels, Page, Parameter, Scenario, TableSpec, Tables, auto, hist, js_action, js_value, load, range_,
    responds, set_)

pytestmark = [pytest.mark.backend("browser"), pytest.mark.layer("integration"), pytest.mark.browser_harness]   # target (4.2c-2c)


def _small(id_, steps, checks):
    return Scenario(id=id_, data=Tables(TableSpec(1000, 3, {"x": "uniform(0,1)"})),
                    page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),)), steps=steps, checks=checks,
                    labels=Labels(family="harness", technique="completion condition"))


def test_delayed_update_completes_by_its_condition(headless_session, headless_out):
    sc = _small("harness/delayed_update_condition",
                (load(), js_action("window.__d = 0; setTimeout(() => { window.__d = 1; }, 60);", until="window.__d === 1")),
                (js_value("window.__d", 1, steps=(1,)),))
    res = run(headless_session, sc, headless_out, test_ref=__file__)
    assert res.ok, res.text()
    assert res.completion[1]["waited_ms"] >= 50                     # it really waited for the delayed update


def test_condition_never_met_is_an_error(headless_session, headless_out):
    sc = _small("harness/condition_never_met",
                (load(), js_action("window.__e = 0;", until="window.__e === 1", timeout_s=0.3)),
                (js_value("window.__e", 1, steps=(1,)),))
    res = run(headless_session, sc, headless_out, test_ref=__file__)
    assert not res.ok
    step1 = [o for o in res.outcomes if o.step == 1]
    # the step itself is an ERROR; the check declared on it was never executed: SKIP (its feature is "Unverified")
    assert [o.outcome for o in step1] == ["ERROR", "SKIP"] and "completion condition not met" in step1[0].detail


def test_stability_rule_alone_accepts_a_stale_value(headless_session):
    page = headless_session.browser.new_page()
    try:
        page.set_content("<html><body></body></html>")
        stale = page.evaluate("""async () => {
            window.__v = 0; setTimeout(() => { window.__v = 1; }, 60);
            await Promise.resolve();
            const a = window.__v; await new Promise(r => setTimeout(r, 0)); const b = window.__v;
            return [a, b]; }""")
        assert stale == [0, 0]                                       # two equal reads: the rule would call it settled
        page.wait_for_function("() => window.__v === 1", timeout=2000)
    finally:
        page.close()


def test_missing_probe_is_an_error(headless_session):
    page = headless_session.browser.new_page()
    try:
        page.set_content("<html><body></body></html>")
        page.add_script_tag(content="window.Bokeh = {documents: [{_all_models: new Map()}]};" + HELPER_JS)
        out = page.evaluate("() => window.__riH.settle([['h_9', 'mean', ['HistoNdProfile']]], 5)")
        assert out["settled"] is False and "h_9.mean — not found" in out["error"]
        # two models of an accepted type with the same name are ambiguous: also an error, never a guess
        page.evaluate("""() => { const mk = () => ({name: 'h', type: 'HistoNdCDS', get_column: () => [1]});
            Bokeh.documents[0]._all_models.set(1, mk()); Bokeh.documents[0]._all_models.set(2, mk()); }""")
        out = page.evaluate("() => window.__riH.settle([['h', 'bin_count', ['HistoNdCDS']]], 5)")
        assert out["settled"] is False and "h.bin_count — ambiguous: 2 models" in out["error"]   # N-4 wording
    finally:
        page.close()


GENERAL_CHECK_JS = """() => {
  const H = window.__riH, M = Bokeh.documents[0]._all_models;
  const range = n => Array.from({length: n}, (_, i) => i);
  const table = n => ({type: 'ColumnDataSource', name: 't' + n, get_length: () => n});
  const lif = (name, n, picked, stale, filters) => ({type: 'LazyIntersectionFilter', name, filters: filters || [{source: table(n)}],
      changed_indices: stale, cached_indices: picked, changed_booleans: stale,
      cached_vector: Array.from({length: 32}, (_, i) => picked.includes(i))});
  const hist = (type, name, n, counts, extra) => Object.assign({type, name, weights: null, source: table(n),
      data: {bin_count: counts}, get_column: () => counts}, extra || {});
  const out = {};
  M.clear(); M.set(1, lif('ok', 12, range(12), false)); M.set(2, hist('HistogramCDS', 'h_ok', 12, [10, 2]));
  out.healthy = H.invariants();
  M.set(3, lif('bad', 12, range(32), false));
  M.set(4, lif('stale', 12, range(32), true));                                   // marked out of date: not judged
  M.set(5, hist('HistogramCDS', 'h_bad', 12, [30, 2]));
  M.set(6, hist('HistogramCDS', 'h_weighted', 12, [30, 2], {weights: 'w'}));      // weighted: totals not bounded
  M.set(7, hist('HistoNdCDS', 'hn_stale', 12, [30, 2], {cached_columns: new Set()}));   // bin_count not current
  M.set(8, hist('HistoNdCDS', 'hn_bad', 12, [7, 6], {cached_columns: new Set(['bin_count'])}));
  // v0.2 (review F3): a current filter whose table cannot be read is a failure, never skipped
  M.set(9, lif('throws', 12, [0], false, [{source: {type: 'ColumnDataSource', get_length: () => { throw new Error('x'); }}}]));
  // v0.2 (review F3): a compressed table not decoded yet is reported as not checked - its get_length() is never called
  M.set(10, lif('lazy', 12, [0], false, [{source: {type: 'CDSCompress', name: 'cz', _length: -1,
      get_length: () => { throw new Error('get_length() of a compressed table must not be called'); }}}]));
  M.set(11, lif('mixed', 12, [0], false, [{source: table(12)}, {source: table(13)}]));   // inconsistent lengths
  out.broken = H.invariants();
  out.unchecked = H.lastUnchecked;
  return out;
}"""


def test_the_general_check_judges_only_current_results(headless_session):
    """PHASE_5 Batch 1 (review C1), Batches 2A/2B v0.2 (review F3): the page helper's strictly observing check, in
    Chromium on stand-in models. A filter that selects past its table's last row, and an unweighted histogram with
    more entries than rows, are reported; results marked out of date and weighted histograms are not judged. A current
    filter whose table cannot be read, or whose filters disagree on the table length, is a failure; a current filter
    on a compressed table not decoded yet is reported as not checked, and its length is never computed. settle()
    then throws."""
    page = headless_session.browser.new_page()
    try:
        page.set_content("<html><body></body></html>")
        page.add_script_tag(content="window.Bokeh = {documents: [{_all_models: new Map()}]};" + HELPER_JS)
        out = page.evaluate(GENERAL_CHECK_JS)
        assert out["healthy"] == []
        assert out["broken"] == [
            "filter bad selects 20 positions outside its table of 12 rows (first 12, 13, 14)",
            "filter bad: position 12 past the last row of its table (12 rows) is selected",
            "HistogramCDS h_bad counts 32 entries from a table of 12 rows",
            "HistoNdCDS hn_bad counts 13 entries from a table of 12 rows",
            "the general check could not read filter throws: x",
            "filter mixed: its filters select from tables of different lengths (12 and 13 rows)"], out["broken"]
        assert out["unchecked"] == ["filter lazy: compressed table cz is not decoded yet"], out["unchecked"]
        err = page.evaluate("""async () => { try { await window.__riH.settle([['h_ok', 'bin_count', ['HistogramCDS']]], 5);
                                               return null; } catch (e) { return String(e.message); } }""")
        assert err is not None and err.startswith("RI invariant violated: filter bad selects 20 positions"), err
    finally:
        page.close()


def test_frozen_probe_fails_responds(headless_session, headless_out):
    sc = Scenario(id="harness/frozen_probe_responds",
                  data=Tables(TableSpec(2000, 4, {"x": "uniform(0,1)", "a": "normal(0,1)"})),
                  page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),), widgets=(range_("a"),)),
                  steps=(load(), set_("a", "full"), set_("a", [-1, 1])),     # step 1 changes nothing
                  checks=(responds("h1", "bin_count", moved=1, restored=2),),
                  labels=Labels(family="harness", technique="negative control"))
    res = run(headless_session, sc, headless_out, test_ref=__file__)
    out = [o for o in res.outcomes if o.check.startswith("h1.bin_count responds")]
    assert not res.ok and [o.outcome for o in out] == ["FAIL"] and "did not change" in out[0].detail


def test_page_build_failure_writes_a_report(headless_session, headless_out):
    """A weights parameter not named literally `weights`, with a profile of that histogram shown, makes
    RootInteractive's builder fail (bokehTools.py 960 looks up paramDict["weights"])."""
    sc = Scenario(id="harness/page_build_failure",
                  data=Tables(TableSpec(500, 5, {"x": "uniform(0,1)", "a": "normal(0,1)", "w1": "uniform(0,1)"})),
                  page=Page(histograms=(hist("h", ["x", "a"], [5, 4], [[0, 1], [-3, 3]], weights="wsel"),),
                            widgets=(range_("a"),), parameters=(Parameter("wsel", ("w1",), ("w1",)),),
                            profiles=("h_1",)),
                  steps=(load(),), checks=(auto(),), labels=Labels(family="harness", technique="negative control"))
    res = run(headless_session, sc, headless_out, test_ref=__file__)
    report = json.loads((Path(res.outdir) / "report.json").read_text())
    assert not res.ok and report["outcomes"][0]["outcome"] == "ERROR"
    assert report["outcomes"][0]["detail"].startswith("page build failed")


def test_load_check_reads_the_page_b1(headless_session, headless_out):
    """Claude3's two-filter page (known defect B1). It FAILED while B1 existed; PHASE_5 Batch 1 fixed B1, so the load
    check now reads the intersection of both filters, equal to the rows with finite widget columns (16273, pandas).
    The check still failing on B1 is proven by the B1 vehicle (test_headless_faults.py)."""
    sc = Scenario(id="harness/load_check_reads_the_page_b1",
                  data=Tables(TableSpec(20_000, 23, {"x": "uniform(0,1)", "y": "normal(0,1)",
                                                     "a": "normal(0,1) + 10% NaN", "b": "normal(0,1) + 10% NaN"})),
                  page=Page(histograms=(hist("h2", ["x", "y"], [10, 2], [[0, 1], [-5, 5]]),),
                            widgets=(range_("a"), range_("b"))),
                  steps=(load(),), checks=(auto(),), labels=Labels(family="harness", technique="negative control"))
    res = run(headless_session, sc, headless_out, test_ref=__file__)
    chk = [o for o in res.outcomes if o.check == LOAD_CHECK]
    assert [o.outcome for o in chk] == ["PASS"], res.text()
    page_n = int(chk[0].detail.split("page selects ")[1].split()[0])
    finite_n = int(chk[0].detail.split("finite widget columns ")[1].split()[0])
    assert page_n == finite_n == 16273                                   # both filters applied (B1 fixed)


def _page_with_widget(id_, steps, checks, expect=()):
    return Scenario(id=id_, data=Tables(TableSpec(2000, 6, {"x": "uniform(0,1)", "a": "normal(0,1)"})),
                    page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),), widgets=(range_("a"),)),
                    steps=steps, checks=checks, expect=expect, labels=Labels(family="harness", technique="negative control"))


def test_loader_never_ready_is_an_error_not_an_xfail(headless_session, headless_out):
    sc = _page_with_widget("harness/loader_never_ready_with_expectation",
                           (load(), js_action("window.__ready = false;", until="window.__ready === true", timeout_s=0.3),
                            set_("a", [-1, 1])),
                           (auto(),),
                           expect=(expected_failure("B0", "h1.bin_count", (1, 2), Mismatch("page > oracle")),))
    res = run(headless_session, sc, headless_out, test_ref=__file__)
    steps12 = [(o.step, o.outcome) for o in res.outcomes if o.step in (1, 2)]
    assert not res.ok and ("XFAIL" not in [x for _, x in steps12]), res.text()
    assert (1, "ERROR") in steps12 and (1, "SKIP") in steps12 and (2, "SKIP") in steps12, res.text()


def test_probe_lost_after_js_step_is_an_error_and_reported(headless_session, headless_out):
    sc = _page_with_widget("harness/probe_lost_after_js_step",
                           (load(), js_action("window.__riH.model('h1', ['HistogramCDS']).name = 'renamed';",
                                              until="true")),
                           (auto(),))
    res = run(headless_session, sc, headless_out, test_ref=__file__)
    err = [o for o in res.outcomes if o.step == 1 and o.outcome == "ERROR"]
    assert not res.ok and err and "h1.bin_count" in err[0].detail and "not found" in err[0].detail, res.text()
    assert (Path(res.outdir) / "report.json").exists()


def test_console_error_not_allowed_is_an_error(headless_session, headless_out):
    sc = _page_with_widget("harness/console_error_policy",
                           (load(), js_action("console.error('an unexpected page error');", until="true"),
                            set_("a", [-1, 1])),
                           (auto(),))
    res = run(headless_session, sc, headless_out, test_ref=__file__)
    pol = [o for o in res.outcomes if o.check == "console error"]
    assert not res.ok and [o.outcome for o in pol] == ["ERROR"] and "an unexpected page error" in pol[0].detail


def test_expected_failure_that_does_not_occur_is_a_strict_xpass(headless_session, headless_out):
    sc = _page_with_widget("harness/stale_expectation_xpass", (load(), set_("a", [-1, 1])), (auto(),),
                           expect=(expected_failure("B0", "h1.bin_count", (1,), Mismatch("page > oracle")),))
    res = run(headless_session, sc, headless_out, test_ref=__file__)
    x = [o for o in res.outcomes if o.check == "h1.bin_count" and o.step == 1]
    assert not res.ok and [o.outcome for o in x] == ["XPASS"] and "did not occur" in x[0].detail


def _two_features(id_, steps):
    return Scenario(id=id_, data=Tables(TableSpec(2000, 8, {"x": "uniform(0,1)", "y": "normal(0,1)",
                                                         "a": "normal(0,1)"})),
                    page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),
                                          hist("h2", ["x", "y"], [10, 4], [[0, 1], [-3, 3]])),
                              widgets=(range_("a"),)),
                    steps=steps, checks=(auto(),), labels=Labels(family="harness", technique="feature attribution"))


def test_stopped_scenario_leaves_every_untested_feature_unverified(headless_session, headless_out):
    res = run(headless_session, _two_features("harness/two_features_stopped",
              (load(), js_action("window.__never = false;", until="window.__never === true", timeout_s=0.3),
               set_("a", [-1, 1]))), headless_out, test_ref=__file__)
    f = res.features()
    assert not res.ok and f["HIST.histogram_1d"] == "Unverified" and f["HIST.histogram_nd"] == "Unverified", res.text()
    skipped = {(o.step, o.check) for o in res.outcomes if o.outcome == "SKIP"}
    assert {(1, "h1.bin_count"), (1, "h2.bin_count"), (2, "h1.bin_count"), (2, "h2.bin_count")} <= skipped


def test_completed_scenario_has_both_features_working(headless_session, headless_out):
    res = run(headless_session, _two_features("harness/two_features_completed", (load(), set_("a", [-1, 1]))),
              headless_out, test_ref=__file__)
    f = res.features()
    assert res.ok and f["HIST.histogram_1d"] == "Working" and f["HIST.histogram_nd"] == "Working", res.text()


def test_error_inside_auto_marks_only_the_affected_check(headless_session, headless_out, monkeypatch):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine import runner
    real = runner.histograms.counts

    def failing_for_nd(columns, nbins, ranges, mask=None, weights=None):
        if len(columns) > 1:
            raise RuntimeError("injected oracle error for the N-D histogram")
        return real(columns, nbins, ranges, mask=mask, weights=weights)
    monkeypatch.setattr(runner.histograms, "counts", failing_for_nd)
    res = run(headless_session, _two_features("harness/error_inside_auto", (load(), set_("a", [-1, 1]))),
              headless_out, test_ref=__file__)
    by = {(o.step, o.check): o for o in res.outcomes}
    assert by[(0, "h2.bin_count")].outcome == "ERROR" and by[(0, "h1.bin_count")].outcome == "PASS", res.text()
    f = res.features()
    assert f["HIST.histogram_nd"] == "Broken" and f["HIST.histogram_1d"] == "Working", res.text()
