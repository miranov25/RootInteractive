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
"""
import pytest

import json
from pathlib import Path

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import HELPER_JS, run
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    Labels, Page, Parameter, Scenario, TableSpec, Tables, auto, hist, js_action, js_value, load, range_, responds, set_)

pytestmark = [pytest.mark.backend("browser"), pytest.mark.layer("integration")]


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
    assert [o.outcome for o in step1] == ["ERROR"] and "completion condition not met" in step1[0].detail


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
        assert out["settled"] is False and "not on the page: h_9.mean" in out["error"]
        # two models of an accepted type with the same name are ambiguous: also an error, never a guess
        page.evaluate("""() => { const mk = () => ({name: 'h', type: 'HistoNdCDS', get_column: () => [1]});
            Bokeh.documents[0]._all_models.set(1, mk()); Bokeh.documents[0]._all_models.set(2, mk()); }""")
        out = page.evaluate("() => window.__riH.settle([['h', 'bin_count', ['HistoNdCDS']]], 5)")
        assert out["settled"] is False and "not on the page: h.bin_count" in out["error"]
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
    """Claude3's two-filter page (known defect B1). Expected to FAIL while B1 exists; stage 4.2b turns it into a strict
    expected failure with a signature."""
    sc = Scenario(id="harness/load_check_reads_the_page_b1",
                  data=Tables(TableSpec(20_000, 23, {"x": "uniform(0,1)", "y": "normal(0,1)",
                                                     "a": "normal(0,1) + 10% NaN", "b": "normal(0,1) + 10% NaN"})),
                  page=Page(histograms=(hist("h2", ["x", "y"], [10, 2], [[0, 1], [-5, 5]]),),
                            widgets=(range_("a"), range_("b"))),
                  steps=(load(),), checks=(auto(),), labels=Labels(family="harness", technique="negative control"))
    res = run(headless_session, sc, headless_out, test_ref=__file__)
    chk = [o for o in res.outcomes if o.check.startswith("load: every finite row selected")]
    assert [o.outcome for o in chk] == ["FAIL"], res.text()
    page_n = int(chk[0].detail.split("page selects ")[1].split()[0])
    finite_n = int(chk[0].detail.split("finite widget columns ")[1].split()[0])
    assert page_n > finite_n                                             # the second filter is not applied
