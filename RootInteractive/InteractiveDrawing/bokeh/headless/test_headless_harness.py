"""Harness tests of the completion conditions (proposal v0.2 §5.3, stages 4.2a/4.2b).

1. A delayed update completes by its stated condition, and the check after it sees the new value.
2. A condition that is never met gives ERROR naming the step — never PASS, never a silent wait.
3. The stability rule alone (one microtask, two equal reads around setTimeout(0)) accepts a stale value when an
   update arrives later (GPT24's counterexample). This is why the engine uses it only for the measured synchronous
   widget path and requires a stated condition (`until=`) for everything else.
"""
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    Labels, Page, Scenario, TableSpec, Tables, hist, js_action, js_value, load)

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
