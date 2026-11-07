"""Deliberate faults with healthy controls (proposal v0.2 §12; stage 4.2b). Each fault is applied exactly once to a
disposable copy of a generated page; the report proves the mutated code ran. A fault counts as detected only if the
healthy control (the same scenario without the fault) passes first, and the fault is caught by the check it is
meant for — for the intended reason — while unrelated checks stay PASS.
"""
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import faults
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios.histograms import PLAIN
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios.known_defects import B1_TWO_FILTERS

pytestmark = [pytest.mark.backend("browser"), pytest.mark.layer("integration")]


def _by(res):
    return {(o.step, o.check): o for o in res.outcomes}


@pytest.fixture(scope="module")
def healthy(headless_session, headless_out):
    res = run(headless_session, PLAIN, headless_out, test_ref=__file__)
    assert res.ok, "healthy control failed: no fault can be judged\n" + res.text()
    return res


def test_update_never_triggered_is_a_fail_not_a_timeout(healthy, headless_session, headless_out):
    res = run(headless_session, PLAIN, headless_out, test_ref=__file__, fault=faults.UPDATE_NEVER_TRIGGERED)
    by = _by(res)
    assert res.fault_executions >= 1 and not res.ok
    assert by[(1, "h1.bin_count")].outcome == "FAIL" and by[(1, "h2.bin_count")].outcome == "FAIL"   # the move is lost
    assert by[(0, "h1.bin_count")].outcome == "PASS" and by[(2, "h2.bin_count")].outcome == "PASS"   # load, move back
    assert not [o for o in res.outcomes if o.outcome == "ERROR"], res.text()                        # completed, no timeout


def test_nd_bin_shifted_is_caught_by_the_nd_histogram_only(healthy, headless_session, headless_out):
    res = run(headless_session, PLAIN, headless_out, test_ref=__file__, fault=faults.ND_BIN_SHIFTED)
    by = _by(res)
    assert res.fault_executions > 1000 and not res.ok
    assert all(by[(k, "h2.bin_count")].outcome == "FAIL" for k in (0, 1, 2)), res.text()
    assert all(by[(k, "h1.bin_count")].outcome == "PASS" for k in (0, 1, 2)), res.text()           # 1-D: other code


def test_fault_on_top_of_a_known_defect_is_a_fail_not_an_xfail(headless_session, headless_out):
    """An unrelated failure inside the B1 scenario: the shifted N-D bins make cells fall BELOW the oracle, which the
    B1 signature ("page > oracle") does not describe -> FAIL ("a different problem"), never XFAIL."""
    res = run(headless_session, B1_TWO_FILTERS, headless_out, test_ref=__file__, fault=faults.ND_BIN_SHIFTED)
    by = _by(res)
    assert all(by[(k, "h2.bin_count")].outcome == "FAIL" and "a different problem" in by[(k, "h2.bin_count")].detail
               for k in (0, 1, 2)), res.text()
    assert by[(0, "load.finite_rows_selected")].outcome == "XFAIL"                                 # untouched by it


def test_fault_that_does_not_match_is_an_error(headless_session, headless_out):
    bad = faults.Fault(id="no_match", description="target absent", source="none", search="no such text", replace="x")
    res = run(headless_session, PLAIN, headless_out, test_ref=__file__, fault=bad)
    assert not res.ok and [o.outcome for o in res.outcomes] == ["ERROR"] and "found 0 times" in res.outcomes[0].detail
