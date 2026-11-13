"""Deliberate faults with healthy controls (proposal v0.2 §12; stage 4.2b). Each fault is applied exactly once to a
disposable copy of a generated page; the report proves the mutated code ran. A fault counts as detected only if the
healthy control (the same scenario without the fault) passes first, and the fault is caught by the check it is
meant for — for the intended reason — while unrelated checks stay PASS.
"""
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import faults
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.assertions import assert_as_declared
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios.histograms import PLAIN
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios import known_defects
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios.known_defects import B1_VEHICLE, with_defect

# PHASE_5 Batch 1 fixed B1: the tests below run B1's old scenario with B1 put back by a fault (a "vehicle")
B1_BACK = faults.B1_REINTRODUCED

pytestmark = [pytest.mark.backend("browser"), pytest.mark.layer("integration"), pytest.mark.browser_harness]   # target (4.2c-2c)


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


def test_the_b1_vehicle_reproduces_b1_exactly_and_fails_without_it(headless_session, headless_out):
    """PHASE_5 Batch 1: with B1 put back, the page equals B1's exact signature at every declared check (XFAIL, the run
    OK); without it, the same expectations are XPASS and the run fails - the fix is in the code."""
    back = run(headless_session, B1_VEHICLE, headless_out, test_ref=__file__, fault=B1_BACK)
    assert back.fault_executions >= 1
    assert_as_declared(B1_VEHICLE, back)
    fixed = run(headless_session, B1_VEHICLE, headless_out / "without_the_fault", test_ref=__file__)
    assert not fixed.ok and {o.outcome for o in fixed.outcomes if o.outcome != "PASS"} == {"XPASS"}, fixed.text()


def test_fault_on_top_of_a_known_defect_is_a_fail_not_an_xfail(headless_session, headless_out):
    """An unrelated failure inside the B1 scenario: the shifted N-D bins make cells fall BELOW the oracle, which the
    B1 signature ("page > oracle") does not describe -> FAIL ("a different problem"), never XFAIL."""
    res = run(headless_session, B1_VEHICLE, headless_out, test_ref=__file__,
              fault=with_defect(B1_BACK, faults.ND_BIN_SHIFTED))
    by = _by(res)
    assert all(by[(k, "h2.bin_count")].outcome == "FAIL" and "a different problem" in by[(k, "h2.bin_count")].detail
               for k in (0, 1, 2)), res.text()
    assert by[(0, "load.finite_rows_selected")].outcome == "XFAIL"                                 # untouched by it


def test_fault_that_does_not_match_is_an_error(headless_session, headless_out):
    bad = faults.Fault(id="no_match", description="target absent", source="none", search="no such text", replace="x")
    res = run(headless_session, PLAIN, headless_out, test_ref=__file__, fault=bad)
    assert not res.ok and [o.outcome for o in res.outcomes] == ["ERROR"] and "found 0 times" in res.outcomes[0].detail


def test_lost_update_on_b1_is_not_masked_as_b1(headless_session, headless_out):
    """4.2b v0.2 (P1-1, Claude3): with v0.1's direction-only B1 signature this run was XFAIL everywhere and OK. With the
    exact signature, a page that ignores the widget moves is a different problem at the moves."""
    res = run(headless_session, B1_VEHICLE, headless_out, test_ref=__file__,
              fault=with_defect(B1_BACK, faults.UPDATE_NEVER_TRIGGERED))
    by = _by(res)
    assert res.fault_executions >= 1 and not res.ok, res.text()
    for k in (1, 2):
        assert by[(k, "h2.bin_count")].outcome == "FAIL" and "a different problem" in by[(k, "h2.bin_count")].detail
    assert by[(0, "h2.bin_count")].outcome == "XFAIL"                           # at load the fault does not act yet


@pytest.mark.parametrize("fault", list(faults.FAULTS.values()), ids=list(faults.FAULTS))
@pytest.mark.parametrize("scenario,defect", known_defects.DEFECT_CASES, ids=[s.id for s, _ in known_defects.DEFECT_CASES])
def test_fault_known_defect_matrix(fault, scenario, defect, headless_session, headless_out):
    """4.2b v0.2 (P2-5, Claude3): every fault on every known-defect scenario must make the scenario fail — never an
    XFAIL with ok = True (a known defect hiding a new problem). PHASE_5: a fixed defect is put back by its fault."""
    res = run(headless_session, scenario, headless_out, test_ref=__file__, fault=with_defect(defect, fault))
    assert res.fault_executions >= 1, res.text()
    assert not res.ok, "fault masked by a known defect:\n" + res.text()
