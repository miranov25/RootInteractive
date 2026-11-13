"""Deliberate faults in the Node model harness (proposal v0.3 §3.7; stage 4.2c-1, acceptance item G).

The same faults as in the browser (engine/faults.py) are applied exactly once to a disposable copy of the model
bundle — never to the stored healthy bundle — and the run proves the mutated code executed. A fault counts as
detected only after the healthy control passed, and only by the check it is meant for; the fault x known-defect
matrix must never end as an expected failure with the scenario OK (no known defect may hide a new problem).
"""
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import faults, node_bundle
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.assertions import assert_as_declared
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import NodeSession
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    Labels, Page, Scenario, TableSpec, Tables, auto, hist, load, range_, set_)
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios import known_defects
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios.histograms import PLAIN
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios.known_defects import B1_VEHICLE, with_defect

# PHASE_5 Batch 1 fixed B1 and B2: their old scenarios run with the defect put back by a fault (a "vehicle")
B1_BACK = faults.B1_REINTRODUCED

pytestmark = [pytest.mark.backend("node"), pytest.mark.layer("integration"), pytest.mark.node_harness]   # target (4.2c-2c)


def _by(res):
    return {(o.step, o.check): o for o in res.outcomes}


class _Sessions(dict):
    """One Node session per fault, made on first use (a composite fault - a defect put back plus an unrelated
    fault - needs its own bundle copy)."""

    def __init__(self, root):
        super().__init__()
        self.root = root

    def for_fault(self, f):
        if f.id not in self:
            self[f.id] = NodeSession(self.root, fault=f)
        return self[f.id]

    def __missing__(self, fid):
        return self.for_fault({**faults.FAULTS, **faults.DEFECT_FAULTS}[fid])


@pytest.fixture(scope="module")
def fault_sessions(node_session, node_bundle_root):
    sessions = _Sessions(node_bundle_root)
    yield sessions
    for s in sessions.values():
        s.close()


@pytest.fixture(scope="module")
def healthy(node_session, headless_out):
    res = run(node_session, PLAIN, headless_out, test_ref=__file__)
    assert res.ok, "healthy control failed: no fault can be judged\n" + res.text()
    return res


def test_each_fault_matches_the_bundle_exactly_once(node_bundle_root):
    text = node_bundle.model_bundle(node_bundle_root).read_text(encoding="utf-8")
    for fid, f in {**faults.FAULTS, **faults.DEFECT_FAULTS}.items():
        assert text.count(f.search) == 1, f"fault {fid}: found {text.count(f.search)} times in the model bundle"


@pytest.mark.parametrize("scenario,defect", known_defects.VEHICLES, ids=[s.id for s, _ in known_defects.VEHICLES])
def test_each_vehicle_reproduces_its_defect_exactly_and_fails_without_it(scenario, defect, node_session,
                                                                          fault_sessions, headless_out):
    """PHASE_5 Batch 1: with the defect put back, the page equals the defect's exact signature at every declared check
    (XFAIL, the run OK); without it, the same expectations are XPASS and the run fails - the fix is in the code."""
    back = run(fault_sessions.for_fault(defect), scenario, headless_out, test_ref=__file__, fault=defect)
    assert back.fault_executions >= 1
    assert_as_declared(scenario, back)
    fixed = run(node_session, scenario, headless_out / "without_the_fault", test_ref=__file__)
    assert not fixed.ok and {o.outcome for o in fixed.outcomes if o.outcome != "PASS"} == {"XPASS"}, fixed.text()


def test_update_never_triggered_is_a_fail_not_a_timeout(healthy, fault_sessions, headless_out):
    res = run(fault_sessions["update_never_triggered"], PLAIN, headless_out, test_ref=__file__,
              fault=faults.UPDATE_NEVER_TRIGGERED)
    by = _by(res)
    assert res.fault_executions >= 1 and not res.ok
    assert by[(1, "h1.bin_count")].outcome == "FAIL" and by[(1, "h2.bin_count")].outcome == "FAIL"
    assert by[(0, "h1.bin_count")].outcome == "PASS" and by[(2, "h2.bin_count")].outcome == "PASS"
    assert not [o for o in res.outcomes if o.outcome == "ERROR"], res.text()


def test_nd_bin_shifted_is_caught_by_the_nd_histogram_only(healthy, fault_sessions, headless_out):
    res = run(fault_sessions["nd_bin_shifted"], PLAIN, headless_out, test_ref=__file__, fault=faults.ND_BIN_SHIFTED)
    by = _by(res)
    assert res.fault_executions > 1000 and not res.ok
    assert all(by[(k, "h2.bin_count")].outcome == "FAIL" for k in (0, 1, 2)), res.text()
    assert all(by[(k, "h1.bin_count")].outcome == "PASS" for k in (0, 1, 2)), res.text()


def test_lost_update_on_b1_is_not_masked_as_b1(fault_sessions, headless_out):
    f = with_defect(B1_BACK, faults.UPDATE_NEVER_TRIGGERED)
    res = run(fault_sessions.for_fault(f), B1_VEHICLE, headless_out, test_ref=__file__, fault=f)
    by = _by(res)
    assert res.fault_executions >= 1 and not res.ok, res.text()
    for k in (1, 2):
        assert by[(k, "h2.bin_count")].outcome == "FAIL" and "a different problem" in by[(k, "h2.bin_count")].detail
    assert by[(0, "h2.bin_count")].outcome == "XFAIL"


def test_session_and_run_must_carry_the_same_fault(node_session, headless_out):
    res = run(node_session, PLAIN, headless_out / "controls", test_ref=__file__, fault=faults.ND_BIN_SHIFTED)
    assert not res.ok and [o.check for o in res.outcomes] == ["fault applied"]


@pytest.mark.parametrize("fault", list(faults.FAULTS.values()), ids=list(faults.FAULTS))
@pytest.mark.parametrize("scenario,defect", known_defects.DEFECT_CASES, ids=[s.id for s, _ in known_defects.DEFECT_CASES])
def test_fault_known_defect_matrix(fault, scenario, defect, fault_sessions, headless_out):
    f = with_defect(defect, fault)
    res = run(fault_sessions.for_fault(f), scenario, headless_out, test_ref=__file__, fault=f)
    assert res.fault_executions >= 1, res.text()
    assert not res.ok, "fault masked by a known defect:\n" + res.text()


ONLY_1D = Scenario(
    id="harness/node_fault_proof_only_1d", data=Tables(TableSpec(5000, 11, {"x": "uniform(0,1)", "a": "normal(0,1)"})),
    page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),), widgets=(range_("a"),)),
    steps=(load(), set_("a", [-1, 1])), checks=(auto(),),
    labels=Labels(family="harness", technique="fault execution proof per document (reviewers Claude3, GPT24)"))


def test_fault_execution_proof_is_per_document(fault_sessions, headless_out):
    """Claude3's and GPT24's counterexample: in one fault session, a page that cannot run the N-D fault must report
    zero executions and the ERROR "the fault proves nothing" also after a page that did run it."""
    s, f = fault_sessions["nd_bin_shifted"], faults.ND_BIN_SHIFTED
    out = headless_out / "controls"
    before = run(s, ONLY_1D, out / "before", test_ref=__file__, fault=f)
    plain = run(s, PLAIN, out / "plain", test_ref=__file__, fault=f)
    after = run(s, ONLY_1D, out / "after", test_ref=__file__, fault=f)
    assert plain.fault_executions > 1000
    for r in (before, after):
        assert r.fault_executions == 0 and [o for o in r.outcomes if o.check == "fault executed"], r.text()
    assert after.isolation["fresh_window"], after.isolation       # fault sessions: a fresh window per document (A-1)
