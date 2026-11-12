"""Every computing scenario in the Node model harness alone (tier T2; stage 4.2c-2b).

The fast default suite runs the browser only on its smoke set, so the scenarios of 4.2a/4.2b — the plain pages, the
deep chain, and the known defects B1, B2 and B8 with their exact descriptions — run here in Node, in about 0.2 s each.
The full suite additionally runs them in the browser (test_headless_scenarios.py) and compares both backends
(test_headless_parity.py).
"""
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.assertions import assert_as_declared
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios import histograms, known_defects

pytestmark = [pytest.mark.backend("node"), pytest.mark.layer("integration"), pytest.mark.node_scenarios]   # target (4.2c-2c)
SCENARIOS = tuple(histograms.SCENARIOS) + tuple(known_defects.SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario_in_node(scenario, node_session, headless_out):
    res = run(node_session, scenario, headless_out, test_ref=__file__)
    assert_as_declared(scenario, res)


def test_a_declared_defect_check_that_never_ran_fails(tmp_path):
    """GPT23 (review of 4.2c-2b): a report holding one unrelated PASS and no B1 outcome must not be accepted."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
    b1 = next(s for s in known_defects.SCENARIOS if s.id.startswith("known_defects/b1_"))
    res = ScenarioResult(id=b1.id + " [node]", outdir=str(tmp_path))
    res.add(0, "load", "unrelated", "PASS", "")
    assert res.ok
    with pytest.raises(AssertionError, match="expected failure B1"):
        assert_as_declared(b1, res)


def test_b1_counts_for_filter_intersection(node_session, headless_out):
    """4.2c-2d A-1 (GPT24): B1 is an intersection defect — with two filters the page's selection check counts for
    FILTER.intersection, not for FILTER.range; the run is healthy."""
    b1 = next(s for s in known_defects.SCENARIOS if s.id.startswith("known_defects/b1_two_filters"))
    res = run(node_session, b1, headless_out, test_ref=__file__)
    f = res.features()
    assert f.get("FILTER.intersection") == "Known Issue" and "FILTER.range" not in f and res.kind == "healthy", f

