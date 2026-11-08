"""Every computing scenario in the Node model harness alone (tier T2; stage 4.2c-2b).

The fast default suite runs the browser only on its smoke set, so the scenarios of 4.2a/4.2b — the plain pages, the
deep chain, and the known defects B1, B2 and B8 with their exact descriptions — run here in Node, in about 0.2 s each.
The full suite additionally runs them in the browser (test_headless_scenarios.py) and compares both backends
(test_headless_parity.py).
"""
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios import histograms, known_defects

pytestmark = [pytest.mark.backend("node"), pytest.mark.layer("integration")]
SCENARIOS = tuple(histograms.SCENARIOS) + tuple(known_defects.SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario_in_node(scenario, node_session, headless_out):
    res = run(node_session, scenario, headless_out, test_ref=__file__)
    assert res.ok, res.text()
