"""Parity of the two backends (proposal v0.3 §5 rule 3; stage 4.2c-1, acceptance item F).

Every scenario that computes is run twice on identical data — in Chromium from the page HTML (tier T3) and in the
Node model harness from the page document (tier T2) — and the two must agree exactly:
  - the same outcome for every check at every step (PASS, XFAIL, ...);
  - the same columns of every probed model after every step (the snapshot the settle rule reads);
  - the same number of selected rows after every step.
This is the standing control that the jsdom stand-ins touch no computation, and that Node runs the page's code.
"""
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios import histograms, known_defects

pytestmark = [pytest.mark.backend("node"), pytest.mark.layer("integration")]
SCENARIOS = tuple(histograms.SCENARIOS) + tuple(known_defects.SCENARIOS)


def _outcomes(res):
    return [(o.step, o.check, o.outcome) for o in res.outcomes]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_node_equals_browser(scenario, headless_session, node_session, headless_out):
    out = headless_out / "parity"
    browser = run(headless_session, scenario, out, test_ref=__file__)
    node = run(node_session, scenario, out, test_ref=__file__)
    assert browser.case_sha256 == node.case_sha256                                   # identical data
    assert _outcomes(node) == _outcomes(browser), f"outcomes differ\nbrowser:\n{browser.text()}\nnode:\n{node.text()}"
    assert node.ok == browser.ok
    assert set(node.probe_values) == set(browser.probe_values) and node.probe_values, node.probe_values.keys()
    for step in browser.probe_values:
        assert node.probe_values[step] == browser.probe_values[step], f"probe values differ at {step}"
