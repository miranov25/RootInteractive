"""Runs the stage-4.2a scenarios in headless Chromium. One pytest test per scenario; the full per-check report is
written to the output folder and printed on failure."""
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios.histograms import SCENARIOS

pytestmark = [pytest.mark.backend("browser"), pytest.mark.layer("integration")]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario(scenario, headless_session, headless_out):
    result = run(headless_session, scenario, headless_out, test_ref=__file__)
    print(result.text())
    assert result.ok, result.text()
