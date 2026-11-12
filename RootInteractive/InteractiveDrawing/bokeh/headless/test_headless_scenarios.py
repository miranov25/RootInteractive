"""Runs every scenario in headless Chromium: the stage-4.2a scenarios (all must PASS) and the known RootInteractive
defects (stage 4.2b: each must FAIL exactly as declared). One pytest test per scenario; the full per-check report is
written to the output folder and printed on failure.

A known-defect scenario passes this test only if (a) every outcome is PASS or XFAIL and (b) every declared expected
failure really occurred, as XFAIL, at each of its steps — an expectation that never shows is a failure of the test,
not a silent success."""
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.assertions import assert_as_declared
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios import histograms, known_defects

pytestmark = [pytest.mark.backend("browser"), pytest.mark.layer("integration"), pytest.mark.browser_scenarios]   # target (4.2c-2c)
SCENARIOS = histograms.SCENARIOS + known_defects.SCENARIOS


SMOKE_ID = "histograms/plain_1d_2d_one_range_two_moves"            # also in target browser_smoke (4.2c-2c)


@pytest.mark.parametrize("scenario", [pytest.param(s, id=s.id, marks=[pytest.mark.browser_smoke] if s.id == SMOKE_ID else [])
                                      for s in SCENARIOS])
def test_scenario(scenario, headless_session, headless_out):
    result = run(headless_session, scenario, headless_out, test_ref=__file__)
    print(result.text())
    assert_as_declared(scenario, result)
