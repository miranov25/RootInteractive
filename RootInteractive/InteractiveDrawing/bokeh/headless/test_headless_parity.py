"""Parity of the two backends (proposal v0.3 §5 rule 3; stage 4.2c-1, acceptance item F; v0.2 after the review).

Every scenario that computes is run twice on identical data — in Chromium from the page HTML (tier T3) and in the
Node model harness from the page document (tier T2) — and `parity_problems` must find nothing:
  - both runs meet their oracles (ok: every check PASS or a declared expected failure) — agreement of two wrong runs
    is not parity (GPT23, GPT24);
  - the same outcome for every check at every step;
  - the same columns of every probed model after every step, with NaN, +-Infinity and -0 kept apart (Claude10, GPT23);
  - the same selected rows after every step: the count AND a hash of the 0/1 selection mask (GPT23, GPT24, Claude3).
The controls at the end are the reviewers' counterexamples: each must be reported as a problem.
"""
from types import SimpleNamespace

import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine.expect import Mismatch, expected_failure
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import run
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    Labels, Page, Scenario, TableSpec, Tables, auto, hist, load, range_, set_)
from RootInteractive.InteractiveDrawing.bokeh.headless.scenarios import histograms, known_defects

pytestmark = [pytest.mark.backend("node"), pytest.mark.layer("integration"), pytest.mark.browser_parity]   # target (4.2c-2c)
SCENARIOS = tuple(histograms.SCENARIOS) + tuple(known_defects.SCENARIOS)


def parity_problems(browser, node) -> list:
    out = []
    if not browser.ok:
        out.append("browser run is not OK against its oracles")
    if not node.ok:
        out.append("Node run is not OK against its oracles")
    if browser.case_sha256 != node.case_sha256:
        out.append("different data")
    ob = [(o.step, o.check, o.outcome) for o in browser.outcomes]
    on = [(o.step, o.check, o.outcome) for o in node.outcomes]
    if ob != on:
        out.append(f"outcomes differ: browser {ob} / node {on}")
    if not browser.probe_values or set(browser.probe_values) != set(node.probe_values):
        out.append(f"probe steps differ: {sorted(browser.probe_values)} / {sorted(node.probe_values)}")
    for k in sorted(set(browser.probe_values) & set(node.probe_values)):
        if browser.probe_values[k] != node.probe_values[k]:
            out.append(f"probe values differ at {k}")
    return out


SMOKE_ID = "histograms/plain_1d_2d_one_range_two_moves"            # also in target browser_smoke (4.2c-2c)


@pytest.mark.parametrize("scenario", [pytest.param(s, id=s.id, marks=[pytest.mark.browser_smoke] if s.id == SMOKE_ID else [])
                                      for s in SCENARIOS])
def test_node_equals_browser(scenario, headless_session, node_session, headless_out):
    out = headless_out / "parity"
    browser = run(headless_session, scenario, out, test_ref=__file__)
    node = run(node_session, scenario, out, test_ref=__file__)
    problems = parity_problems(browser, node)
    assert not problems, "\n".join(problems) + f"\nbrowser:\n{browser.text()}\nnode:\n{node.text()}"


# ------------------------------------------------------------------- controls: the reviewers' counterexamples
def _fake(ok=True, values=None):
    return SimpleNamespace(ok=ok, case_sha256="c", outcomes=[], probe_values=values or {"0": "[[1]]"})


def test_equal_counts_with_different_rows_are_not_parity():
    b = _fake(values={"0": "[[1]]", "0:selected": {"count": 1, "mask_fnv": 111}})
    n = _fake(values={"0": "[[1]]", "0:selected": {"count": 1, "mask_fnv": 222}})
    assert parity_problems(b, n) == ["probe values differ at 0:selected"]


def test_nan_and_infinity_are_not_parity(node_session):
    nan = node_session.request("eval", code="() => window.__riH.encode([[NaN, -0]])")["result"]
    inf = node_session.request("eval", code="() => window.__riH.encode([[Infinity, 0]])")["result"]
    assert nan != inf and "NaN" in nan and "-0" in nan and "Infinity" in inf
    assert parity_problems(_fake(values={"0": nan}), _fake(values={"0": inf})) == ["probe values differ at 0"]


def test_mask_hash_sees_which_rows(node_session):
    h = [node_session.request("eval", code=f"() => window.__riH.maskHash({m})")["result"]
         for m in ("[1, 0, 0]", "[0, 1, 0]", "[1, 0, 0]")]
    assert h[0] != h[1] and h[0] == h[2]


def test_two_runs_that_are_both_wrong_are_not_parity(headless_session, node_session, headless_out):
    """GPT24's counterexample: a wrong expectation makes both backends fail identically; parity must refuse it."""
    sc = Scenario(id="harness/parity_both_wrong", data=Tables(TableSpec(1000, 5, {"x": "uniform(0,1)", "a": "normal(0,1)"})),
                  page=Page(histograms=(hist("h1", ["x"], 10, [0, 1]),), widgets=(range_("a"),)),
                  steps=(load(), set_("a", [-1, 1])), checks=(auto(),),
                  expect=(expected_failure("B0", "h1.bin_count", (1,), Mismatch("page > oracle")),),
                  labels=Labels(family="harness", technique="parity control"))
    out = headless_out / "controls"
    browser, node = run(headless_session, sc, out, test_ref=__file__), run(node_session, sc, out, test_ref=__file__)
    problems = parity_problems(browser, node)
    assert "browser run is not OK against its oracles" in problems and "Node run is not OK against its oracles" in problems
