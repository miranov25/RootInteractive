"""Unit tests of the comparison helpers (proposal v0.2 §7.3) and of scenario validation (§4) — no browser.
Includes the stage-4.2a fault: a comparator that always says "equal" must be caught by these tests."""
import numpy as np
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import compare as compare_mod
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    Labels, Page, Scenario, ScenarioError, TableSpec, Tables, Transport, auto, hist, js_action, load, range_, set_)

pytestmark = [pytest.mark.layer("unit"), pytest.mark.backend("python")]

INF, NAN = np.inf, np.nan
# (page, oracle, kind, kwargs, expected ok)
CASES = [
    ([1, 2, 3], [1, 2, 3], "count", {}, True),
    ([1, 2, 3], [1, 2, 4], "count", {}, False),                         # a perturbed count
    ([1, 2], [1, 2, 3], "count", {}, False),                            # shape mismatch
    ([NAN], [0], "count", {}, False),                                   # a NaN count is never a count
    ([0.0], [-0.0], "float", {}, True),                                 # -0 equals +0
    ([INF], [INF], "float", {}, True),
    ([INF], [-INF], "float", {}, False),                                # opposite infinities differ
    ([INF], [1e308], "float", {"rel_tol": 1.0}, False),                 # infinite vs finite always differs
    ([NAN], [NAN], "float", {}, False),                                 # NaN = NaN only when declared
    ([NAN], [NAN], "float", {"nan_equal": True}, True),
    ([NAN], [1.0], "float", {"nan_equal": True}, False),
    ([1e-13], [0.0], "float", {"rel_tol": 1e-12}, False),               # zero reference: relative tolerance alone fails
    ([1e-13], [0.0], "float", {"abs_tol": 1e-12}, True),
    ([1.0 + 5e-13], [1.0], "float", {"rel_tol": 1e-12}, True),          # tolerance edge, inside
    ([1.0 + 3e-12], [1.0], "float", {"rel_tol": 1e-12}, False),         # tolerance edge, outside
]


def _run_table(fn):
    return [fn(p, o, kind=k, **kw).ok == expect for p, o, k, kw, expect in CASES]


def test_comparator_rules():
    results = _run_table(compare_mod.compare)
    assert all(results), [c for c, ok in zip(CASES, results) if not ok]


def test_fault_always_equal_comparator_is_caught():
    """Fault injection on test code: replace the comparator by one that always says 'equal'; the rule table above
    must notice it (the healthy control is test_comparator_rules)."""
    def always_equal(page, oracle, kind="count", **kw):
        return compare_mod.Comparison(True, kind, len(oracle), 0, "always equal")
    assert not all(_run_table(always_equal))


def test_comparison_reports_worst_cell_and_ratio():
    c = compare_mod.compare([10, 20, 31], [10, 20, 30])
    assert not c.ok and c.n_diff == 1 and c.worst == (2, 31.0, 30.0)
    assert c.ratio_range == (1.0, pytest.approx(31 / 30))


def _scenario(**over):
    base = dict(id="unit/valid", data=Tables(TableSpec(10, 1, {"x": "uniform(0,1)", "a": "normal(0,1)"})),
                page=Page(histograms=(hist("h", ["x"], 10, [0, 1]),), widgets=(range_("a"),)),
                steps=(load(), set_("a", [-1, 1])), checks=(auto(),), labels=Labels(family="unit"))
    base.update(over)
    return Scenario(**base)


def test_valid_scenario_passes_validation():
    assert _scenario().validate().id == "unit/valid"


@pytest.mark.parametrize("over,match", [
    ({"id": "nofamily"}, "family"),
    ({"steps": (set_("a", [0, 1]),)}, "first step"),
    ({"steps": (load(), set_("b", [0, 1]))}, "no widget"),
    ({"steps": (load(), js_action("window.x = 1"))}, "completion condition"),
    ({"steps": (load(), load())}, "only as the first step"),
    ({"page": Page(histograms=(hist("h", ["q"], 10, [0, 1]),))}, "unknown variable"),
    ({"page": Page(widgets=(range_("zz"),))}, "unknown column"),
    ({"page": Page(), "steps": (load(),), "checks": (auto(),)}, "at least one supported model"),
    ({"transport": Transport(((".*", (("relative", 8),)),))}, "lossless"),
    ({"expect": ("B1",)}, "4.2b"),
])
def test_invalid_scenarios_are_refused_with_a_readable_message(over, match):
    with pytest.raises(ScenarioError, match=match):
        _scenario(**over).validate()
