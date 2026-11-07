"""Unit tests of expected failures, signatures, classification, feature aggregation and fault application
(stage 4.2b) — no browser. Every rule is tested both ways: a case that must hold and one that must not."""
import numpy as np
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import faults, report
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.expect import (
    EqualsUnselected, Error, ExcessInCells, Mismatch, classify, expected_failure, feature_status)
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    LOAD_CHECK, Labels, Page, Scenario, ScenarioError, TableSpec, Tables, auto, hist, load, range_, set_)

pytestmark = [pytest.mark.layer("unit"), pytest.mark.backend("python")]
A = lambda *v: np.array(v, dtype=np.float64)


# ------------------------------------------------------------------------------------------- signatures
@pytest.mark.parametrize("sig,page,oracle,ctx,holds", [
    (Mismatch("page > oracle"), A(5, 3), A(4, 3), {}, True),
    (Mismatch("page > oracle"), A(5, 2), A(4, 3), {}, False),          # one cell below: another problem
    (Mismatch("page > oracle"), A(4, 3), A(4, 3), {}, False),          # nothing differs
    (Mismatch("page > oracle", "all"), A(5, 3), A(4, 3), {}, False),   # not every non-empty cell differs
    (Mismatch("page < oracle"), A(3, 3), A(4, 3), {}, True),
    (EqualsUnselected(), A(10, 10), A(6, 7), {"unselected": A(10, 10)}, True),
    (EqualsUnselected(), A(10, 9), A(6, 7), {"unselected": A(10, 10)}, False),
    (EqualsUnselected(), A(10, 10), A(10, 10), {"unselected": A(10, 10)}, False),   # selection keeps all rows
    (ExcessInCells((0,), "nan_rows"), A(7, 3, 2), A(5, 3, 2), {"nan_rows": 2.0}, True),
    (ExcessInCells((0,), "nan_rows"), A(7, 4, 2), A(5, 3, 2), {"nan_rows": 2.0}, False),   # another cell too
    (ExcessInCells((0,), "nan_rows"), A(6, 3, 2), A(5, 3, 2), {"nan_rows": 2.0}, False),   # wrong amount
    (ExcessInCells((0,), "nan_rows"), A(5, 3, 2), A(5, 3, 2), {"nan_rows": 0.0}, False),   # cannot show
    (Error("KeyError: 'weights'"), None, None, {"error": "page build failed: KeyError: 'weights'"}, True),
    (Error("KeyError: 'weights'"), None, None, {"error": "TypeError"}, False),
])
def test_signature(sig, page, oracle, ctx, holds):
    assert sig.holds(page, oracle, ctx)[0] is holds


# ------------------------------------------------------------------------------------------- classification
EXP = [expected_failure("B1", "h.bin_count", (0, 1), Mismatch("page > oracle"))]


@pytest.mark.parametrize("check,step,ok,page,oracle,outcome", [
    ("h.bin_count", 2, True, A(4), A(4), "PASS"),       # step without expectation
    ("h.bin_count", 2, False, A(5), A(4), "FAIL"),
    ("g.bin_count", 0, False, A(5), A(4), "FAIL"),       # other check
    ("h.bin_count", 0, False, A(5), A(4), "XFAIL"),      # the known defect, as described
    ("h.bin_count", 0, False, A(3), A(4), "FAIL"),       # fails, but not as described: a different problem
    ("h.bin_count", 1, True, A(4), A(4), "XPASS"),       # expected failure did not occur: strict
])
def test_classify(check, step, ok, page, oracle, outcome):
    assert classify(check, step, ok, page, oracle, {}, EXP)[0] == outcome


# ------------------------------------------------------------------------------------------- aggregation
def _result(*outs):
    r = report.ScenarioResult(id="unit/agg", outdir="/tmp")
    for feature, outcome in outs:
        r.add(0, "load", f"{feature} check", outcome, "", feature=feature)
    return r


@pytest.mark.parametrize("outs,status,ok", [
    ([("F1", "PASS"), ("F2", "XFAIL")], {"F1": "Working", "F2": "Known Issue"}, True),      # PASS + XFAIL
    ([("F1", "PASS"), ("F1", "FAIL")], {"F1": "Broken"}, False),                           # PASS + unexpected FAIL
    ([("F1", "PASS"), ("F1", "SKIP")], {"F1": "Unverified"}, False),                       # PASS + skipped required
    ([("F1", "XFAIL"), ("F2", "XFAIL")], {"F1": "Known Issue", "F2": "Known Issue"}, True),  # all XFAIL
    ([("F1", "PASS"), ("F1", "XPASS")], {"F1": "Broken"}, False),                          # strict XPASS
    ([], {}, False),                                                                       # zero checks executed
])
def test_feature_aggregation_and_scenario_outcome(outs, status, ok):
    r = _result(*outs)
    assert r.features() == status and r.ok is ok


# ------------------------------------------------------------------------------------------- validation
def _sc(**over):
    base = dict(id="unit/expect", data=Tables(TableSpec(10, 1, {"x": "uniform(0,1)", "a": "normal(0,1)"})),
                page=Page(histograms=(hist("h", ["x"], 10, [0, 1]),), widgets=(range_("a"),)),
                steps=(load(), set_("a", [-1, 1])), checks=(auto(),), labels=Labels(family="unit"))
    base.update(over)
    return Scenario(**base)


def test_valid_expectations_pass_validation():
    sc = _sc(expect=(expected_failure("B2", "h.bin_count", (0,), EqualsUnselected()),
                     expected_failure("B1", LOAD_CHECK, (0,), Mismatch("page > oracle"))))
    assert sc.validate().id == "unit/expect"


@pytest.mark.parametrize("over,match", [
    ({"expect": ("B1",)}, "use expected_failure"),
    ({"expect": (expected_failure("", "h.bin_count", (0,), Mismatch("page > oracle")),)}, "bug id"),
    ({"expect": (expected_failure("B1", "q.bin_count", (0,), Mismatch("page > oracle")),)}, "does not produce"),
    ({"expect": (expected_failure("B1", "h.bin_count", (7,), Mismatch("page > oracle")),)}, "outside"),
    ({"page": Page(histograms=(hist("h", ["x"], 10, [0, 1]),), widgets=(range_("a", (-1, 1)),)),
      "expect": (expected_failure("B1", LOAD_CHECK, (0,), Mismatch("page > oracle")),)}, "does not produce"),
    ({"labels": Labels(family="unit", models=("HistogramCDS",), verified=("HistoNdCDS",))}, "labels.verified"),
])
def test_invalid_expectations_and_labels_are_refused(over, match):
    with pytest.raises(ScenarioError, match=match):
        _sc(**over).validate()


def test_alias_with_scientific_notation_is_accepted():
    """Closure note N-5 (Claude10): `x*1e-3` was refused with the reason "uses 'e'"."""
    sc = _sc(page=Page(histograms=(hist("h", ["xa"], 10, [0, 1]),), widgets=(range_("a"),),
                       aliases=(("xa", "x*1e-3 + 2.5E+1*x - .5"),)))
    assert sc.validate().id == "unit/expect"


# ------------------------------------------------------------------------------------------- faults
def test_fault_applies_exactly_once():
    f = faults.ND_BIN_SHIFTED
    once = "a; " + f.search + " b;"
    assert faults.apply(once, f).count(faults.COUNTER) == 1 and f.search not in faults.apply(once, f)
    with pytest.raises(faults.FaultNotApplied, match="found 0 times"):
        faults.apply("nothing here", f)
    with pytest.raises(faults.FaultNotApplied, match="found 2 times"):
        faults.apply(f.search + f.search, f)
