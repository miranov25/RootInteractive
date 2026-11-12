"""Unit tests of expected failures, signatures, classification, feature aggregation and fault application
(stage 4.2b) — no browser. Every rule is tested both ways: a case that must hold and one that must not."""
import numpy as np
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import faults, report
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.expect import (
    EqualsSelection, EqualsUnselected, Error, ExcessInCells, Mismatch, classify, expected_failure, feature_status)
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import filters
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    LOAD_CHECK, Labels, Page, Scenario, ScenarioError, TableSpec, Tables, auto, hist, load, range_, set_)

pytestmark = [pytest.mark.layer("unit"), pytest.mark.backend("python"), pytest.mark.headless_unit]   # target (4.2c-2c)
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
    (EqualsSelection("r"), A(7, 3), A(5, 3), {"selection:r": A(7, 3)}, True),
    (EqualsSelection("r"), A(8, 3), A(5, 3), {"selection:r": A(7, 3)}, False),     # one extra count: not the defect
    (EqualsSelection("r"), A(5, 3), A(5, 3), {"selection:r": A(5, 3)}, False),     # defect cannot show here
    (EqualsSelection("r"), A(7, 3), A(5, 3), {}, False),                           # no reference
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



# ------------------------------------------------------------------------------------------- stage 4.2b v0.2
def test_signature_uses_the_checks_tolerance():
    """Claude3's / Claude10's weighted B8: bin 0 exceeds the oracle by the NaN weight sum only within ~1e-12."""
    oracle = A(2232.2802212641473, 10.0)
    page = A(2232.2802212641473 + 2486.554153600955, 10.0)
    sig, ctx = ExcessInCells((0,), "nan_rows"), {"nan_rows": 2486.554153600963}
    assert sig.holds(page, oracle, ctx, (0.0, 0.0))[0] is False            # exact comparison: false FAIL (v0.1)
    assert sig.holds(page, oracle, ctx, (1e-9, 1e-12))[0] is True          # with the check's tolerance: the defect
    assert classify("h.bin_count", 0, False, page, oracle, ctx,
                    [expected_failure("B8", "h.bin_count", (0,), sig)], (1e-9, 1e-12))[0] == "XFAIL"


def test_b1_masking_counterexamples_are_failures():
    """GPT24 / GPT22 / Claude3: with the exact B1 signature, an overcount that B1 does not produce stays FAIL."""
    oracle, b1 = A(823, 410, 200), A(938, 470, 230)
    exp = [expected_failure("B1", "h.bin_count", (0,), EqualsSelection("r"))]
    ctx = {"selection:r": b1}
    assert classify("h.bin_count", 0, False, b1, oracle, ctx, exp)[0] == "XFAIL"                  # healthy control
    assert classify("h.bin_count", 0, False, b1 + A(1, 0, 0), oracle, ctx, exp)[0] == "FAIL"      # one extra count
    assert classify("h.bin_count", 0, False, A(970, 480, 240), oracle, ctx, exp)[0] == "FAIL"     # a lost update
    out = classify("h.bin_count", 0, False, A(np.inf, np.inf, np.inf), oracle, ctx, exp)            # infinities
    assert out[0] == "FAIL" and "non-finite" in out[1]


def test_mismatch_is_reported_as_loose():
    assert "loose signature" in Mismatch("page > oracle").holds(A(5, 3), A(4, 3), {})[1]


def test_b1_selection_rule():
    """First filter applied; a later filter keeps the rows it excluded at load (here: the missing value of b)."""
    a_now, b_now = np.array([True, True, False, True]), np.array([True, False, True, False])
    b_load = np.array([True, True, True, False])          # row 3: b missing, excluded at load
    assert filters.b1_selection([a_now, b_now], [a_now, b_load]).tolist() == [True, False, False, True]
    assert filters.b1_selection([a_now], [a_now]).tolist() == a_now.tolist()


@pytest.mark.parametrize("over,match", [
    ({"expect": (expected_failure("B1", "h.bin_count", (0,), Mismatch("page > oracle")),
                 expected_failure("B8", "h.bin_count", (0, 1), ExcessInCells((0,), "nan_rows")))}, "two expected failures"),
    ({"page": Page(histograms=(hist("h", ["x"], 10, [0, 1]),), widgets=(range_("a"), range_("x"))),
      "expect": (expected_failure("B1", "h.bin_count", (0,), EqualsSelection("no_such_rule")),)}, "selection rule"),
    ({"expect": (expected_failure("B1", "h.bin_count", (0,),
                                  EqualsSelection("b1_later_filters_keep_rows_excluded_at_load")),)}, "several filters"),
])
def test_v02_expectation_validation(over, match):
    with pytest.raises(ScenarioError, match=match):
        _sc(**over).validate()
