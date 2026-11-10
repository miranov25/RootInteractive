"""Controls for the headless evidence in the capability matrix (stage 4.2c-2d, option (a) with amendments A-1 … A-7).

`scripts/generate_capability_matrix.py --from-headless <folder>` reads the headless engine's own verdict per capability
from the scenario reports of the folder's session. Each test builds a small synthetic output folder (session.json plus
reports) and checks one rule; the report-kind rule of the engine is checked at the end.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.headless_unit]   # target (4.2c-2c)
ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("ri_matrix_generator", ROOT / "scripts" / "generate_capability_matrix.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)
TAX = {"FILTER.range": {}, "FILTER.intersection": {}, "HIST.histogram_1d": {}, "HIST.histogram_nd": {}, "HIST.profile": {}}


def _out(folder: Path, session="s1", reports=()):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "session.json").write_text(json.dumps({"session": session, "complete": True}))
    for i, r in enumerate(reports):
        d = folder / "main" / f"r{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.json").write_text(r if isinstance(r, str) else json.dumps(r))
    return folder


_CHECK_OF = {"Working": "PASS", "Known Issue": "XFAIL", "Unverified": "SKIP", "Broken": "FAIL"}


def _report(sid, features, kind="healthy", session="s1", outcomes=(), backend="node"):
    """A synthetic report whose checks agree with its declared statuses, as the engine's reports always do."""
    return {"id": sid, "kind": kind, "session": session, "backend": backend, "features": features,
            "outcomes": list(outcomes) or [{"feature": f, "outcome": _CHECK_OF[st], "detail": "", "assertion": True,
                                            "scope": "value"} for f, st in features.items()]}


def _status(tmp_path, *reports, session="s1"):
    ev = gen.load_headless_evidence([str(_out(tmp_path / "o", session, reports))], TAX)
    return {f: gen.headless_matrix_status(e) for f, e in ev.items()}, ev


def test_b1_counts_as_a_known_intersection_defect_with_its_id(tmp_path):
    xf = {"feature": "FILTER.intersection", "outcome": "XFAIL", "detail": "known defect B1: rows kept", "assertion": True,
          "scope": "value"}
    st, ev = _status(tmp_path, _report("known_defects/b1 [node]", {"FILTER.intersection": "Known Issue"}, outcomes=[xf]))
    assert st["FILTER.intersection"] == "⚠️ Known Issue"
    assert ev["FILTER.intersection"]["scenarios"]["known_defects/b1"]["defects"] == {"B1"}


def test_working_and_response_only(tmp_path):
    resp = {"feature": "HIST.profile", "outcome": "PASS", "detail": "", "assertion": True, "scope": "response"}
    st, _ = _status(tmp_path, _report("histograms/plain", {"FILTER.range": "Working"}),
                    _report("chain/deep", {"HIST.profile": "Working"}, outcomes=[resp]))
    assert st["FILTER.range"] == "✅ Working"
    assert st["HIST.profile"] == "❓ Partly verified (response only)"


def test_fault_and_harness_runs_are_ignored_but_a_healthy_broken_counts(tmp_path):
    st, _ = _status(tmp_path, _report("x [fault f1]", {"FILTER.range": "Broken"}, kind="fault"),
                    _report("harness/h", {"FILTER.range": "Broken"}, kind="harness"),
                    _report("histograms/ok", {"FILTER.range": "Working"}),
                    _report("histograms/bad", {"HIST.histogram_1d": "Broken"}))
    assert st["FILTER.range"] == "✅ Working" and st["HIST.histogram_1d"] == "🧨 Broken"


def test_a_known_issue_is_never_hidden_by_an_unverified_result(tmp_path):
    st, _ = _status(tmp_path, _report("a", {"HIST.histogram_nd": "Known Issue"}), _report("b", {"HIST.histogram_nd": "Unverified"}))
    assert st["HIST.histogram_nd"] == "⚠️ Known Issue"


def test_engine_labels_are_skipped_and_another_sessions_reports_ignored(tmp_path):
    st, _ = _status(tmp_path, _report("a", {"ENGINE.completion": "Broken", "FILTER.range": "Working"}),
                    _report("old", {"FILTER.range": "Broken"}, session="s0"))
    assert st == {"FILTER.range": "✅ Working"}


@pytest.mark.parametrize("case", ["unknown capability", "no session.json", "unreadable report", "no kind", "no report of the session"])
def test_bad_input_fails_the_generator(tmp_path, case):
    folder = tmp_path / "o"
    if case == "unknown capability":
        _out(folder, reports=[_report("a", {"HIST.nonexistent": "Working"})])
    elif case == "no session.json":
        _out(folder, reports=[_report("a", {"FILTER.range": "Working"})]).joinpath("session.json").unlink()
    elif case == "unreadable report":
        _out(folder, reports=["{not json"])
    elif case == "no kind":
        r = _report("a", {"FILTER.range": "Working"}); r.pop("kind")
        _out(folder, reports=[r])
    else:
        _out(folder, reports=[_report("a", {"FILTER.range": "Working"}, session="other")])
    with pytest.raises(SystemExit):
        gen.load_headless_evidence([str(folder)], TAX)


def test_the_worse_of_decorator_and_headless_status_counts(tmp_path):
    _, ev = _status(tmp_path, _report("a", {"HIST.histogram_1d": "Known Issue"}))
    status, count = gen.combine_status("✅ Working", 2, ["t1", "t2"], ev["HIST.histogram_1d"])
    assert status == "⚠️ Known Issue" and count == 3


def test_report_kind_comes_from_explicit_facts():
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import _kind
    sc = lambda sid, fam: SimpleNamespace(id=sid, labels=SimpleNamespace(family=fam))
    assert _kind(sc("histograms/x", "histograms"), None) == "healthy"
    assert _kind(sc("histograms/x", "histograms"), object()) == "fault"
    assert _kind(sc("harness/x", "harness"), None) == "harness"
    with pytest.raises(ValueError):
        _kind(sc("harness/x", "histograms"), None)            # the id namespace and the family must agree


def _mixed_b1():
    return _report("known_defects/b1 [node]", {"FILTER.intersection": "Unverified"}, outcomes=[
        {"feature": "FILTER.intersection", "outcome": "XFAIL", "detail": "known defect B1: rows kept", "assertion": True,
         "scope": "value"},
        {"feature": "FILTER.intersection", "outcome": "SKIP", "detail": "not executed", "assertion": True, "scope": "value"}])


def test_a_known_defect_and_a_skip_in_one_report_stay_a_known_issue(tmp_path):
    """GPT23, GPT24: the engine rates such a report Unverified; the matrix must keep the confirmed B1 visible and note
    the skipped check."""
    st, ev = _status(tmp_path, _mixed_b1())
    sc = ev["FILTER.intersection"]["scenarios"]["known_defects/b1"]
    assert st["FILTER.intersection"] == "⚠️ Known Issue" and sc["defects"] == {"B1"} and sc["skipped"]


def test_the_same_through_the_real_generator_command(tmp_path):
    import subprocess, sys
    folder = _out(tmp_path / "o", reports=[_mixed_b1()])
    (tmp_path / "pytest_report.json").write_text(json.dumps({"tests": []}))
    md = tmp_path / "m.md"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_capability_matrix.py"), "--from-reports",
                        str(tmp_path / "pytest_report.json"), "--from-headless", str(folder), "--output", str(md)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    text = md.read_text()
    assert "| `FILTER.intersection` | P1 | ⚠️ Known Issue |" in text and "known defect B1 — also skipped checks" in text
    assert md.with_suffix(".json").exists()


def test_every_variant_of_a_parametrized_test_counts():
    """PHASE_4_3 J4: a failing variant must not hide behind a passing first variant (the join tests are parametrized
    over the four join modes); the status is the worst over all variants."""
    st, n, failed = gen.compute_status("F", ["t.py::test_x"], {"t.py::test_x[a]": "passed", "t.py::test_x[b]": "failed"}, {})
    assert st == "🧨 Broken" and failed == ["t.py::test_x[b]"] and n == 1
    assert gen.compute_status("F", ["t.py::test_x"], {"t.py::test_x[a]": "passed", "t.py::test_x[b]": "xfailed"}, {})[0] \
        == "⚠️ Known Issue"
    assert gen.compute_status("F", ["t.py::test_x"], {"t.py::test_x[a]": "passed", "t.py::test_x[b]": "passed"}, {})[0] \
        == "✅ Working"                                                       # control: all variants pass
    assert gen.compute_status("F", ["t.py::test_y"], {"t.py::test_y": "passed", "t.py::test_y_other[a]": "failed"}, {})[0] \
        == "✅ Working"                                                       # control: a similar name is not a variant


FAKE_TESTS = '''import pytest


class Port:                                   # a helper class before the tests, as in test_headless_node_joins.py
    def run(self):
        return 1


@pytest.mark.feature("JOIN.change_propagation")
@pytest.mark.parametrize("how", ["inner", "outer"])
def test_x(how):
    assert how != "{fail}"


class TestGroup:
    @pytest.mark.feature("JOIN.change_propagation")
    def test_in_class(self):
        assert True


@pytest.mark.feature("JOIN.change_propagation")
def test_after_classes():
    """@pytest.mark.feature("JOIN.recursive") inside a docstring is not a marker"""
    assert True
'''


def _source_to_matrix(tmp_path, fail: str, pytest_args=()) -> dict:
    """The REAL path (J4 review M-1): a test file with a helper class -> pytest's own JSON report -> the generator's
    command line -> its JSON output. Returns the feature's entry. `pytest_args`: extra selection (-k ...)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "test_fake_matrix.py"
    src.write_text(FAKE_TESTS.replace("{fail}", fail))
    report = tmp_path / "report.json"
    subprocess.run([sys.executable, "-m", "pytest", str(src), "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
                    "--rootdir", str(tmp_path), "-c", os.devnull, *pytest_args,
                    "--json-report", f"--json-report-file={report}"],
                   cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert report.is_file()
    out = tmp_path / "matrix.md"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_capability_matrix.py"), "--test-dir", str(src), "--from-reports", str(report),
                        "--output", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(out.with_suffix(".json").read_text())["features"]["JOIN.change_propagation"]


def test_a_failed_later_variant_after_a_helper_class_makes_the_feature_broken(tmp_path):
    """J4 review M-1 (GPT25, reproduced by GPT24): after a helper class, a module-level parametrized test whose SECOND
    variant fails must give Broken - before, the misnamed test id fell through to fuzzy matching and gave Working.
    Control: with no failing variant the same file gives Working. Also: no class segment on module-level tests, a
    class segment inside a test class, docstring markers ignored, and cases counted per variant."""
    broken = _source_to_matrix(tmp_path / "a", "outer")
    assert broken["status"] == "🧨 Broken" and [t.split("::")[-1] for t in broken["failed_tests"]] == ["test_x[outer]"]
    assert [t.split("/")[-1] for t in broken["tests"]] == ["test_fake_matrix.py::test_x",
                                                          "test_fake_matrix.py::TestGroup::test_in_class",
                                                          "test_fake_matrix.py::test_after_classes"]
    assert broken["test_count"] == 3 and broken["cases"] == 4 and broken["without_result"] == []
    working = _source_to_matrix(tmp_path / "b", "never")
    assert working["status"] == "✅ Working" and working["cases"] == 4


def test_a_feature_with_a_test_or_case_that_did_not_run_is_never_working():
    """PHASE_4_5 A2 review F1 (GPT28, reproduced by GPT24): one passing test must not vouch for a registered test
    without a result, or for a skipped test or case. The reviewers' counterexamples, then the controls."""
    two = ["t.py::test_good", "t.py::test_missing"]
    partly = gen.PARTLY_NOT_RUN
    assert gen.compute_status("F", two, {"t.py::test_good": "passed"}, {})[0] == partly            # no result
    assert gen.compute_status("F", two, {"t.py::test_good": "passed", "t.py::test_missing": "skipped"}, {})[0] == partly
    assert gen.compute_status("F", ["t.py::test_x"], {"t.py::test_x[a]": "passed", "t.py::test_x[b]": "skipped"},
                              {})[0] == partly                                                     # a skipped case
    assert gen.compute_status("F", two, {"t.py::test_good": "passed", "t.py::test_missing": "passed"}, {})[0] \
        == "✅ Working"                                                                            # control: all ran
    assert gen.compute_status("F", two, {"t.py::test_good": "passed", "t.py::test_missing": "failed"}, {})[0] \
        == "🧨 Broken"
    assert gen.compute_status("F", two, {"t.py::test_good": "passed", "t.py::test_missing": "xfailed"}, {})[0] \
        == "⚠️ Known Issue"
    assert gen.compute_status("F", two, {}, {})[0] == "❓ Unknown"
    assert "Working" not in partly and gen._matrix_rank(partly) > gen._matrix_rank("✅ Working")   # never hidden by headless


def test_a_deselected_test_makes_the_feature_partly_verified_through_the_real_path(tmp_path):
    """The same through pytest's own report and the generator's command line: the fake file's last test is deselected
    (-k), so it has no result; the other tests all pass. Control: the full selection gives Working."""
    partly = _source_to_matrix(tmp_path / "a", "never", ["-k", "not after_classes"])
    assert partly["status"] == gen.PARTLY_NOT_RUN and partly["cases"] == 3
    assert [t.split("::")[-1] for t in partly["without_result"]] == ["test_after_classes"]
    assert _source_to_matrix(tmp_path / "b", "never")["status"] == "✅ Working"


def test_a_test_without_result_is_unknown_and_listed_never_a_pass():
    """Exact matching (J4 review M-1): a similar name is not a result, and a missing result is shown."""
    hits = gen.matched_outcomes("d/t.py::test_x", {"d/t.py::test_x_other": "failed", "t.py::test_x_other": "failed"})
    assert hits == []
    assert gen.compute_status("F", ["d/t.py::test_x"], {"d/t.py::test_x_other": "passed"}, {})[0] == "❓ Unknown"
    assert gen.matched_outcomes("d/t.py::test_x", {"t.py::test_x[a]": "passed"}) == [("t.py::test_x[a]", "passed")]


def test_a_not_executed_selection_check_is_attributed_like_an_executed_one():
    """GPT24 (follow-up finding 4): with several filters a skipped or failed selection check counts for intersection."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import _selection_feature
    page = lambda *cols: SimpleNamespace(page=SimpleNamespace(widgets=[SimpleNamespace(column=c) for c in cols]))
    assert _selection_feature(page("a", "b")) == "FILTER.intersection"
    assert _selection_feature(page("a")) == "FILTER.range"
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine import runner
    src = Path(runner.__file__).read_text()
    assert "(LOAD_CHECK, _selection_feature(scenario))" in src and "sel_feature = _selection_feature(scenario)" in src

