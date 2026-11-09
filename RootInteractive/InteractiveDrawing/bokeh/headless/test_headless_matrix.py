"""Controls for the headless evidence in the capability matrix (stage 4.2c-2d, option (a) with amendments A-1 … A-7).

`scripts/generate_capability_matrix.py --from-headless <folder>` reads the headless engine's own verdict per capability
from the scenario reports of the folder's session. Each test builds a small synthetic output folder (session.json plus
reports) and checks one rule; the report-kind rule of the engine is checked at the end.
"""
import importlib.util
import json
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


def _report(sid, features, kind="healthy", session="s1", outcomes=(), backend="node"):
    return {"id": sid, "kind": kind, "session": session, "backend": backend, "features": features,
            "outcomes": list(outcomes) or [{"feature": f, "outcome": "PASS", "detail": "", "assertion": True, "scope": "value"}
                                           for f in features]}


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
