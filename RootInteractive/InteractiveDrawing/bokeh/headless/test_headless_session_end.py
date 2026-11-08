"""Controls for the end of a headless session (stage 4.2c-2b v0.2, review findings GPT24 / GPT23).

The headless conftest writes the outcome list and the timings at the end of every session, and checks the required
scenarios when RI_HEADLESS_CHECK_REQUIRED=1. A failure of any of these must fail the pytest run itself, even when every
test passed. Each control runs a real pytest session in a subprocess on a quick unit-test file.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.headless_unit]   # target (4.2c-2c)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
QUICK = HERE / "test_headless_unit_expect.py"            # passes in about a second, no browser and no Node


def _session(out: Path, **env):
    e = {k: v for k, v in os.environ.items()                  # the caller's headless settings must not leak in
         if k not in ("PYTEST_XDIST_WORKER", "RI_HEADLESS_CHECK_REQUIRED", "PYTEST_ADDOPTS", "RI_HEADLESS_SESSION_START", "RI_HEADLESS_TARGET")}
    e.update({"RI_HEADLESS_OUT": str(out), **env})
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(QUICK)],
                          cwd=ROOT, env=e, capture_output=True, text=True, timeout=300)


def test_healthy_session_writes_outcomes_and_timings(tmp_path):
    r = _session(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "outcomes.txt").exists() and (tmp_path / "timings.txt").exists()
    assert "wall time" in (tmp_path / "timings.txt").read_text()


def test_a_broken_outcome_export_fails_the_run(tmp_path):
    bad = tmp_path / "main" / "some_scenario"
    bad.mkdir(parents=True)
    (bad / "report.json").write_text(json.dumps({"id": "x"}))          # no case checksum, no outcomes
    future = time.time() + 600                                          # counts as written in the coming session
    os.utime(bad / "report.json", (future, future))
    r = _session(tmp_path)
    assert r.returncode != 0 and "outcome list failed" in r.stdout, r.stdout + r.stderr


def test_missing_required_scenarios_fail_the_run(tmp_path):
    r = _session(tmp_path, RI_HEADLESS_CHECK_REQUIRED="1")
    assert r.returncode != 0 and "required scenarios missing" in r.stdout, r.stdout + r.stderr


# ------------------------------------------------- C-1 / C-2 of the 4.2c-2b confirmation summary (stage 4.2c-2c)
def _result(sid, outdir, outcome):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.report import ScenarioResult
    r = ScenarioResult(id=sid, outdir=str(outdir))
    r.add(1, "step", "h1.bin_count", outcome, "")
    return r


def test_fail_then_pass_in_one_session_keeps_both(tmp_path, monkeypatch):
    """C-1 (GPT22): one session writes FAIL, then PASS, then PASS for one scenario. Both outcomes stay in the outcome
    list; the repeated, identical PASS counts once."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine import report
    monkeypatch.setenv("RI_HEADLESS_SESSION_START", repr(time.time() - 10))
    for outcome in ("FAIL", "PASS", "PASS"):
        _result("probe/one_scenario", tmp_path, outcome).write()
    lines = [l for l in report.outcomes(tmp_path, since=report.session_start()) if "h1.bin_count" in l]
    assert sorted(lines) == ["probe/one_scenario | step 1 | h1.bin_count | FAIL",
                             "probe/one_scenario | step 1 | h1.bin_count | PASS"], lines
    assert sorted(p.name for p in tmp_path.glob("report*.json")) == ["report.1.json", "report.json"]


def test_a_report_of_an_earlier_session_is_neither_kept_nor_listed(tmp_path, monkeypatch):
    """C-2 (Claude10): a re-used output folder holds a differing report from an EARLIER session. The new session
    overwrites it (no numbered copy) and its outcome list holds only what the new session produced."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine import report
    monkeypatch.setenv("RI_HEADLESS_SESSION_START", "0")
    _result("probe/reused_folder", tmp_path, "FAIL").write()                    # session 1
    old = time.time() - 3600
    os.utime(tmp_path / "report.json", (old, old))
    monkeypatch.setenv("RI_HEADLESS_SESSION_START", repr(time.time() - 1))     # session 2
    _result("probe/reused_folder", tmp_path, "PASS").write()
    assert [p.name for p in tmp_path.glob("report*.json")] == ["report.json"]
    lines = [l for l in report.outcomes(tmp_path, since=report.session_start()) if "h1.bin_count" in l]
    assert lines == ["probe/reused_folder | step 1 | h1.bin_count | PASS"], lines


def test_a_real_session_ignores_reports_left_by_an_earlier_one(tmp_path):
    """C-2 in a real pytest session: an earlier session's report in the output folder is not in the new outcome list."""
    old_dir = tmp_path / "main" / "earlier"
    old_dir.mkdir(parents=True)
    (old_dir / "report.json").write_text(json.dumps({"id": "earlier/scenario", "case_sha256": "x", "outcomes": [
        {"step": 0, "check": "c", "outcome": "FAIL"}]}))
    old = time.time() - 3600
    os.utime(old_dir / "report.json", (old, old))
    r = _session(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "earlier/scenario" not in (tmp_path / "outcomes.txt").read_text()

