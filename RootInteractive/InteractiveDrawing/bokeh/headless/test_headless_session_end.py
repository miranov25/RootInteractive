"""Controls for the end of a headless session (stage 4.2c-2b v0.2, review findings GPT24 / GPT23).

The headless conftest writes the outcome list and the timings at the end of every session, and checks the required
scenarios when RI_HEADLESS_CHECK_REQUIRED=1. A failure of any of these must fail the pytest run itself, even when every
test passed. Each control runs a real pytest session in a subprocess on a quick unit-test file.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
QUICK = HERE / "test_headless_unit_expect.py"            # passes in about a second, no browser and no Node


def _session(out: Path, **env):
    e = {k: v for k, v in os.environ.items()                  # the caller's headless settings must not leak in
         if k not in ("PYTEST_XDIST_WORKER", "RI_HEADLESS_CHECK_REQUIRED", "PYTEST_ADDOPTS")}
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
    r = _session(tmp_path)
    assert r.returncode != 0 and "outcome list failed" in r.stdout, r.stdout + r.stderr


def test_missing_required_scenarios_fail_the_run(tmp_path):
    r = _session(tmp_path, RI_HEADLESS_CHECK_REQUIRED="1")
    assert r.returncode != 0 and "required scenarios missing" in r.stdout, r.stdout + r.stderr


def _collect(marker):
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-qq", "-p", "no:cacheprovider", "-m", marker,
                        str(HERE)], cwd=ROOT, capture_output=True, text=True, timeout=300)
    return [l for l in r.stdout.splitlines() if "::" in l]


def test_backend_markers_select_their_tests():
    """GPT23 (review of 4.2c-2b): `-m "backend(name='node')"` selected nothing. The plain markers node_backend and
    browser_backend select by the fixtures a test uses; a parity test has both."""
    node, browser = _collect("node_backend"), _collect("browser_backend")
    assert len(node) >= 80 and len(browser) >= 50, (len(node), len(browser))     # measured at v0.2: 84 and 56
    both = set(node) & set(browser)
    assert both and all("test_headless_parity.py::" in t for t in both), sorted(both)[:5]   # only parity needs both

