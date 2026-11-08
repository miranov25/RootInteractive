"""Runner controls for `run_tests.sh --target` (stage 4.2c-2c v0.2; review findings GPT22, GPT23, GPT24).

Each test runs the real run_tests.sh in a miniature checkout — like RootInteractive/MLpipeline/tests/
test_phase_2_1_runner.py does for the run without targets — with two demo targets registered in a small pytest.ini.
A failing test, a forced failure of the evidence zip, an unreadable pytest report, a misspelled name next to "all",
and a joins run with zero tests must each make the run fail; a passing target must succeed with its evidence written.
"""
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.headless_unit]   # target (4.2c-2c)
RUNNER = Path(__file__).resolve().parents[4] / "run_tests.sh"
HEADLESS = "RootInteractive/InteractiveDrawing/bokeh/headless"


@pytest.fixture
def checkout(tmp_path):
    shutil.copy2(RUNNER, tmp_path / "run_tests.sh")
    (tmp_path / "pytest.ini").write_text(textwrap.dedent("""\
        [pytest]
        addopts = --strict-markers
        markers =
            demo_pass: target: a passing demo target
            demo_fail: target: a demo target with one failing test
        """))
    h = tmp_path / HEADLESS
    h.mkdir(parents=True)
    (h / "test_demo.py").write_text(textwrap.dedent("""\
        import pytest

        @pytest.mark.demo_pass
        def test_passes():
            assert True

        @pytest.mark.demo_fail
        def test_fails():
            assert False, "deliberate failure"
        """))
    (h / "conftest.py").write_text(textwrap.dedent("""\
        import os

        def pytest_unconfigure(config):          # makes the pytest report unreadable after pytest wrote it
            path = getattr(config.option, "json_report_file", None)
            if os.environ.get("CORRUPT_REPORT") and path and not hasattr(config, "workerinput"):
                open(path, "w").write("{not json")
        """))
    for directory in ("RootInteractive/InteractiveDrawing/bokeh", "RootInteractive/Tools", "RootInteractive/onnx",
                      "RootInteractive/MLpipeline/tests"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def _run(root, *args, **env):
    e = {**os.environ, "RootInteractive_DOCS": str(root / "docs"), **env}
    for k in ("RI_HEADLESS_OUT", "RI_HEADLESS_SESSION", "RI_HEADLESS_TARGET", "RI_HEADLESS_CHECK_REQUIRED",
              "PYTEST_XDIST_WORKER", "PYTEST_ADDOPTS"):
        e.pop(k, None)
    return subprocess.run(["bash", "./run_tests.sh", *args, "-n", "1"], cwd=root, env=e, capture_output=True,
                          text=True, timeout=240)


def test_a_passing_target_succeeds_with_its_evidence(checkout):
    r = _run(checkout, "--target", "demo_pass")
    assert r.returncode == 0 and "Evidence:" in r.stdout, r.stdout + r.stderr
    assert list((checkout / "test_logs").glob("targets_*.zip"))


def test_a_failing_test_fails_the_target_and_keeps_the_evidence(checkout):
    r = _run(checkout, "--target", "demo-fail")
    line = [l for l in r.stdout.splitlines() if l.startswith("demo-fail ")]
    assert r.returncode != 0 and line and line[0].split()[3] == "1", r.stdout + r.stderr   # 1 failed
    assert list((checkout / "test_logs").glob("targets_*.zip"))


def test_a_failed_evidence_zip_fails_the_run(checkout, tmp_path_factory):
    fake = tmp_path_factory.mktemp("fakebin")
    (fake / "zip").write_text("#!/bin/sh\nexit 17\n")
    (fake / "zip").chmod(0o755)
    r = _run(checkout, "--target", "demo_pass", PATH=f"{fake}:{os.environ['PATH']}")
    assert r.returncode != 0 and "Evidence:" not in r.stdout and "evidence zip could not be written" in r.stdout, \
        r.stdout + r.stderr


def test_an_unreadable_report_fails_the_run(checkout):
    r = _run(checkout, "--target", "demo_pass", CORRUPT_REPORT="1")
    assert r.returncode != 0 and "REPORT UNREADABLE" in r.stdout, r.stdout + r.stderr


def test_a_misspelled_name_next_to_all_runs_nothing(checkout):
    r = _run(checkout, "--target", "all", "--target", "demo_passs")
    assert r.returncode != 0 and "unknown target(s): demo_passs" in r.stdout, r.stdout + r.stderr
    assert "Testing:" not in r.stdout and "Target:" not in r.stdout, r.stdout                  # nothing was run


def test_a_joins_run_with_zero_tests_fails(checkout, tmp_path_factory):
    """GPT24: a readable TAP summary with zero tests must not pass. A stand-in `node` first in PATH reports exactly that
    (a real `node --test` counts even an empty test file as one passing test)."""
    joins = checkout / "RootInteractive/joins"
    joins.mkdir(parents=True)
    (joins / "test_empty.mjs").write_text("// no tests\n")
    (joins / "ao2d_fixture.json").write_text("{}")
    fake = tmp_path_factory.mktemp("fakenode")
    (fake / "node").write_text('#!/bin/sh\ncase "$*" in *--help*) echo "--test-reporter";; '
                               '*) printf "TAP version 13\\n# tests 0\\n# pass 0\\n# fail 0\\n";; esac\nexit 0\n')
    (fake / "node").chmod(0o755)
    r = _run(checkout, "--target", "joins", PATH=f"{fake}:{os.environ['PATH']}")
    assert r.returncode != 0 and "NO TESTS" in r.stdout, r.stdout + r.stderr
