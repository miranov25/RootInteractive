"""The collection contract of the headless targets (stage 4.2c-2c, specification v0.1 §5 and acceptance check 4).

Targets are pytest markers registered with the description "target: …" (pytest.ini and the root conftest.py);
`run_tests.sh --target <name>` and `pytest -m <name>` select the same
tests. These tests collect the headless folder once in a subprocess and check that every test belongs to a target and
every target selects a test; their controls show that an unmarked test and a misspelled marker are caught.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import targets

pytestmark = [pytest.mark.headless_unit]   # target (4.2c-2c)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DUMP = '''
import json, os
def pytest_collection_finish(session):
    names = sorted({l.split(":")[0].strip() for l in session.config.getini("markers") if ": target:" in l})
    data = {i.nodeid: sorted(m.name for m in i.iter_markers() if m.name in names) for i in session.items}
    open(os.environ["TARGET_DUMP"], "w").write(json.dumps({"targets": names, "collected": data}))
'''


def _collect(tmp_path, folder=HERE):
    (tmp_path / "target_dump_plugin.py").write_text(DUMP)
    out = tmp_path / "collected.json"
    env = {**__import__("os").environ, "TARGET_DUMP": str(out),
           "PYTHONPATH": f"{tmp_path}:{__import__('os').environ.get('PYTHONPATH', '')}"}
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-qq", "-p", "no:cacheprovider",
                        "-p", "target_dump_plugin", str(folder)], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=300)
    if not out.exists():
        return r, [], {}
    d = json.loads(out.read_text())
    return r, d["targets"], {k: set(v) for k, v in d["collected"].items()}


def test_every_headless_test_belongs_to_a_target(tmp_path):
    r, names, collected = _collect(tmp_path)
    assert r.returncode == 0 and collected, r.stdout[-2000:] + r.stderr[-2000:]
    assert len(names) == 21, names                             # the targets registered in pytest.ini (node_joins PHASE_4_3, node_aliases A1, node_widgets and browser_widgets A3, node_a4 and browser_a4 A4, node_stats C1, node_display and browser_display C2, node_transport D1p2)
    assert not targets.problems(collected, names), targets.problems(collected, names)
    over = targets.overlaps(collected)                         # only the smoke set is intended to overlap
    assert over and all(m == ["browser_parity", "browser_smoke"] or m == ["browser_scenarios", "browser_smoke"]
                        for m in over.values()), over


def test_the_contract_catches_an_unmarked_test_and_an_empty_target():
    collected = {"a.py::test_marked": {"node_histograms"}, "a.py::test_forgotten": set()}
    found = targets.problems(collected, ["node_histograms", "node_filters"])
    assert "test without a target marker: a.py::test_forgotten" in found
    assert "target selects no test: node_filters" in found


def test_a_misspelled_target_marker_is_a_collection_error(tmp_path):
    """--strict-markers (pytest.ini) turns an unregistered marker into an error; run from the repository root."""
    bad = tmp_path / "test_typo.py"
    bad.write_text("import pytest\n\n@pytest.mark.node_histogramz\ndef test_x():\n    pass\n")
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-qq", "-p", "no:cacheprovider",
                        "-c", str(ROOT / "pytest.ini"), "--rootdir", str(ROOT), str(bad)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "node_histogramz" in (r.stdout + r.stderr), r.stdout + r.stderr


def test_every_target_has_its_own_registration_in_both_places():
    """R-6 (GPT23, GPT25): each target marker is registered in pytest.ini AND by its own call in the root conftest.py;
    no call holds two adjacent strings (Python would glue them into one marker description)."""
    import re
    root = Path(__file__).resolve().parents[4]
    ini = {m.group(1) for m in re.finditer(r"^\s+([a-z_]+): target:", (root / "pytest.ini").read_text(), re.M)}
    calls = re.findall(r'addinivalue_line\(\s*"markers",\s*((?:"[^"]*"\s*)+)\)', (root / "conftest.py").read_text())
    assert all(c.count('"') == 2 for c in calls), [c for c in calls if c.count('"') != 2]
    conf = {re.match(r'"([a-z_]+): target:', c.strip()).group(1) for c in calls if re.match(r'"[a-z_]+: target:', c.strip())}
    assert ini and ini == conf, (sorted(ini ^ conf))

