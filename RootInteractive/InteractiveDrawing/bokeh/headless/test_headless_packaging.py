"""Packaging checks (proposal v0.2 D-14): the engine is never installed for users.

- No `.ts` file anywhere under headless/ (a `.ts` would be shipped through setup.py's package_data). Always runs.
- The built wheel contains no file of the headless tree. Builds a wheel of the repository, so it runs only when
  RI_HEADLESS_WHEEL=1 is set (stage acceptance runs).
"""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.layer("unit"), pytest.mark.backend("python")]

HEADLESS = Path(__file__).resolve().parent
REPO = HEADLESS.parents[3]


def test_no_typescript_file_under_headless():
    assert sorted(p.relative_to(HEADLESS).as_posix() for p in HEADLESS.rglob("*.ts")) == []


@pytest.mark.skipif(os.environ.get("RI_HEADLESS_WHEEL") != "1", reason="wheel build only with RI_HEADLESS_WHEEL=1")
def test_wheel_contains_no_headless_file(tmp_path):
    out = subprocess.run([sys.executable, "-m", "pip", "wheel", str(REPO), "--no-deps", "--no-build-isolation",
                          "-w", str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, wheels
    names = zipfile.ZipFile(wheels[0]).namelist()
    found = [n for n in names if "headless" in n]
    print(f"wheel {wheels[0].name}: {len(names)} files, {len(found)} headless files")
    assert found == []
