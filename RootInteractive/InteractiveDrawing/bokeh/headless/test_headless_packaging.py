"""Packaging checks (proposal v0.2 D-14): the engine is never installed for users.

- No `.ts` file anywhere under headless/ (a `.ts` would be shipped through setup.py's package_data). Always runs.
- The wheel built from the committed tree contains no file of the headless tree. Builds a wheel, so it runs only
  when RI_HEADLESS_WHEEL=1 is set (stage acceptance runs).
"""
import io
import os
import subprocess
import tarfile
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
    """Built from the committed tree (`git archive HEAD`), never from the working tree, so untracked files of a
    checkout cannot change the result (v0.3, GPT22: the alma2 wheel had 146 files from untracked leftovers).
    The build needs pip, setuptools and wheel in the running interpreter; without them the test is skipped with that
    reason rather than failing as if the packaging were wrong (closure note N-7, Claude10)."""
    import importlib.util
    missing = [m for m in ("pip", "setuptools", "wheel") if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"wheel build needs {missing} in this interpreter")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    assert head.returncode == 0, "not a git checkout"
    src = tmp_path / "src"
    src.mkdir()
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD"], capture_output=True)
    assert archive.returncode == 0, archive.stderr[-500:]
    tarfile.open(fileobj=io.BytesIO(archive.stdout)).extractall(src)
    out = subprocess.run([sys.executable, "-m", "pip", "wheel", str(src), "--no-deps", "--no-build-isolation",
                          "-w", str(tmp_path / "wheel")], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    wheels = list((tmp_path / "wheel").glob("*.whl"))
    assert len(wheels) == 1, wheels
    names = zipfile.ZipFile(wheels[0]).namelist()
    found = [n for n in names if "headless" in n]
    print(f"\nwheel of commit {head.stdout.strip()}: {wheels[0].name}: {len(names)} files, {len(found)} headless files")
    assert found == []
