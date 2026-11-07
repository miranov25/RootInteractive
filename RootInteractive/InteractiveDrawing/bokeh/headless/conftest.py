"""pytest fixtures of the headless engine (proposal v0.2 §5.1, §8, §10.2). The repository's root conftest.py and
bokeh/conftest.py stay unchanged.

Missing browser or missing pinned scripts: the browser tests are SKIPPED locally with a clear message, but are an
ERROR when RI_HEADLESS_REQUIRED=1 is set (CI and acceptance runs), so a missing browser can never pass.
Output folder: RI_HEADLESS_OUT, default `headless/_out/` (git-ignored); one sub-folder per pytest-xdist worker.
"""
import os
from pathlib import Path

import pytest

os.environ.setdefault("BOKEH_RESOURCES", "inline")
os.environ.setdefault("BOKEH_BROWSER", "none")

HERE = Path(__file__).resolve().parent


def _unavailable(reason: str):
    if os.environ.get("RI_HEADLESS_REQUIRED") == "1":
        pytest.fail(f"headless tests required (RI_HEADLESS_REQUIRED=1) but unavailable: {reason}", pytrace=False)
    pytest.skip(f"headless browser tests skipped: {reason}")


@pytest.fixture(scope="session")
def headless_out() -> Path:
    root = Path(os.environ.get("RI_HEADLESS_OUT", HERE / "_out"))
    out = root / os.environ.get("PYTEST_XDIST_WORKER", "main")
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope="session")
def headless_session():
    try:
        import playwright  # noqa: F401
    except ImportError:
        _unavailable("Playwright is not installed (pip install playwright==1.56.0; python -m playwright install chromium)")
    from .engine import vendor
    problems = vendor.check()
    if problems:
        _unavailable("pinned page scripts missing: " + "; ".join(problems) +
                     " — run: python -m RootInteractive.InteractiveDrawing.bokeh.headless.engine.vendor fetch")
    from .engine.runner import Session
    try:
        session = Session()
    except Exception as exc:                                   # Chromium not downloaded, missing system libraries
        _unavailable(f"Chromium could not start: {exc}")
    yield session
    session.close()
