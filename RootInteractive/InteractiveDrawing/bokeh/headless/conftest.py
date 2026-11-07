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


def _node_unavailable(reason: str):
    if os.environ.get("RI_HEADLESS_REQUIRED") == "1":
        pytest.fail(f"Node model harness required (RI_HEADLESS_REQUIRED=1) but unavailable: {reason}", pytrace=False)
    pytest.skip(f"Node model harness tests skipped: {reason}")


@pytest.fixture(scope="session")
def node_bundle_root() -> Path:
    """Where the stored model bundle lives: the output root, shared by all workers (written atomically)."""
    root = Path(os.environ.get("RI_HEADLESS_OUT", HERE / "_out"))
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def node_session(node_bundle_root):
    """The Node model harness (proposal v0.3 D-26): missing node or jsdom -> skipped locally, ERROR when required."""
    import shutil
    if shutil.which("node") is None:
        _node_unavailable("node is not installed")
    if not (HERE.parent / "node_modules" / "jsdom").exists():
        _node_unavailable("jsdom 24.1.3 is not installed — run: npm install   (in RootInteractive/InteractiveDrawing/bokeh)")
    from .engine import vendor
    problems = vendor.check()
    if problems:
        _node_unavailable("pinned page scripts missing: " + "; ".join(problems))
    from .engine.node_session import NodeSession, NodeUnavailable
    from .engine.node_session import JSDOM_PIN
    try:
        session = NodeSession(node_bundle_root)
    except NodeUnavailable as exc:
        _node_unavailable(str(exc))
    if os.environ.get("RI_HEADLESS_REQUIRED") == "1" and session.versions.get("jsdom") != JSDOM_PIN:
        session.close()
        pytest.fail(f"jsdom {session.versions.get('jsdom')} installed, {JSDOM_PIN} required (proposal v0.3 D-26) "
                    f"— run: npm install   (in RootInteractive/InteractiveDrawing/bokeh)", pytrace=False)
    yield session
    session.close()

