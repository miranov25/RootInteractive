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
    root = Path(os.environ.get("RI_HEADLESS_BUNDLE", os.environ.get("RI_HEADLESS_OUT", HERE / "_out")))
    root.mkdir(parents=True, exist_ok=True)              # RI_HEADLESS_BUNDLE: a persistent store (run_tests.sh, 4.2c-2b)
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
    try:
        session = NodeSession(node_bundle_root)
    except NodeUnavailable as exc:
        _node_unavailable(str(exc))
    yield session
    session.close()


# ------------------------------------------------------------------ fast default suite and full suite (stage 4.2c-2b)
# The fast default suite (`pytest -m "not slow"`, the default of run_tests.sh) keeps every check but samples the
# mass cases (the family tests mark their large variants `slow`) and runs the browser only on its smoke set: ONE page,
# so the browser compiles ONE set of models. Every other browser test is marked `slow` here; the full suite
# (`run_tests.sh --full`, and CI at GSI on every pull request) runs everything.
BROWSER_SMOKE = (
    "test_headless_scenarios.py::test_scenario[histograms/plain_1d_2d_one_range_two_moves]",
    "test_headless_parity.py::test_node_equals_browser[histograms/plain_1d_2d_one_range_two_moves]",
)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if HERE not in Path(str(item.fspath)).resolve().parents:
            continue                                       # only the tests of this folder
        if "headless_session" in getattr(item, "fixturenames", ()) and not item.nodeid.endswith(BROWSER_SMOKE):
            item.add_marker(pytest.mark.slow)

