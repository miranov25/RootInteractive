"""pytest fixtures of the headless engine (proposal v0.2 §5.1, §8, §10.2). The repository's root conftest.py and
bokeh/conftest.py stay unchanged.

Missing browser or missing pinned scripts: the browser tests are SKIPPED locally with a clear message, but are an
ERROR when RI_HEADLESS_REQUIRED=1 is set (CI and acceptance runs), so a missing browser can never pass.
Output folder: RI_HEADLESS_OUT, default `headless/_out/` (git-ignored); one sub-folder per pytest-xdist worker.
"""
import json
import os
import time
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
    t0 = time.perf_counter()
    try:
        session = Session()
    except Exception as exc:                                   # Chromium not downloaded, missing system libraries
        _unavailable(f"Chromium could not start: {exc}")
    _session_timing("browser_session_start", time.perf_counter() - t0)
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
    from .engine import node_bundle
    warm = (node_bundle_root / "node_bundle" / f"ri_models_{node_bundle.current_key()[:16]}.js").exists()
    t0 = time.perf_counter()
    try:
        session = NodeSession(node_bundle_root)
    except NodeUnavailable as exc:
        _node_unavailable(str(exc))
    _session_timing("node_session_start", time.perf_counter() - t0, bundle="warm" if warm else "cold")
    yield session
    session.close()


# ------------------------------------------------------------------ fast default suite and full suite (stage 4.2c-2b)
# The fast default suite (`pytest -m "not slow"`, the default of run_tests.sh) keeps every check but samples the
# mass cases (the family tests mark their large variants `slow`) and runs the browser only on its smoke set: ONE page,
# so the browser compiles ONE set of models. Every other browser test is marked `slow` here; the full suite
# (`run_tests.sh --full`, and CI at GSI on every pull request) runs everything.
# The smoke set is the tests marked browser_smoke (a target marker, registered in pytest.ini).


def pytest_collection_modifyitems(config, items):
    for item in items:
        if HERE not in Path(str(item.fspath)).resolve().parents:
            continue                                       # only the tests of this folder
        names = getattr(item, "fixturenames", ())
        if "headless_session" in names and not item.get_closest_marker("browser_smoke"):
            item.add_marker(pytest.mark.slow)
        if "node_session" in names or "node_bundle_root" in names:
            item.add_marker(pytest.mark.node_backend)          # select with -m node_backend (4.2c-2b v0.2)
        if "headless_session" in names:
            item.add_marker(pytest.mark.browser_backend)       # parity tests carry both markers




# --------------------------------------- measurements, outcome list and required scenarios (stage 4.2c-2b v0.2)
# At the end of every headless session — run_tests.sh, a reviewer's pytest, CI — the main pytest process writes into
# the output root (RI_HEADLESS_OUT, default headless/_out):
#   outcomes.txt    the normalised outcome list reviewers compare across machines;
#   timings.jsonl   one line per test: worker, start, stop, duration of setup + call + teardown, outcome;
#   timings.txt     wall time, sum, busy time per worker, the slowest tests, session starts (Node bundle cold/warm),
#                   and the browser page builds per worker (a build over 3 s compiled a new set of models);
# and, when RI_HEADLESS_CHECK_REQUIRED=1 (run_tests.sh), checks required_scenarios.txt. A failure of any of these
# fails the pytest run itself, so it can never pass unnoticed (review of 4.2c-2b: an ignored export failure).
_TIMES = {}


def pytest_configure(config):
    # target markers are registered in pytest.ini and the root conftest.py (4.2c-2c), like every other marker
    # the session's id: NEW in every main pytest process — an id inherited from the shell is never reused (4.2c-2c F-3,
    # GPT22) — and inherited by its xdist workers (started after this hook); reports of another session are stale.
    # A test may fix the id only through the explicitly test-only RI_HEADLESS_TEST_SESSION_ID.
    import uuid
    if not hasattr(config, "workerinput"):
        os.environ["RI_HEADLESS_SESSION"] = os.environ.get("RI_HEADLESS_TEST_SESSION_ID") or uuid.uuid4().hex
    # backend selectors (4.2c-2b v0.2; kept as the approved specification says — their future is a later consensus)
    config.addinivalue_line("markers", "node_backend: runs the Node model harness (set by the headless conftest)")
    config.addinivalue_line("markers", "browser_backend: runs Chromium (set by the headless conftest)")


def _out_root() -> Path:
    return Path(os.environ.get("RI_HEADLESS_OUT", HERE / "_out"))


def _session_timing(what: str, seconds: float, **extra) -> None:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    folder = _out_root() / worker
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "session_timing.jsonl", "a") as fh:
        fh.write(json.dumps({"worker": worker, "what": what, "seconds": round(seconds, 3),
                             "session": os.environ.get("RI_HEADLESS_SESSION"), **extra}) + "\n")


def pytest_runtest_logreport(report):
    if "bokeh/headless/" not in report.nodeid:
        return
    node = getattr(report, "node", None)
    worker = getattr(getattr(node, "gateway", None), "id", None) or os.environ.get("PYTEST_XDIST_WORKER", "main")
    t = _TIMES.setdefault(report.nodeid, {"test": report.nodeid.split("bokeh/headless/")[-1], "worker": worker,
                                           "duration": 0.0, "start": None, "stop": None, "outcome": "passed"})
    t["duration"] += report.duration
    start, stop = getattr(report, "start", None), getattr(report, "stop", None)
    if start is not None:
        t["start"] = start if t["start"] is None else min(t["start"], start)
    if stop is not None:
        t["stop"] = stop if t["stop"] is None else max(t["stop"], stop)
    if report.outcome != "passed":
        t["outcome"] = report.outcome


def _timing_summary(out: Path, times: list) -> str:
    lines = []
    starts = [t["start"] for t in times if t["start"] is not None]
    stops = [t["stop"] for t in times if t["stop"] is not None]
    wall = (max(stops) - min(starts)) if starts and stops else float("nan")
    total = sum(t["duration"] for t in times)
    workers = sorted({t["worker"] for t in times})
    lines.append(f"tests: {len(times)}; workers: {len(workers)}; wall time (first test start to last test end): "
                 f"{wall:.1f} s; sum of test times: {total:.1f} s; occupancy (sum / wall, average busy workers): {total / wall:.2f}")
    lines.append("busy time per worker (s, tests): " + ", ".join(
        f"{w} {sum(t['duration'] for t in times if t['worker'] == w):.1f} ({sum(t['worker'] == w for t in times)})"
        for w in workers))
    lines.append("slowest tests (s, worker):")
    for t in sorted(times, key=lambda t: -t["duration"])[:15]:
        lines.append(f"  {t['duration']:7.2f}  {t['worker']:5s} {t['test'][:110]}")
    lines.append("session starts (s):")
    for f in sorted(out.glob("*/session_timing.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("session") != os.environ.get("RI_HEADLESS_SESSION"):
                continue                                   # another session's entry (re-used output folder)
            lines.append(f"  {r['worker']:5s} {r['what']}: {r['seconds']:.2f}" + (f" (model bundle {r['bundle']})" if "bundle" in r else ""))
    lines.append("browser page builds per worker (count, sum s, builds over 3 s = new model set compiled):")
    builds = {}
    from .engine.report import _reports, session_id
    for f in _reports(out, session_id()):
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        if r.get("backend", "browser") == "browser" and "build_s" in r.get("timings", {}):
            worker = f.relative_to(out).parts[0]
            builds.setdefault(worker, []).append(r["timings"]["build_s"])
    for w, b in sorted(builds.items()):
        big = sorted((x for x in b if x > 3), reverse=True)
        lines.append(f"  {w:5s} {len(b)} builds, {sum(b):.1f} s; compiles: {len(big)} = {sum(big):.1f} s {[round(x, 1) for x in big]}")
    return "\n".join(lines) + "\n"


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput") or not _TIMES:
        return                                             # workers, and sessions without headless tests
    from .engine import report
    out = _out_root()
    out.mkdir(parents=True, exist_ok=True)
    times = sorted(_TIMES.values(), key=lambda t: (t["start"] or 0))
    problems = []
    # the run artifact the capability-matrix generator reads (4.2c-2d A-5): which session wrote the reports here,
    # and whether the run was complete (no marker or keyword deselection) — only a complete run may update the matrix
    markexpr = session.config.getoption("markexpr") or ""
    keyword = session.config.getoption("keyword") or ""
    try:
        (out / "session.json").write_text(json.dumps({
            "session": os.environ.get("RI_HEADLESS_SESSION", ""), "markexpr": markexpr, "keyword": keyword,
            "target": os.environ.get("RI_HEADLESS_TARGET", ""), "tests": len(times),
            "complete": (not markexpr and not keyword and not os.environ.get("RI_HEADLESS_TARGET")
                         and [Path(a).resolve() for a in session.config.args] == [HERE])}, indent=1))
    except Exception as exc:
        problems.append(f"session artifact failed: {exc!r}")
    try:
        (out / "timings.jsonl").write_text("".join(json.dumps(t) + "\n" for t in times))
        (out / "timings.txt").write_text(_timing_summary(out, times))
    except Exception as exc:                               # timings are evidence: their failure fails the run
        problems.append(f"timing summary failed: {exc!r}")
    try:
        lines = report.outcomes(out, session=report.session_id())
        (out / "outcomes.txt").write_text("\n".join(lines) + "\n")
    except Exception as exc:
        problems.append(f"outcome list failed: {exc!r}")
    if os.environ.get("RI_HEADLESS_CHECK_REQUIRED") == "1":
        full = "not slow" not in (session.config.getoption("markexpr") or "")
        try:
            missing = report.required(out, HERE / "required_scenarios.txt", full=full, session=report.session_id(),
                                      target=os.environ.get("RI_HEADLESS_TARGET", ""))
        except Exception as exc:
            missing = [f"(check failed: {exc!r})"]
        if missing:
            problems.append(f"required scenarios missing ({'full' if full else 'fast'} suite): {missing}")
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    for p in problems:
        if reporter:
            reporter.write_line(f"HEADLESS: {p}", red=True)
    if problems and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    elif reporter:
        reporter.write_line(f"HEADLESS: outcome list, timings{' and required scenarios' if os.environ.get('RI_HEADLESS_CHECK_REQUIRED') == '1' else ''} written to {out}")

