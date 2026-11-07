"""Browser runner (proposal v0.2 §5, §6, §7.4, §7.5) — stage 4.2a.

One Chromium per worker process (`Session`); a fresh browser context and page per scenario, closed on success and
on failure. Error capture, console capture and request routing start before the page opens. Every script the page
requests from the internet is served from the pinned local copies (engine.vendor); any other external request is
blocked and reported.

Completion conditions (§5.3):
  load      — Bokeh document present and idle, then the stability rule below;
  set_      — the measured synchronous widget path: set the value inside one async call, await one microtask,
              then read the probed model columns until two consecutive reads around setTimeout(0) agree
              (at most 50 rounds);
  js_action — only with a stated condition (`until`), polled until true or until its timeout (then ERROR).
The engine never waits for a value to equal the oracle.
"""
import importlib.metadata as md
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import data as data_mod
from . import page as page_mod
from . import transport, vendor
from .compare import compare
from .report import ScenarioResult
from .scenario import Audit, Auto, JsAction, JsValue, Load, Scenario, SetRange
from ..oracles import filters, histograms

REPO_ROOT = Path(__file__).resolve().parents[5]
LOAD_TIMEOUT_MS = 20_000
SETTLE_ROUNDS = 50

HELPER_JS = r"""
window.__riH = {
  models() { return (window.Bokeh && Bokeh.documents.length) ? Array.from(Bokeh.documents[0]._all_models.values()) : []; },
  byName(n) { return this.models().find(m => m.name === n); },
  slider(title) { return this.models().find(m => m.type === 'RangeSlider' && m.title === title); },
  col(n, c) { const m = this.byName(n); if (!m) return null; const v = m.get_column(c); return v == null ? null : Array.from(v); },
  snapshot(probes) {
    return JSON.stringify(probes.map(([n, c]) => { const m = this.byName(n); if (!m) return 'missing';
      const v = m.get_column(c); return v == null ? null : Array.from(v); }));
  },
  async settle(probes, maxRounds) {
    await Promise.resolve();
    let prev = this.snapshot(probes);
    for (let r = 1; r <= maxRounds; r++) {
      await new Promise(res => setTimeout(res, 0));
      const cur = this.snapshot(probes);
      if (cur === prev) return {settled: true, rounds: r};
      prev = cur;
    }
    return {settled: false, rounds: maxRounds};
  },
  limits(title) { const s = this.slider(title); return s ? {start: s.start, end: s.end, value: Array.from(s.value)} : null; },
  decoded(cols) {
    const src = this.byName('cdsOrig'); const out = {};
    if (!src) return out;
    for (const c of cols) { const v = src.get_column(c); if (v != null) out[c] = {type: v.constructor.name, values: Array.from(v)}; }
    return out;
  },
};
"""

SET_JS = """async ([title, value, probes, rounds]) => {
  const s = window.__riH.slider(title);
  if (!s) return {error: 'no range widget titled ' + title};
  s.value = (value === 'full') ? [s.start, s.end] : value;
  return await window.__riH.settle(probes, rounds);
}"""

JS_ACTION_JS = """async ([script, until, timeoutMs]) => {
  (new Function(script))();
  const cond = new Function('return (' + until + ');');
  const t0 = performance.now();
  while (!cond()) {
    if (performance.now() - t0 > timeoutMs) return {settled: false, waited_ms: performance.now() - t0};
    await new Promise(r => setTimeout(r, 5));
  }
  return {settled: true, waited_ms: performance.now() - t0};
}"""


class StepNotComplete(RuntimeError):
    pass


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except Exception:
        return "not installed"


class Session:
    """One Chromium per worker process."""

    def __init__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch()
        self.versions = {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "bokeh": _version("bokeh"), "playwright": _version("playwright"),
            "chromium": self.browser.version, "machine": platform.machine(),
            "rootinteractive": f"{_git('rev-parse', '--short', 'HEAD')} ({_git('rev-parse', '--abbrev-ref', 'HEAD')})",
            **{f"vendor:{k}": v for k, v in vendor.versions().items()},
        }

    def close(self) -> None:
        try:
            self.browser.close()
        finally:
            self._pw.stop()


# ------------------------------------------------------------------------------------------- oracle helpers
def _widget_mask(pg, scenario: Scenario, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, dict]]:
    states = {}
    masks = []
    for w in scenario.page.widgets:
        lim = pg.evaluate("t => window.__riH.limits(t)", w.column)
        if lim is None:
            raise StepNotComplete(f"range widget {w.column!r} not found on the page")
        states[w.column] = lim
        masks.append(filters.range_mask(df[w.column].to_numpy(), lim["value"][0], lim["value"][1]))
    return filters.intersection(masks, len(df)), states


def _variable(df: pd.DataFrame, scenario: Scenario, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy()
    expr = dict(scenario.page.aliases)[name]
    return np.asarray(df.eval(expr), dtype=np.float64)       # aliases: arithmetic subset (RB-6), evaluated by pandas


def _probes(scenario: Scenario) -> List[List[str]]:
    p = [[name, "bin_count"] for name, _, _ in page_mod.histogram_models(scenario.page)]
    p += [[name, "mean"] for name in scenario.page.profiles]
    return p


def _applies(check, step: int) -> bool:
    return check.steps is None or step in check.steps


# ------------------------------------------------------------------------------------------- the run
def run(session: Session, scenario: Scenario, out_root: Path, test_ref: str = "") -> ScenarioResult:
    scenario.validate()
    outdir = Path(out_root) / scenario.id.replace("/", "__")
    outdir.mkdir(parents=True, exist_ok=True)
    res = ScenarioResult(id=scenario.id, outdir=str(outdir), versions=dict(session.versions),
                         reproduce=f"pytest {test_ref} -k '{scenario.id}'" if test_ref else scenario.id)
    t0 = time.time()
    df = data_mod.generate(scenario.data.main)
    res.case_sha256 = data_mod.save_case(df, outdir / "case.npz")
    html = page_mod.build(scenario, df, outdir / "page.html")
    res.timings["build_s"] = round(time.time() - t0, 3)

    routes = {glob.replace("**/", "/"): path for glob, path in vendor.routes().items()}
    ctx = session.browser.new_context()
    try:
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: res.page_errors.append(str(e)[:300]))
        pg.on("console", lambda m: res.console_errors.append(m.text[:300]) if m.type == "error" else None)

        def route(r):
            url = r.request.url
            if url.startswith("file:"):
                return r.continue_()
            for suffix, path in routes.items():
                if url.split("?")[0].endswith(suffix):
                    res.requests_served_locally.append(url)
                    return r.fulfill(path=str(path), content_type="text/javascript")
            res.requests_blocked.append(url)
            return r.abort()
        pg.route("**/*", route)
        pg.add_init_script(HELPER_JS)
        probes = _probes(scenario)

        for i, step in enumerate(scenario.steps):
            name = step.describe()
            ts = time.time()
            try:
                if isinstance(step, Load):
                    pg.goto(html.resolve().as_uri(), timeout=LOAD_TIMEOUT_MS)
                    pg.wait_for_function("() => window.Bokeh && Bokeh.documents.length > 0 && Bokeh.documents[0].is_idle",
                                         timeout=LOAD_TIMEOUT_MS)
                    done = pg.evaluate("([p, r]) => window.__riH.settle(p, r)", [probes, SETTLE_ROUNDS])
                    _transport_check(pg, scenario, df, res)
                elif isinstance(step, SetRange):
                    value = step.value if step.value == "full" else list(step.value)
                    done = pg.evaluate(SET_JS, [step.column, value, probes, SETTLE_ROUNDS])
                elif isinstance(step, JsAction):
                    done = pg.evaluate(JS_ACTION_JS, [step.script, step.until, step.timeout_s * 1000])
                else:
                    raise StepNotComplete(f"unsupported step {type(step).__name__}")
                if done.get("error"):
                    raise StepNotComplete(done["error"])
                if not done.get("settled"):
                    raise StepNotComplete(f"completion condition not met ({done})")
                res.completion.append({"step": i, "name": name, **done, "seconds": round(time.time() - ts, 3)})
            except Exception as exc:                                     # the step itself failed: ERROR, stop
                res.add(i, name, "step completion", "ERROR", f"{type(exc).__name__}: {str(exc)[:300]}")
                break
            snapshot = pg.evaluate("p => window.__riH.snapshot(p)", probes)
            for check in scenario.checks:
                if not _applies(check, i):
                    continue
                try:
                    if isinstance(check, Auto):
                        _auto(pg, scenario, df, res, i, name)
                    elif isinstance(check, Audit):
                        pg.wait_for_timeout(int(check.delay_s * 1000))   # audit mode only: deliberate delay
                        again = pg.evaluate("p => window.__riH.snapshot(p)", probes)
                        res.add(i, name, f"audit after {check.delay_s} s", "PASS" if again == snapshot else "ERROR",
                                "page unchanged after completion" if again == snapshot else
                                "page changed after the completion condition was met: the condition is insufficient")
                    elif isinstance(check, JsValue):
                        got = pg.evaluate(f"() => ({check.expr})")
                        res.add(i, name, f"js value {check.expr}", "PASS" if got == check.expected else "FAIL",
                                f"page {got!r}, expected {check.expected!r}")
                except Exception as exc:
                    res.add(i, name, type(check).__name__, "ERROR", f"{type(exc).__name__}: {str(exc)[:300]}")
        for e in res.page_errors:
            res.add(-1, "page", "page errors", "ERROR", e)
        for u in res.requests_blocked:
            res.add(-1, "page", "external request", "ERROR", f"blocked, not a pinned local script: {u}")
    finally:
        ctx.close()
    res.timings["total_s"] = round(time.time() - t0, 3)
    res.write()
    return res


def _transport_check(pg, scenario: Scenario, df: pd.DataFrame, res: ScenarioResult) -> None:
    cols = page_mod.input_columns(scenario)
    expected = transport.expected_decoded(df, cols, scenario.transport.array_compression)
    decoded = pg.evaluate("c => window.__riH.decoded(c)", cols)
    for col, ok, msg in transport.check(decoded, expected):
        res.add(0, "load", f"transport {col}", "PASS" if ok else "FAIL", msg)


def _auto(pg, scenario: Scenario, df: pd.DataFrame, res: ScenarioResult, step: int, name: str) -> None:
    mask, states = _widget_mask(pg, scenario, df)
    if step == 0 and states and all(s["value"] == [s["start"], s["end"]] for s in states.values()):
        finite = np.ones(len(df), dtype=bool)
        for col in states:
            finite &= np.isfinite(df[col].to_numpy())
        kept = int((mask & finite).sum())
        res.add(step, name, "load: every finite row kept at default limits",
                "PASS" if kept == int(finite.sum()) else "FAIL", f"{kept} of {int(finite.sum())} finite rows kept")
    for model, h, wcol in page_mod.histogram_models(scenario.page):
        cols = [_variable(df, scenario, v) for v in h.variables]
        weights = None if wcol is None else df[wcol].to_numpy()
        expect = histograms.counts(cols, h.nbins, h.range, mask=mask, weights=weights)
        got = pg.evaluate("([n, c]) => window.__riH.col(n, c)", [model, "bin_count"])
        if got is None:
            res.add(step, name, f"{model}.bin_count", "ERROR", "model or column not found on the page")
            continue
        cmp = compare(got, expect, kind="count") if weights is None else \
            compare(got, expect, kind="float", abs_tol=1e-9, rel_tol=1e-12)
        res.add(step, name, f"{model}.bin_count", "PASS" if cmp.ok else "FAIL",
                f"{cmp.message}; selected rows {int(mask.sum())}" + (f"; ratio page/oracle {cmp.ratio_range}" if cmp.ratio_range else ""))
