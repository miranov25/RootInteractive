"""Browser runner (proposal v0.2 §5, §6, §7.4, §7.5, §8, §12) — stage 4.2b.

One Chromium per worker process (`Session`); a fresh browser context and page per scenario, closed on success and
on failure. Error capture, console capture and request routing start before the page opens. Every script the page
requests from the internet is served from the pinned local copies (engine.vendor); any other external request is
blocked and reported as an ERROR.

Completion conditions (§5.3):
  load      — Bokeh document present and idle, then the stability rule below;
  set_      — the measured synchronous widget path: set the value inside one async call, await one microtask,
              then read the probed model columns until two consecutive reads around setTimeout(0) agree
              (at most 50 rounds);
  js_action — only with a stated condition (`until`), polled until true or until its timeout (then ERROR).
After every step the probed models must all exist before anything is read. The engine never waits for a value to
equal the oracle.

Probes name their model types (contract since v0.3): several page models can share a name — a histogram and the
DownsamplerCDS that plots it (bokehTools.py 2333) — so every lookup states the accepted types and must find exactly
one model; none or several is an ERROR, never a guess. The same exactly-one rule holds for widgets (by title) and
for the transport source `cdsOrig`.

Outcomes (engine/expect.py): PASS, FAIL, XFAIL (a declared known defect whose signature holds), XPASS (strict), ERROR,
SKIP (a declared check on a step never reached). Prerequisites (transport, build, fault application, console policy,
page errors, blocked requests) never make a scenario pass on their own.
"""
import importlib.metadata as md
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import data as data_mod
from . import faults as faults_mod
from . import page as page_mod
from . import transport, vendor
from .compare import compare
from .expect import EqualsSelection, classify
from .report import ScenarioResult
from .scenario import LOAD_CHECK, Audit, Auto, JsAction, JsValue, Load, Responds, Scenario, SetRange
from ..oracles import filters, histograms
from ..oracles.filters import SELECTION_RULES

REPO_ROOT = Path(__file__).resolve().parents[5]
LOAD_TIMEOUT_MS = 20_000
SETTLE_ROUNDS = 50

HIST_TYPES = ["HistogramCDS", "HistoNdCDS"]
PROFILE_TYPES = ["HistoNdProfile", "CDSStack"]
# models whose output a scenario could verify; any of them on the page without a check is listed in the report
INVENTORY_TYPES = ["HistogramCDS", "HistoNdCDS", "HistoNdProfile", "CDSStack", "HistoStatsCDS", "CDSJoin",
                   "ClientLinearFitter", "CustomJSNAryFunction", "OrtFunction"]

# Console-error policy (stage 4.2b): every console error is an ERROR unless it is listed here, with its reason.
CONSOLE_ALLOWED = (
    (r"^\[bokeh\] Bad column name on label",
     "Bokeh legend validation on the weights-stacked profile figure (factor_cmap on 'weights', bokehTools.py 960); "
     "cosmetic, the figure renders; seen only in chain/deep_widget_filter_alias_stack_profile"),
)

HELPER_JS = r"""
window.__riH = {
  models() { return (window.Bokeh && Bokeh.documents.length) ? Array.from(Bokeh.documents[0]._all_models.values()) : []; },
  find(n, types) { return this.models().filter(m => m.name === n && (!types || types.includes(m.type))); },
  model(n, types) { const c = this.find(n, types); return c.length === 1 ? c[0] : null; },
  why(n, types) {
    const c = this.find(n, types);
    if (c.length > 1) return 'ambiguous: ' + c.length + ' models of type ' + (types || []).join('|') + ' named ' + n;
    const all = this.models().filter(m => m.name === n).map(m => m.type);
    return 'not found: no model of type ' + (types || []).join('|') + ' named ' + n +
           (all.length ? ' (models with that name: ' + all.join(', ') + ')' : '');
  },
  sliders(title) { return this.models().filter(m => m.type === 'RangeSlider' && m.title === title); },
  slider(title) { const c = this.sliders(title); return c.length === 1 ? c[0] : null; },
  col(n, c, types) { const m = this.model(n, types); if (!m) return null; const v = m.get_column(c); return v == null ? null : Array.from(v); },
  missing(probes) {
    const out = [];
    for (const [n, c, t] of probes) {
      const m = this.model(n, t);
      if (!m) out.push(n + '.' + c + ' — ' + this.why(n, t));
      else if (m.get_column(c) == null) out.push(n + '.' + c + ' — column not found on ' + m.type + ' ' + n);
    }
    return out;
  },
  encode(x) {                       // JSON that keeps NaN, +-Infinity and -0 apart (4.2c-1 v0.2, parity)
    return JSON.stringify(x, (k, v) => typeof v === 'number' && (!Number.isFinite(v) || Object.is(v, -0))
                                       ? (Object.is(v, -0) ? '-0' : String(v)) : v);
  },
  maskHash(mask) {                  // 64-bit fingerprint of the 0/1 selection mask: which rows, not only how many —
    let a = 0x811c9dc5, b = 0x050c5d1f;   // two FNV-1a passes, forwards and backwards, different offsets (4.2c-2a)
    const n = mask.length;
    for (let i = 0; i < n; i++) {
      a ^= mask[i] ? 1 : 0; a = Math.imul(a, 0x01000193) >>> 0;
      b ^= mask[n - 1 - i] ? 1 : 0; b = Math.imul(b, 0x01000193) >>> 0;
    }
    return a.toString(16).padStart(8, '0') + b.toString(16).padStart(8, '0') + ':' + n;
  },
  snapshot(probes) {
    return this.encode(probes.map(([n, c, t]) => Array.from(this.model(n, t).get_column(c))));
  },
  async settle(probes, maxRounds) {
    await Promise.resolve();
    const miss = this.missing(probes);
    if (miss.length) return {settled: false, error: 'probed model or column: ' + miss.join('; ')};
    let prev = this.snapshot(probes);
    for (let r = 1; r <= maxRounds; r++) {
      await new Promise(res => setTimeout(res, 0));
      const cur = this.snapshot(probes);
      if (cur === prev) return {settled: true, rounds: r};
      prev = cur;
    }
    return {settled: false, rounds: maxRounds};
  },
  selected(name, types, n) {
    const m = this.model(name, types);
    if (!m) return {error: name + ': ' + this.why(name, types)};
    const f = m.filter;
    if (f == null || !f.active) return {count: n, filter: null, mask_fnv: this.maskHash(new Array(n).fill(1))};
    const v = f.v_compute();
    if (v == null || v.length < n) return {error: 'filter vector of length ' + (v == null ? 'null' : v.length) +
                                                    ' is shorter than the table (' + n + ' rows)'};
    let c = 0;
    for (let i = 0; i < n; i++) if (v[i]) c++;
    return {count: c, filter: f.type, length: v.length, mask_fnv: this.maskHash(Array.from(v).slice(0, n))};
  },
  limits(title) {
    const s = this.slider(title);
    return s ? {start: s.start, end: s.end, value: Array.from(s.value)} : {error: 'range widget titled ' + title +
           ' found ' + this.sliders(title).length + ' times (must be exactly once)'};
  },
  decoded(cols) {
    const src = this.find('cdsOrig', null);
    if (src.length !== 1) return {__error: 'transport source cdsOrig found ' + src.length + ' times (must be exactly once)'};
    const out = {};
    for (const c of cols) { const v = src[0].get_column(c); if (v != null) out[c] = {type: v.constructor.name, values: Array.from(v)}; }
    return out;
  },
  inventory(types) { return this.models().filter(m => types.includes(m.type)).map(m => [m.type, m.name]); },
};
"""

SET_JS = """async ([title, value, probes, rounds]) => {
  const s = window.__riH.slider(title);
  if (!s) return {error: 'range widget titled ' + title + ' found ' + window.__riH.sliders(title).length +
                         ' times (must be exactly once)'};
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
def _widget_mask(pg, scenario: Scenario, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, dict], List[np.ndarray]]:
    """Intersection of all range filters, the widget states read from the page, and the mask of each widget in the
    order of the page declaration."""
    states = {}
    masks = []
    for w in scenario.page.widgets:
        lim = pg.evaluate("t => window.__riH.limits(t)", w.column)
        if lim.get("error"):
            raise StepNotComplete(lim["error"])
        states[w.column] = lim
        masks.append(filters.range_mask(df[w.column].to_numpy(), lim["value"][0], lim["value"][1]))
    return filters.intersection(masks, len(df)), states, masks


def _variable(df: pd.DataFrame, scenario: Scenario, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy()
    expr = dict(scenario.page.aliases)[name]
    return np.asarray(df.eval(expr), dtype=np.float64)       # aliases: arithmetic subset (RB-6), evaluated by pandas


def _types(scenario: Scenario, model: str) -> List[str]:
    return PROFILE_TYPES if model in scenario.page.profiles else HIST_TYPES


def _probes(scenario: Scenario) -> List[list]:
    p = [[name, "bin_count", HIST_TYPES] for name, _, _ in page_mod.histogram_models(scenario.page)]
    p += [[name, "mean", PROFILE_TYPES] for name in scenario.page.profiles]
    return p


def _applies(check, step: int) -> bool:
    return check.steps is None or step in check.steps


def _hist_feature(h) -> str:
    return "HIST.histogram_1d" if len(h.variables) == 1 else "HIST.histogram_nd"


def _load_check_applies(scenario: Scenario, step: int) -> bool:
    return step == 0 and bool(scenario.page.widgets) and all(w.limits is None for w in scenario.page.widgets)


def _concrete_checks(scenario: Scenario, check, step: int) -> List[Tuple[str, str]]:
    """The concrete checks (report name, feature) a declared check produces at `step` — used to record SKIP and ERROR
    for exactly the outputs that were not verified (4.2b v0.2, GPT24 P1-2), never for the scenario's first feature
    only."""
    if isinstance(check, Auto):
        out = [(LOAD_CHECK, _selection_feature(scenario))] if _load_check_applies(scenario, step) else []
        return out + [(f"{m}.bin_count", _hist_feature(h)) for m, h, _ in page_mod.histogram_models(scenario.page)]
    if isinstance(check, Audit):
        return [(f"audit after {check.delay_s} s", "ENGINE.completion")]
    if isinstance(check, JsValue):
        return [(f"js value {check.expr}", "ENGINE.harness")]
    if isinstance(check, Responds):
        if step != max(check.steps):
            return []
        return [(f"{check.model}.{check.column} responds to step {check.moved} and returns at step {check.restored}",
                 _feature_of(scenario, check))]
    return [(type(check).__name__, "")]


def _responds(res: ScenarioResult, c: Responds, values: Dict[int, list], step: int, name: str, feature: str) -> None:
    label = f"{c.model}.{c.column} responds to step {c.moved} and returns at step {c.restored}"
    if any(values.get(k) is None for k in c.steps):
        res.add(step, name, label, "ERROR", f"probed column missing at step(s) "
                f"{[k for k in c.steps if values.get(k) is None]}", feature=feature, scope="response")
        return
    ref, moved, back = (np.asarray(values[k], dtype=np.float64) for k in (c.reference, c.moved, c.restored))
    changed = not compare(moved, ref, kind="float", rel_tol=c.rel_tol, nan_equal=True).ok
    restored = compare(back, ref, kind="float", rel_tol=c.rel_tol, nan_equal=True)
    ok = changed and restored.ok and ref.size > 0 and np.isfinite(ref).any()
    why = [] if ok else (["did not change on the move"] if not changed else []) + \
        ([f"not restored: {restored.message}"] if not restored.ok else []) + \
        (["no finite value at the reference step"] if not (ref.size > 0 and np.isfinite(ref).any()) else [])
    res.add(step, name, label, "PASS" if ok else "FAIL",
            f"{ref.size} values; changed on the move, restored within relative {c.rel_tol:g}" if ok else "; ".join(why),
            feature=feature, scope="response")


# ------------------------------------------------------------------------------------------- the run
def run(session: Session, scenario: Scenario, out_root: Path, test_ref: str = "",
        fault: Optional[faults_mod.Fault] = None) -> ScenarioResult:
    """Run one scenario; with `fault`, on a disposable copy of its page with that fault applied exactly once.
    The report is written in every case, also after a build failure or an unexpected exception."""
    scenario.validate()
    node = getattr(session, "backend", "browser") == "node"
    outdir = Path(out_root) / (scenario.id.replace("/", "__") + ("__node" if node else "")
                               + (f"__fault_{fault.id}" if fault else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    res = ScenarioResult(id=scenario.id + (" [node]" if node else "") + (f" [fault {fault.id}]" if fault else ""),
                         outdir=str(outdir), versions=dict(session.versions),
                         reproduce=_reproduce(test_ref, scenario.id))
    res.fault = fault.id if fault else ""
    res.backend = "node" if node else "browser"
    res.kind = _kind(scenario, fault)
    t0 = time.time()
    try:
        df = data_mod.generate(scenario.data.main)
        res.case_sha256 = data_mod.save_case(df, outdir / "case.npz")
        if node:                                                      # tier T2: the page document, no HTML
            if (fault.id if fault else None) != (session.fault.id if session.fault else None):
                res.add(-1, "fault", "fault applied", "ERROR", "the Node session does not carry this fault "
                        "(a fault is applied to the session's bundle copy)", assertion=False)
                return res
            try:
                target = page_mod.build_document(scenario, df, outdir / "page_document.json")
            except Exception as exc:
                res.add(-1, "build", "page build", "ERROR",
                        f"page build failed: {type(exc).__name__}: {str(exc)[:300]}", assertion=False)
                return res
            res.timings["build_s"] = round(time.time() - t0, 3)
            _drive(session.new_page(res), scenario, df, res, fault, target)
            return res
        try:
            html = page_mod.build(scenario, df, outdir / "page.html")
        except Exception as exc:                                      # Claude3 C-2: never a bare crash
            res.add(-1, "build", "page build", "ERROR", f"page build failed: {type(exc).__name__}: {str(exc)[:300]}",
                    assertion=False)
            return res
        if fault is not None:
            try:
                mutated = faults_mod.apply(html.read_text(encoding="utf-8"), fault)
            except faults_mod.FaultNotApplied as exc:
                res.add(-1, "fault", "fault applied", "ERROR", str(exc), assertion=False)
                return res
            html = outdir / "page_fault.html"
            html.write_text(mutated, encoding="utf-8")
        res.timings["build_s"] = round(time.time() - t0, 3)
        _browser_run(session, scenario, df, html, res, fault)
    except Exception as exc:                                          # Claude3 N-1: anything else still reports
        res.add(-1, "engine", "engine", "ERROR", f"{type(exc).__name__}: {str(exc)[:300]}", assertion=False)
    finally:
        res.timings["total_s"] = round(time.time() - t0, 3)
        res.write()
    return res


def _browser_run(session: Session, scenario: Scenario, df: pd.DataFrame, html: Path, res: ScenarioResult,
                 fault: Optional[faults_mod.Fault]) -> None:
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
        _drive(pg, scenario, df, res, fault, html.resolve().as_uri())
    finally:
        ctx.close()


BROWSER_LOAD_CONDITION = "() => window.Bokeh && Bokeh.documents.length > 0 && Bokeh.documents[0].is_idle"


def _drive(pg, scenario: Scenario, df: pd.DataFrame, res: ScenarioResult, fault: Optional[faults_mod.Fault],
           target) -> None:
    """The step loop and the checks after it, shared by both backends. `pg` is a Playwright page (browser) or a
    NodePage (Node model harness); both offer goto, wait_for_function, evaluate and wait_for_timeout. The load
    condition differs: the browser waits until the document is idle (its views rendered); Node has no views, and
    `Document.from_json` builds, initialises and connects every model before it returns. In both, the settle rule on
    the probed models then decides when the load is complete."""
    probes = _probes(scenario)
    recorded: Dict[Tuple[str, str], Dict[int, list]] = {}
    state: Dict[str, list] = {}               # widget masks at load (selection rules of known defects need them)

    for i, step in enumerate(scenario.steps):
        name = step.describe()
        ts = time.time()
        try:
            if isinstance(step, Load):
                pg.goto(target, timeout=LOAD_TIMEOUT_MS)
                pg.wait_for_function(getattr(pg, "load_condition", BROWSER_LOAD_CONDITION), timeout=LOAD_TIMEOUT_MS)
                done = pg.evaluate("([p, r]) => window.__riH.settle(p, r)", [probes, SETTLE_ROUNDS])
                if done.get("settled"):
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
            miss = pg.evaluate("p => window.__riH.missing(p)", probes)     # Claude3 N-1: also after js_action
            if miss:
                raise StepNotComplete("probed model or column: " + "; ".join(miss))
            res.completion.append({"step": i, "name": name, **done, "seconds": round(time.time() - ts, 3)})
            if isinstance(step, Load) and scenario.page.widgets:
                state["load_masks"] = _widget_mask(pg, scenario, df)[2]
        except Exception as exc:                                   # the step itself failed: ERROR, then SKIP
            res.add(i, name, "step completion", "ERROR", f"{type(exc).__name__}: {str(exc)[:300]}", assertion=False)
            for k in range(i, len(scenario.steps)):
                for check in scenario.checks:
                    if _applies(check, k):
                        for cname, feat in _concrete_checks(scenario, check, k):
                            res.add(k, scenario.steps[k].describe(), cname, "SKIP",
                                    "not executed: an earlier step did not complete", feature=feat)
            break
        snapshot = pg.evaluate("p => window.__riH.snapshot(p)", probes)
        res.probe_values[str(i)] = snapshot
        if scenario.page.widgets and page_mod.histogram_models(scenario.page):
            sel = pg.evaluate("([m, t, n]) => window.__riH.selected(m, t, n)",
                              [page_mod.histogram_models(scenario.page)[0][0], HIST_TYPES, len(df)])
            res.probe_values[f"{i}:selected"] = ({"count": sel["count"], "mask_fnv": sel["mask_fnv"]}
                                                 if "count" in sel else sel.get("error"))
        for check in scenario.checks:
            if isinstance(check, Responds) and i in check.steps:
                recorded.setdefault((check.model, check.column), {})[i] = \
                    pg.evaluate("([n, c, t]) => window.__riH.col(n, c, t)",
                                [check.model, check.column, _types(scenario, check.model)])
            if not _applies(check, i):
                continue
            try:
                if isinstance(check, Auto):
                    _auto(pg, scenario, df, res, i, name, state)
                elif isinstance(check, Audit):
                    pg.wait_for_timeout(int(check.delay_s * 1000))   # audit mode only: deliberate delay
                    miss = pg.evaluate("p => window.__riH.missing(p)", probes)
                    if miss:
                        raise StepNotComplete("probed model or column: " + "; ".join(miss))
                    again = pg.evaluate("p => window.__riH.snapshot(p)", probes)
                    res.add(i, name, f"audit after {check.delay_s} s", "PASS" if again == snapshot else "ERROR",
                            "page unchanged after completion" if again == snapshot else
                            "page changed after the completion condition was met: the condition is insufficient",
                            feature="ENGINE.completion")
                elif isinstance(check, JsValue):
                    got = pg.evaluate(f"() => ({check.expr})")
                    res.add(i, name, f"js value {check.expr}", "PASS" if got == check.expected else "FAIL",
                            f"page {got!r}, expected {check.expected!r}", feature="ENGINE.harness")
                elif isinstance(check, Responds) and i == max(check.steps):
                    _responds(res, check, recorded.get((check.model, check.column), {}), i, name,
                              _feature_of(scenario, check))
            except Exception as exc:                                # ERROR on each affected concrete check
                for cname, feat in _concrete_checks(scenario, check, i):
                    res.add(i, name, cname, "ERROR", f"{type(exc).__name__}: {str(exc)[:300]}", feature=feat)

    if fault is not None:                                          # proof that the mutated code ran
        res.fault_executions = int(pg.evaluate("() => window.__riFault || 0"))
        if res.fault_executions == 0:
            res.add(-1, "fault", "fault executed", "ERROR",
                    f"fault {fault.id!r} applied but the mutated code never ran: the fault proves nothing",
                    assertion=False)
    probed = {(n, tuple(t)) for n, _, t in probes}
    for typ, nm in pg.evaluate("t => window.__riH.inventory(t)", INVENTORY_TYPES):
        if not any(nm == n and typ in t for n, t in probed):
            res.unchecked_models.append(f"{typ} {nm}")
    for e in res.console_errors:                                   # console-error policy
        allowed = next((why for pat, why in CONSOLE_ALLOWED if re.search(pat, e)), None)
        if allowed:
            res.console_allowed.append(f"{e[:120]} — allowed: {allowed}")
        else:
            res.add(-1, "page", "console error", "ERROR", f"not on the allowed list: {e}", assertion=False)
    for e in res.page_errors:
        res.add(-1, "page", "page errors", "ERROR", e, assertion=False)
    for u in res.requests_blocked:
        res.add(-1, "page", "external request", "ERROR", f"blocked, not a pinned local script: {u}", assertion=False)


def _feature_of(scenario: Scenario, check) -> str:
    if isinstance(check, Audit):
        return "ENGINE.completion"
    if isinstance(check, JsValue):
        return "ENGINE.harness"
    if isinstance(check, Responds):
        return "HIST.profile" if check.model in scenario.page.profiles else "HIST.histogram_nd"
    return scenario.labels.features[0] if scenario.labels.features else ""


def _transport_check(pg, scenario: Scenario, df: pd.DataFrame, res: ScenarioResult) -> None:
    cols = page_mod.input_columns(scenario)
    expected = transport.expected_decoded(df, cols, scenario.transport.array_compression)
    decoded = pg.evaluate("c => window.__riH.decoded(c)", cols)
    if decoded.get("__error"):
        raise StepNotComplete(decoded["__error"])
    for col, ok, msg in transport.check(decoded, expected):
        res.add(0, "load", f"transport {col}", "PASS" if ok else "FAIL", msg, assertion=False)


def _auto(pg, scenario: Scenario, df: pd.DataFrame, res: ScenarioResult, step: int, name: str,
          state: Dict[str, list]) -> None:
    """Automatic oracle for every declared histogram (and the load check). Each concrete check is handled on its own:
    an error in one marks only that check, with its own feature."""
    try:
        mask, states, masks = _widget_mask(pg, scenario, df)
    except Exception as exc:
        for cname, feat in _concrete_checks(scenario, Auto(), step):
            res.add(step, name, cname, "ERROR", f"widget state not readable: {type(exc).__name__}: {str(exc)[:200]}",
                    feature=feat)
        return
    # selections of known defects, only where an expectation asks for one (computed from data and widget values)
    rules = {e.signature.rule for e in scenario.expect if isinstance(e.signature, EqualsSelection) and step in e.steps}
    rule_masks = {r: SELECTION_RULES[r](masks, state.get("load_masks", masks)) for r in rules}

    # 4.2c-2d A-1: with two or more filters the page's selection is their intersection (LazyIntersectionFilter),
    # so the check counts for FILTER.intersection; with one filter for FILTER.range
    sel_feature = _selection_feature(scenario)
    if _load_check_applies(scenario, step) and all(s["value"] == [s["start"], s["end"]] for s in states.values()):
        # the page's own selected-row count, read from the filter every histogram consumes, against the number of
        # rows whose widget columns are finite; only at RootInteractive's default limits (RB-1.4)
        try:
            finite = np.ones(len(df), dtype=bool)
            for col in states:
                finite &= np.isfinite(df[col].to_numpy())
            first = page_mod.histogram_models(scenario.page)[0][0]
            sel = pg.evaluate("([m, t, n]) => window.__riH.selected(m, t, n)", [first, HIST_TYPES, len(df)])
            if sel.get("error"):
                res.add(step, name, LOAD_CHECK, "ERROR", sel["error"], feature=sel_feature)
            else:
                n_fin = int(finite.sum())
                ctx = {f"selection:{r}": [int(m.sum())] for r, m in rule_masks.items()}
                outcome, why = classify(LOAD_CHECK, step, sel["count"] == n_fin, [sel["count"]], [n_fin], ctx,
                                        scenario.expect, (0.0, 0.0))
                res.add(step, name, LOAD_CHECK, outcome,
                        f"page selects {sel['count']} rows ({sel['filter']}); rows with finite widget columns {n_fin} "
                        f"of {len(df)}" + (f"; {why}" if why else ""), feature=sel_feature)
        except Exception as exc:
            res.add(step, name, LOAD_CHECK, "ERROR", f"{type(exc).__name__}: {str(exc)[:300]}", feature=sel_feature)

    for model, h, wcol in page_mod.histogram_models(scenario.page):
        check = f"{model}.bin_count"
        feature = _hist_feature(h)
        try:
            cols = [_variable(df, scenario, v) for v in h.variables]
            weights = None if wcol is None else df[wcol].to_numpy()
            expect = histograms.counts(cols, h.nbins, h.range, mask=mask, weights=weights)
            got = pg.evaluate("([n, c, t]) => window.__riH.col(n, c, t)", [model, "bin_count", HIST_TYPES])
            if got is None:
                why = pg.evaluate("([n, t]) => window.__riH.why(n, t)", [model, HIST_TYPES])
                res.add(step, name, check, "ERROR", f"{why} (or no column bin_count)", feature=feature)
                continue
            tol = (0.0, 0.0) if weights is None else (1e-9, 1e-12)            # the check's own tolerance
            cmp = compare(got, expect, kind="count") if weights is None else \
                compare(got, expect, kind="float", abs_tol=tol[0], rel_tol=tol[1])
            ctx = {}
            if any(e.check == check and step in e.steps for e in scenario.expect):    # context for signatures
                ctx["unselected"] = histograms.counts(cols, h.nbins, h.range, mask=None, weights=weights)
                if len(cols) == 1:
                    nan_rows = mask & np.isnan(cols[0])
                    ctx["nan_rows"] = float(nan_rows.sum() if weights is None else weights[nan_rows].sum())
                for r, m in rule_masks.items():
                    ctx[f"selection:{r}"] = histograms.counts(cols, h.nbins, h.range, mask=m, weights=weights)
            outcome, why = classify(check, step, cmp.ok, got, expect, ctx, scenario.expect, tol)
            res.add(step, name, check, outcome,
                    f"{cmp.message}; selected rows {int(mask.sum())}" +
                    (f"; ratio page/oracle {cmp.ratio_range}" if cmp.ratio_range else "") + (f"; {why}" if why else ""),
                    feature=feature)
        except Exception as exc:
            res.add(step, name, check, "ERROR", f"{type(exc).__name__}: {str(exc)[:300]}", feature=feature)


def _reproduce(test_ref, scenario_id) -> str:
    """An executable reproduction line: the exact pytest node id of the running test (4.2c-2b review, GPT23: `-k` with
    a scenario id selected nothing when several scenarios run inside one test, as in the B1 sweep)."""
    import os
    current = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
    if current:
        return f"pytest '{current}'   # scenario {scenario_id}"
    return f"pytest {test_ref}   # scenario {scenario_id}" if test_ref else scenario_id


def _selection_feature(scenario) -> str:
    """The capability of the page's selection check (4.2c-2d A-1, follow-up finding 4): with two or more filtered columns
    the selection is their intersection (LazyIntersectionFilter) — FILTER.intersection; with one, FILTER.range. Used
    for executed AND not-executed checks, so a skipped or failed check is attributed the same way."""
    return "FILTER.intersection" if len({w.column for w in scenario.page.widgets}) >= 2 else "FILTER.range"


def _kind(scenario, fault) -> str:
    """The report's kind from explicit facts (4.2c-2d A-7): a run with a deliberate fault is "fault"; a scenario of the
    family "harness" (a control of the engine itself) is "harness"; every other run is "healthy". The id namespace
    must agree with the family, so the two can never diverge silently."""
    harness = scenario.labels.family == "harness"
    if scenario.id.startswith("harness/") != harness:
        raise ValueError(f"scenario {scenario.id}: the harness/ id namespace and the family {scenario.labels.family!r} disagree")
    return "fault" if fault else ("harness" if harness else "healthy")

