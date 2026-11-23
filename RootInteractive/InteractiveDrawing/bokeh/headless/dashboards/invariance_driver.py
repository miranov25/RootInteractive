"""PHASE_5 Group 7: the driver of the invariance dashboards in Chromium, and the exact comparisons (Claude21.RI,
2026-11-22).

In plain words: one page is opened once and shared by the twenty tests of its dashboard. A test sets a state the way a
user's action ends: it writes the new value into the control's own model (a slider's value, a list's chosen items, a
select, a spinner, a text field, a query's on/off switch, the active tab); a few tests use the real mouse and keys
instead. After every state the driver waits until the page has drawn twice and its results stopped changing, then
reads the page's own arrays, exactly (as float64 bytes):
- the selection: one 0/1 value per row, from the page's intersection filter;
- each histogram's counts and, per axis, its bin edges;
- each profile's columns (entries, mean, std, quantiles);
- the alias columns of the table (for example diffY, the live difference, ratio or log-ratio);
- what a plot draws: the columns of its drawn table.
Each test ends by setting every control back to its value at load (`restore`): the page must then give exactly the
arrays it gave at load (compared through a fingerprint of their bytes). A test that leaves the page different fails,
and the page is opened again, so the next test starts clean.

Every state's timing is recorded: the time the page took (set + draw + settle), the time to read, the time to compare.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

DRIVER_JS = r"""
window.__g7 = {
  all() { return window.__riH.models() },
  one(pred, what) {
    const c = this.all().filter(pred);
    if (c.length !== 1) throw new Error(what + ': ' + c.length + ' models found, exactly one expected');
    return c[0] },
  switchOf(k) {
    return this.one(m => m.type === 'Toggle' && m.label === 'disable' &&
      ((m.js_property_callbacks['change:active'] || [])[0]?.args?.filter?.name === 'customSelect' + k),
      'the on/off switch of text query ' + k) },
  widget(kind, key) {
    if (kind === 'range') return this.one(m => m.type === 'RangeSlider' && m.title === key, 'range slider ' + key);
    if (kind === 'list') return this.one(m => m.type === 'MultiSelect' && m.title === key, 'list ' + key);
    if (kind === 'select') return this.one(m => m.type === 'Select' && m.title === key, 'select ' + key);
    if (kind === 'spinner') return this.one(m => m.type === 'Spinner' && m.title === key, 'spinner ' + key);
    if (kind === 'text') return this.one(m => m.type === 'TextAreaInput' && m.name === key, 'text field ' + key);
    if (kind === 'query') return this.switchOf(key).js_property_callbacks['change:active'][0].args.widget;
    if (kind === 'off') return this.switchOf(key);
    if (kind === 'tabs') return this.one(m => m.type === 'LazyTabs' && m.tabs.length > 0 && m.tabs[0].title === key,
                                         'the tabs whose first tab is ' + key);
    // controls found by their name (the ToyMC page, D3): a range slider, an on/off switch, a set of tabs
    if (kind === 'slider') return this.one(m => m.type === 'RangeSlider' && m.name === key, 'range slider ' + key);
    if (kind === 'switch') return this.one(m => m.type === 'Toggle' && m.name === key, 'switch ' + key);
    if (kind === 'seltabs') return this.one(m => m.type === 'Tabs' && m.name === key, 'the tabs ' + key);
    throw new Error('unknown kind of control: ' + kind) },
  prop(kind) { return ['off', 'tabs', 'switch', 'seltabs'].includes(kind) ? 'active' : 'value' },
  bits(v, n, enc) {                         // the first n values of a filter's vector as 0/1, and the selected ones past n
    const u = new Uint8Array(n);
    let extra = 0;
    for (let i = 0; i < n; i++) u[i] = v[i] ? 1 : 0;
    for (let i = n; i < v.length; i++) if (v[i]) extra++;
    return {rows: this.pack(u, enc), n: n, padding_selected: extra} },
  values(list) {
    return list.map(([kind, key]) => { const v = this.widget(kind, key)[this.prop(kind)];
                                        return Array.isArray(v) ? Array.from(v) : v }) },
  async frames() { await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => r()))) },
  async wait(probes) {
    const t0 = performance.now();
    await this.frames();
    const s = await window.__riH.settle(probes, 400);
    await this.frames();
    return {ms: performance.now() - t0, settled: s.settled, rounds: s.rounds, error: s.error || null} },
  async set(changes, probes) {
    const t0 = performance.now();
    for (const [kind, key, value] of changes) this.widget(kind, key)[this.prop(kind)] = value;
    const s = await this.wait(probes);
    s.ms = performance.now() - t0;
    return s },
  // bytes -> base64 text (enc 'b64') or a 64-bit FNV-1a fingerprint of the bytes (enc 'fp')
  pack(u8, enc) {
    if (enc === 'fp') {
      let a = 0x811c9dc5, b = 0x050c5d1f;
      for (let i = 0; i < u8.length; i++) {
        a = Math.imul(a ^ u8[i], 0x01000193) >>> 0; b = Math.imul(b ^ u8[u8.length - 1 - i], 0x01000193) >>> 0 }
      return a.toString(16).padStart(8, '0') + b.toString(16).padStart(8, '0') + ':' + u8.length }
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s) },
  f64(v, enc) {
    const a = new Float64Array(v.length);
    for (let i = 0; i < v.length; i++) a[i] = v[i] == null ? NaN : v[i];
    return this.pack(new Uint8Array(a.buffer), enc) },
  read(spec, enc) {
    const out = {};
    if (spec.mask) {
      const n = this.one(m => m.name === 'cdsOrig', 'the table cdsOrig').get_length();
      const f = this.one(m => m.type === 'LazyIntersectionFilter', 'the selection (intersection filter)');
      // the filter's vector is longer than the table (whole words of 32 rows; its length may change): only the
      // table's rows are compared, and no row past the end may be selected
      out.mask = this.bits(f.v_compute(), n, enc) }
    for (const [target, table] of spec.masks || []) {      // a named mask per table (D3: selection_mask_<target>)
      const f = this.one(m => m.type === 'LazyIntersectionFilter' && m.name === 'selection_mask_' + target, 'the mask of ' + target);
      const n = this.one(m => m.type === 'CDSAlias' && m.name === table, 'the table ' + table).get_length();
      out['mask:' + target] = this.bits(f.v_compute(), n, enc) }
    for (const [table, cols] of spec.columns || []) {      // columns of a named table, as its figures read them
      const t = this.one(m => m.type === 'CDSAlias' && m.name === table, 'the table ' + table), o = {};
      for (const c of cols) { const v = t.get_column(c); o[c] = v == null ? null : this.f64(v, enc) }
      out['col:' + table] = o }
    for (const name of spec.hist || []) {
      const h = this.one(m => m.type === 'HistoNdCDS' && m.name === name, 'histogram ' + name);
      const nb = Array.from(h.nbins), edges = [];
      const count = h.get_column('bin_count');
      let stride = 1;
      for (let a = 0; a < nb.length; a++) {
        const bot = h.get_column('bin_bottom_' + a), top = h.get_column('bin_top_' + a), e = new Float64Array(nb[a] + 1);
        for (let k = 0; k < nb[a]; k++) e[k] = bot[k * stride];
        e[nb[a]] = top[(nb[a] - 1) * stride];
        edges.push(this.pack(new Uint8Array(e.buffer), enc));
        stride *= nb[a] }
      out['hist:' + name] = {nbins: nb, vars: Array.from(h.sample_variables), count: this.f64(count, enc), edges: edges} }
    for (const [name, cols] of spec.hist1 || []) {        // a 1D histogram (HistogramCDS), with extra counts `cols`
      const h = this.one(m => m.type === 'HistogramCDS' && m.name === name, '1D histogram ' + name);
      const bot = h.get_column('bin_bottom'), top = h.get_column('bin_top'), nb = bot.length, e = new Float64Array(nb + 1);
      for (let k = 0; k < nb; k++) e[k] = bot[k];
      e[nb] = top[nb - 1];
      const o = {nbins: [nb], vars: [h.sample], count: this.f64(h.get_column('bin_count'), enc), edges: [this.pack(new Uint8Array(e.buffer), enc)]};
      for (const c of cols) { const v = h.get_column(c); o[c] = v == null ? null : this.f64(v, enc) }
      out['hist:' + name] = o }
    for (const name of spec.stack || []) {                  // which series a stacked table holds, and its length
      const s = this.one(m => m.type === 'CDSStack' && m.name === name, 'stacked table ' + name);
      out['stack:' + name] = {active: Array.from(s.activeSources), length: s.get_length()} }
    for (const [name, cols] of spec.prof || []) {
      const p = this.one(m => m.type === 'HistoNdProfile' && m.name === name, 'profile ' + name), o = {};
      for (const c of cols) { const v = p.get_column(c); o[c] = v == null ? null : this.f64(v, enc) }
      out['prof:' + name] = o }
    if ((spec.alias || []).length) {
      const t = this.one(m => m.name === 'cdsFull' && m.type === 'CDSAlias', 'the alias table cdsFull');
      for (const c of spec.alias) { const v = t.get_column(c); out['alias:' + c] = v == null ? null : this.f64(v, enc) } }
    for (const [name, cols] of spec.drawn || []) {
      const d = this.one(m => m.type === 'DownsamplerCDS' && m.name === name, 'drawn table ' + name), o = {};
      for (const c of cols) { const v = d.get_column(c); o[c] = v == null ? null : this.f64(v, enc) }   // as its plots read
      o.__watched = d.watched; o.__needs_update = d._needs_update; o.__trivial = d._is_trivial; o.__nPoints = d.nPoints;
      // the rows of the source it draws (with a filter, or fewer points than rows): read after its columns
      o.__rows = d._is_trivial ? null : this.f64(d._downsampled_indices, enc);
      out['drawn:' + name] = o }
    return out },
  inventory() {
    const hist = this.all().filter(m => m.type === 'HistoNdCDS').map(m => m.name).sort();
    const prof = this.all().filter(m => m.type === 'HistoNdProfile').map(m => [m.name, Array.from(m.quantiles || []).length]).sort();
    const hist1 = this.all().filter(m => m.type === 'HistogramCDS').map(m => [m.name, Object.keys(m.histograms || {}).sort()]).sort();
    return {hist, prof, hist1} },
  view(kind, key) {
    const m = this.widget(kind, key);
    const v = [...Bokeh.index.query(v => v.model === m)][0];
    if (!v) throw new Error('no view of ' + kind + ' ' + key + ' (is its tab shown?)');
    return v },
};
"""


def _f64(b: Optional[str]) -> Optional[np.ndarray]:
    return None if b is None else np.frombuffer(base64.b64decode(b), dtype="<f8").copy()


def decode(raw: dict) -> dict:
    """The arrays of a `read` in numpy: mask (bool per row), mask:<target> (the same for a named mask; its padding
    in mask:<target>:padding), col:<table> {column: array}, hist:<name> {count, edges[], nbins, vars, and the extra
    count columns of a 1D histogram}, prof:<name> {column: array}, alias:<column> array, drawn:<name> {column: array,
    and __rows: the rows of its source it draws, None if all}, stack:<name> {active, length}."""
    out = {}
    for k, v in raw.items():
        if k == "mask" or k.startswith("mask:"):
            out[k] = np.frombuffer(base64.b64decode(v["rows"]), dtype=np.uint8).astype(bool)
            out["mask_padding_selected" if k == "mask" else k + ":padding"] = v["padding_selected"]
        elif k.startswith("col:"):
            out[k] = {c: _f64(x) for c, x in v.items()}
        elif k.startswith("hist:"):
            out[k] = {"count": _f64(v["count"]), "edges": [_f64(e) for e in v["edges"]], "nbins": v["nbins"],
                      "vars": v["vars"]}
            out[k].update({c: _f64(x) for c, x in v.items() if c not in ("count", "edges", "nbins", "vars")})
        elif k.startswith("stack:"):
            out[k] = v
        elif k.startswith("prof:") or k.startswith("drawn:"):
            out[k] = {c: (_f64(x) if isinstance(x, str) or (x is None and c != "__rows") else x) for c, x in v.items()}
        elif k.startswith("alias:"):
            out[k] = _f64(v)
    return out


class KnownDefect(AssertionError):
    """Raised only when the page shows the exact signature of a registered defect: the test is then a strict expected
    failure (pytest.mark.xfail(strict=True, raises=KnownDefect)); any other wrong result is an ordinary failure, and a
    page without the defect makes the strict mark fail, so the mark is removed with the fix."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


# ------------------------------------------------------------------ exact comparisons (plain messages)
def same_rows(page: np.ndarray, expected: np.ndarray, where: str) -> None:
    """Row by row: the same selected rows, not only the same number."""
    page, expected = np.asarray(page, bool), np.asarray(expected, bool)
    assert page.shape == expected.shape, f"{where}: the page has {page.size} rows, the table {expected.size}"
    bad = np.flatnonzero(page != expected)
    assert bad.size == 0, (f"{where}: the page selects {int(page.sum())} rows, pandas {int(expected.sum())}; "
                           f"{bad.size} rows differ, the first: {bad[:10].tolist()}")


def same_values(page: np.ndarray, expected: np.ndarray, where: str, atol: float = 0.0) -> None:
    """Element by element; NaN equals NaN; atol = 0 means bit for bit equal values (0.0 == -0.0)."""
    page, expected = np.asarray(page, np.float64), np.asarray(expected, np.float64)
    assert page.shape == expected.shape, f"{where}: {page.size} values on the page, {expected.size} expected"
    both_nan = np.isnan(page) & np.isnan(expected)
    if atol == 0:
        bad = np.flatnonzero(~both_nan & (page != expected))
    else:
        bad = np.flatnonzero(~both_nan & ~(np.abs(page - expected) <= atol))
    assert bad.size == 0, (f"{where}: {bad.size} of {page.size} values differ" + (f" by more than {atol:g}" if atol else "")
                           + "; the first (index, page, expected): "
                           + str([(int(i), float(page[i]), float(expected[i])) for i in bad[:6]]))


def every_value(page: np.ndarray, value: float, where: str) -> None:
    page = np.asarray(page, np.float64)
    bad = np.flatnonzero(page != value)
    assert bad.size == 0, (f"{where}: {bad.size} of {page.size} values are not exactly {value:g}; the first "
                           f"(index, value): {[(int(i), float(page[i])) for i in bad[:6]]}")


def same_histogram(page: dict, expected: dict, where: str) -> None:
    """The same edges on every axis, bit for bit, and the same count in every bin."""
    assert len(page["edges"]) == len(expected["edges"]), f"{where}: {len(page['edges'])} axes, {len(expected['edges'])} expected"
    for a, (pe, ee) in enumerate(zip(page["edges"], expected["edges"])):
        same_values(pe, ee, f"{where}, the bin edges of axis {a} (range expected {ee[0]!r} .. {ee[-1]!r})")
    same_values(page["count"], expected["count"], f"{where}, the counts (first axis fastest)")


def scaled_close(page, expected, where: str, rel: float) -> None:
    """For means, standard deviations and quantiles computed in another order: |page - expected| <= rel * max(1, |e|)."""
    page, expected = np.asarray(page, np.float64), np.asarray(expected, np.float64)
    tol = rel * np.maximum(1.0, np.abs(np.nan_to_num(expected)))
    both_nan = np.isnan(page) & np.isnan(expected)
    bad = np.flatnonzero(~both_nan & ~(np.abs(page - expected) <= tol))
    assert bad.size == 0, (f"{where}: {bad.size} of {page.size} values differ by more than {rel:g} (relative); the first "
                           f"(index, page, expected): {[(int(i), float(page[i]), float(expected[i])) for i in bad[:6]]}")


# ------------------------------------------------------------------ the page
class Page:
    """One dashboard page in its own Chromium context, shared by the tests of a module."""

    def __init__(self, session, html: Path, kinds: Dict[str, str], controls: Sequence[tuple], fingerprint_spec: dict,
                 probes: Sequence, out_dir: Path):
        self.session, self.html, self.kinds, self.controls = session, Path(html), dict(kinds), list(controls)
        self.fingerprint_spec, self.probes, self.out_dir = fingerprint_spec, [list(p) for p in probes], Path(out_dir)
        self.timings: List[dict] = []
        self.notes: List[str] = []              # problems after a test that failed: reported again by the last test
        self.expected_notes: List[str] = []     # the same after a test that met its registered defect
        self.test = "load"
        self._open()
        self.at_load = self.values()
        self.fingerprint_at_load = self.fingerprint()

    # -- opening
    def _open(self):
        from ..engine import vendor
        from ..engine.runner import BROWSER_LOAD_CONDITION, HELPER_JS
        routes = {g.replace("**/", "/"): p for g, p in vendor.routes().items()}
        self.errors: List[str] = []
        self.ctx = self.session.browser.new_context(viewport={"width": 1500, "height": 1000})
        self.pg = self.ctx.new_page()
        self.pg.on("pageerror", lambda e: self.errors.append("pageerror: " + str(e)[:400]))
        self.pg.on("console", lambda m: self.errors.append("console.error: " + m.text[:400]) if m.type == "error" else None)

        def route(r):
            url = r.request.url
            if url.startswith("file:"):
                return r.continue_()
            for suffix, path in routes.items():
                if url.split("?")[0].endswith(suffix):
                    return r.fulfill(path=str(path), content_type="text/javascript")
            self.errors.append("blocked request: " + url)
            return r.abort()
        self.pg.route("**/*", route)
        self.pg.add_init_script(HELPER_JS)
        self.pg.add_init_script(DRIVER_JS)
        t0 = time.perf_counter()
        self.pg.goto(self.html.resolve().as_uri())
        self.pg.wait_for_function(BROWSER_LOAD_CONDITION, timeout=300_000)
        w = self.pg.evaluate("p => window.__g7.wait(p)", self.probes)
        self.load_seconds = round(time.perf_counter() - t0, 3)
        assert w["settled"], f"the page did not settle after loading: {w}"

    def reopen(self):
        self.ctx.close()
        self._open()
        again = self.fingerprint()
        if again != self.fingerprint_at_load:
            self.notes.append("the page opened again does not give the arrays of the first load: "
                              + ", ".join(k for k in again if again[k] != self.fingerprint_at_load.get(k)))

    def close(self):
        self.ctx.close()

    # -- controls
    def values(self) -> Dict[tuple, object]:
        got = self.pg.evaluate("l => window.__g7.values(l)", [list(c) for c in self.controls])
        return {tuple(c): v for c, v in zip(self.controls, got)}

    def changes(self, **kw) -> list:
        """Control changes from keywords: a range (lo, hi), a list [...], a select or spinner value, a text, a query
        ("queryK": text, or None to switch it off), a tab ("tabs:<first tab title>": index)."""
        out = []
        for key, value in kw.items():
            m = re.fullmatch(r"query(\d)", key)
            if m:
                k = int(m.group(1))
                if value is None:
                    out.append(["off", k, True])
                else:
                    out += [["query", k, value], ["off", k, False]]
                continue
            kind = self.kinds[key]
            out.append([kind, key, list(value) if isinstance(value, (tuple, list)) else value])
        return out

    def set(self, label: str, **kw) -> dict:
        return self.apply(label, self.changes(**kw))

    def apply(self, label: str, changes: list) -> dict:
        t0 = time.perf_counter()
        r = self.pg.evaluate("([c, p]) => window.__g7.set(c, p)", [changes, self.probes])
        wall = time.perf_counter() - t0
        assert r["settled"], f"{label}: the page did not settle: {r}"
        self.timings.append({"test": self.test, "state": label, "page_ms": round(r["ms"], 1),
                             "wall_s": round(wall, 4)})
        return r

    def waited(self, label: str) -> dict:
        """After a real mouse or key action: wait as after a set, and record the time."""
        t0 = time.perf_counter()
        r = self.pg.evaluate("p => window.__g7.wait(p)", self.probes)
        assert r["settled"], f"{label}: the page did not settle: {r}"
        self.timings.append({"test": self.test, "state": label, "page_ms": round(r["ms"], 1),
                             "wall_s": round(time.perf_counter() - t0, 4), "real_input": True})
        return r

    # -- reading
    def read(self, mask: bool = True, hist: Sequence[str] = (), prof: Sequence = (), alias: Sequence[str] = (),
             drawn: Sequence = (), hist1: Sequence = (), stack: Sequence[str] = (), masks: Sequence = (),
             columns: Sequence = ()) -> dict:
        """The arrays of the page (see `decode`); hist1: 1D histograms [(name, extra count columns)]; stack: stacked
        tables, their chosen series and length; masks: named masks [(target, its table)]; columns: [(table,
        columns)]."""
        t0 = time.perf_counter()
        spec = {"mask": mask, "hist": list(hist), "prof": [[n, list(c)] for n, c in prof], "alias": list(alias),
                "drawn": [[n, list(c)] for n, c in drawn], "hist1": [[n, list(c)] for n, c in hist1],
                "stack": list(stack), "masks": [list(m) for m in masks], "columns": [[n, list(c)] for n, c in columns]}
        out = decode(self.pg.evaluate("([s, e]) => window.__g7.read(s, e)", [spec, "b64"]))
        if self.timings:
            self.timings[-1]["read_s"] = round(self.timings[-1].get("read_s", 0) + time.perf_counter() - t0, 4)
        return out

    def fingerprint(self) -> Dict[str, object]:
        return self.pg.evaluate("([s, e]) => window.__g7.read(s, e)", [self.fingerprint_spec, "fp"])

    # -- the end of a test
    def restore(self) -> List[str]:
        """Every control back to its value at load; returns the arrays that then differ from those at load."""
        now = self.values()
        changes = []
        for (kind, key), v in self.at_load.items():
            if now[(kind, key)] != v and kind != "off":
                changes.append([kind, key, v])
        for (kind, key), v in self.at_load.items():                 # the query switches last
            if kind == "off" and now[(kind, key)] != v:
                changes.append([kind, key, v])
        self.last_restore = [[kind, key, str(v)[:60]] for kind, key, v in changes]
        if changes:
            self.apply("restore", changes)
        fp = self.fingerprint()
        return [k for k in self.fingerprint_at_load if fp.get(k) != self.fingerprint_at_load[k]]

    @contextmanager
    def case(self, name: str, allowed_errors: Sequence[str] = ()):
        """One test on the shared page: at its end every control goes back to its load value and the page must give
        exactly its arrays at load; new page or console errors fail the test unless `allowed_errors` (regular
        expressions) names them. A page left different is opened again for the next test."""
        self.test = name
        n_errors = len(self.errors)
        t0 = time.perf_counter()
        failed = None
        try:
            yield self
        except BaseException as exc:                                # noqa: B902 - re-raised below
            failed = exc
        problems = []
        try:
            dirty = self.restore()
        except Exception as exc:                                    # the restore itself failed
            dirty = [f"the restore failed: {exc}"]
        if dirty:
            problems.append("the page did not return to its state at load; these arrays differ: " + ", ".join(dirty)
                            + f" (the controls set back: {getattr(self, 'last_restore', None)})")
        if dirty or os.environ.get("RI_INVARIANCE_FRESH_PAGE") == "1":
            self.reopen()                       # RI_INVARIANCE_FRESH_PAGE=1: every test on a page opened again
        new = [e for e in self.errors[n_errors:] if not any(re.search(p, e) for p in allowed_errors)]
        if new:
            problems.append(f"{len(new)} new page or console errors: {new[:3]}")
        self.timings.append({"test": name, "state": "(whole test)", "wall_s": round(time.perf_counter() - t0, 3)})
        self.test = "between tests"
        if failed is not None:
            if problems:
                (self.expected_notes if isinstance(failed, KnownDefect) else self.notes).append(
                    f"{name}: also: " + "; ".join(problems))
            raise failed
        assert not problems, f"{name}: " + "; ".join(problems)

    # -- real mouse and keys (shadow-aware: the element is taken from the widget's own view)
    def element(self, kind: str, key, part: str):
        js = {"input": "v => v.input_el", "lower": "v => v.shadow_el.querySelector('.noUi-handle-lower')",
              "upper": "v => v.shadow_el.querySelector('.noUi-handle-upper')",
              "button": "v => v.shadow_el.querySelector('button') ?? v.el.querySelector('button')"}[part]
        return self.pg.evaluate_handle(f"([k, x]) => ({js})(window.__g7.view(k, x))", [kind, key]).as_element()

    def click_tab(self, title: str, which: Optional[str] = None):
        """A real click on the tab `title`; `which` = "first" or "last" when two sets of tabs hold that title (the
        ToyMC page: the figure tabs above, the selection tabs below)."""
        tabs = self.pg.locator(".bk-tab").filter(has_text=re.compile(rf"^{re.escape(title)}$"))
        (tabs.first if which == "first" else tabs.last if which == "last" else tabs).click()

    def keys(self, kind: str, key, part: str, press: str, n: int):
        el = self.element(kind, key, part)
        el.focus()
        for _ in range(n):
            self.pg.keyboard.press(press)

    def type_into(self, kind: str, key, text: str, commit: str = "Enter"):
        """Click into the field, put the text in with one input event (Playwright's fill: typing key by key makes the
        page lay itself out again after every key, about 0.3 s each), then the real key `commit`."""
        el = self.element(kind, key, "input")
        el.click()
        el.fill(text)
        el.press(commit)

    # -- the record
    def write_timings(self, path: Path) -> dict:
        states = [t for t in self.timings if t["state"] not in ("(whole test)",)]
        page_ms = sorted(t["page_ms"] for t in states if "page_ms" in t)
        tests = [t for t in self.timings if t["state"] == "(whole test)"]
        summary = {
            "page": self.html.name, "load_s": self.load_seconds, "states": len(page_ms),
            "page_ms_median": page_ms[len(page_ms) // 2] if page_ms else None,
            "page_ms_p90": page_ms[int(len(page_ms) * 0.9)] if page_ms else None,
            "page_ms_max": page_ms[-1] if page_ms else None,
            "read_s_total": round(sum(t.get("read_s", 0) for t in states), 2),
            "tests_s_total": round(sum(t["wall_s"] for t in tests), 2),
            "per_test_s": {t["test"]: t["wall_s"] for t in tests}, "notes": self.notes,
            "notes_after_known_defects": self.expected_notes,
        }
        path.write_text(json.dumps({"summary": summary, "states": self.timings}, indent=1), encoding="utf-8")
        return summary
