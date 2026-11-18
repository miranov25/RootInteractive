"""PHASE_4_6 C1 -- test-only statistics safety net for RootInteractive.

Independent reference values are calculated from event rows using NumPy.  All
Node tests construct Bokeh documents through the user API (fromArray), load the
production models in the Node harness, and read numerical data from those models.

v0.2 candidate after GPT24 consolidated review: P1 F1/F2 test-only fixes.
Source inspected at RI 33ab605 (older than requested 175687e).  No RootInteractive product code or shared test files edited.
The Node cases REQUIRE Bokeh 3.3.2, jsdom 24.1.3 and a compiled model bundle;
they have NOT been executed by the contributor in this environment.

Open policy: Q-H1 (q=1), Q-H2 (quantile within its bin), ridge intercept,
selection effect on fit, stack ordering; do not certify these by characterization.

Integration (Claude20.RI, on 175687e + A4 b2e7b75; GPT25.RI and GPT26.RI approved v0.2 for test-only integration).
Run on the pinned toolchain: 20 of 31 cases passed as delivered. Changes, each measured:
  * test-side: the engine's console policy (engine/runner.py CONSOLE_ALLOWED) is applied, so Bokeh's known cosmetic
    "Bad column name on label" message on a weights-stacked figure no longer fails the four stack tests;
  * test-side: feature markers written as @pytest.mark.feature("...") - the capability matrix reads only that form;
  * test-side: the fit-data-change test runs uncompressed (its mutation needs a plain ColumnDataSource cdsOrig), and
    with one predictor too; the frequency-replication case compares two pages that both use the WEIGHTED path
    (repeated rows with weight 1), because the weighted and unweighted unbinned paths use different quantile
    definitions (no interpolation versus interpolation) - which is the intended one is the Architect's Q-H2;
  * four defects found, each with an exact signature, a strict expected-failure partner and a healthy control:
    ST-D1  a linear fit does not send its response or weight column to the page (bokehTools.py builds the
           ClientLinearFitter without registering varY/weights as used columns): reading the fitted alias throws
           "Cannot read properties of undefined (reading 'decodeProgram')" in CDSCompress.get_column.
           Fixed in PHASE_5 Group 4: the page builder registers every column a fit reads; the signature test is
           removed, the strict test (response and weight column) is an ordinary test, the signature a REGRESSION;
    ST-D2  with two or more predictors the fit is wrong: MathUtils.ts chol updates every off-diagonal element of the
           packed matrix twice (once in each inner loop); one predictor (2 x 2) is correct.
           Fixed in PHASE_5 Group 1: the signature test is removed, the strict tests are ordinary tests, and the exact
           signature (the fit equals the transcription of the old solver) is a REGRESSION check;
    ST-D3  an alias whose variables are chosen by a parameter fails the page build: bokehTools.py bokehDrawArray reads
           `context` in the function-form alias branch, where it is set only by an earlier expression alias
           (UnboundLocalError). Fixed in PHASE_5 Batch 3 (with A-D3): the signature test is removed, the exact signature
           is a REGRESSION check, and the strict test now waits for ST-D2, which the page shows once it builds;
    ST-D5  (PHASE_5 Group 1, new) the solver treats a diagonal element below the absolute number 1e-6 as zero, so a
           fit with small predictors is wrong without a message, also with one predictor. Fixed in PHASE_5 Group 1
           with ST-D2: a diagonal element must be positive, there is no threshold, and a system that cannot be solved
           gives NaN for every coefficient.
    ST-D4  the statistics table computes quantiles in one pass that assumes an ascending list; a list in another order
           (the table keeps the order the histograms request) gives wrong values under the right titles.
           Fixed in PHASE_5 Batch 2B: the signature test is removed, the strict test is an ordinary test, and the exact
           signature is a REGRESSION check.
  The fit tests show the fit's columns in a second figure (ST-D1 workaround), as A1 does for A-D4.
"""
from __future__ import annotations

import re

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NODE = pytest.mark.node_stats
UNIT = pytest.mark.headless_unit


# --------- Pure independent mathematical references (NO RI implementation calls)
def _bin_ref(values, edges, weights=None):
    a = np.asarray(values, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    return np.histogram(a, bins=np.asarray(edges, dtype=float), weights=w)[0].astype(float)


def _integral_ref(counts, edges, interval):
    """Uniform density in every bin, clipping the physical overlap per bin."""
    c, e = np.asarray(counts, float), np.asarray(edges, float)
    lo, hi = map(float, interval)
    if hi < lo:
        raise ValueError("reversed integral interval")
    overlap = np.maximum(0., np.minimum(e[1:], hi) - np.maximum(e[:-1], lo))
    return float(np.dot(c, overlap / (e[1:] - e[:-1])))


def _stats_ref(counts, edges, probs=(), intervals=()):
    c, e = np.asarray(counts, float), np.asarray(edges, float)
    assert c.shape == (len(e) - 1,)
    assert np.all(c >= 0) and np.all(np.diff(e) > 0)
    n = float(c.sum())
    out = {"entries": n, "isOK": n > 0}
    if n == 0:
        out.update(mean=np.nan, std=np.nan)
    else:
        centers = (e[:-1] + e[1:]) / 2
        mean = float(c @ centers / n)
        out.update(mean=mean, std=float(np.sqrt(c @ (centers - mean)**2 / n)))
    for q in probs:
        if not 0 <= q < 1 or n == 0:
            out[f"q:{q}"] = np.nan
        else:
            below = 0.
            value = np.nan
            for i, amount in enumerate(c):
                if below + amount > q * n:
                    value = float(e[i] + (q*n - below) / amount * (e[i+1] - e[i]))
                    break
                below += float(amount)
            out[f"q:{q}"] = value
    for i, rng in enumerate(intervals):
        integ = _integral_ref(c, e, rng)
        out[f"sum_{i}"] = integ
        out[f"sum_normed_{i}"] = integ/n if n else np.nan
    return out


def _profile_ref(df, xedges, yedges, intervals, weight=None):
    """Profiles are y statistics for each x bin, not raw y moments."""
    c = np.histogramdd(df[["x", "y"]].to_numpy(), bins=[xedges, yedges],
                       weights=None if weight is None else df[weight].to_numpy())[0]
    return [_stats_ref(row, yedges, intervals=intervals) for row in c]


def _fit_ref(df, predictors, response="target", weight=None, alpha=0., ridge_intercept=True):
    """Independent least-squares normal equation; ridge convention explicit.

    Current ClientLinearFitter.ts adds alpha to ALL diagonal elements, including
    the intercept.  That is a characterization, NOT an Architect-approved policy.
    """
    x = df[list(predictors)].to_numpy(dtype=float)
    a = np.column_stack((x, np.ones(len(df))))
    y = df[response].to_numpy(dtype=float)
    w = np.ones(len(df)) if weight is None else df[weight].to_numpy(dtype=float)
    assert np.all(w >= 0) and w.sum() > 0
    if alpha:
        reg = np.eye(a.shape[1])
        if not ridge_intercept:
            reg[-1, -1] = 0
        beta = np.linalg.solve(a.T @ (w[:, None]*a) + alpha*reg, a.T @ (w*y))
    else:
        beta = np.linalg.lstsq(np.sqrt(w)[:, None]*a, np.sqrt(w)*y, rcond=None)[0]
    return a @ beta, beta


def _js_number(x):
    """Format numeric labels as JavaScript Number.toString() for exact row names."""
    x = float(x)
    return str(int(x)) if x.is_integer() else str(x)


def _values_equal(got, want, *, atol=1e-11):
    np.testing.assert_allclose(np.asarray(got, float), np.asarray(want, float),
                               rtol=1e-11, atol=atol, equal_nan=True)


# --------- Known defects (integration, Claude20.RI): exact signatures; the ONLY failure a strict test accepts
class KnownDefect(AssertionError):
    """Raised only when production shows the exact signature of a registered defect."""


def known_defect(condition: bool, message: str) -> None:
    if condition:
        raise KnownDefect(message)


def regression(condition: bool, message: str) -> None:
    """PHASE_5 Batch 2A/2B fix commit: the exact signature of a fixed defect is a regression - an ordinary failure
    that names it."""
    if condition:
        raise AssertionError("REGRESSION, a fixed defect is back: " + message)


STRICT = dict(strict=True, raises=KnownDefect)


def _ri_solve_prediction(x, rhs):
    """ST-D2 PREDICTION - what production computes today, transcribed from MathUtils.ts `chol` and `solve` (packed
    lower triangle, row by row). Used only to recognise the defect, never as the oracle. In `chol` the first inner
    loop updates row j, columns i+1..j, and the second updates column j, rows j+1..n-1; an element (a, b) with
    a > b > i is updated in both, once with an already scaled multiplier."""
    x, y = list(map(float, x)), list(map(float, rhs))
    n = len(y)
    i_row = 0
    for i in range(n):
        pivot_diag = 0. if x[i + i_row] < 1e-6 else 1. / x[i + i_row]
        j_row = i_row + i + 1
        for j in range(i + 1, n):
            pivot_row = pivot_diag * x[i + j_row]
            k_row = i_row + i + 1
            for k in range(i + 1, j + 1):
                x[k + j_row] -= pivot_row * x[i + k_row]
                k_row += k + 1
            for k in range(j + 1, n):
                x[j + k_row] -= pivot_row * x[i + k_row]
                k_row += k + 1
            x[i + j_row] = pivot_row
            j_row += j + 1
        i_row += i + 1
    i_row = 0
    for i in range(n):
        for j in range(i):
            y[i] -= y[j] * x[i_row + j]
        i_row += i + 1
    i_diag = 0
    for i in range(n):
        y[i] /= x[i_diag]
        i_diag += i + 2
    for i in range(n - 1, -1, -1):
        j_row = 1 + ((i * (i + 5)) >> 1)
        for j in range(i + 1, n):
            y[i] -= y[j] * x[j_row]
            j_row += j + 1
    return y


def _ri_fit_prediction(df, predictors, weight=None, alpha=0.):
    """ST-D2 PREDICTION of the fitted values: the normal equations as ClientLinearFitter.fit packs them, alpha on
    every diagonal element, solved by the transcription above."""
    cols = [df[c].to_numpy(dtype=float) for c in predictors]
    w = np.ones(len(df)) if weight is None else df[weight].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)
    x = [float(np.sum(cols[i] * cols[j] * w)) for i in range(len(cols)) for j in range(i + 1)]
    x += [float(np.sum(c * w)) for c in cols] + [float(w.sum())]
    rhs = [float(np.sum(c * y * w)) for c in cols] + [float(np.sum(y * w))]
    if alpha > 0:
        i_row = 0
        for i in range(len(rhs)):
            x[i_row] += alpha
            i_row += i + 2
    beta = _ri_solve_prediction(x, rhs)
    return sum(b * c for b, c in zip(beta, cols)) + beta[-1], np.asarray(beta)


def _is_st_d2(vals, df, predictors, weight=None, alpha=0.) -> bool:
    ref = _fit_ref(df, predictors, weight=weight, alpha=alpha)[0]
    pred = _ri_fit_prediction(df, predictors, weight, alpha)[0]
    vals = np.asarray(vals, dtype=float)
    return (vals.shape == pred.shape and np.allclose(vals, pred, rtol=1e-9, atol=1e-9)
            and not np.allclose(pred, ref, rtol=1e-6, atol=1e-6))


def _check_fit(vals, df, predictors, weight=None, alpha=0.):
    """The oracle (numpy), after the ST-D2 signature (fixed in PHASE_5 Group 1: a regression)."""
    regression(_is_st_d2(vals, df, predictors, weight, alpha),
               f"ST-D2: the fit with predictors {list(predictors)} equals the double-update solver prediction")
    _values_equal(vals, _fit_ref(df, predictors, weight=weight, alpha=alpha)[0])


def _single_pass_quantiles(counts, edges, quantiles):
    """ST-D4 PREDICTION - HistoStatsCDS.ts's one-pass quantile loop applied to the list in the model's order (it
    assumes ascending order). Used only to recognise the defect."""
    c, e = np.asarray(counts, float), np.asarray(edges, float)
    entries = float(c.sum())
    out = [np.nan] * len(quantiles)
    if entries <= 0:
        return out
    cumulative = np.cumsum(c)
    i_q, low = 0, 0.
    while i_q < len(quantiles) and quantiles[i_q] < 0:
        i_q += 1
    for j in range(len(cumulative)):
        if i_q >= len(quantiles):
            break
        high = float(cumulative[j])
        while i_q < len(quantiles) and high > quantiles[i_q] * entries:
            m = (quantiles[i_q] * entries - low) / (high - low)
            out[i_q] = float(e[j] * (1 - m) + e[j + 1] * m)
            i_q += 1
        low = high
    return out


# --------- Fixed dyadic and hand-auditable input. Interior bin coordinates.
XEDGE = [0., 1., 2., 3., 4.]
YEDGE = [0., 1., 2., 3., 4.]
BOXES = [[-2., .5], [.5, 1.25], [1., 3.], [0., 4.], [3.75, 6.]]


def _df():
    return pd.DataFrame({
        "x": [.25, .25, .25, 1.25, 1.25, 1.25, 2.25, 2.25, 3.25, 3.25, 3.25, 3.25],
        "y": [.25, 1.25, 2.25, .25, 1.25, 3.25, .25, 2.25, .25, 1.25, 2.25, 3.25],
        "a": [-1.5, -.5, .5, -.25, .25, .75, -1., 1.5, -1.25, 1.25, -.75, 1.75],
        "wA": [1., 2., 3., 2., 1., 4., 2., 3., 1., 2., 3., 4.],
        "wB": [4., 3., 2., 1., 1., 2., 3., 4., 2., 1., 2., 1.],
        "z": [1., 0., 1., 2., 1., 0., 2., 3., 1., 2., 3., 4.],
    }).assign(target=lambda d: 1.5*d.x - .75*d.z + 2 + .25*d.y)


# --------- Actual public fromArray -> serialized model -> Node harness
# No imports of RootInteractive at module import/collection time: pure Python
# controls remain executable without pinned Bokeh or node_modules.
def _page(tmp_path, df, figures, sources=(), params=(), widgets=(), js=(), aliases=(),
          *, select_types=(), suffix="page", compression=True):
    from bokeh.document import Document
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import LOSSLESS
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    before = sa.show
    sa.show = lambda *a, **kw: None
    try:
        options = dict(sourceArray=list(sources), parameterArray=list(params),
                       jsFunctionArray=list(js), aliasArray=list(aliases),
                       layout=[list(range(len(figures)))],
                       widgetLayout=[list(range(len(widgets)))])
        if compression:
            options["arrayCompression"] = [(p, list(actions)) for p, actions in LOSSLESS]
        obj = sa.bokehDrawSA.fromArray(df, None, list(figures), list(widgets), **options)
    finally:
        sa.show = before
    doc = Document()
    if select_types:
        for kind in select_types:
            matching = [m for m in obj.pAll.references() if type(m).__name__ == kind]
            assert len(matching) == 1, f"expected one {kind} from the public page: {[m.name for m in matching]}"
            doc.add_root(matching[0])
    else:
        doc.add_root(obj.pAll)       # required for actual widget callbacks
    out = tmp_path / f"stat_{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


def _unexpected_console(r):
    """Integration: the engine's console policy - every console error fails unless engine/runner.py CONSOLE_ALLOWED
    lists it with its reason (the weights-stacked figure's cosmetic Bokeh legend message)."""
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import CONSOLE_ALLOWED
    return [m for m in (r.get("console_errors") or []) if not any(re.search(p, m) for p, _ in CONSOLE_ALLOWED)]


def _node_open(session, path):
    r = session.request("new_doc", path=str(path))
    assert r.get("ok") and not r.get("page_errors") and not _unexpected_console(r), r


def _eval_raw(session, code, arg=None):
    """The harness reply as it is (for a defect signature that expects an error)."""
    return session.request("eval", code=code, arg=arg)


def _eval(session, code, arg=None):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = _eval_raw(session, code, arg)
    assert r.get("ok") and not r.get("page_errors") and not _unexpected_console(r), r
    return decode(r["result"])


READ = r"""(a) => {
 const all = window.__riH.models();
 const matches = all.filter(m => m.type === a.kind && (a.name == null || m.name === a.name));
 if (matches.length !== 1) throw Error('Expected unique model: ' + a.kind + ': ' + matches.length);
 const m = matches[0], data = {};
 for (const key of a.keys) {
   const v = typeof m.get_column === 'function' ? m.get_column(key) : m.data?.[key];
   if (v == null) data[key] = null;
   else data[key] = Array.from(v);
 }
 const extra = {};
 if (a.kind === 'HistoStatsCDS') {
   extra.quantiles = Array.from(m.quantiles);
   extra.names = Array.from(m.names);
   extra.rowwise = m.rowwise;
   extra.sum_range = Array.from(m.sum_range, v => Array.from(v));
   const tables = all.filter(t => t.type === 'DataTable' && t.source === m);
   extra.tableColumns = tables.map(t => t.columns.map(c => [c.field,c.title]));
 }
 if (a.kind === 'ClientLinearFitter') {
   // Fit is lazy. Do not inspect parameters until an alias has been evaluated.
   extra.varX = Array.from(m.varX);
   extra.parameters = m.parameters == null ? null : Array.from(m.parameters);
 }
 return {data,extra};
}"""


def _read(session, kind, keys=(), name=None):
    x = _eval(session, READ, {"kind": kind, "keys": list(keys), "name": name})
    assert all(v is not None for v in x["data"].values()), f"Missing {kind} column: {x}"
    return x


# Async Bokeh 3.3.2 callbacks: await each bound CustomJS.execute Promise.
# This is *model-driven*, not a real mouse/keyboard browser event.
SET_WIDGET = r"""async (a) => {
 const all = window.__riH.models();
 const possible = all.filter(w => 'value' in w && w.js_property_callbacks &&
     (w.name === a.name || (w.name == null && w.title === a.name)));
 if (possible.length !== 1) throw Error('widget identity: ' + a.name + ': ' + possible.length);
 const w = possible[0], cbs = (w.js_property_callbacks || {})['change:value'] || [];
 const pending = [], methods = [];
 for (const cb of cbs) {
   const original = cb.execute;
   methods.push([cb,original]);
   cb.execute = function (...args) {
     const p = Promise.resolve(original.apply(this,args)); pending.push(p); return p;
   };
 }
 try {
   w.value = a.value;
   const done = await Promise.allSettled(pending);
   if (done.some(x => x.status === 'rejected')) throw Error('callback rejected: ' + done.filter(x=>x.status==='rejected').map(x=>String(x.reason)));
   await new Promise(resolve => setTimeout(resolve,0));
   return {callbacks: done.length, value: Array.isArray(w.value) ? Array.from(w.value) : w.value};
 } finally {
   for (const [cb,original] of methods) cb.execute=original;
 }
}"""


def _set(session, name, value):
    return _eval(session, SET_WIDGET, {"name": name, "value": value})


def _hist_page(tmp_path, df, *, rowwise=False, sources=None, boxes=None, suffix="table"):
    if sources is None:
        sources = [{"name": "ha", "variables": ["x"], "nbins": 4, "range": [0, 4],
                    "quantiles": [.25,.5,.75], "sum_range": BOXES if boxes is None else boxes}]
    plots = [[["bin_center"], ["bin_count"], {"source": s["name"]}] for s in sources]
    plots += [["tableHisto", {"rowwise": rowwise}]]
    return _page(tmp_path, df, plots, sources=sources, suffix=suffix)


def _profile_page(tmp_path, df, *, intervals=BOXES, unbinned=False, weighted=False, suffix="profile"):
    hist = {"name": "hp", "variables": ["x", "y"], "nbins": [4, 4],
            "range": [[0,4],[0,4]], "sum_range": intervals, "unbinned_projections": unbinned}
    if weighted:
        hist["weights"] = "wA"
    return _page(tmp_path, df, [[["bin_center_0"], ["mean"], {"source": "hp_1"}]],
                 sources=[hist], select_types=["HistoNdProfile"], suffix=suffix)


def _fit_page(tmp_path, df, predictors, *, weight=None, alpha=0., parameter=False, suffix="fit",
              send="all", expression_first=False, compression=True):
    """send: which of the fit's own columns a second figure shows - "all" (response, predictors, weight: the ST-D1
    workaround), "response" (not the weight), "none" (the page as a user writes it). expression_first: an expression
    alias declared before the fit alias (the ST-D3 control)."""
    fit = {"name": "C1fit", "type": "linearFit", "varX": "predictors" if parameter else list(predictors),
           "varY": "target", "alpha": alpha, "weights": weight}
    params = [{"name": "predictors", "value": list(predictors), "options": ["x", "z", "y"]}] if parameter else []
    widgets = [["multiSelect", ["predictors"]]] if parameter else []
    figures = [[["x"], ["fitted"]]]
    if send in ("all", "response"):
        figures.append([["target"], ["x", "z", "y"] + ([weight] if weight and send == "all" else [])])
    aliases = [("xdouble", "x*2")] if expression_first else []
    aliases.append({"name":"fitted", "variables": "predictors" if parameter else list(predictors), "transform":"C1fit"})
    return _page(tmp_path, df, figures, params=params, widgets=widgets, js=[fit], aliases=aliases, suffix=suffix,
                 compression=compression)


def _stack_page(tmp_path, df, *, multi=True, with_filter=False, compression=True, suffix="stack"):
    p = {"name":"weightChoice", "value": ["wA","wB"] if multi else "wA",
         "options":["wA","wB"]}
    hist = {"name":"hs", "variables":["x"], "nbins":4,"range":[0,4],"weights":"weightChoice"}
    widget = ["multiSelect" if multi else "select", ["weightChoice"]]
    widgets = [widget] + ([["range", ["a"]]] if with_filter else [])
    return _page(tmp_path, df, [[["bin_center"],["bin_count"],{"source":"hs"}]],
                 sources=[hist], params=[p], widgets=widgets, suffix=suffix,
                 compression=compression)


# ==========================================================================
# INDEPENDENT PYTHON ORACLES / NON-VACUOUS CONTROLS -- executable without Node
# ==========================================================================
@UNIT
def test_stat_ref_binned_statistics_and_falsification():
    c, edges = [2.,1.,3.,0.], XEDGE
    ref = _stats_ref(c, edges, probs=[.25,.5,.75], intervals=BOXES)
    assert ref["entries"] == 6 and ref["isOK"]
    assert ref["mean"] == 10/6
    assert ref["q:0.25"] == .75
    assert ref["q:0.5"] == 2.
    assert ref["sum_0"] == 1.
    assert ref["sum_2"] == 4.
    assert ref["sum_3"] == 6.
    with pytest.raises(AssertionError):
        _values_equal([ref["entries"]+1], [6.])
    with pytest.raises(AssertionError):
        _values_equal([ref["sum_2"]+.25], [4.])
    empty = _stats_ref([0,0,0,0], edges, [.25], BOXES)
    assert not empty["isOK"] and empty["entries"] == 0
    assert np.isnan(empty["mean"]) and np.isnan(empty["std"]) and np.isnan(empty["q:0.25"])
    assert np.isnan(empty["sum_normed_0"])


@UNIT
def test_stat_ref_interval_overlap_and_profile_controls():
    assert _integral_ref([2,1,3,0], XEDGE, [-5,5]) == 6
    assert _integral_ref([2,1,3,0], XEDGE, [-5,-1]) == 0
    assert _integral_ref([2,1,3,0], XEDGE, [6,8]) == 0
    assert _integral_ref([2,1,3,0], XEDGE, [.5,2.5]) == 3.5
    result = _profile_ref(_df().query('x < 3'), XEDGE, YEDGE, BOXES)
    assert [r["entries"] for r in result] == [3,3,2,0]
    assert not result[-1]["isOK"] and np.isnan(result[-1]["sum_normed_0"])
    with pytest.raises(AssertionError):
        _values_equal([r["entries"] for r in result], [3,3,3,0])


@UNIT
def test_stat_ref_weighted_and_ridge_falsification():
    d = _df()
    for weight in (None,"wA"):
        expected, beta = _fit_ref(d, ["x","z"], weight=weight)
        x = np.column_stack((d.x,d.z,np.ones(len(d))))
        if weight is None:
            oracle = np.linalg.lstsq(x,d.target,rcond=None)[0]
        else:
            root = np.sqrt(d.wA.to_numpy())
            oracle = np.linalg.lstsq(x*root[:,None],d.target*root,rcond=None)[0]
        _values_equal(beta, oracle)
        with pytest.raises(AssertionError):
            _values_equal(expected + .25, d.target.to_numpy()*0 + expected)
    for alpha in (.25, 2.):
        _, beta = _fit_ref(d, ["x","z"], weight="wA", alpha=alpha)
        a = np.column_stack((d[["x","z"]],np.ones(len(d))))
        rhs = a.T @ (d.wA.to_numpy()*d.target.to_numpy())
        lhs = a.T @ (d.wA.to_numpy()[:,None]*a) + alpha*np.eye(3)
        _values_equal(lhs@beta,rhs)


@UNIT
def test_stat_ref_stack_weight_options_catch_wrong_labels():
    d = _df()
    a = _bin_ref(d.x,XEDGE,d.wA)
    b = _bin_ref(d.x,XEDGE,d.wB)
    assert a.sum() == d.wA.sum() and b.sum() == d.wB.sum()
    assert not np.array_equal(a,b)
    with pytest.raises(AssertionError):
        _values_equal(a,b)


# ==========================================================================
# NODE: HistoStatsCDS and its visible DataTable columns
# ==========================================================================
@NODE
@pytest.mark.feature("STAT.histogram_table")
@pytest.mark.parametrize("rowwise", [False, True], ids=["columnwise","rowwise"])
def test_stat_table_binned_stats_quantiles_integrals(node_session,tmp_path,rowwise):
    df = _df()
    path = _hist_page(tmp_path,df,rowwise=rowwise,suffix=f"stats_{int(rowwise)}")
    _node_open(node_session,path)
    keys = (["description","ha"] if rowwise else
            ["name","mean","std","entries","isOK","quantile_0","quantile_1","quantile_2"]+
            [f"{p}_{i}" for i in range(len(BOXES)) for p in ("sum","sum_normed")])
    got = _read(node_session,"HistoStatsCDS",keys)
    ref = _stats_ref(_bin_ref(df.x,XEDGE),XEDGE,[.25,.5,.75],BOXES)
    assert got["extra"]["rowwise"] == rowwise
    if rowwise:
        fields = dict(zip(got["data"]["description"],got["data"]["ha"]))
        for k in ("mean","std","entries"):
            _values_equal([fields[k]],[ref[k]])
        for q in [.25,.5,.75]:
            _values_equal([fields[f"Quantile {q}"]],[ref[f"q:{q}"]])
        for i,(lo,hi) in enumerate(BOXES):
            _values_equal([fields[f"Σ({_js_number(lo)},{_js_number(hi)})"]],[ref[f"sum_{i}"]])
            _values_equal([fields[f"Σ_normed({_js_number(lo)},{_js_number(hi)})"]],[ref[f"sum_normed_{i}"]])
    else:
        assert got["data"]["name"] == ["ha"]
        for k in ("mean","std","entries","isOK"):
            _values_equal([got["data"][k][0]],[ref[k]])
        for i,q in enumerate(got["extra"]["quantiles"]):
            _values_equal([got["data"][f"quantile_{i}"][0]],[ref[f"q:{q}"]])
        for i in range(len(BOXES)):
            for pref in ("sum", "sum_normed"):
                _values_equal([got["data"][f"{pref}_{i}"][0]],[ref[f"{pref}_{i}"]])


MIXED_QUANTILES = [{"name":"ha","variables":["x"],"nbins":4,"range":[0,4],"quantiles":[.75,.25]},
                   {"name":"hb","variables":["y"],"nbins":4,"range":[0,4],"quantiles":[.5,.125]}]
ASCENDING_QUANTILES = [{"name":"ha","variables":["x"],"nbins":4,"range":[0,4],"quantiles":[.125,.25]},
                       {"name":"hb","variables":["y"],"nbins":4,"range":[0,4],"quantiles":[.5,.75]}]


def _quantile_table(node_session, tmp_path, sources, suffix):
    d=_df()
    _node_open(node_session,_hist_page(tmp_path,d,sources=sources,suffix=suffix))
    got=_read(node_session,"HistoStatsCDS",["name","entries","quantile_0","quantile_1","quantile_2","quantile_3"])
    return d, got


def _is_st_d4(d, got) -> bool:
    """ST-D4's exact signature: the model's list is not ascending, every quantile value equals the one-pass prediction
    for that order, and at least one differs from the reference."""
    q = got["extra"]["quantiles"]
    if q == sorted(q):
        return False
    wrong = False
    for i, col in enumerate(("x", "y")):
        counts = _bin_ref(d[col], XEDGE)
        pred = _single_pass_quantiles(counts, XEDGE, q)
        for j, qj in enumerate(q):
            value = got["data"][f"quantile_{j}"][i]
            if not np.allclose([value], [pred[j]], rtol=1e-11, atol=1e-11, equal_nan=True):
                return False
            ref = _stats_ref(counts, XEDGE, [qj])[f"q:{qj}"]
            wrong |= not np.allclose([value], [ref], rtol=1e-11, atol=1e-11, equal_nan=True)
    return wrong


@NODE
@pytest.mark.feature("STAT.histogram_table")
def test_control_st_d4_an_ascending_merged_list_is_correct(node_session,tmp_path):
    """The same two histograms and data; the requested lists merge into an ascending list: every value is right."""
    d, got = _quantile_table(node_session, tmp_path, ASCENDING_QUANTILES, "quantile_order_control")
    q = got["extra"]["quantiles"]
    assert q == sorted(q) == [.125, .25, .5, .75], q
    for j, qj in enumerate(q):
        for i, col in enumerate(("x", "y")):
            _values_equal([got["data"][f"quantile_{j}"][i]], [_stats_ref(_bin_ref(d[col], XEDGE), XEDGE, [qj])[f"q:{qj}"]])


@NODE
@pytest.mark.feature("STAT.histogram_table")
def test_stat_table_mixed_quantile_requests_preserve_column_titles(node_session,tmp_path):
    d, got = _quantile_table(node_session, tmp_path, MIXED_QUANTILES, "quantile_order")
    assert got["data"]["name"] == ["ha","hb"]
    assert got["extra"]["quantiles"] == [.75, .25, .5, .125], got["extra"]["quantiles"]   # the requested order, kept
    regression(_is_st_d4(d, got), "ST-D4: one-pass quantile values for an unsorted list")
    qlist=got["extra"]["quantiles"]
    assert len(qlist)==4 and set(qlist)=={.125,.25,.5,.75}
    for j,q in enumerate(qlist):
        for i,col in enumerate(("x","y")):
            want=_stats_ref(_bin_ref(d[col],XEDGE),XEDGE,[q])[f"q:{q}"]
            _values_equal([got["data"][f"quantile_{j}"][i]],[want])
        titles=[cols for group in got["extra"]["tableColumns"] for cols in group
                if cols[0]==f"quantile_{j}"]
        assert len(titles)==1 and titles[0][1]==f"Quantile {q}", (j, titles, q)


@NODE
@pytest.mark.feature("STAT.histogram_table")
def test_stat_table_empty_histogram_reports_invalid(node_session,tmp_path):
    d=_df().copy();d["x"]=8.25
    _node_open(node_session,_hist_page(tmp_path,d,suffix="empty"))
    got=_read(node_session,"HistoStatsCDS",["entries","mean","std","isOK","quantile_0"])["data"]
    assert got["entries"]==[0] and got["isOK"]==[False]
    for name in ("mean","std","quantile_0"):
        assert np.isnan(got[name][0]), (name,got)


@NODE
@pytest.mark.feature("STAT.histogram_table")
def test_stat_table_filter_changes_and_restores_exact_numbers(node_session,tmp_path):
    d=_df()
    src=[{"name":"ha","variables":["x"],"nbins":4,"range":[0,4],"sum_range":BOXES}]
    fig=[[["bin_center"],["bin_count"],{"source":"ha"}], ["tableHisto",{"rowwise":False}]]
    path=_page(tmp_path,d,fig,sources=src,widgets=[["range",["a"]]],suffix="table_filter")
    _node_open(node_session,path)
    def verify(cut=None):
        sel=d if cut is None else d.loc[d.a.between(*cut)]
        ref=_stats_ref(_bin_ref(sel.x,XEDGE),XEDGE,intervals=BOXES)
        got=_read(node_session,"HistoStatsCDS",["entries","mean","std","sum_1"])["data"]
        for key in got:
            _values_equal([got[key][0]],[ref[key]])
    verify()
    _set(node_session,"a",[-1.,.5]); verify((-1.,.5))
    _set(node_session,"a",[3.,4.]); verify((3.,4.))
    _set(node_session,"a",[-1.5,1.75]); verify()


@NODE
@pytest.mark.feature("STAT.sum_range")
@pytest.mark.parametrize("trim_last,weighted",[(False,False),(True,False),(False,True),(True,True)],
                         ids=["plain","empty_slice","weighted","weighted_empty_slice"])
def test_stat_profile_integral_efficiency_matches_uniform_bins(node_session,tmp_path,trim_last,weighted):
    d=_df().copy()
    if trim_last:d=d.loc[d.x < 3].reset_index(drop=True)
    _node_open(node_session,_profile_page(tmp_path,d,weighted=weighted,
                                         suffix=f"profile_sum_{int(trim_last)}_{int(weighted)}"))
    keys=["entries"]+[f"{p}_{i}" for i in range(len(BOXES)) for p in ("sum","sum_normed")]
    got=_read(node_session,"HistoNdProfile",keys,name="hp_1")["data"]
    ref=_profile_ref(d,XEDGE,YEDGE,BOXES,weight="wA" if weighted else None)
    for j in range(4):
        for k in keys:
            _values_equal([got[k][j]],[ref[j][k]])


# ==========================================================================
# NODE: production fit (unweighted/weighted/ridge/parameter changes)
# ==========================================================================
FIT_CASES = dict(argnames="weight,alpha", argvalues=[(None,0.),("wA",0.),("wA",.25)],
                 ids=["unweighted","weighted","ridge_all_diagonal"])


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.parametrize(**FIT_CASES)
def test_stat_client_linear_fit_matches_numpy(node_session,tmp_path,weight,alpha):
    d=_df()
    path=_fit_page(tmp_path,d,["x","z"],weight=weight,alpha=alpha,
                   suffix=f"fit_{weight}_{alpha}")
    _node_open(node_session,path)
    vals=_read(node_session,"CDSAlias",["fitted"],name="cdsFull")["data"]["fitted"]
    ref,beta=_fit_ref(d,["x","z"],weight=weight,alpha=alpha)
    _check_fit(vals,d,["x","z"],weight,alpha)
    got=_read(node_session,"ClientLinearFitter",[])["extra"]
    _values_equal(got["parameters"],beta)


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.parametrize(**FIT_CASES)
def test_control_st_d2_one_predictor_fit_equals_numpy(node_session,tmp_path,weight,alpha):
    """ST-D2's control (the defect is fixed in PHASE_5 Group 1): with one predictor (a 2 x 2 system) the same page and
    solver give numpy's fit."""
    d=_df()
    _node_open(node_session,_fit_page(tmp_path,d,["x"],weight=weight,alpha=alpha,suffix=f"fit_one_{weight}_{alpha}"))
    vals=_read(node_session,"CDSAlias",["fitted"],name="cdsFull")["data"]["fitted"]
    ref,beta=_fit_ref(d,["x"],weight=weight,alpha=alpha)
    _values_equal(vals,ref)
    _values_equal(_read(node_session,"ClientLinearFitter",[])["extra"]["parameters"],beta)


# --------- PHASE_5 Group 1: more fits against numpy (ST-D2), and the solver's absolute threshold (ST-D5)
GROUP1_SLOPES = [2., 3., -1.5, .5, -2.5]


def _group1_frame(case):
    """Exact data for the fit cases of Group 1: 40 rows, target = 1 + 2*p0 + 3*p1 - 1.5*p2 + 0.5*p3 - 2.5*p4 (as many
    terms as the case has predictors), so the fit must return these numbers. `w` is a weight column."""
    rng = np.random.default_rng(20261116)
    n = 40
    if case == "parabola":                       # x and x squared, x from 0 to 1
        x = rng.uniform(0., 1., n)
        cols = {"p0": x, "p1": x ** 2}
    else:
        count, scale = {"three": (3, 1.), "five": (5, 1.), "small": (2, 1e-2), "tiny": (2, 1e-5),
                        "tiny_line": (1, 1e-5), "zero_predictor": (2, 1.)}[case]
        cols = {f"p{i}": rng.normal(1., 1., n) * scale for i in range(count)}
        if case == "zero_predictor":             # a predictor that is 0 in every row: the system cannot be solved
            cols["p1"] = np.zeros(n)
    d = pd.DataFrame(cols)
    d["target"] = 1. + sum(s * d[c] for s, c in zip(GROUP1_SLOPES, cols))
    d["w"] = rng.uniform(.5, 2., n)
    return d, list(cols)


def _group1_fit_page(tmp_path, df, predictors, *, weight=None, suffix="g1", alpha=0.):
    """A fit page for any predictor names; a second figure shows the fit's own columns (the ST-D1 workaround)."""
    fit = {"name": "C1fit", "type": "linearFit", "varX": list(predictors), "varY": "target", "alpha": alpha,
           "weights": weight}
    figures = [[[predictors[0]], ["fitted"]], [["target"], list(predictors) + ([weight] if weight else [])]]
    aliases = [{"name": "fitted", "variables": list(predictors), "transform": "C1fit"}]
    return _page(tmp_path, df, figures, js=[fit], aliases=aliases, suffix=suffix)


def _group1_read(node_session, tmp_path, case, weight):
    d, predictors = _group1_frame(case)
    _node_open(node_session, _group1_fit_page(tmp_path, d, predictors, weight=weight, suffix=f"g1_{case}_{weight}"))
    vals = _read(node_session, "CDSAlias", ["fitted"], name="cdsFull")["data"]["fitted"]
    got = _read(node_session, "ClientLinearFitter", [])["extra"]["parameters"]
    return d, predictors, np.asarray(vals, dtype=float), np.asarray(got, dtype=float)


def _is_old_solver(vals, d, predictors, weight) -> bool:
    """ST-D2 / ST-D5, exact: the fitted values equal the transcription of the solver before the fix, and that
    transcription is not numpy's fit."""
    ref = _fit_ref(d, predictors, weight=weight)[0]
    pred = _ri_fit_prediction(d, predictors, weight, 0.)[0]
    return bool(np.allclose(vals, pred, rtol=1e-9, atol=1e-9, equal_nan=True)
                and not np.allclose(pred, ref, rtol=1e-6, atol=1e-6, equal_nan=True))


GROUP1_FITS = dict(argnames="case,weight", argvalues=[("three", None), ("three", "w"), ("five", None), ("five", "w"),
                                                      ("parabola", None), ("small", None), ("tiny", None),
                                                      ("tiny_line", None)],
                   ids=["three_predictors", "three_predictors_weighted", "five_predictors", "five_predictors_weighted",
                        "parabola_x_and_x_squared", "predictors_of_size_0.01", "predictors_of_size_0.00001",
                        "one_predictor_of_size_0.00001"])


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.parametrize(**GROUP1_FITS)
def test_stat_client_fit_group1_cases_equal_numpy_and_the_true_coefficients(node_session, tmp_path, case, weight):
    """Exact data, so the fit must return numpy's fit AND the coefficients the data were made with. The last case has
    one predictor: before the fix it was wrong only because of the absolute threshold (ST-D5), not because of ST-D2.
    The tolerances are those of the review of Group 1 (finding P2-1): see `_coefficient_tolerance`."""
    d, predictors, vals, got = _group1_read(node_session, tmp_path, case, weight)
    regression(_is_old_solver(vals, d, predictors, weight), f"ST-D2 / ST-D5: the fit of case {case} is the old solver's")
    ref, beta = _fit_ref(d, predictors, weight=weight)
    # Group 3 (the review of Group 1, finding P2-1): the fitted values are well determined in every case, so they are
    # compared within 1e-12. The coefficients within the standard error bound of a stable solver of the normal
    # equations, 100 * cond(A) * eps of the case's own matrix, but not looser than needed (at least 1e-12).
    tol = _coefficient_tolerance(d, predictors, weight)
    np.testing.assert_allclose(vals, ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got, beta, rtol=tol, atol=tol)
    np.testing.assert_allclose(got, GROUP1_SLOPES[:len(predictors)] + [1.], rtol=tol, atol=tol)


@NODE
@pytest.mark.feature("STAT.linear_fit")
def test_control_st_d5_a_fit_that_cannot_be_solved_shows_no_fit(node_session, tmp_path):
    """ST-D5's control, the same before and after the fix: a predictor that is 0 in every row has no fit, and every
    fitted value is NaN, so the page shows no fit instead of a wrong one."""
    d, predictors, vals, got = _group1_read(node_session, tmp_path, "zero_predictor", None)
    assert len(vals) == len(d) and np.isnan(vals).all(), vals
    assert np.isnan(got[:len(predictors)]).all(), got


@NODE
@pytest.mark.feature("STAT.linear_fit")
def test_group1_reference_data_are_exact_and_the_old_solver_prediction_is_wrong_on_them():
    """No page: before the cases above are used, numpy returns the true coefficients on every data set, and the
    transcription of the solver before the fix does not (so its signature can be told from the right answer)."""
    for case, weight in GROUP1_FITS["argvalues"]:
        d, predictors = _group1_frame(case)
        ref, beta = _fit_ref(d, predictors, weight=weight)
        np.testing.assert_allclose(beta, GROUP1_SLOPES[:len(predictors)] + [1.], rtol=1e-6)
        with np.errstate(all="ignore"):
            old = _ri_fit_prediction(d, predictors, weight, 0.)[0]
        assert not np.allclose(old, ref, rtol=1e-6, atol=1e-6, equal_nan=True), case



# --------- PHASE_5 Group 3 (Claude21.RI, 2026-11-17): a fit on predictors with a large offset, and two more controls
# In plain words. The fit forms the normal equations from sums over the rows (sum of x*x, sum of x, ...) and solves
# them. When a predictor has a large offset (a position far from 0, a run number, a timestamp), these sums lose the
# digits that carry the information: with an offset of 1e6 the fit is silently wrong, with 1e9 it is meaningless.
# Found by the review of Group 1 (Claude20.RI); register entry ST-D6 proposed. The standard remedy is to centre (and
# scale) the predictors before the sums are formed and to transform the coefficients back, as
# numpy.polynomial.Polynomial.fit does. Two more cases from the same review: exactly collinear predictors (a control
# that holds before and after), and fewer rows than unknowns, which can give finite, meaningless coefficients
# (register entry ST-D7 proposed). Both are FIXED in Group 3: their tests are ordinary tests, and their exact
# signatures are REGRESSION checks.
def _coefficient_tolerance(d, predictors, weight=None):
    """100 * cond(A) * eps of the normal matrix A = X^T W X of the case (X with the intercept column), at least 1e-12:
    the standard error bound of a stable solver of the normal equations (the review of Group 1, P2-1)."""
    x = np.column_stack([d[c].to_numpy(dtype=float) for c in predictors] + [np.ones(len(d))])
    w = np.ones(len(d)) if weight is None else d[weight].to_numpy(dtype=float)
    a = x.T @ (w[:, None] * x)
    return max(1e-12, 100. * np.linalg.cond(a) * np.finfo(float).eps)


def _js_sum(values):
    """A sum in the order of a JavaScript loop: left to right, one double addition at a time (never numpy's pairwise
    sum, which rounds differently)."""
    acc = 0.
    for v in values:
        acc += float(v)
    return acc


def _current_fit_prediction(d, predictors, weight=None):
    """ST-D6 PREDICTION - what the page computes before Group 3, transcribed from ClientLinearFitter.fit (the raw sums,
    in the same order of operations) and from MathUtils.ts after Group 1 (no threshold: a pivot that is not positive
    gives NaN). Used only to recognise the defect, never as the oracle. Returns the fitted values and coefficients."""
    cols = [[float(v) for v in d[c]] for c in predictors]
    y = [float(v) for v in d["target"]]
    w = None if weight is None else [float(v) for v in d[weight]]
    n, rows = len(cols), len(y)
    x, rhs = [], []
    for i in range(n):
        for j in range(i + 1):
            x.append(_js_sum(cols[i][k] * cols[j][k] if w is None else cols[i][k] * cols[j][k] * w[k]
                             for k in range(rows)))
    for i in range(n):
        x.append(_js_sum(cols[i][k] if w is None else cols[i][k] * w[k] for k in range(rows)))
        rhs.append(_js_sum(cols[i][k] * y[k] if w is None else cols[i][k] * y[k] * w[k] for k in range(rows)))
    x.append(float(rows) if w is None else _js_sum(w))
    rhs.append(_js_sum(y) if w is None else _js_sum(y[k] * w[k] for k in range(rows)))
    m = len(rhs)
    i_row = 0                                     # chol, as after Group 1
    for i in range(m):
        if not x[i + i_row] > 0:
            x[i + i_row] = float("nan")
        pivot_diag = 1. / x[i + i_row]
        j_row = i_row + i + 1
        for j in range(i + 1, m):
            pivot_row = pivot_diag * x[i + j_row]
            k_row = i_row + i + 1
            for k in range(i + 1, j + 1):
                if k == j:
                    x[k + j_row] -= pivot_row * x[i + k_row]
                k_row += k + 1
            for k in range(j + 1, m):
                x[j + k_row] -= pivot_row * x[i + k_row]
                k_row += k + 1
            x[i + j_row] = pivot_row
            j_row += j + 1
        i_row += i + 1
    i_diag = 0                                    # solve, as after Group 1
    for i in range(m):
        if not x[i_diag] > 0:
            rhs = [float("nan")] * m
            break
        i_diag += i + 2
    else:
        i_row = 0
        for i in range(m):
            for j in range(i):
                rhs[i] -= rhs[j] * x[i_row + j]
            i_row += i + 1
        i_diag = 0
        for i in range(m):
            rhs[i] /= x[i_diag]
            i_diag += i + 2
        for i in range(m - 1, -1, -1):
            j_row = 1 + ((i * (i + 5)) >> 1)
            for j in range(i + 1, m):
                rhs[i] -= rhs[j] * x[j_row]
                j_row += j + 1
    beta = np.asarray(rhs)
    fitted = np.full(rows, beta[-1])              # as v_compute: the intercept, then one predictor after the other
    for i in range(n):
        fitted = fitted + np.asarray(cols[i]) * beta[i]
    return fitted, beta


def _centred_fit_ref(d, predictors, weight=None):
    """The oracle for large offsets: numpy's least squares (LAPACK, not the normal equations) on the centred and
    scaled predictors, transformed back. Returns the fitted values and the coefficients (predictors, then the
    intercept)."""
    x = d[list(predictors)].to_numpy(dtype=float)
    y = d["target"].to_numpy(dtype=float)
    w = np.ones(len(d)) if weight is None else d[weight].to_numpy(dtype=float)
    mean = (w[:, None] * x).sum(axis=0) / w.sum()
    scale = np.sqrt((w[:, None] * (x - mean) ** 2).sum(axis=0) / w.sum())
    u = np.column_stack(((x - mean) / scale, np.ones(len(d))))
    root = np.sqrt(w)
    gamma = np.linalg.lstsq(root[:, None] * u, root * y, rcond=None)[0]
    slopes = gamma[:-1] / scale
    return u @ gamma, np.append(slopes, gamma[-1] - (slopes * mean).sum())


OFFSET_SLOPES = [2., -3.]


def _offset_frame(case):
    """200 rows; target = 1 + 2*p0 (- 3*p1), exactly as computed in floating point. p0 = offset + uniform(0, 1);
    p1, if there is one, is normal around 0. `w` is a weight column."""
    offset, count = {"offset_1e3": (1e3, 1), "offset_1e6": (1e6, 1), "offset_1e9": (1e9, 1),
                     "offset_1e6_two": (1e6, 2)}[case]
    rng = np.random.default_rng(20261117)
    n = 200
    cols = {"p0": offset + rng.uniform(0., 1., n)}
    if count == 2:
        cols["p1"] = rng.normal(0., 1., n)
    d = pd.DataFrame(cols)
    d["target"] = 1. + sum(s * d[c] for s, c in zip(OFFSET_SLOPES, cols))
    d["w"] = rng.uniform(.5, 2., n)
    return d, list(cols)


def _offset_read(node_session, tmp_path, case, weight):
    d, predictors = _offset_frame(case)
    _node_open(node_session, _group1_fit_page(tmp_path, d, predictors, weight=weight, suffix=f"g3_{case}_{weight}"))
    vals = _read(node_session, "CDSAlias", ["fitted"], name="cdsFull")["data"]["fitted"]
    got = _read(node_session, "ClientLinearFitter", [])["extra"]["parameters"]
    return d, predictors, np.asarray(vals, dtype=float), np.asarray(got, dtype=float)


def _is_st_d6(vals, d, predictors, weight) -> bool:
    """ST-D6, exact: the fitted values equal the transcription of today's fit on the raw sums, and that transcription
    is not the right fit (it is NaN, or it differs from the oracle by more than 1e-12 relative)."""
    pred = _current_fit_prediction(d, predictors, weight)[0]
    ref = _centred_fit_ref(d, predictors, weight)[0]
    return bool(np.allclose(vals, pred, rtol=1e-12, atol=0, equal_nan=True)
                and not np.allclose(pred, ref, rtol=1e-12, atol=0))


def _assert_offset_fit(vals, got, d, predictors, weight):
    ref, beta = _centred_fit_ref(d, predictors, weight)
    np.testing.assert_allclose(vals, ref, rtol=1e-12, atol=0)
    # the slopes: the data are rounded to the last digit of the target, which limits them to about 1e-7 relative
    # at an offset of 1e9; the intercept is checked through the fitted values
    np.testing.assert_allclose(got[:-1], beta[:-1], rtol=1e-6, atol=0)
    np.testing.assert_allclose(got[:-1], OFFSET_SLOPES[:len(predictors)], rtol=1e-6, atol=0)


OFFSET_FITS = dict(argnames="case,weight", argvalues=[("offset_1e3", None), ("offset_1e6", None),
                                                      ("offset_1e6", "w"), ("offset_1e9", None),
                                                      ("offset_1e6_two", None)],
                   ids=["offset_1e3", "offset_1e6", "offset_1e6_weighted", "offset_1e9",
                        "offset_1e6_and_a_second_predictor"])


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.parametrize(**OFFSET_FITS)
def test_stat_client_fit_on_a_predictor_with_a_large_offset_equals_numpy(node_session, tmp_path, case, weight):
    """The fitted values equal numpy's least squares within 1e-12 relative, and the slopes are the ones the data were
    made with. Before Group 3: an offset of 1e3 lost digits, 1e6 gave a wrong slope and intercept without a
    message, 1e9 a meaningless fit."""
    d, predictors, vals, got = _offset_read(node_session, tmp_path, case, weight)
    regression(_is_st_d6(vals, d, predictors, weight),
                 f"ST-D6: the fit of case {case} is the fit on the raw sums: "
                 f"{'no fit (NaN)' if np.isnan(vals).all() else 'slopes ' + str(got[:-1])}")
    _assert_offset_fit(vals, got, d, predictors, weight)


@NODE
@pytest.mark.feature("STAT.linear_fit")
def test_control_st_d6_a_predictor_without_offset_is_fitted_exactly(node_session, tmp_path):
    """ST-D6's control, right before and after the fix: the same kind of data without the offset."""
    d, predictors = _offset_frame("offset_1e3")
    d["p0"] = d["p0"] - 1e3 - .5
    d["target"] = 1. + 2. * d["p0"]
    _node_open(node_session, _group1_fit_page(tmp_path, d, predictors, suffix="g3_no_offset"))
    vals = np.asarray(_read(node_session, "CDSAlias", ["fitted"], name="cdsFull")["data"]["fitted"], dtype=float)
    got = np.asarray(_read(node_session, "ClientLinearFitter", [])["extra"]["parameters"], dtype=float)
    _assert_offset_fit(vals, got, d, predictors, None)
    np.testing.assert_allclose(got, [2., 1.], rtol=1e-12, atol=1e-12)


COLLINEAR = {"double": 2., "triple": 3., "tenth": .1}


def _fit_of_rows(node_session, tmp_path, d, predictors, suffix):
    _node_open(node_session, _group1_fit_page(tmp_path, d, predictors, suffix=suffix))
    vals = np.asarray(_read(node_session, "CDSAlias", ["fitted"], name="cdsFull")["data"]["fitted"], dtype=float)
    got = np.asarray(_read(node_session, "ClientLinearFitter", [])["extra"]["parameters"], dtype=float)
    return vals, got


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.parametrize("factor", list(COLLINEAR), ids=[f"predictor_and_its_{k}" for k in COLLINEAR])
def test_control_exactly_collinear_predictors_give_no_fit_or_the_right_fitted_values(node_session, tmp_path, factor):
    """The review of Group 1, P2-2: a predictor and a multiple of it cannot be separated, so the coefficients are not
    determined. In floating point the last pivot is then a rounding residue: zero or negative (no fit: every fitted
    value and coefficient is NaN) or tiny and positive (one of the many exact solutions). Which of the two happens
    depends on the rounding, before and after Group 3. What must hold in both: no fit, or fitted values equal to
    numpy's within 1e-12. The coefficients are not checked: they are not unique."""
    d, _ = _group1_frame("three")
    d = d[["p0"]].copy()
    d["p1"] = COLLINEAR[factor] * d["p0"]
    d["target"] = 1. + 2. * d["p0"] + 3. * d["p1"]
    vals, got = _fit_of_rows(node_session, tmp_path, d, ["p0", "p1"], f"g3_collinear_{factor}")
    assert len(vals) == len(d)
    if np.isnan(vals).all():
        assert np.isnan(got).all(), got
    else:
        assert np.isfinite(vals).all() and np.isfinite(got).all(), (vals[:5], got)
        np.testing.assert_allclose(vals, _fit_ref(d, ["p0", "p1"])[0], rtol=1e-12, atol=1e-12)


TWO_ROWS = {"exact_in_binary": ([1., 2.], [.5, -1.5], [3., 4.]),
            "not_exact_in_binary": ([.13, .37], [.71, 1.93], [3.3, 4.1])}


def _is_st_d7(vals, got, d) -> bool:
    """ST-D7, exact: with fewer rows than unknowns the page returns finite coefficients, and they are those of the
    transcription of the fit before Group 3 (a rounding residue taken as a pivot)."""
    with np.errstate(all="ignore"):
        pred_vals, pred = _current_fit_prediction(d, ["p0", "p1"])
    return bool(np.isfinite(got).all() and np.allclose(got, pred, rtol=1e-12, atol=0)
                and np.allclose(vals, pred_vals, rtol=1e-12, atol=0))


@NODE
@pytest.mark.feature("STAT.linear_fit")
def test_control_fewer_rows_than_unknowns_exact_in_binary_give_no_fit(node_session, tmp_path):
    """The review of Group 1, P2-2: two rows, two predictors and the intercept (three unknowns). The fit is not
    determined, so the page shows no fit. With these values the last pivot is exactly zero: no fit before and after
    Group 3."""
    p0, p1, y = TWO_ROWS["exact_in_binary"]
    d = pd.DataFrame({"p0": p0, "p1": p1, "target": y})
    vals, got = _fit_of_rows(node_session, tmp_path, d, ["p0", "p1"], "g3_two_rows_exact")
    assert len(vals) == 2 and np.isnan(vals).all(), vals
    assert np.isnan(got).all(), got


@NODE
@pytest.mark.feature("STAT.linear_fit")
def test_fewer_rows_than_unknowns_give_no_fit(node_session, tmp_path):
    """Two rows whose values are not exact in binary: the last pivot is a rounding residue that happens to be
    positive, and before Group 3 the page showed finite, meaningless coefficients. The fit is not determined, so the
    page must show no fit."""
    p0, p1, y = TWO_ROWS["not_exact_in_binary"]
    d = pd.DataFrame({"p0": p0, "p1": p1, "target": y})
    vals, got = _fit_of_rows(node_session, tmp_path, d, ["p0", "p1"], "g3_two_rows_inexact")
    regression(_is_st_d7(vals, got, d), f"ST-D7: two rows give the coefficients {got}")
    assert len(vals) == 2 and np.isnan(vals).all(), vals
    assert np.isnan(got).all(), got


@UNIT
@pytest.mark.feature("STAT.linear_fit")
def test_group3_offset_reference_and_prediction():
    """No page: the oracle returns the slopes the data were made with at every offset, and its fitted values equal
    those of numpy's plain least squares (the fitted values are well determined; at 1e9 the split between slope and
    intercept is not, so the intercept is only checked through the fitted values). The transcription of today's fit
    is wrong on every case (so its signature can be told from the right answer); on data without an offset it is
    right."""
    for case, weight in OFFSET_FITS["argvalues"]:
        d, predictors = _offset_frame(case)
        ref, beta = _centred_fit_ref(d, predictors, weight)
        np.testing.assert_allclose(beta[:-1], OFFSET_SLOPES[:len(predictors)], rtol=1e-6)
        with np.errstate(all="ignore"):
            pred = _current_fit_prediction(d, predictors, weight)[0]
        assert not np.allclose(pred, ref, rtol=1e-12, atol=0), case
        np.testing.assert_allclose(_fit_ref(d, predictors, weight=weight)[0], ref, rtol=1e-12, atol=0)
    d, predictors = _group1_frame("three")
    np.testing.assert_allclose(_current_fit_prediction(d, predictors)[0], _fit_ref(d, predictors)[0],
                               rtol=1e-12, atol=1e-12)
    with np.errstate(all="ignore"):                # ST-D7: two rows not exact in binary give finite coefficients
        for key, finite in (("not_exact_in_binary", True), ("exact_in_binary", False)):
            p0, p1, y = TWO_ROWS[key]
            beta = _current_fit_prediction(pd.DataFrame({"p0": p0, "p1": p1, "target": y}), ["p0", "p1"])[1]
            assert np.isfinite(beta).all() if finite else np.isnan(beta).all(), (key, beta)

# ==========================================================================
# PHASE_5 Group 5: the linear fit WITH the ridge term (review of Groups 3 and 4, findings G3-P1-A and G3-P1-B)
# ==========================================================================
# The ridge term alpha penalises every coefficient, the intercept included (today's definition, kept):
#     sum_k w_k (y_k - b0 - sum_i b_i x_ki)^2 + alpha (b0^2 + sum_i b_i^2).
# The oracle is the exact minimiser of that function for the stored numbers, in rational arithmetic (no rounding).
RIDGE_ALPHAS = [1e-3, .25, 10.]
RIDGE_OFFSETS = [1e6, 1e7, 1e8, 1e9]
GPT27_ALPHA = .25

FIT_NOW = """() => {
  const f = window.__riH.models().filter(m => m.type === 'ClientLinearFitter');
  if (f.length !== 1) throw Error('linear fitters on the page: ' + f.length);
  f[0].fit();
  return Array.from(f[0].parameters, v => Number.isNaN(v) ? 'NaN' : v);
}"""


def _exact_ridge(d, predictors, weight=None, alpha=0.):
    """The exact minimiser (predictors..., intercept) and the exact fitted values, both rounded once to doubles at
    the end; None if the minimiser is not unique."""
    from fractions import Fraction
    names = list(predictors)
    cols = [[Fraction(float(v)) for v in d[c]] for c in names]
    rows = [[col[k] for col in cols] + [Fraction(1)] for k in range(len(d))]
    y = [Fraction(float(v)) for v in d["target"]]
    w = [Fraction(1)] * len(d) if weight is None else [Fraction(float(v)) for v in d[weight]]
    p = len(names) + 1
    m = [[Fraction(0)] * (p + 1) for _ in range(p)]
    for k, row in enumerate(rows):
        if w[k] == 0:
            continue
        for i in range(p):
            m[i][p] += w[k] * row[i] * y[k]
            for j in range(p):
                m[i][j] += w[k] * row[i] * row[j]
    for i in range(p):
        m[i][i] += Fraction(alpha)
    for c in range(p):
        pivot = next((r for r in range(c, p) if m[r][c] != 0), None)
        if pivot is None:
            return None
        m[c], m[pivot] = m[pivot], m[c]
        for r in range(p):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [m[r][k] - f * m[c][k] for k in range(p + 1)]
    beta = [m[i][p] / m[i][i] for i in range(p)]
    fitted = [sum(row[i] * beta[i] for i in range(p)) for row in rows]
    return np.array([float(b) for b in beta]), np.array([float(f) for f in fitted])


def _ridge_tolerance(offset):
    """The fitted values must equal the exact ones within this fraction of the target's spread. The predictors near
    the offset are stored with an error of up to 1.1e-16 * offset, and no solver in double precision can do much
    better than a few of these units: the bound is 1e-12 plus 36 of them (4e-15 * offset). The new fit's worst case
    on a reviewer's 360 cases was 7.3e-16 * offset."""
    return 1e-12 + 4e-15 * offset


def _group3_ridge_prediction(d, predictors, weight, alpha):
    """G3-P1-A/B PREDICTION - what the page computes with the ridge term after Group 3, transcribed from
    ClientLinearFitter.fit at 7aa4423 (centred and scaled sums in the same order of operations, the penalty
    alpha * T^T T added to them) and from MathUtils.solve. Used only to recognise the defect, never as the oracle.
    Returns the fitted values and the coefficients (predictors, then the intercept)."""
    cols = [[float(v) for v in d[c]] for c in predictors]
    y = [float(v) for v in d["target"]]
    w = None if weight is None else [float(v) for v in d[weight]]
    nx, rows = len(cols), len(y)
    sum_w = 0.
    for k in range(rows):
        sum_w += 1. if w is None else w[k]
    mean, scale = [], []
    for col in cols:
        acc = 0.
        for k in range(rows):
            acc += col[k] if w is None else col[k] * w[k]
        m = acc / sum_w if sum_w != 0 else float("nan")
        acc = 0.
        for k in range(rows):
            dk = col[k] - m
            acc += dk * dk if w is None else dk * dk * w[k]
        s = np.sqrt(acc / sum_w) if sum_w != 0 else float("nan")
        mean.append(m)
        scale.append(s if s > 0 else 1.)
    x = [0.] * ((nx + 1) * (nx + 2) // 2)
    rhs = [0.] * (nx + 1)
    last = nx * (nx + 1) // 2
    for k in range(rows):
        wk = 1. if w is None else w[k]
        u = [(cols[i][k] - mean[i]) / scale[i] for i in range(nx)]
        i_row = 0
        for i in range(nx):
            uw = u[i] * wk
            for j in range(i + 1):
                x[i_row + j] += uw * u[j]
            x[last + i] += uw
            rhs[i] += uw * y[k]
            i_row += i + 1
        x[last + nx] += wk
        rhs[nx] += wk * y[k]
    i_row = 0
    for i in range(nx):
        ri = mean[i] / scale[i]
        for j in range(i + 1):
            x[i_row + j] += alpha * ri * mean[j] / scale[j]
        x[i_row + i] += alpha / (scale[i] * scale[i])
        x[last + i] -= alpha * ri
        i_row += i + 1
    x[last + nx] += alpha
    gamma = _js_solve_packed(x, rhs)
    beta, intercept = [], gamma[nx]
    for i in range(nx):
        b = gamma[i] / scale[i]
        beta.append(b)
        intercept -= b * mean[i]
    beta = np.asarray(beta + [intercept])
    fitted = np.full(rows, beta[-1])              # as v_compute: the intercept, then one predictor after the other
    for i in range(nx):
        fitted = fitted + np.asarray(cols[i]) * beta[i]
    return fitted, beta


def _js_solve_packed(x, rhs):
    """MathUtils.solve after Group 1, transcribed (chol of the packed lower triangle, then the substitutions); a
    pivot that is not positive gives NaN for every unknown. Works on copies."""
    x, rhs = list(x), list(rhs)
    m = len(rhs)
    i_row = 0
    for i in range(m):
        if not x[i + i_row] > 0:
            x[i + i_row] = float("nan")
        pivot_diag = 1. / x[i + i_row]
        j_row = i_row + i + 1
        for j in range(i + 1, m):
            pivot_row = pivot_diag * x[i + j_row]
            k_row = i_row + i + 1
            for k in range(i + 1, j + 1):
                if k == j:
                    x[k + j_row] -= pivot_row * x[i + k_row]
                k_row += k + 1
            for k in range(j + 1, m):
                x[j + k_row] -= pivot_row * x[i + k_row]
                k_row += k + 1
            x[i + j_row] = pivot_row
            j_row += j + 1
        i_row += i + 1
    i_diag = 0
    for i in range(m):
        if not x[i_diag] > 0:
            return [float("nan")] * m
        i_diag += i + 2
    i_row = 0
    for i in range(m):
        for j in range(i):
            rhs[i] -= rhs[j] * x[i_row + j]
        i_row += i + 1
    i_diag = 0
    for i in range(m):
        rhs[i] /= x[i_diag]
        i_diag += i + 2
    for i in range(m - 1, -1, -1):
        j_row = 1 + ((i * (i + 5)) >> 1)
        for j in range(i + 1, m):
            rhs[i] -= rhs[j] * x[j_row]
            j_row += j + 1
    return rhs


def _ridge_frame(offset, count, n=60, seed=20261118):
    """n rows; every predictor is the offset plus a uniform number in [-1, 1]; target = 1 + 2 p0 + p1 - 3 p2 (as many
    terms as predictors), as computed in floating point; `w` is a weight column."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: offset + rng.uniform(-1., 1., n) for c in ("p0", "p1", "p2")})
    predictors = ["p0", "p1", "p2"][:count]
    d["target"] = 1. + d[predictors].to_numpy() @ np.array([2., 1., -3.][:count])
    d["w"] = rng.uniform(.5, 2., n)
    return d, predictors


def _gpt27_frame():
    """GPT27.RI's case: 8 rows, two predictors near 1e9."""
    t = np.linspace(-1., 1., 8)
    d = pd.DataFrame({"p0": 1e9 + t, "p1": 1e9 + np.sin(6. * t), "w": np.linspace(.5, 2., 8)})
    d["target"] = 1. + 2. * d["p0"] + d["p1"]
    return d, ["p0", "p1"]


def _ridge_read(node_session, tmp_path, d, predictors, weight, alpha, suffix):
    _node_open(node_session, _group1_fit_page(tmp_path, d, predictors, weight=weight, alpha=alpha, suffix=suffix))
    vals = np.asarray(_read(node_session, "CDSAlias", ["fitted"], name="cdsFull")["data"]["fitted"], dtype=float)
    got = np.asarray(_read(node_session, "ClientLinearFitter", [])["extra"]["parameters"], dtype=float)
    return vals, got


def _ridge_error(vals, d, predictors, weight, alpha):
    """The largest deviation of the fitted values from the exact ones, as a fraction of the target's spread (inf for
    no fit)."""
    exact = _exact_ridge(d, predictors, weight, alpha)[1]
    if not np.isfinite(vals).all():
        return float("inf")
    return float(np.max(np.abs(vals - exact)) / np.ptp(d["target"].to_numpy()))


def _is_g3_p1_b(cases) -> bool:
    """G3-P1-B, exact: on every case the page's fitted values are those of the transcription of the fit after
    Group 3, and on at least one case that transcription is outside the tolerance (no fit, or wrong)."""
    same, wrong = True, False
    for vals, d, predictors, weight, alpha, offset in cases:
        with np.errstate(all="ignore"):
            pred = _group3_ridge_prediction(d, predictors, weight, alpha)[0]
        same = same and bool(np.allclose(vals, pred, rtol=1e-12, atol=0, equal_nan=True))
        wrong = wrong or _ridge_error(pred, d, predictors, weight, alpha) > _ridge_tolerance(offset)
    return same and wrong


@UNIT
@pytest.mark.feature("STAT.linear_fit")
def test_group5_exact_ridge_reference_and_prediction():
    """No page: the exact reference equals numpy's least squares of the augmented rows [sqrt(w) X; sqrt(alpha) I] on
    a well-conditioned case; with no row of positive weight its minimum is every coefficient 0 (and with alpha = 0
    there is none). The transcription of the fit after Group 3 is right on that well-conditioned case, and gives no
    fit on GPT27.RI's case (so the signature of the defect can be told from the right answer)."""
    d, predictors = _ridge_frame(0., 2)
    w = d["w"].to_numpy()
    x = np.column_stack([d[predictors].to_numpy(), np.ones(len(d))])
    aug = np.vstack([np.sqrt(w)[:, None] * x, np.sqrt(GPT27_ALPHA) * np.eye(3)])
    target = np.concatenate([np.sqrt(w) * d["target"].to_numpy(), np.zeros(3)])
    beta, fitted = _exact_ridge(d, predictors, "w", GPT27_ALPHA)
    np.testing.assert_allclose(beta, np.linalg.lstsq(aug, target, rcond=None)[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fitted, x @ beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_group3_ridge_prediction(d, predictors, "w", GPT27_ALPHA)[0], fitted,
                               rtol=1e-12, atol=1e-12)
    d0 = d.assign(w=0.)
    np.testing.assert_array_equal(_exact_ridge(d0, predictors, "w", GPT27_ALPHA)[0], [0., 0., 0.])
    assert _exact_ridge(d0, predictors, "w", 0.) is None
    d, predictors = _gpt27_frame()
    for weight in (None, "w"):
        with np.errstate(all="ignore"):
            assert np.isnan(_group3_ridge_prediction(d, predictors, weight, GPT27_ALPHA)[1]).all(), weight


NO_POSITIVE_WEIGHT = dict(argnames="case", argvalues=["all_weights_zero", "no_row"])


def _no_positive_weight_fit(node_session, tmp_path, case, alpha):
    """The fit's coefficients with every weight 0 (8 rows) or without any row. Without rows the page computes no
    fitted value and never runs the fit, so the fit is run directly."""
    if case == "all_weights_zero":
        d = pd.DataFrame({"p0": np.arange(8.), "target": 1. + 2. * np.arange(8.), "w": np.zeros(8)})
        weight = "w"
    else:
        d = pd.DataFrame({"p0": np.zeros(0), "target": np.zeros(0)})
        weight = None
    _node_open(node_session, _group1_fit_page(tmp_path, d, ["p0"], weight=weight, alpha=alpha,
                                              suffix=f"g5_{case}_{alpha}"))
    got = np.asarray([float(v) for v in _eval(node_session, FIT_NOW)])
    return d, got


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.xfail(**STRICT, reason="G3-P1-A: with the ridge term and no row of positive weight, the fit gives NaN")
@pytest.mark.parametrize(**NO_POSITIVE_WEIGHT)
def test_ridge_fit_without_a_row_of_positive_weight_gives_zero_coefficients(node_session, tmp_path, case):
    """With the ridge term and no row of positive weight, only the penalty is left, and its only minimum is every
    coefficient 0. Before Group 3 the page gave 0; after Group 3 it gives NaN (the mean of the predictor is 0/0)."""
    d, got = _no_positive_weight_fit(node_session, tmp_path, case, GPT27_ALPHA)
    known_defect(len(got) == 2 and np.isnan(got).all(), f"G3-P1-A: the {case} fit gives {got}")
    np.testing.assert_array_equal(got, [0., 0.])


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.parametrize(**NO_POSITIVE_WEIGHT)
def test_control_no_row_of_positive_weight_without_ridge_gives_no_fit(node_session, tmp_path, case):
    """G3-P1-A's control, right before and after the fix: without the ridge term every coefficient minimises the
    function equally, so there is no fit."""
    d, got = _no_positive_weight_fit(node_session, tmp_path, case, 0.)
    assert len(got) == 2 and np.isnan(got).all(), got


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.xfail(**STRICT, reason="G3-P1-B: GPT27.RI's ridge case near 1e9 gives no fit")
@pytest.mark.parametrize("weight", [None, "w"], ids=["unweighted", "weighted"])
def test_ridge_fit_of_two_predictors_near_1e9_equals_the_exact_answer(node_session, tmp_path, weight):
    """GPT27.RI's case: 8 rows, two predictors near 1e9, alpha 0.25. The exact answer is finite (slopes about 1.97
    and 1.03: the penalty moves the intercept to almost 0). After Group 3 the page gives no fit."""
    d, predictors = _gpt27_frame()
    vals, got = _ridge_read(node_session, tmp_path, d, predictors, weight, GPT27_ALPHA, f"g5_gpt27_{weight}")
    known_defect(_is_g3_p1_b([(vals, d, predictors, weight, GPT27_ALPHA, 1e9)]),
                 f"G3-P1-B: GPT27.RI's case ({weight}) gives {got}")
    beta = _exact_ridge(d, predictors, weight, GPT27_ALPHA)[0]
    assert _ridge_error(vals, d, predictors, weight, GPT27_ALPHA) <= _ridge_tolerance(1e9), (got, beta)
    np.testing.assert_allclose(got[:-1], beta[:-1], rtol=1e-6, atol=0)


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.xfail(**STRICT, reason="G3-P1-B: with the ridge term, predictors far from 0 give no fit or a wrong fit")
@pytest.mark.parametrize("offset", RIDGE_OFFSETS, ids=[f"offset_{o:.0e}" for o in RIDGE_OFFSETS])
def test_ridge_fit_of_predictors_far_from_zero_equals_the_exact_answer(node_session, tmp_path, offset):
    """Two and three predictors at the offset, 60 rows, alpha 1e-3, 0.25 and 10, with and without weights (12 pages).
    The fitted values must equal the exact ones within _ridge_tolerance(offset) of the target's spread. After Group 3
    the penalty of the intercept swamped the data: a wrong fit without a message from 1e6 on (a reviewer measured
    0.5 % of the spread at 1e7 and more than the spread at 1e8 and 1e9), or no fit."""
    cases, errors = [], []
    for count in (2, 3):
        d, predictors = _ridge_frame(offset, count)
        for alpha in RIDGE_ALPHAS:
            for weight in (None, "w"):
                vals, got = _ridge_read(node_session, tmp_path, d, predictors, weight, alpha,
                                        f"g5_{offset:.0e}_{count}_{alpha}_{weight}")
                cases.append((vals, d, predictors, weight, alpha, offset))
                errors.append((count, alpha, weight, _ridge_error(vals, d, predictors, weight, alpha)))
    known_defect(_is_g3_p1_b(cases), f"G3-P1-B at offset {offset:g}: the worst error is "
                                     f"{max(e[-1] for e in errors):.3g} of the spread")
    bad = [e for e in errors if not e[-1] <= _ridge_tolerance(offset)]
    assert not bad, (offset, _ridge_tolerance(offset), bad)


# ST-D1: the fit's response and weight columns are not sent
READ_SENT = r"""() => {
  const t = window.__riH.models().filter(m => m.type === 'CDSCompress' && m.name === 'cdsOrig');
  if (t.length !== 1) throw Error('compressed main table cdsOrig: ' + t.length);
  return Object.keys(t[0].inputData).sort();
}"""
ST_D1_ERROR = "Cannot read properties of undefined (reading 'decodeProgram')"


def _st_d1_state(node_session, tmp_path, send, weight, suffix):
    d=_df()
    _node_open(node_session,_fit_page(tmp_path,d,["x"],weight=weight,send=send,suffix=suffix))
    sent=_eval(node_session,READ_SENT)
    r=_eval_raw(node_session,READ,{"kind":"CDSAlias","keys":["fitted"],"name":"cdsFull"})
    return d, sent, r


def _is_st_d1(sent, r, column) -> bool:
    return column not in sent and not r.get("ok") and r.get("error") == ST_D1_ERROR


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.parametrize("send,weight,column",[("none",None,"target"),("response","wA","wA")],
                         ids=["response_column","weight_column"])
def test_strict_a_fit_on_a_page_as_a_user_writes_it_equals_numpy(node_session,tmp_path,send,weight,column):
    """One predictor (no ST-D2); nothing else on the page shows the response column (or, in the second case, the
    weight column). PHASE_5 Group 4: the weight column is a case of its own."""
    d, sent, r = _st_d1_state(node_session, tmp_path, send, weight, f"fit_d1_strict_{column}")
    regression(_is_st_d1(sent, r, column), f"ST-D1: the fit's column {column} is not sent")
    assert column in sent, sent
    assert r.get("ok"), r
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    _values_equal(decode(r["result"])["data"]["fitted"], _fit_ref(d,["x"],weight=weight)[0])


def _fit_build_error(tmp_path, df, predictors, **kw):
    """None if the page builds, else (exception type, message, file:function where it was raised)."""
    import traceback
    try:
        _fit_page(tmp_path, df, predictors, **kw)
    except Exception as exc:                                                # noqa: BLE001 - the signature decides
        last = traceback.extract_tb(exc.__traceback__)[-1]
        return type(exc).__name__, str(exc), f"{Path(last.filename).name}:{last.name}"
    return None


def _is_st_d3(err) -> bool:
    return (err is not None and err[0] == "UnboundLocalError" and "'context'" in err[1]
            and err[2] == "bokehTools.py:bokehDrawArray")


@NODE
@pytest.mark.feature("STAT.linear_fit")
def test_control_st_d3_after_an_expression_alias_the_page_builds_and_follows_the_predictor(node_session,tmp_path):
    """ST-D3's control: the same page with an expression alias declared first builds (that alias sets `context`), and
    the fitted alias follows the predictor parameter (one predictor at a time, so ST-D2 does not apply)."""
    d=_df()
    _node_open(node_session,_fit_page(tmp_path,d,["x"],parameter=True,expression_first=True,suffix="varX_control"))
    for chosen in (["x"], ["z"], ["x"]):
        _set(node_session,"predictors",chosen)
        vals=_read(node_session,"CDSAlias",["fitted"],name="cdsFull")["data"]["fitted"]
        _values_equal(vals,_fit_ref(d,chosen)[0])


@NODE
@pytest.mark.feature("STAT.linear_fit")
def test_stat_client_fit_predictor_count_change_recomputes(node_session,tmp_path):
    d=_df()
    err=_fit_build_error(tmp_path,d,["x"],parameter=True,suffix="varX")
    regression(_is_st_d3(err), "ST-D3: the page does not build")
    assert err is None, err
    _node_open(node_session,_fit_page(tmp_path,d,["x"],parameter=True,suffix="varX"))
    before=_read(node_session,"CDSAlias",["fitted"],name="cdsFull")["data"]["fitted"]
    _values_equal(before,_fit_ref(d,["x"])[0])
    _set(node_session,"predictors",["x","z"])
    after=_read(node_session,"CDSAlias",["fitted"],name="cdsFull")["data"]["fitted"]
    _check_fit(after,d,["x","z"])
    _set(node_session,"predictors",["z"])
    last=_read(node_session,"CDSAlias",["fitted"],name="cdsFull")["data"]["fitted"]
    _values_equal(last,_fit_ref(d,["z"])[0])
    assert not np.allclose(before,after) and not np.allclose(after,last)


@NODE
@pytest.mark.feature("STAT.linear_fit")
@pytest.mark.parametrize("predictors",[pytest.param(["x"],id="one_predictor"),
                                       pytest.param(["x","z"],id="two_predictors")])
def test_stat_client_fit_reacts_to_source_data_change(node_session,tmp_path,predictors):
    """Integration: uncompressed page (the mutation replaces a column of the plain ColumnDataSource cdsOrig)."""
    d=_df()
    _node_open(node_session,_fit_page(tmp_path,d,predictors,suffix=f"mutate_fit_{len(predictors)}",compression=False))
    initial=_read(node_session,"CDSAlias",["fitted"],name="cdsFull")["data"]["fitted"]
    _check_fit(initial,d,predictors)
    new_y=(d.target + d.x*.75).to_list()
    _eval(node_session,"""(a) => {
      const matches=window.__riH.models().filter(m => m.type==='ColumnDataSource' && m.name==='cdsOrig');
      if(matches.length!==1)throw Error('cdsOrig count '+matches.length);
      const m=matches[0];m.data={...m.data,target:a};return m.get_length();
    }""",new_y)
    revised=d.assign(target=new_y)
    updated=_read(node_session,"CDSAlias",["fitted"],name="cdsFull")["data"]["fitted"]
    assert not np.allclose(updated, initial), "the fit did not change after its response changed"
    _check_fit(updated,revised,predictors)


# ==========================================================================
# NODE: weighted unbinned quantiles (no Q-H1/Q-H2 oracle imposed)
# ==========================================================================
# Deliberately use a dedicated public-page builder for requested quantiles:
# the ordinary profile helper does not request them.
def _weighted_q_page(tmp_path,d,*,weight=None,suffix="q"):
    hist={"name":"hp","variables":["x","y"],"nbins":[4,4],
          "range":[[0,4],[0,4]],"unbinned_projections":True,
          "quantiles":[.2,.45,.7]}
    if weight is not None:hist["weights"]=weight
    return _page(tmp_path,d,[[["bin_center_0"],["quantile_0"],{"source":"hp_1"}]],
                 sources=[hist],select_types=["HistoNdProfile"],suffix=suffix)


def _weighted_quantiles(session,tmp_path,d,weight,suffix):
    _node_open(session,_weighted_q_page(tmp_path,d,weight=weight,suffix=suffix))
    got=_read(session,"HistoNdProfile",["quantile_0","quantile_1","quantile_2"],name="hp_1")
    return np.asarray([got["data"][f"quantile_{j}"] for j in range(3)],float)


def _expected_quantile_occupied(df, weight):
    """Oracle for defined x slices, independent of the quantile convention.

    Only rows carrying positive weight can contribute.  This is an independent
    event-level occupancy oracle, not a readback of the production profile.
    """
    if weight is None:
        active = np.ones(len(df), dtype=bool)
    else:
        raw_weights = np.asarray(df[weight], dtype=float)
        assert np.all(np.isfinite(raw_weights)) and np.all(raw_weights >= 0)
        active = raw_weights > 0
    occupancy = np.histogram(np.asarray(df.loc[active, "x"], float), bins=XEDGE)[0]
    return occupancy > 0


def _assert_weighted_quantile_pair(actual, reference, occupied):
    """F2: invalid/all-NaN results cannot satisfy the metamorphic check.

    We deliberately DO NOT specify an interpolation convention or a q=1 rule.
    The chosen q values (.2, .45, .7) are interior and the first two x slices
    contain positive-weight rows, so their quantiles must be finite.  Empty
    x slices have no quantile and must remain NaN, not +/-Infinity or zero.
    """
    a = np.asarray(actual, dtype=float)
    b = np.asarray(reference, dtype=float)
    occupied = np.asarray(occupied, dtype=bool)
    shape = (3, len(XEDGE) - 1)  # three requested q values, four x slices
    assert a.shape == shape, f"weighted quantile shape: {a.shape}, expected {shape}"
    assert b.shape == shape, f"reference quantile shape: {b.shape}, expected {shape}"
    assert occupied.shape == (shape[1],), f"occupancy shape: {occupied.shape}"
    assert occupied.any(), "F2 control requires at least one occupied x slice"
    valid = np.broadcast_to(occupied[np.newaxis, :], shape)
    np.testing.assert_array_equal(np.isfinite(a), valid,
                                  err_msg="production finite-quantile mask")
    np.testing.assert_array_equal(np.isfinite(b), valid,
                                  err_msg="reference finite-quantile mask")
    assert np.isnan(a[~valid]).all(), "empty slices must have NaN quantiles"
    assert np.isnan(b[~valid]).all(), "empty reference slices must have NaN quantiles"
    _values_equal(a[valid], b[valid])


@NODE
@pytest.mark.feature("HIST.profile_unbinned")
@pytest.mark.parametrize("mode",["all_equal","repeated_rows","reversed","zero_row"],
                         ids=["equal_weights","frequency_replication","row_permutation","zero_weight"])
def test_stat_unbinned_weighted_quantile_invariants(node_session,tmp_path,mode):
    d=_df().loc[lambda z: z.x<2].reset_index(drop=True)
    # A fixed set of positive integer frequency weights; do not use q=1.
    d["freq"]=[1,2,1,3,1,2]
    if mode=="all_equal":
        a=_weighted_quantiles(node_session,tmp_path,d.assign(freq=1),"freq","q_equal_w")
        b=_weighted_quantiles(node_session,tmp_path,d,None,"q_equal_plain")
    elif mode=="repeated_rows":
        a=_weighted_quantiles(node_session,tmp_path,d,"freq","q_freq")
        # Integration: the repeated rows go through the same WEIGHTED path (weight 1); the unweighted path uses another
        # quantile definition (interpolation), which one is intended is Q-H2 - an observation in the CRR, not a test.
        repeated=d.loc[d.index.repeat(d.freq.astype(int))].reset_index(drop=True).assign(freq=1)
        b=_weighted_quantiles(node_session,tmp_path,repeated,"freq","q_repeat")
    elif mode=="reversed":
        a=_weighted_quantiles(node_session,tmp_path,d,"freq","q_order_a")
        b=_weighted_quantiles(node_session,tmp_path,d.iloc[::-1].reset_index(drop=True),"freq","q_order_b")
    else:
        d2=d.copy();d2.loc[3,"freq"]=0
        a=_weighted_quantiles(node_session,tmp_path,d2,"freq","q_zero")
        b=_weighted_quantiles(node_session,tmp_path,d2[d2.freq>0].reset_index(drop=True),"freq","q_without")
    # F2: both real production outputs must be finite on the x slices with
    # strictly positive-weight rows; equal all-NaN arrays are NOT a PASS.
    source = d2 if mode == "zero_row" else d
    _assert_weighted_quantile_pair(a, b, _expected_quantile_occupied(source, "freq"))


# ==========================================================================
# NODE: multiselect CDSStack / single-select changes, F7
# ==========================================================================
def _expected_parts(df,options):
    return {opt:_bin_ref(df.x,XEDGE,df[opt]) for opt in options}


def _assert_stack(session,df,options):
    got=_read(session,"CDSStack",["bin_count","_source_index"],name="hs")["data"]
    counts=np.asarray(got["bin_count"],float)
    labels=list(got["_source_index"])
    assert len(counts)==len(labels)==4*len(options)
    assert set(labels)==set(options)
    ref=_expected_parts(df,options)
    for label in options:
        observed=counts[np.asarray(labels,dtype=str)==label]
        assert len(observed)==4, (label,labels)
        _values_equal(observed,ref[label])
    # Option ordering vs selection ordering is Architect policy: record actual
    # label blocks in CRR; avoid falsely certifying either convention here.
    blocks=[labels[0]] if labels else []
    for label in labels[1:]:
        if label != blocks[-1]:blocks.append(label)
    assert len(blocks)==len(options) and set(blocks)==set(options), blocks
    return blocks


@NODE
@pytest.mark.feature("HIST.stack")
def test_stat_multiselect_weights_stack_and_reverse_order(node_session,tmp_path):
    d=_df();_node_open(node_session,_stack_page(tmp_path,d,suffix="multistack"))
    first=_assert_stack(node_session,d,["wA","wB"])
    _set(node_session,"weightChoice",["wB","wA"])
    reversed_labels=_assert_stack(node_session,d,["wB","wA"])
    assert set(first)==set(reversed_labels)=={"wA","wB"}


@NODE
@pytest.mark.feature("HIST.stack")
def test_stat_multiselect_stack_selection_change(node_session,tmp_path):
    d=_df();_node_open(node_session,_stack_page(tmp_path,d,suffix="stack_change"))
    _assert_stack(node_session,d,["wA","wB"])
    _set(node_session,"weightChoice",["wB"])
    _assert_stack(node_session,d,["wB"])
    _set(node_session,"weightChoice",["wA"])
    _assert_stack(node_session,d,["wA"])
    _set(node_session,"weightChoice",["wA","wB"])
    _assert_stack(node_session,d,["wA","wB"])


@NODE
@pytest.mark.feature("HIST.stack")
def test_stat_single_select_weights_recomputes(node_session,tmp_path):
    d=_df();_node_open(node_session,_stack_page(tmp_path,d,multi=False,suffix="single_weights"))
    initial=_read(node_session,"HistogramCDS",["bin_count"],name="hs")["data"]["bin_count"]
    _values_equal(initial,_bin_ref(d.x,XEDGE,d.wA))
    _set(node_session,"weightChoice","wB")
    changed=_read(node_session,"HistogramCDS",["bin_count"],name="hs")["data"]["bin_count"]
    _values_equal(changed,_bin_ref(d.x,XEDGE,d.wB))
    _set(node_session,"weightChoice","wA")
    restored=_read(node_session,"HistogramCDS",["bin_count"],name="hs")["data"]["bin_count"]
    _values_equal(restored,initial)


@NODE
@pytest.mark.feature("HIST.stack")
def test_stat_real_page_mutation_falsifies_stack_oracle(node_session,tmp_path):
    """Real-model negative control: wrong fixture *must* fail at the bin assertion."""
    d=_df();_node_open(node_session,_stack_page(tmp_path,d,suffix="mutation_control",compression=False))
    _assert_stack(node_session,d,["wA","wB"])
    _eval(node_session,"""() => {const a=window.__riH.models().filter(m=>m.type==='ColumnDataSource' && m.name==='cdsOrig');
       if(a.length!==1)throw Error('transport source not unique');
       const m=a[0];const w=Array.from(m.get_column('wA'));w[0]+=8;
       m.data={...m.data,wA:w};return true;}""")
    with pytest.raises(AssertionError,match="Not equal|not equal|Mismatched|mismatch"):
        _assert_stack(node_session,d,["wA","wB"])

@NODE
@pytest.mark.feature("STAT.histogram_table")
def test_stat_table_weight_choice_updates(node_session,tmp_path):
    """One select parameter controls source weights, which must update table entries/moments."""
    d=_df()
    source=[{"name":"hw","variables":["x"],"nbins":4,"range":[0,4],"weights":"weightChoice"}]
    plots=[[["bin_center"],["bin_count"],{"source":"hw"}],
           ["tableHisto",{"rowwise":False}]]
    param=[{"name":"weightChoice","value":"wA","options":["wA","wB"]}]
    path=_page(tmp_path,d,plots,sources=source,params=param,widgets=[["select",["weightChoice"]]],
               suffix="stats_weight_choice")
    _node_open(node_session,path)
    for label in ("wA","wB","wA"):
        _set(node_session,"weightChoice",label)
        got=_read(node_session,"HistoStatsCDS",["entries","mean","std"])["data"]
        ref=_stats_ref(_bin_ref(d.x,XEDGE,d[label]),XEDGE)
        for key in ("entries","mean","std"):
            _values_equal([got[key][0]],[ref[key]])


@NODE
@pytest.mark.feature("STAT.histogram_table")
def test_stat_table_nd_axes_have_independent_statistics_no_quantiles(node_session,tmp_path):
    """N-D table: one row per axis, with population binned means over *all* rows."""
    d=_df()
    src=[{"name":"hnd","variables":["x","y"],"nbins":[4,4],
          "range":[[0,4],[0,4]]}]
    plots=[[["bin_center_0"],["bin_center_1"],{"source":"hnd","colorZvar":"bin_count"}],
           ["tableHisto",{"rowwise":False}]]
    _node_open(node_session,_page(tmp_path,d,plots,sources=src,suffix="nd_table"))
    got=_read(node_session,"HistoStatsCDS",["name","entries","mean","std","isOK"])
    assert got["data"]["name"] == ["hnd_0","hnd_1"]
    for idx,name in enumerate(("x","y")):
        ref=_stats_ref(_bin_ref(d[name],XEDGE),XEDGE)
        for key in ("entries","mean","std","isOK"):
            _values_equal([got["data"][key][idx]],[ref[key]])
    assert got["extra"]["quantiles"] == []


@NODE
@pytest.mark.feature("HIST.stack")
def test_stat_multiselect_stack_tracks_range_filter(node_session,tmp_path):
    d=_df()
    path=_stack_page(tmp_path,d,with_filter=True,suffix="stack_filter")
    _node_open(node_session,path)
    _assert_stack(node_session,d,["wA","wB"])
    cut=[-1.,.5]
    _set(node_session,"a",cut)
    _assert_stack(node_session,d.loc[d.a.between(*cut)],["wA","wB"])
    _set(node_session,"a",[-1.5,1.75])
    _assert_stack(node_session,d,["wA","wB"])

@NODE
@pytest.mark.feature("STAT.histogram_table")
def test_stat_table_include_exclude_selects_named_histograms(node_session,tmp_path):
    """The public include/exclude regex contract must select only requested histogram rows."""
    d=_df()
    sources=[{"name":"ha","variables":["x"],"nbins":4,"range":[0,4]},
             {"name":"hb","variables":["y"],"nbins":4,"range":[0,4]}]
    figs=[[["bin_center"],["bin_count"],{"source":"ha"}],
          [["bin_center"],["bin_count"],{"source":"hb"}],
          ["tableHisto",{"include":"h.*","exclude":"hb", "rowwise":False}]]
    _node_open(node_session,_page(tmp_path,d,figs,sources=sources,suffix="table_regex"))
    actual=_read(node_session,"HistoStatsCDS",["name","entries"])["data"]
    assert actual["name"]==["ha"], actual
    _values_equal(actual["entries"],[len(d)])

@UNIT
def test_stat_ref_integer_weight_replication_zero_and_order_controls():
    """Independent distributional foundation for quantile metamorphisms.

    Compares empirical mass below thresholds, not an unapproved quantile rule.
    """
    values = np.array([.25, 1.25, 2.25, 3.25])
    weights = np.array([1, 2, 3, 1])
    repeat = np.repeat(values, weights)
    perm = [2, 0, 3, 1]
    for cut in (0., 1., 2., 3., 4.):
        count = int(weights[values < cut].sum())
        assert count == int((repeat < cut).sum())
        assert count == int(weights[perm][values[perm] < cut].sum())
    augmented_values = np.append(values, 0.5)
    augmented_weights = np.append(weights, 0)
    np.testing.assert_array_equal(
        np.histogram(augmented_values, bins=XEDGE, weights=augmented_weights)[0],
        np.histogram(values, bins=XEDGE, weights=weights)[0])
    with pytest.raises(AssertionError):
        _values_equal([int(weights[values < 2.].sum()) + 1], [int((repeat < 2.).sum())])

    # F2 non-vacuity/shape falsifiers.  These controls execute without Node.
    occupied = np.array([True, True, False, False])
    assert _expected_quantile_occupied(_df().loc[lambda z: z.x < 2], None).tolist() == occupied.tolist()
    good = np.array([[.25, 1.25, np.nan, np.nan],
                     [.50, 1.50, np.nan, np.nan],
                     [.75, 1.75, np.nan, np.nan]])
    _assert_weighted_quantile_pair(good, good.copy(), occupied)
    all_nan = np.full((3, 4), np.nan)
    with pytest.raises(AssertionError):
        _assert_weighted_quantile_pair(all_nan, all_nan, occupied)
    with pytest.raises(AssertionError):
        _assert_weighted_quantile_pair(good[:, :3], good[:, :3], occupied)
    bad_mask = good.copy(); bad_mask[0, 0] = np.nan
    with pytest.raises(AssertionError):
        _assert_weighted_quantile_pair(bad_mask, bad_mask, occupied)
    bad_empty = good.copy(); bad_empty[0, 2] = 0.
    with pytest.raises(AssertionError):
        _assert_weighted_quantile_pair(bad_empty, bad_empty, occupied)
    bad_value = good.copy(); bad_value[0, 0] += .5
    with pytest.raises(AssertionError):
        _assert_weighted_quantile_pair(bad_value, good, occupied)
