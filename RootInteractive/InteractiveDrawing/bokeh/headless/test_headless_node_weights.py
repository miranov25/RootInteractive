"""PHASE_4_7 stage 1 - weights in Node: exact reproductions of the weight defects found in the audit of the
multi-select and weights dashboard (test_interactiveTemplateWeights) and in its discovery (Claude20.RI, 2026-11-12).

In plain words. The Weights dashboard draws the same profiles (mean, std, quantiles per X bin) for several weights at
once. A weight says how much a row counts; a row with weight 0 counts for nothing (Architect, answer A-W3). These
tests build small pages with RootInteractive's public builder (bokehDrawSA.fromArray), load the real TypeScript models
in Node and compare every number with numpy. Each defect has:
- an exact signature (what the page computes today, predicted from the data, never a loose "something is wrong");
- a strict expected failure (the test of the desired behaviour; it turns into a pass when PHASE_5 fixes the defect);
  PHASE_5 Batches 2A and 2B fixed WT-D1, WT-D2, WT-D3 and WT-D4: those tests are now ordinary tests, and the exact
  signature of each is a REGRESSION check (an ordinary failure that names the defect). WT-D5 is still a known defect;
- a healthy control (the same feature where it works today, so the test is not vacuous).

The defects, all in production code that this phase does not change (DEC-01):
    WT-D1  a weight that is a Boolean column, for example the dashboard's own "A>.5" and "B>C", cannot be read in a
           compressed page: the column is sent with dtype "bool", and SerializationUtils.ts getElementSize throws
           "Unknown dtype: bool" (both with lossless compression and with relative16, the dashboard's compression).
           Every weighted histogram and profile on the Weights dashboard fails this way.
    WT-D2  a row with weight 0 that is the FIRST row visited in an X bin makes that bin's mean and std NaN
           (HistoNdProfile.ts unbinned branch: entries_ += 0, then (v - mean) * 0 / 0). The audit's C1.
    WT-D3  a row with weight 0 can be returned as a weighted quantile (MathUtils.ts weighted_kth_value returns the
           pivot when the target lies on a cumulative boundary, also when the pivot's weight is 0).
    WT-D4  in a profile with several weights, every series after the first is computed with the weights paired to the
           wrong rows: the first series' quantile pass reorders the shared value array inside each bin
           (kth_value / weighted_kth_value partition in place), but the next series reads its weights in the old order
           (HistoNdProfile.ts: sorted_entries fetched once, sorted_weights per series). A series' numbers depend on
           which series come before it (decision DEC-21 says they must not).
    WT-D5  a multi-select weight parameter must be called "weights": any other name fails the page build with
           KeyError: 'weights' (bokehTools.py bokehDrawArray reads paramDict["weights"] for the colour and marker).
The quantile tests do not choose a quantile convention (open: inverse CDF, midpoint): the expected value is one of the
values a standard definition gives; a zero-weight row's value is none of them.
"""
from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
UNIT = pytest.mark.headless_unit
XEDGE = np.array([0., 1., 2., 3., 4.])
YEDGE = np.array([0., 1., 2., 3., 4.])
BOOL_ERROR = "Unknown dtype: bool"


# ------------------------------------------------------------------ data
def _zero_rows(first_zero: bool = True, middle_zero: bool = True) -> pd.DataFrame:
    """Hand-checkable rows; bins are X slices [0,1), [1,2), [2,3), [3,4) and Y bins of width 1.
    Slice 0: y 0.25 (w 0) is the first row visited (alone in the lowest Y bin), then 1.25 (w 2), 2.25 (w 3);
             with first_zero=False the zero-weight row moves to y 3.25, the last Y bin.
             Weighted mean (1.25*2 + 2.25*3)/5 = 1.85, std sqrt(0.24).
    Slice 1: y 1.125 (w 1), 1.375 (w 0), 1.875 (w 1), all in Y bin [1,2); weighted median: 1.125 (inverse CDF),
             1.5 (midpoint) or 1.875 (upper) - never 1.375. With middle_zero=False the zero-weight row is left out.
    Slices 2 and 3: positive weights only."""
    rows = [(.5, .25 if first_zero else 3.25, 0.), (.5, 1.25, 2.), (.5, 2.25, 3.),
            (1.5, 1.125, 1.), (1.5, 1.375, 0.), (1.5, 1.875, 1.),
            (2.5, .25, 1.), (2.5, 1.25, 2.), (2.5, 2.75, 1.),
            (3.5, .5, 1.), (3.5, 1.5, 2.), (3.5, 2.5, 1.)]
    if not middle_zero:
        rows = [r for r in rows if not (r[0] == 1.5 and r[2] == 0.)]
    return pd.DataFrame(rows, columns=["x", "y", "w"])


def _random_rows(seed: int = 7, n: int = 400) -> pd.DataFrame:
    """Rows for WT-D4: an indicator weight w (0 or 1) and a positive weight v. With seed 7 the first row visited in
    every X slice has w = 1, so WT-D2 does not interfere (checked by test_wt_random_rows_avoid_wt_d2)."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"x": rng.random(n) * 4, "y": rng.random(n) * 4})
    d["w"] = (rng.random(n) > .5).astype(float)
    d["v"] = rng.random(n) + .5
    return d


# ------------------------------------------------------------------ independent oracles (numpy only)
def _slices(df):
    return [df[(df.x >= XEDGE[i]) & (df.x < XEDGE[i + 1])] for i in range(len(XEDGE) - 1)]


def _weighted_ref(df, weight):
    """Per X slice: entries = sum of the weights; mean and std (population) over the rows with weight > 0."""
    out = {"entries": [], "mean": [], "std": []}
    for s in _slices(df):
        w = s[weight].to_numpy(float)
        y = s.y.to_numpy(float)
        out["entries"].append(float(w.sum()))
        if w.sum() > 0:
            m = float(np.average(y, weights=w))
            out["mean"].append(m)
            out["std"].append(float(np.sqrt(np.average((y - m) ** 2, weights=w))))
        else:
            out["mean"].append(np.nan)
            out["std"].append(np.nan)
    return {k: np.asarray(v) for k, v in out.items()}


def _quantile_candidates(values, weights, q):
    """The values standard weighted-quantile definitions give from the rows with weight > 0: the inverse of the
    weighted CDF (lowest x with F(x) >= q), the next value up when q*W falls exactly on a boundary, and their midpoint.
    A zero-weight row never enters."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    keep = w > 0
    v, w = v[keep], w[keep]
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    target = q * cum[-1]
    i = int(np.argmax(cum >= target))
    out = {float(v[i])}
    if np.isclose(cum[i], target, rtol=0, atol=1e-12) and i + 1 < len(v):
        out |= {float(v[i + 1]), float((v[i] + v[i + 1]) / 2)}
    return out


def _first_visited_weight(s, weight):
    """The weight of the row the page visits first in a slice: the lowest occupied Y bin, the first row in table order."""
    yb = np.searchsorted(YEDGE, s.y.to_numpy(float), side="right") - 1
    first = int(np.argmin(yb))                     # argmin returns the first occurrence = table order
    return float(s[weight].to_numpy(float)[first])


# ------------------------------------------------------------------ signatures (what production computes today)
def _is_wt_d2(got, df, weight) -> bool:
    """WT-D2, exact: mean and std are NaN in exactly the slices whose first visited row has weight 0 and whose
    entries are positive; every other slice equals the oracle; entries equal the sum of the weights everywhere."""
    ref = _weighted_ref(df, weight)
    hit = np.array([_first_visited_weight(s, weight) == 0 and s[weight].sum() > 0 for s in _slices(df)])
    mean, std, ent = (np.asarray(got[k], float) for k in ("mean", "std", "entries"))
    return (hit.any() and np.allclose(ent, ref["entries"], rtol=1e-12, atol=0)
            and np.isnan(mean[hit]).all() and np.isnan(std[hit]).all()
            and np.allclose(mean[~hit], ref["mean"][~hit], rtol=1e-12, atol=1e-12, equal_nan=True)
            and np.allclose(std[~hit], ref["std"][~hit], rtol=1e-12, atol=1e-12, equal_nan=True))


def _is_wt_d3(got_q, df, weight, q, slice_index) -> bool:
    """WT-D3, exact: the slice's quantile is the value of a zero-weight row of that slice and not a standard value."""
    s = _slices(df)[slice_index]
    zero_values = set(s.y[s[weight] == 0].astype(float))
    value = float(got_q[slice_index])
    return value in zero_values and value not in _quantile_candidates(s.y, s[weight], q)


def _is_wt_d4(alone, first, later) -> bool:
    """WT-D4, exact: the series computed first equals the same weight computed alone (entries, mean, std, quantile);
    computed after another series it has the same entries but another mean, std or quantile."""
    same = all(np.allclose(first[k], alone[k], rtol=1e-12, atol=1e-12, equal_nan=True)
               for k in ("entries", "mean", "std", "quantile_0"))
    entries_kept = np.allclose(later["entries"], alone["entries"], rtol=1e-12, atol=1e-12)
    moved = any(not np.allclose(later[k], alone[k], rtol=1e-12, atol=1e-12, equal_nan=True)
                for k in ("mean", "std", "quantile_0"))
    return same and entries_kept and moved


def _is_wt_d5(error) -> bool:
    """WT-D5, exact: the build raises KeyError('weights') inside bokehTools.py bokehDrawArray."""
    return error is not None and error[0] == "KeyError" and error[1] == "'weights'" and error[2] == \
        "bokehTools.py:bokehDrawArray"


def _is_wt_d1(read, spec, compression) -> bool:
    """WT-D1, exact: every profile column read throws 'Unknown dtype: bool', the weight column is sent as bool
    (lossless: dtype "bool"; relative16: the code table's dtype1 "bool"), and the other columns are float64 or float32."""
    sent_bool = spec.get("dtype") == "bool" if compression == "lossless" else \
        (spec.get("decodeDope") or {}).get("dtype1") == "bool"
    return sent_bool and all(read[k] == "ERR " + BOOL_ERROR for k in ("entries", "mean", "std", "quantile_0"))


# ------------------------------------------------------------------ the public page builder -> Node
def _page(tmp_path, df, weights_value, options, *, param="weights", compression="lossless", quantiles=(.5,),
          suffix="p"):
    from bokeh.document import Document
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import LOSSLESS
    from RootInteractive.Tools.compressArray import arrayCompressionRelative16
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    before = sa.show
    sa.show = lambda *a, **kw: None
    try:
        hist = {"name": "hp", "variables": ["x", "y"], "nbins": [4, 4], "range": [[0, 4], [0, 4]],
                "unbinned_projections": True, "quantiles": list(quantiles), "weights": param}
        comp = [(p, list(a)) for p, a in LOSSLESS] if compression == "lossless" else arrayCompressionRelative16
        obj = sa.bokehDrawSA.fromArray(df, None, [[["bin_center_0"], ["mean"], {"source": "hp_1"}]],
                                       [["multiSelect", [param]]],
                                       parameterArray=[{"name": param, "value": list(weights_value),
                                                        "options": list(options)}],
                                       histogramArray=[hist], layout=[[0]], widgetLayout=[[0]], arrayCompression=comp)
    finally:
        sa.show = before
    doc = Document()
    doc.add_root(obj.pAll)
    out = tmp_path / f"weights_{suffix}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


def _build_error(tmp_path, *args, **kw):
    """None if the page builds, else (exception type, message, file:function where it was raised)."""
    try:
        _page(tmp_path, *args, **kw)
        return None
    except Exception as exc:                                    # noqa: BLE001 - the signature names the exception
        last = traceback.extract_tb(exc.__traceback__)[-1]
        return type(exc).__name__, str(exc), f"{Path(last.filename).name}:{last.name}"


def _unexpected_console(r):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import CONSOLE_ALLOWED
    return [m for m in (r.get("console_errors") or []) if not any(re.search(p, m) for p, _ in CONSOLE_ALLOWED)]


def _open(session, path):
    r = session.request("new_doc", path=str(path))
    assert r.get("ok") and not r.get("page_errors") and not _unexpected_console(r), r


PROFILE = r"""() => {
  const ms = window.__riH.models().filter(m => m.type === 'HistoNdProfile');
  if (ms.length !== 1) throw Error('expected one HistoNdProfile, found ' + ms.length);
  const r = {};
  for (const k of ['weights', 'entries', 'mean', 'std', 'quantile_0']) {
    try { r[k] = Array.from(ms[0].get_column(k)); } catch (e) { r[k] = 'ERR ' + e.message; }
  }
  return r; }"""

SPEC = r"""(c) => { const o = window.__riH.model('cdsOrig', ['CDSCompress']); if (!o) throw Error('no cdsOrig');
  const s = o.inputData[c]; if (!s) return null;
  const r = {}; for (const k of ['dtype', 'decodeDope', 'actionArray']) if (s[k] !== undefined) r[k] = s[k]; return r; }"""


def _profile(session):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = session.request("eval", code=PROFILE)
    assert r.get("ok") and not r.get("page_errors") and not _unexpected_console(r), r
    return decode(r["result"])


def _series(got, name):
    """One series of a multi-weight profile, selected by its name (the 'weights' column), never by position."""
    names = np.asarray([str(x) for x in got["weights"]])
    assert (names == name).sum() == len(XEDGE) - 1, (name, list(names))
    return {k: np.asarray(got[k], float)[names == name] for k in ("entries", "mean", "std", "quantile_0")}


def _equal(a, b, what):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-12, atol=1e-12, equal_nan=True,
                               err_msg=what)


# ------------------------------------------------------------------ python-only: oracles and signatures are not vacuous
@UNIT
def test_wt_oracle_hand_values():
    ref = _weighted_ref(_zero_rows(), "w")
    _equal(ref["entries"], [5., 2., 4., 4.], "entries")
    _equal(ref["mean"][:2], [1.85, 1.5], "mean")
    _equal(ref["std"][:2], [np.sqrt(.24), .375], "std")
    s1 = _slices(_zero_rows())[1]
    assert _quantile_candidates(s1.y, s1.w, .5) == {1.125, 1.5, 1.875}
    assert 1.375 not in _quantile_candidates(s1.y, s1.w, .5)
    assert _quantile_candidates([1., 2., 3.], [1., 1., 1.], .5) == {2.}
    assert _first_visited_weight(_slices(_zero_rows())[0], "w") == 0
    assert _first_visited_weight(_slices(_zero_rows(first_zero=False))[0], "w") == 2


@UNIT
def test_wt_signatures_are_exact():
    d = _zero_rows()
    ref = _weighted_ref(d, "w")
    today = {k: v.copy() for k, v in ref.items()}
    today["mean"][0] = today["std"][0] = np.nan
    assert _is_wt_d2(today, d, "w")
    assert not _is_wt_d2(ref, d, "w")                           # the right answer is not the defect
    wrong = {k: v.copy() for k, v in today.items()}
    wrong["mean"][2] += 1e-6
    assert not _is_wt_d2(wrong, d, "w")                         # another error elsewhere is not WT-D2
    assert not _is_wt_d2(today, _zero_rows(first_zero=False), "w")
    assert _is_wt_d3([2.25, 1.375, 1.25, 1.5], d, "w", .5, 1)
    assert not _is_wt_d3([2.25, 1.125, 1.25, 1.5], d, "w", .5, 1)
    assert not _is_wt_d3([2.25, 1.4, 1.25, 1.5], d, "w", .5, 1)
    a = {"entries": np.ones(4), "mean": np.ones(4), "std": np.ones(4), "quantile_0": np.ones(4)}
    b = {k: v.copy() for k, v in a.items()}
    b["mean"][1] = 2.
    assert _is_wt_d4(a, a, b) and not _is_wt_d4(a, a, a) and not _is_wt_d4(a, b, b)
    assert _is_wt_d5(("KeyError", "'weights'", "bokehTools.py:bokehDrawArray"))
    assert not _is_wt_d5(None) and not _is_wt_d5(("KeyError", "'x'", "bokehTools.py:bokehDrawArray"))
    read = {k: "ERR " + BOOL_ERROR for k in ("entries", "mean", "std", "quantile_0")}
    assert _is_wt_d1(read, {"dtype": "bool"}, "lossless")
    assert _is_wt_d1(read, {"decodeDope": {"dtype1": "bool"}}, "relative16")
    assert not _is_wt_d1(read, {"dtype": "float64"}, "lossless")
    assert not _is_wt_d1({**read, "mean": [1.]}, {"dtype": "bool"}, "lossless")


@UNIT
def test_wt_random_rows_avoid_wt_d2():
    d = _random_rows()
    for name in ("w", "v"):
        assert all(_first_visited_weight(s, name) > 0 for s in _slices(d)), name
        assert np.isfinite(_weighted_ref(d, name)["mean"]).all()


# ------------------------------------------------------------------ Node: WT-D2 zero weight first
NODE_STATS = [pytest.mark.node_stats, pytest.mark.backend("node"), pytest.mark.layer("integration")]


def _marks(*marks):
    def wrap(f):
        for m in reversed(NODE_STATS + list(marks)):
            f = m(f)
        return f
    return wrap


@_marks(pytest.mark.feature("HIST.profile_unbinned"))
def test_wt_zero_weight_not_first_contributes_nothing(node_session, tmp_path):
    """Healthy control: a zero-weight row visited after a positive one changes nothing (entries, mean, std)."""
    d = _zero_rows(first_zero=False)
    _open(node_session, _page(tmp_path, d, ["w"], [None, "w"], suffix="zero_last"))
    got, ref = _series(_profile(node_session), "w"), _weighted_ref(d, "w")
    for k in ("entries", "mean", "std"):
        _equal(got[k], ref[k], k)


@_marks(pytest.mark.feature("HIST.profile_unbinned"))
def test_wt_zero_weight_first_contributes_nothing(node_session, tmp_path):
    d = _zero_rows()
    _open(node_session, _page(tmp_path, d, ["w"], [None, "w"], suffix="zero_first"))
    got, ref = _series(_profile(node_session), "w"), _weighted_ref(d, "w")
    regression(_is_wt_d2(got, d, "w"), "WT-D2: mean and std NaN where the first visited row has weight 0")
    for k in ("entries", "mean", "std"):
        _equal(got[k], ref[k], k)


# ------------------------------------------------------------------ Node: WT-D3 zero weight as a quantile
@_marks(pytest.mark.feature("HIST.profile_unbinned"))
def test_wt_weighted_median_without_zero_weight_rows(node_session, tmp_path):
    """Healthy control: the same slice without its zero-weight row gives a standard weighted median."""
    d = _zero_rows(middle_zero=False)
    _open(node_session, _page(tmp_path, d, ["w"], [None, "w"], suffix="median_plain"))
    got = _series(_profile(node_session), "w")
    s1 = _slices(d)[1]
    assert float(got["quantile_0"][1]) in _quantile_candidates(s1.y, s1.w, .5), got["quantile_0"]


@_marks(pytest.mark.feature("HIST.profile_unbinned"))
def test_wt_weighted_median_ignores_zero_weight_rows(node_session, tmp_path):
    d = _zero_rows()
    _open(node_session, _page(tmp_path, d, ["w"], [None, "w"], suffix="median_zero"))
    got = _series(_profile(node_session), "w")
    regression(_is_wt_d3(got["quantile_0"], d, "w", .5, 1), "WT-D3: the median is a zero-weight row's value")
    s1 = _slices(d)[1]
    assert float(got["quantile_0"][1]) in _quantile_candidates(s1.y, s1.w, .5), got["quantile_0"]


# ------------------------------------------------------------------ Node: WT-D4 series depend on the series before them
def _three_orders(session, tmp_path, d, name, other):
    _open(session, _page(tmp_path, d, [name], [None, "w", "v"], suffix=f"alone_{name}"))
    alone = _series(_profile(session), name)
    _open(session, _page(tmp_path, d, [name, other], [None, "w", "v"], suffix=f"first_{name}"))
    first = _series(_profile(session), name)
    _open(session, _page(tmp_path, d, [other, name], [None, "w", "v"], suffix=f"later_{name}"))
    later = _series(_profile(session), name)
    return alone, first, later


@_marks(pytest.mark.feature("HIST.stack"))
@pytest.mark.parametrize("name", ["w", "v"])
def test_wt_first_series_equals_the_weight_alone_and_numpy(node_session, tmp_path, name):
    """Healthy control: a weight alone equals numpy, and computed first next to another series it is unchanged."""
    d = _random_rows()
    other = None if name == "w" else "w"
    alone, first, _ = _three_orders(node_session, tmp_path, d, name, other)
    ref = _weighted_ref(d, name)
    for k in ("entries", "mean", "std"):
        _equal(alone[k], ref[k], f"{name} alone {k}")
    for k in ("entries", "mean", "std", "quantile_0"):
        _equal(first[k], alone[k], f"{name} first {k}")


@_marks(pytest.mark.feature("HIST.stack"))
@pytest.mark.parametrize("name,other", [("w", None), ("v", "w")], ids=["None_then_w", "w_then_v"])
def test_wt_series_do_not_depend_on_the_series_before_them(node_session, tmp_path, name, other):
    d = _random_rows()
    alone, first, later = _three_orders(node_session, tmp_path, d, name, other)
    regression(_is_wt_d4(alone, first, later), f"WT-D4: series {name} changes when {other} is computed before it")
    for k in ("entries", "mean", "std", "quantile_0"):
        _equal(later[k], alone[k], f"{name} after {other}: {k}")


# ------------------------------------------------------------------ Node: WT-D5 the parameter name
@_marks(pytest.mark.feature("HIST.stack"))
def test_wt_weight_parameter_named_weights_builds(node_session, tmp_path):
    """Healthy control: the same page with the parameter called "weights" builds and computes."""
    d = _zero_rows(first_zero=False)
    assert _build_error(tmp_path, d, ["w"], [None, "w"], param="weights", suffix="named") is None
    _open(node_session, _page(tmp_path, d, ["w"], [None, "w"], param="weights", suffix="named"))
    _equal(_series(_profile(node_session), "w")["entries"], _weighted_ref(d, "w")["entries"], "entries")


@_marks(pytest.mark.feature("HIST.stack"),
        pytest.mark.xfail(**STRICT, reason="WT-D5: a weight parameter not called 'weights' fails the build"))
def test_wt_weight_parameter_may_have_any_name(node_session, tmp_path):
    d = _zero_rows(first_zero=False)
    err = _build_error(tmp_path, d, ["w"], [None, "w"], param="cut", suffix="cut")
    known_defect(_is_wt_d5(err), "WT-D5: KeyError 'weights' in bokehTools.py bokehDrawArray")
    assert err is None, err
    _open(node_session, _page(tmp_path, d, ["w"], [None, "w"], param="cut", suffix="cut"))
    _equal(_series(_profile(node_session), "w")["entries"], _weighted_ref(d, "w")["entries"], "entries")


# ------------------------------------------------------------------ Node: WT-D1 Boolean weights in a compressed page
NODE_TRANSPORT = [pytest.mark.node_transport, pytest.mark.backend("node"), pytest.mark.layer("integration"),
                  pytest.mark.feature("ENC.cds_compress.page_decode")]


def _transport(*marks):
    def wrap(f):
        for m in reversed(NODE_TRANSPORT + list(marks)):
            f = m(f)
        return f
    return wrap


@_transport()
@pytest.mark.parametrize("compression", ["lossless", "relative16"])
def test_wt_numeric_indicator_weight_in_a_compressed_page(node_session, tmp_path, compression):
    """Healthy control: the same cut as a float column (0.0 / 1.0) is read and equals numpy."""
    d = _zero_rows(first_zero=False).assign(cut=lambda z: (z.x > 2).astype(float))
    _open(node_session, _page(tmp_path, d, ["cut"], [None, "cut"], compression=compression, suffix=f"num_{compression}"))
    got, ref = _series(_profile(node_session), "cut"), _weighted_ref(d, "cut")
    _equal(got["entries"], ref["entries"], "entries")
    ok = ref["entries"] > 0
    _equal(got["mean"][ok], ref["mean"][ok], "mean")


@_transport()
@pytest.mark.parametrize("compression", ["lossless", "relative16"])
def test_wt_boolean_weight_in_a_compressed_page(node_session, tmp_path, compression):
    """The dashboard's own kind of weight: an expression that is true or false ("A>.5"), here "x>2"."""
    d = _zero_rows(first_zero=False)
    _open(node_session, _page(tmp_path, d, ["x>2"], [None, "x>2"], compression=compression, suffix=f"bool_{compression}"))
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = node_session.request("eval", code=SPEC, arg="x>2")
    assert r.get("ok"), r
    spec = decode(r["result"]) or {}
    got = _profile(node_session)
    regression(_is_wt_d1(got, spec, compression), f"WT-D1: {BOOL_ERROR} ({compression}, spec {spec})")
    ref = _weighted_ref(d.assign(cut=lambda z: (z.x > 2).astype(float)), "cut")
    series = _series(got, "x>2")
    _equal(series["entries"], ref["entries"], "entries")
