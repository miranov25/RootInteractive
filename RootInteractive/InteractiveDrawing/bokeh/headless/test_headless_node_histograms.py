"""Histograms and profiles in Node — PHASE_4_5 stage A2 (basic). Oracle and invariant tests only.

Plain words first:
  * a HISTOGRAM counts rows (or sums their weights) in bins: HistogramCDS (1-D), HistoNdCDS (N-D);
  * a PROFILE summarises an N-D histogram slice by slice along one axis: entries, mean, standard deviation and
    quantiles of the other variable in each slice (HistoNdProfile). BINNED statistics use the bin centres and counts;
    UNBINNED statistics (`unbinned_projections`) use the rows' raw values;
  * a range FILTER (range widget) selects rows; the histograms follow the selection.
The production models are built by RootInteractive's own page builder (engine/page.py, or `bokehDrawSA.fromArray`
directly where the scenario language has no option for it), loaded in the Node harness, and read back.

Two sources:
  * tests 1-10 (the counts, filter, binned profile, binned quantile and empty-slice tests and the hand-calculated
    reference) were contributed by GPT24.RI as a candidate (PHASE_4_5_Histogram_SafetyNet_Candidate_GPT24_RI_20261110.py,
    MD5 c29d3ded…) and are integrated here with the same data and oracles. Changes on integration: counts compared
    exactly (they are integers or sums of integer weights), capability features added, the binned profile also for
    the weighted histogram, a filter case with both limits ON row values (the range filter is inclusive), output in
    the test's own temporary folder;
  * the unbinned-profile tests (B4, B5, B5b) are new: the bug report BUG_RootInteractive_20261106 lists these three
    defects, but the suite had no test for them.

Every input value of the counting tests lies inside a bin, never on an edge: the independent numpy oracle then needs
no knowledge of RootInteractive's edge arithmetic (edges are covered by test_headless_node_families.py: B8, B10).
A known-defect check passes only if production equals the predicted wrong result exactly AND the prediction differs
from the oracle; then it raises `KnownDefect`, the only failure the strict expected-failure tests accept.

Findings (ids of the bug report; reproduced here in Node, exact signatures):
  B4   an unbinned WEIGHTED profile is NaN (mean and standard deviation) when the slice's first row (in the model's
       order) has weight 0: HistoNdProfile.ts divides by the running weight sum, 0 at that row.
  B5   an unbinned quantile that falls in a bin with exactly TWO rows is (a + b) * q instead of a + (b - a) * q
       (MathUtils.ts `quantile`); it can lie outside the bin's values (19.5 for the values 12 and 14).
  B5b  an unbinned quantile at the lower edge of its bin's share returns the bin's first row in row order, not the
       bin's smallest value: the result depends on the row order.
Question (not a test here): q = 1 gave NaN, binned and unbinned, not the largest value. Fixed in PHASE_5 Batch 2B
(q = 1 is the largest value) and tested in test_headless_batch2b_quantiles.py; the reference below is never asked q = 1.
PHASE_5 Batches 2A and 2B fixed B4 (as WT-D2), B5 and B5b: their signature tests are removed, the strict tests are
ordinary tests, and the exact signature of each defect is a REGRESSION check (an ordinary failure that names it).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import page as page_mod
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.runner import SET_JS, SETTLE_ROUNDS
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import (
    LOSSLESS, Labels, Page, Scenario, TableSpec, Tables, auto, hist, load, range_)

NODE = pytest.mark.node_histograms          # target (Node model tests)
UNIT = pytest.mark.headless_unit            # target (Python only)


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
XEDGES = np.array([0., 1., 2., 3.])
YEDGES = np.array([0., 1., 2., 3., 4.])


# ===================================================================================== data and oracles (GPT24)
def _table():
    """Hand-auditable 3 x 4 fixture; no value is close to a bin edge."""
    return pd.DataFrame({
        "x": [.25, .25, .25, 1.25, 1.25, 1.25, 2.25, 2.25, 2.25, .25, 1.25, 2.25, .25, 1.25],
        "y": [.25, 1.25, 2.25, .25, 1.25, 3.25, 1.25, 2.25, 3.25, 3.25, 2.25, .25, .25, 2.25],
        "a": [-1.5, -1., -.5, .25, .5, .75, 1., 1.25, 1.5, -.25, -.75, 1.75, -1.25, 1.6],
        "w": [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.],
    })


def _reference(df, cut=None):
    """Independent numpy histograms; the flat N-D layout runs x fastest. `cut`: rows with lo <= a <= hi."""
    m = np.ones(len(df), dtype=bool)
    if cut is not None:
        lo, hi = cut
        m &= (df.a.to_numpy() >= lo) & (df.a.to_numpy() <= hi)
    sel = df.loc[m]
    xy = np.histogramdd(sel[["x", "y"]].to_numpy(), bins=[XEDGES, YEDGES])[0]
    xyw = np.histogramdd(sel[["x", "y"]].to_numpy(), bins=[XEDGES, YEDGES], weights=sel.w)[0]
    return {"hx": np.histogram(sel.x, bins=XEDGES)[0], "hy": np.histogram(sel.y, bins=YEDGES)[0],
            "hxy": xy.ravel(order="F"), "hxyw": xyw.ravel(order="F"), "matrix": xy, "matrix_w": xyw}


def _profile_reference(df, cut=None, probs=(), weighted=False):
    """BINNED profile along y per x slice: each y bin is its centre with its count (or summed weight); population
    moments; quantiles by linear interpolation of the cumulative count inside the bin that holds the share."""
    counts = _reference(df, cut)["matrix_w" if weighted else "matrix"]
    centers = 0.5 * (YEDGES[:-1] + YEDGES[1:])
    out = {"entries": [], "mean": [], "std": [], "isOK": []}
    for p in probs:
        out[f"q{p}"] = []
    for row in counts:
        n = float(row.sum())
        out["entries"].append(n)
        out["isOK"].append(n > 0)
        if not n:
            out["mean"].append(np.nan)
            out["std"].append(np.nan)
            for p in probs:
                out[f"q{p}"].append(np.nan)
            continue
        mu = float(np.dot(row, centers) / n)
        out["mean"].append(mu)
        out["std"].append(np.sqrt(float(np.dot(row, (centers - mu) ** 2) / n)))
        cumulative = np.cumsum(row)
        for p in probs:
            if not 0 <= p < 1:
                out[f"q{p}"].append(np.nan)
                continue
            target = p * n
            j = int(np.searchsorted(cumulative, target, side="right"))
            if j == len(row):
                out[f"q{p}"].append(np.nan)
                continue
            left = float(cumulative[j - 1]) if j else 0.
            out[f"q{p}"].append(YEDGES[j] + (target - left) / row[j] * (YEDGES[j + 1] - YEDGES[j]))
    return out


def _scenario(with_widget=False, profiles=()):
    models = [hist("hx", ["x"], 3, [0, 3]), hist("hy", ["y"], 4, [0, 4]),
              hist("hxy", ["x", "y"], [3, 4], [[0, 3], [0, 4]]),
              hist("hxyw", ["x", "y"], [3, 4], [[0, 3], [0, 4]], weights="w")]
    return Scenario(
        id="histograms/phase45_a2_weighted_projection_safety_net",
        data=Tables(TableSpec(14, 24, {"x": "uniform(0,3)", "y": "uniform(0,4)",
                                       "a": "uniform(-2,2)", "w": "uniform(1,12)"})),
        page=Page(histograms=tuple(models), widgets=(range_("a", (-2, 2)),) if with_widget else (),
                  profiles=tuple(profiles)),
        steps=(load(),), checks=(auto(),),
        labels=Labels(family="histograms", models=("HistogramCDS", "HistoNdCDS"),
                      verified=("HistogramCDS", "HistoNdCDS"),
                      features=("HIST.histogram_1d", "HIST.histogram_nd"),
                      technique="independent numpy oracle; conservation; selected-state sequence", priority="p0"),
    )


# ===================================================================================== harness
def _document(scenario, df, tmp_path: Path, models_only=()):
    return page_mod.build_document(scenario, df, tmp_path / "histogram_document.json", models_only=models_only)


def _node_open(node_session, path):
    r = node_session.request("new_doc", path=str(path))
    assert r.get("ok"), r
    assert not r.get("page_errors") and not r.get("console_errors"), r


def _node_eval(node_session, expr, arg=None):
    r = node_session.request("eval", code=expr, arg=arg)
    assert r.get("ok") and not r.get("page_errors") and not r.get("console_errors"), r
    return decode(r["result"])


def _read_histograms(node_session):
    return _node_eval(node_session, """() => ({
      hx: window.__riH.col('hx','bin_count',['HistogramCDS']),
      hy: window.__riH.col('hy','bin_count',['HistogramCDS']),
      hxy: window.__riH.col('hxy','bin_count',['HistoNdCDS']),
      hxyw: window.__riH.col('hxyw','bin_count',['HistoNdCDS'])
    })""")


PROFILE_JS = """(a) => { const o = {};
  for (const k of a.keys) o[k] = window.__riH.col(a.name, k, ['HistoNdProfile']);
  o.__model = window.__riH.model(a.name, ['HistoNdProfile']) ? 'found' : window.__riH.why(a.name, ['HistoNdProfile']);
  return o; }"""


def _read_profile(node_session, name, keys):
    out = _node_eval(node_session, PROFILE_JS, {"name": name, "keys": list(keys)})
    assert out.pop("__model") == "found", out
    for k in keys:
        assert out[k] is not None, f"missing profile column {name}.{k}"
    return out


def _assert_histograms(actual, expected):
    """Every bin exact, and the 2-D histogram's projections equal the 1-D histograms (conservation)."""
    assert set(actual) == {"hx", "hy", "hxy", "hxyw"}
    for key in ("hx", "hy", "hxy", "hxyw"):
        assert actual[key] is not None, f"missing production histogram model/column: {key}"
        np.testing.assert_array_equal(np.asarray(actual[key], dtype=float), np.asarray(expected[key], dtype=float), err_msg=key)
    hxy = np.asarray(actual["hxy"]).reshape((3, 4), order="F")
    np.testing.assert_array_equal(hxy.sum(axis=1), actual["hx"])
    np.testing.assert_array_equal(hxy.sum(axis=0), actual["hy"])
    assert hxy.sum() == np.asarray(actual["hx"]).sum()


def _profile_page(df, tmp_path: Path, **declaration) -> Path:
    """A real user-level page (`bokehDrawSA.fromArray`) with one 2-D histogram `h` of (x, y) declared in sourceArray
    — the scenario language has no quantile or unbinned option — and its profile along y, `h_1`. Only the profile
    and what it references go into the document."""
    from bokeh.document import Document
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    original = sa.show
    sa.show = lambda *args, **kwargs: None
    try:
        obj = sa.bokehDrawSA.fromArray(
            df, None, [[['bin_center_0'], ['mean'], {'source': 'h_1'}]], [],
            sourceArray=[{'name': 'h', 'variables': ['x', 'y'], **declaration}],
            layout=[[0]], widgetLayout=[[]],
            arrayCompression=[(pat, list(actions)) for pat, actions in LOSSLESS])
    finally:
        sa.show = original
    matching = [m for m in obj.pAll.references() if m.name == 'h_1' and type(m).__name__ == 'HistoNdProfile']
    assert len(matching) == 1, f'expected exactly one HistoNdProfile h_1, got {len(matching)}'
    document = Document()
    document.add_root(matching[0])
    path = tmp_path / 'profile_document.json'
    path.write_text(json.dumps(document.to_json(deferred=False)), encoding='utf-8')
    return path


def _profile(node_session, tmp_path, df, keys, **declaration):
    _node_open(node_session, _profile_page(df, tmp_path, **declaration))
    return _read_profile(node_session, "h_1", keys)


# ===================================================================================== 1. reference (GPT24)
@UNIT
def test_reference_is_hand_calculated_and_detects_faults():
    """The oracles on the hand-calculated table, and three kinds of wrong result the comparison must reject."""
    df = _table()
    ref = _reference(df)
    assert ref["hx"].tolist() == [5, 5, 4]
    assert ref["hy"].tolist() == [4, 3, 4, 3]
    assert ref["matrix"].tolist() == [[2, 1, 1, 1], [1, 1, 2, 1], [1, 1, 1, 1]]
    assert ref["matrix_w"].sum() == 105
    assert ref["matrix_w"][0, 3] == 10
    prof = _profile_reference(df, probs=(.25, .5, .75))
    assert prof["entries"] == [5., 5., 4.]
    np.testing.assert_allclose(prof["mean"], [1.7, 2.1, 2.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(prof["q0.25"], [.625, 1.25, 1.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(prof["q0.5"], [1.5, 2.25, 2.], rtol=0, atol=1e-12)
    good = {k: np.array(ref[k]) for k in ("hx", "hy", "hxy", "hxyw")}
    shifted = dict(ref); shifted["hxy"] = np.roll(ref["hxy"], 1)
    with pytest.raises(AssertionError):
        _assert_histograms(good, shifted)
    missing = dict(ref); missing["hx"] = ref["hx"] - np.array([1, 0, 0])
    with pytest.raises(AssertionError):
        _assert_histograms(good, missing)
    wrong_weights = dict(ref); wrong_weights["hxyw"] = ref["hxyw"] + 1
    with pytest.raises(AssertionError):
        _assert_histograms(good, wrong_weights)


# ===================================================================================== 2-4. counts (GPT24)
@NODE
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.parametrize("case", ["baseline", "permuted", "duplicated"])
def test_histogram_counts_weights_and_projections_equal_numpy(case, node_session, tmp_path):
    """1-D, 2-D and weighted 2-D histograms equal numpy; invariants: a row permutation changes nothing, duplicating
    every row doubles every bin; the 2-D histogram's projections are the 1-D histograms."""
    df = _table()
    if case == "permuted":
        df = df.iloc[[13, 11, 3, 9, 1, 6, 2, 8, 5, 0, 12, 10, 7, 4]].reset_index(drop=True)
    elif case == "duplicated":
        df = pd.concat([df, df], ignore_index=True)
    path = _document(_scenario(), df, tmp_path, models_only=("hx", "hy", "hxy", "hxyw"))
    _node_open(node_session, path)
    _assert_histograms(_read_histograms(node_session), _reference(df))


# ===================================================================================== 5-7 (+1). filter (GPT24)
@NODE
@pytest.mark.feature("FILTER.range")
@pytest.mark.feature("HIST.histogram_1d")
@pytest.mark.feature("HIST.histogram_nd")
@pytest.mark.parametrize("cut", [pytest.param((-.99, .01), id="negative-side"),
                                 pytest.param((0., 1.), id="upper-limit-on-a-value"),
                                 pytest.param((-.25, .5), id="both-limits-on-values"),
                                 pytest.param((1.8, 1.9), id="empty")])
def test_histograms_follow_a_range_filter_and_its_restoration(cut, node_session, tmp_path):
    """A real range widget moved in Node: the histograms equal numpy on the selected rows (limits inclusive; an empty
    selection gives empty histograms), and equal the full table again after the widget is reset."""
    df = _table()
    path = _document(_scenario(with_widget=True), df, tmp_path)
    _node_open(node_session, path)
    _assert_histograms(_read_histograms(node_session), _reference(df))
    probes = [[name, "bin_count", ["HistogramCDS" if name in ("hx", "hy") else "HistoNdCDS"]]
              for name in ("hx", "hy", "hxy", "hxyw")]
    for value, expected in ((cut, _reference(df, cut)), ("full", _reference(df))):
        r = node_session.request("eval", code=SET_JS, arg=["a", list(value) if value != "full" else value, probes, SETTLE_ROUNDS])
        assert r.get("ok") and not r.get("page_errors") and not r.get("console_errors"), r
        result = decode(r["result"])
        assert result.get("settled") is True and not result.get("error"), result
        _assert_histograms(_read_histograms(node_session), expected)


# ===================================================================================== 8. binned profile (GPT24)
@NODE
@pytest.mark.feature("HIST.profile")
@pytest.mark.parametrize("name", ["hxy_1", "hxyw_1"])
def test_binned_profile_uses_bin_moments_not_raw_values(name, node_session, tmp_path):
    """Entries, mean and standard deviation per x slice equal the binned formula (weighted for hxyw) — and differ
    from the raw-value mean, so the check cannot pass with the other convention."""
    df = _table()
    weighted = name == "hxyw_1"
    path = _document(_scenario(profiles=(name,)), df, tmp_path, models_only=(name,))
    _node_open(node_session, path)
    result = _read_profile(node_session, name, ("entries", "mean", "std", "isOK"))
    oracle = _profile_reference(df, weighted=weighted)
    for key in ("entries", "mean", "std"):
        np.testing.assert_allclose(result[key], oracle[key], rtol=1e-13, atol=1e-13, err_msg=key)
    assert result["isOK"] == oracle["isOK"]
    raw_mean = df.groupby(pd.cut(df.x, XEDGES, include_lowest=True), observed=True).y.mean().to_numpy()
    assert not np.allclose(raw_mean, oracle["mean"])


# ===================================================================================== 9. binned quantiles (GPT24)
@NODE
@pytest.mark.feature("HIST.profile")
def test_binned_profile_quantiles_interpolate_the_cumulative_count(node_session, tmp_path):
    """Quantiles 0.25, 0.5, 0.75 per slice equal the interpolation of the cumulative bin count — not numpy's quantile
    of the raw values, which is a different convention (checked to differ)."""
    df = _table()
    got = _profile(node_session, tmp_path, df, ("entries", "quantile_0", "quantile_1", "quantile_2"),
                   nbins=[3, 4], range=[[0, 3], [0, 4]], quantiles=[.25, .5, .75])
    ref = _profile_reference(df, probs=(.25, .5, .75))
    np.testing.assert_array_equal(got['entries'], ref['entries'])
    for i, q in enumerate((.25, .5, .75)):
        np.testing.assert_allclose(got[f'quantile_{i}'], ref[f'q{q}'], rtol=0, atol=1e-12, err_msg=f"q={q}")
    assert np.quantile(df[df.x == .25].y, .25) != ref['q0.25'][0]


# ===================================================================================== 10. empty slice (GPT24)
@NODE
@pytest.mark.feature("HIST.profile")
def test_empty_profile_slice_reports_no_valid_statistic(node_session, tmp_path):
    """A slice without rows: entries 0, isOK false, NaN mean and standard deviation; the other slices stay valid."""
    df = _table().loc[lambda d: d.x < 2].reset_index(drop=True)
    path = _document(_scenario(profiles=("hxy_1",)), df, tmp_path, models_only=("hxy_1",))
    _node_open(node_session, path)
    actual = _read_profile(node_session, "hxy_1", ("entries", "mean", "std", "isOK"))
    ref = _profile_reference(df)
    np.testing.assert_array_equal(actual['entries'], ref['entries'])
    assert actual['isOK'] == [True, True, False]
    np.testing.assert_allclose(actual['mean'], ref['mean'], equal_nan=True)
    np.testing.assert_allclose(actual['std'], ref['std'], equal_nan=True)


# ===================================================================================== unbinned profiles (new)
# One x slice; y in two bins [0, 10) and [10, 20]. Unbinned statistics use the raw y values of the slice.
Y5 = [1., 2., 3., 12., 14.]
UNBINNED = dict(nbins=[1, 2], range=[[0, 1], [0, 20]], unbinned_projections=True)


def _one_slice(y, w=None):
    d = {"x": [.5] * len(y), "y": list(y)}
    if w is not None:
        d["w"] = list(w)
    return pd.DataFrame(d)


def _subbin_quantile(y, q):
    """RootInteractive's unbinned quantile, as specified by its design: find the y bin that holds the q-th share of the
    slice's rows, then take numpy's linear quantile of that bin's raw values at the local share."""
    y = np.asarray(y, dtype=float)
    bins = [np.sort(y[(y >= lo) & (y < hi if hi < 20 else y <= hi)]) for lo, hi in ((0, 10), (10, 20))]
    n, below = len(y), 0
    for values in bins:
        if below + len(values) > q * n:
            return float(np.quantile(values, (q * n - below) / len(values)))
        below += len(values)
    return float("nan")


@NODE
@pytest.mark.feature("HIST.profile_unbinned")
def test_unbinned_profile_moments_equal_numpy(node_session, tmp_path):
    """Unweighted and weighted (all weights positive) unbinned mean and standard deviation per x slice equal numpy on
    the raw values; GPT24's table, three slices."""
    df = _table()
    decl = dict(nbins=[3, 4], range=[[0, 3], [0, 4]], unbinned_projections=True)
    plain = _profile(node_session, tmp_path, df, ("entries", "mean", "std"), **decl)
    weighted = _profile(node_session, tmp_path, df, ("entries", "mean", "std"), weights="w", **decl)
    groups = [g for _, g in df.groupby(pd.cut(df.x, XEDGES), observed=True)]
    np.testing.assert_array_equal(plain["entries"], [len(g) for g in groups])
    np.testing.assert_allclose(plain["mean"], [g.y.mean() for g in groups], rtol=1e-13)
    np.testing.assert_allclose(plain["std"], [g.y.std(ddof=0) for g in groups], rtol=1e-12)
    np.testing.assert_array_equal(weighted["entries"], [g.w.sum() for g in groups])
    means = [np.average(g.y, weights=g.w) for g in groups]
    np.testing.assert_allclose(weighted["mean"], means, rtol=1e-13)
    np.testing.assert_allclose(weighted["std"], [np.sqrt(np.average((g.y - m) ** 2, weights=g.w)) for g, m in zip(groups, means)],
                               rtol=1e-12)


@NODE
@pytest.mark.feature("HIST.profile_unbinned")
def test_unbinned_quantiles_in_a_bin_of_three_rows(node_session, tmp_path):
    """Healthy: quantiles that fall in the bin with three rows equal the sub-bin definition (1.333…, 2.666…)."""
    got = _profile(node_session, tmp_path, _one_slice(Y5), ("quantile_0", "quantile_1"), quantiles=[.1, .5], **UNBINNED)
    np.testing.assert_allclose([got["quantile_0"][0], got["quantile_1"][0]],
                               [_subbin_quantile(Y5, .1), _subbin_quantile(Y5, .5)], rtol=1e-13)


# --------------------------------------------------------------------------- B4 zero weight first
B4_W = [0., 1., 1., 1., 1.]          # the first row of the slice (y = 1, bin 0) has weight 0


def _weighted_moments(y, w):
    y, w = np.asarray(y), np.asarray(w)
    m = np.average(y, weights=w)
    return m, np.sqrt(np.average((y - m) ** 2, weights=w))


@NODE
@pytest.mark.feature("HIST.profile_unbinned")
def test_strict_zero_weight_rows_do_not_spoil_the_weighted_moments(node_session, tmp_path):
    got = _profile(node_session, tmp_path, _one_slice(Y5, B4_W), ("entries", "mean", "std"), weights="w", **UNBINNED)
    m, s = _weighted_moments(Y5, B4_W)
    # A2 v0.2 (review F5): the correct result needs the right entries (the summed weights) as well as mean and std
    if got["entries"] == [float(sum(B4_W))] and np.allclose([got["mean"][0], got["std"][0]], [m, s], rtol=1e-13):
        return
    regression(got["entries"] == [4.0] and np.isnan(got["mean"][0]) and np.isnan(got["std"][0]), "B4 (WT-D2)")
    raise AssertionError(f"B4: neither numpy nor the B4 signature: {got}")


@NODE
@pytest.mark.feature("HIST.profile_unbinned")
def test_control_b4_a_zero_weight_after_the_first_row_is_harmless(node_session, tmp_path):
    """Control: the same values with the zero weight on the SECOND row give the correct weighted moments."""
    w = [1., 0., 1., 1., 1.]
    got = _profile(node_session, tmp_path, _one_slice(Y5, w), ("entries", "mean", "std"), weights="w", **UNBINNED)
    np.testing.assert_allclose([got["mean"][0], got["std"][0]], _weighted_moments(Y5, w), rtol=1e-13)


# --------------------------------------------------------------------------- B5 two rows in the quantile's bin
B5_Q = (.7, .9)                      # both fall in bin 1, which holds the two rows 12 and 14


def _b5_prediction(q):
    """MathUtils.ts `quantile` for two values: (a + b) * q_local."""
    local = (q * 5 - 3) / 2
    return (12. + 14.) * local


@NODE
@pytest.mark.feature("HIST.profile_unbinned")
def test_strict_a_quantile_lies_among_its_bin_values_and_equals_the_definition(node_session, tmp_path):
    got = _profile(node_session, tmp_path, _one_slice(Y5), ("quantile_0", "quantile_1"), quantiles=list(B5_Q), **UNBINNED)
    values = [got["quantile_0"][0], got["quantile_1"][0]]
    if all(12. <= v <= 14. for v in values) and np.allclose(values, [_subbin_quantile(Y5, q) for q in B5_Q], rtol=1e-13):
        return
    regression(values == [_b5_prediction(q) for q in B5_Q], "B5")
    raise AssertionError(f"B5: neither the definition nor the B5 signature: {values}")


# --------------------------------------------------------------------------- B5b row order
B5B_ORDERS = ([1., 2., 15., 12., 18.], [1., 2., 12., 15., 18.])   # the same values; bin 1 holds 15, 12, 18
B5B_Q = .4                            # 0.4 * 5 = 2 rows: exactly the lower edge of bin 1's share (local share 0)


def _b5b(node_session, tmp_path):
    out = []
    for order in B5B_ORDERS:
        got = _profile(node_session, tmp_path, _one_slice(order), ("quantile_0",), quantiles=[B5B_Q], **UNBINNED)
        out.append(got["quantile_0"][0])
    return out


@NODE
@pytest.mark.feature("HIST.profile_unbinned")
def test_strict_a_quantile_does_not_depend_on_the_row_order(node_session, tmp_path):
    got = _b5b(node_session, tmp_path)
    if got == [12., 12.]:
        return
    regression(got == [15., 12.], "B5b")
    raise AssertionError(f"B5b: neither the definition nor the B5b signature: {got}")


# ===================================================================================== the harness itself
@UNIT
def test_control_the_sub_bin_quantile_reference():
    """The unbinned reference on hand-calculated values: bin 0 = 1, 2, 3 and bin 1 = 12, 14 (five rows)."""
    assert _subbin_quantile(Y5, .1) == pytest.approx(1 + 2 * (0.5 / 3))      # local share 1/6 of [1, 2, 3]
    assert _subbin_quantile(Y5, .7) == 12.5 and _subbin_quantile(Y5, .9) == 13.5
    assert _subbin_quantile(Y5, 1.0) != _subbin_quantile(Y5, 1.0)            # q = 1: outside this reference (NaN)
    assert _subbin_quantile([1., 2., 15., 12., 18.], .4) == 12.
