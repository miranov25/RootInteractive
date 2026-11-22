"""PHASE_5_2 stage 3: the filled band between two columns (FR-RI-2), and dfdraw's quantile estimators for profiles.

In plain words.
- Profile quantiles. A profile's quantile columns (quantile_0, ...) were computed only with RootInteractive's present
  method (DEC-10, "subbin": the level rescaled inside the one sub-bin where the cumulative count passes q n). They
  differ from NumPy's linear quantile and change with the sub-binning (review of the proposal, Claude3 R1). The
  Architect's rule of 2026-11-13: methods that differ become options; RootInteractive's present method stays the
  default. The histogram option "quantile_method" now chooses:
  - "subbin": the present method, the default (existing pages and tests unchanged);
  - "linear": over all the rows of a cell, NumPy's linear without weights; with weights dfdraw's weighted linear,
    h = q (W - w1) + w1 over the cumulative weights;
  - "inverted_cdf": the smallest value whose cumulative weight share reaches q (NumPy 2's weighted inverted_cdf,
    dfdraw's default with weights);
  - a weight of 0 carries no mass, a negative weight gives NaN, a cell without positive weight NaN.
  It needs unbinned projections (it reads the rows); a wrong method is refused by name.
- The band. "visualization_type": "band" fills the area between the Y column (the lower edge) and the column of the
  option "upper", at the X of the rows in row order, with the order and filter rules of the line (stage 1). One band
  per weight series. A NaN in either edge breaks it: the band is drawn as polygons (BandPolygons and Bokeh's Patches),
  because Bokeh 3.3.2's VArea joins the neighbours of a NaN (measured). "band_style" gives the colour, the alpha (0.25 by default, as
  dfdraw) and a legend label. visualization_type, line_style and band_style may be lists, one per Y column, so one
  figure holds nested bands (outer first, drawn in the order given) and a line on top.
- Before the fix (commit 1 of stage 3, the tests only): the band failed with NotImplementedError naming "band"; quantile_method
  was silently ignored (the quantiles equal the page without it). Strict expected failures (KnownDefect); since the
  fix, regressions that name them.
Author: Claude10.RI, 2026-11-22.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_phase5_2_lines import (
    NBX, MIN_ENTRIES, _build_error, _close, _from_array, _glyphs, _node, _open, _page, _set, _table,
    _validation_errors, _eval)


def regression(condition: bool, message: str) -> None:
    """After the fix, the exact signature of a fixed defect is a regression: an ordinary failure that names it."""
    if condition:
        raise AssertionError("REGRESSION, a fixed defect is back: " + message)


UNIT = pytest.mark.headless_unit
LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


MISSING_BAND = ("Visualization type not supported: band", "Visualization type not supported: ['band'")


def _is_missing_band(error) -> bool:
    """FR-RI-2 at the snapshot (stage 1b), exact: the NotImplementedError of bokehDrawArray names the type "band", alone
    or first in a list of types per Y column."""
    return error is not None and error[0] == "NotImplementedError" and error[1].startswith(MISSING_BAND) and \
        error[2] == "bokehTools.py:bokehDrawArray"


def _band_or_defect(error):
    regression(_is_missing_band(error), "FR-RI-2: NotImplementedError 'Visualization type not supported: band'")
    assert error is None, error


def _hist(weights=None, method="linear", nbins_b=4, **extra):
    h = {"name": "hp", "variables": ["A", "B"], "nbins": [NBX, nbins_b], "range": [[0, NBX], [0, 4]],
         "unbinned_projections": True, "quantiles": list(LEVELS)}
    if method is not None:
        h["quantile_method"] = method
    if weights is not None:
        h["weights"] = weights
    h.update(extra)
    return h


# ------------------------------------------------------------------ the reference estimators (NumPy and dfdraw)
def _pairs(values, weights):
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        return None
    keep = np.isfinite(v) & (w > 0)
    order = np.argsort(v[keep], kind="stable")
    return v[keep][order], w[keep][order]


def ref_quantile(values, weights, q, method):
    """dfdraw's definitions: NumPy linear (no weights); weighted linear h = q (W - w1) + w1; NumPy 2's weighted
    inverted_cdf (cumulative sum divided by the total, first index with share >= q)."""
    p = _pairs(values, weights)
    if p is None or len(p[0]) == 0 or not 0 <= q <= 1:
        return float("nan")
    xs, ws = p
    if method == "linear" and (weights is None or np.all(ws == 1)):
        return float(np.quantile(xs, q, method="linear"))
    cdf = np.cumsum(ws)
    if method == "inverted_cdf":
        return float(xs[np.searchsorted(cdf / cdf[-1], q, side="left")])
    h = q * (cdf[-1] - ws[0]) + ws[0]
    k = int(np.searchsorted(cdf, h, side="left"))
    if k == 0:
        return float(xs[0])
    return float(xs[k - 1] + (h - cdf[k - 1]) / (cdf[k] - cdf[k - 1]) * (xs[k] - xs[k - 1]))


@UNIT
def test_control_the_reference_estimators():
    """NumPy linear; unit weights give NumPy linear; inverted_cdf at a boundary takes the value whose share reaches q
    (Claude3 R2's example: values [-2.134, 0.9, 0.699], weights [0, 2, 2], q = 0.5 -> 0.699); empty and negative NaN."""
    x = [5., 1., 3., 2., 4.]
    assert ref_quantile(x, None, .25, "linear") == 2.0 and ref_quantile(x, [1] * 5, .3, "linear") == pytest.approx(2.2)
    assert ref_quantile([-2.134, 0.9, 0.699], [0, 2, 2], .5, "inverted_cdf") == 0.699
    assert ref_quantile([1., 2., 3.], [1., 1., 2.], .5, "inverted_cdf") == 2.0
    assert ref_quantile([1., 2., 3.], [1., 1., 2.], .5, "linear") == pytest.approx(2.25)   # h = .5*(4-1)+1 = 2.5
    assert np.isnan(ref_quantile([], None, .5, "linear")) and np.isnan(ref_quantile([1., 2.], [1, -1], .5, "linear"))


# ------------------------------------------------------------------ Node: the profile's quantile methods
PROFILE = r"""(a) => {
  const p = window.__riH.models().find(m => m.type === 'HistoNdProfile' && m.name === a.name);
  const out = {};
  for (const c of a.columns) out[c] = Array.from(p.get_column(c) ?? []).map(Number);
  return out;
}"""


def _profile_columns(session, name="hp_1", columns=("bin_center_0", "entries", "weights")):
    cols = list(columns) + [f"quantile_{k}" for k in range(len(LEVELS))]
    r = _eval(session, PROFILE, {"name": name, "columns": [c for c in cols if c != "weights"]})
    return r


def _cells(df, weight_column=None, lo=0.0, hi=float(NBX)):
    """Per A bin: (B values, weights) of the rows with lo <= A <= hi and B inside [0, 4)."""
    rows = df[(df.A >= lo) & (df.A <= hi) & (df.B >= 0) & (df.B < 4)]
    out = []
    for k in range(NBX):
        cell = rows[np.floor(rows.A) == k]
        out.append((cell.B.to_numpy(), None if weight_column is None else cell[weight_column].to_numpy()))
    return out


def _weighted_table():
    """_table() with a real weight column wr: positive, some zeros, ties of B values in bin 1."""
    d = _table()
    rng = np.random.default_rng(11)
    d["wr"] = np.round(rng.random(len(d)) * 3, 1)
    d.loc[d.index[::7], "wr"] = 0.0
    ties = d.index[(np.floor(d.A) == 1)][:6]
    d.loc[ties, "B"] = 2.0
    return d


@_node
@pytest.mark.feature("HIST.profile_unbinned")
@pytest.mark.parametrize("nbins_b", [2, 20, 200])
def test_linear_profile_quantiles_equal_numpy_for_any_sub_binning(node_session, tmp_path, nbins_b):
    """quantile_method "linear": every cell's quantiles equal NumPy's linear quantile of the cell's rows, for 2, 20 and
    200 sub-bins of B (the present method changes with the sub-binning); after the slider cuts A too."""
    df = _table()
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1"}]
    _open(node_session, _page(tmp_path, f"qlin_{nbins_b}", df, [figure], histograms=[_hist(nbins_b=nbins_b)]))
    for lo, hi in ((None, None), (1.5, 4.5)):
        if lo is not None:
            _set(node_session, "A", [lo, hi])
        lo_, hi_ = (0.0, float(NBX)) if lo is None else (lo, hi)
        got = _profile_columns(node_session)
        want = [[ref_quantile(v, None, q, "linear") for q in LEVELS] for v, _ in _cells(df, None, lo_, hi_)]
        got_cells = [[got[f"quantile_{j}"][k] for j in range(len(LEVELS))] for k in range(NBX)]
        if lo is None:
            present = _present_quantiles(node_session, tmp_path, df, nbins_b)
            regression(np.allclose(got_cells, present, equal_nan=True) and not np.allclose(got_cells, want, equal_nan=True),
                         "quantile_method 'linear' is ignored: the quantiles are the present method's")
        np.testing.assert_allclose(got_cells, want, rtol=1e-12, atol=1e-12, equal_nan=True)


def _present_quantiles(node_session, tmp_path, df, nbins_b):
    """The same page without quantile_method (the present method), read in the same session, then the page again."""
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1"}]
    _open(node_session, _page(tmp_path, f"qsub_{nbins_b}", df, [figure], histograms=[_hist(method=None, nbins_b=nbins_b)]))
    got = _profile_columns(node_session)
    out = [[got[f"quantile_{j}"][k] for j in range(len(LEVELS))] for k in range(NBX)]
    _open(node_session, _page(tmp_path, f"qlin_{nbins_b}_again", df, [figure], histograms=[_hist(nbins_b=nbins_b)]))
    return out


@_node
@pytest.mark.feature("HIST.profile_unbinned")
@pytest.mark.parametrize("method", ["linear", "inverted_cdf"])
def test_weighted_profile_quantiles_follow_dfdraws_definitions(node_session, tmp_path, method):
    """A real weight column (zeros and ties included): each cell's quantiles equal dfdraw's definition."""
    df = _weighted_table()
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1"}]
    _open(node_session, _page(tmp_path, f"qw_{method}", df, [figure], widgets=(),
                              histograms=[_hist(weights="wr", method=method, nbins_b=20)]))
    got = _profile_columns(node_session)
    want = [[ref_quantile(v, w, q, method) for q in LEVELS] for v, w in _cells(df, "wr")]
    got_cells = [[got[f"quantile_{j}"][k] for j in range(len(LEVELS))] for k in range(NBX)]
    _open(node_session, _page(tmp_path, f"qw_{method}_present", df, [figure], widgets=(),
                              histograms=[_hist(weights="wr", method=None, nbins_b=20)]))
    present = _profile_columns(node_session)
    present_cells = [[present[f"quantile_{j}"][k] for j in range(len(LEVELS))] for k in range(NBX)]
    regression(np.allclose(got_cells, present_cells, equal_nan=True) and not np.allclose(got_cells, want, equal_nan=True),
                 f"quantile_method {method!r} is ignored: the quantiles are the present method's")
    np.testing.assert_allclose(got_cells, want, rtol=1e-12, atol=1e-12, equal_nan=True)


@_node
@pytest.mark.feature("HIST.profile_unbinned")
def test_a_negative_weight_gives_nan_and_a_cell_without_weight_gives_nan(node_session, tmp_path):
    df = _weighted_table()
    df.loc[df.index[np.floor(df.A) == 2][:1], "wr"] = -1.0            # bin 2: one negative weight
    df.loc[np.floor(df.A) == 4, "wr"] = 0.0                            # bin 4: no positive weight
    figure = [["bin_center_0"], ["mean"], {"source": "hp_1"}]
    _open(node_session, _page(tmp_path, "qneg", df, [figure], widgets=(),
                              histograms=[_hist(weights="wr", method="inverted_cdf")]))
    got = _profile_columns(node_session)
    cells = [[got[f"quantile_{j}"][k] for j in range(len(LEVELS))] for k in range(NBX)]
    regression(not all(np.isnan(cells[2])) and not np.isnan(cells[0][0]) and all(np.isfinite(cells[2])),
                 "quantile_method is ignored: a negative weight still gives numbers")
    assert all(np.isnan(cells[2])) and all(np.isnan(cells[4])), cells
    assert all(np.isfinite(cells[k]).all() for k in (0, 1, 3, 5)), cells


@UNIT
@pytest.mark.feature("HIST.profile_unbinned")
@pytest.mark.parametrize("hist,named", [(_hist(method="median"), ["'median'", "'subbin'", "'linear'", "'inverted_cdf'"]),
                                        (_hist(unbinned_projections=False), ["'linear'", "unbinned_projections"])])
def test_a_wrong_quantile_method_is_refused_by_name(hist, named):
    error = _build_error(_table(), [[["bin_center_0"], ["mean"], {"source": "hp_1"}]], widgets=(), histograms=[hist])
    regression(error is None, "quantile_method is ignored: a wrong value builds a page")
    assert error is not None and error[0] == "ValueError" and all(n in error[1] for n in named), error


# ------------------------------------------------------------------ Python: the band when the page is built
NESTED = [["bin_center_0"], ["quantile_0", "quantile_1", "quantile_2"],
          {"source": "hp_1", "filter": "entries >= 3", "upper": ["quantile_4", "quantile_3"],
           "visualization_type": ["band", "band", "line"],
           "band_style": [{"alpha": 0.125}, {"alpha": 0.25}], "line_style": {"color": "black", "width": 2}}]


@UNIT
@pytest.mark.feature("DISPLAY.band")
@pytest.mark.parametrize("weighted", [False, True])
def test_nested_bands_and_a_line_on_one_figure_are_built_and_valid(weighted):
    hist = _hist(weights="weights" if weighted else None)
    error = _build_error(_table(), [NESTED], histograms=[hist], weighted=weighted)
    _band_or_defect(error)
    page = _from_array(_table(), [NESTED], histograms=[hist], weighted=weighted)
    n = 3 if weighted else 1
    assert _glyphs(page) == sorted(["Patches"] * 2 * n + ["Line"] * n), _glyphs(page)
    assert _validation_errors(page) == []


@UNIT
@pytest.mark.feature("DISPLAY.band")
@pytest.mark.parametrize("options,named", [
    ({}, ["upper", "lower"]),
    ({"upper": "quantile_4", "band_style": {"alhpa": 0.3}}, ["band_style", "'alhpa'", "'alpha'"]),
    ({"upper": "quantile_4", "band_style": {"alpha": 1.5}}, ["band_style", "alpha", "1.5"])])
def test_a_wrong_band_is_refused_by_name(options, named):
    figure = [["bin_center_0"], ["quantile_0"], {"source": "hp_1", "visualization_type": "band", **options}]
    error = _build_error(_table(), [figure], histograms=[_hist()])
    regression(_is_missing_band(error), "FR-RI-2: NotImplementedError 'Visualization type not supported: band'")
    assert error is not None and error[0] == "ValueError" and all(n in error[1] for n in named), error


@UNIT
@pytest.mark.feature("DISPLAY.band")
def test_a_band_on_a_table_is_refused_by_name():
    error = _build_error(_table(), [[["A"], ["B"], {"visualization_type": "band", "upper": "B"}]], widgets=())
    regression(_is_missing_band(error), "FR-RI-2: NotImplementedError 'Visualization type not supported: band'")
    assert error is not None and error[0] == "ValueError" and "band" in error[1] and "projection" in error[1], error


# ------------------------------------------------------------------ Node: the band's edges, series, order
STATE = r"""() => {
  const all = window.__riH.models();
  const scalar = (c) => (c != null && typeof c === 'object' && 'value' in c) ? c.value : c;
  const nums = (ds, spec) => spec == null || spec.field == null ? null : Array.from(ds.get_column(spec.field) ?? []).map(Number);
  const strs = (ds, f) => { try { const c = ds.get_column(f); return c == null ? null : Array.from(c).map(String); } catch (e) { return null; } };
  const plot = all.find(m => m.type === 'Figure' || m.type === 'Plot');
  return plot.renderers.filter(r => r.type === 'GlyphRenderer').map(r => {
    const g = r.glyph, ds = r.data_source;
    const rows = g.type === 'Patches' ? ds.source : ds;
    const out = {glyph: g.type, visible: r.visible, x: nums(ds, g.x), weights: strs(rows, 'weights')};
    if (g.type === 'Patches') Object.assign(out, {xs: Array.from(ds.get_column('xs')).map(a => Array.from(a).map(Number)),
                                                   ys: Array.from(ds.get_column('ys')).map(a => Array.from(a).map(Number)),
                                                   color: String(scalar(g.fill_color)), alpha: Number(scalar(g.fill_alpha))});
    if (g.type === 'Line') Object.assign(out, {y: nums(ds, g.y), color: String(scalar(g.line_color))});
    return out;
  });
}"""


def _edges(r):
    """A band's polygons back to (x, lower, upper) per point, in order: each polygon is the upper edge forward and the
    lower edge backward."""
    x, lower, upper = [], [], []
    for xs, ys in zip(r["xs"], r["ys"]):
        k = len(xs) // 2
        assert xs[:k] == xs[k:][::-1], xs
        x += xs[:k]
        upper += ys[:k]
        lower += ys[k:][::-1]
    return x, lower, upper


def _oracle_edges(df, lo, hi, weight=None, levels=(0, 4)):
    """Per A bin with at least MIN_ENTRIES rows: (centre, NumPy linear at the two levels) of the selected rows."""
    out = []
    for k, (v, _) in enumerate(_cells(df[df[weight] == 1] if weight else df, None, lo, hi)):
        if len(v) >= MIN_ENTRIES:
            out.append((k + 0.5, [float(np.quantile(v, LEVELS[j], method="linear")) for j in levels]))
    return out


@_node
@pytest.mark.feature("DISPLAY.band")
@pytest.mark.feature("HIST.profile_unbinned")
@pytest.mark.parametrize("weighted", [False, True])
def test_the_band_edges_equal_the_quantile_columns_and_numpy_for_each_series(node_session, tmp_path, weighted):
    """Nested bands (10-90 % outer, 25-75 % inner) and the median line: in the order given, the outer band first and
    the line last; their edges equal NumPy's linear quantiles of each series' selected rows, before and after the
    slider cuts A and (weighted) after the selection changes; each series only its own bins."""
    df = _table()
    hist = _hist(weights="weights" if weighted else None)
    error = _build_error(df, [NESTED], histograms=[hist], weighted=weighted)
    _band_or_defect(error)
    _open(node_session, _page(tmp_path, f"band_{weighted}", df, [NESTED], histograms=[hist], weighted=weighted))
    steps = [((0.0, float(NBX)), ["w0", "w1"]), ((1.5, 4.5), ["w0", "w1"]), ((1.5, 4.5), ["w2", "w0"])]
    for (lo, hi), selection in steps:
        if (lo, hi) != (0.0, float(NBX)):
            _set(node_session, "A", [lo, hi])
        if weighted and selection != ["w0", "w1"]:
            _set(node_session, "weights", selection)
        rs = [r for r in _eval(node_session, STATE) if r["visible"]]
        kinds = [r["glyph"] for r in rs]
        n = len(selection) if weighted else 1
        assert kinds == ["Patches"] * n + ["Patches"] * n + ["Line"] * n, kinds      # outer bands, inner bands, lines
        for r in rs:
            w = r["weights"][0] if weighted else None
            assert not weighted or set(r["weights"]) == {w}, r["weights"]
            if r["glyph"] == "Patches":
                levels = (0, 4) if rs.index(r) < n else (1, 3)
                want = _oracle_edges(df, lo, hi, w, levels)
                x, lower, upper = _edges(r)
                assert len(r["xs"]) == 1, r["xs"]                                    # no NaN: one piece
                _close(list(zip(x, lower)), [(x_, e[0]) for x_, e in want])
                _close(list(zip(x, upper)), [(x_, e[1]) for x_, e in want])
            else:
                want = _oracle_edges(df, lo, hi, w, (2, 2))
                _close(list(zip(r["x"], r["y"])), [(x, e[0]) for x, e in want])
        alphas = [r["alpha"] for r in rs if r["glyph"] == "Patches"]
        assert alphas == [0.125] * n + [0.25] * n, alphas


@_node
@pytest.mark.feature("DISPLAY.band")
def test_a_band_keeps_the_nan_of_an_empty_bin(node_session, tmp_path):
    """Without a filter: w0's band is two pieces, bins 0 to 2 and bins 4 to 5 (its bin 3 is empty, NaN)."""
    df = _table()
    figure = [["bin_center_0"], ["quantile_0"], {"source": "hp_1", "visualization_type": "band", "upper": "quantile_4"}]
    _band_or_defect(_build_error(df, [figure], histograms=[_hist(weights="weights")], weighted=True))
    _open(node_session, _page(tmp_path, "band_nan", df, [figure], histograms=[_hist(weights="weights")], weighted=True))
    w0 = [r for r in _eval(node_session, STATE) if r["visible"] and r["weights"] and r["weights"][0] == "w0"][0]
    assert [xs[:len(xs) // 2] for xs in w0["xs"]] == [[0.5, 1.5, 2.5], [4.5, 5.5]], w0["xs"]
    assert all(np.isfinite(y).all() for y in w0["ys"]), w0["ys"]


# ------------------------------------------------------------------ Chromium: drawn, broken at NaN, line on top
@pytest.mark.browser_display
@pytest.mark.feature("DISPLAY.band")
def test_browser_a_band_breaks_at_nan_and_the_line_is_drawn_on_top(headless_session, tmp_path):
    """Only w0, no filter: a solid green band with NaN in bin 3 is drawn in two pieces, with no green over the NaN bin
    (from 2.5 to 4.5); with the filter "entries >= 3" one piece without a gap. A black median line drawn after it stays black inside the band."""
    import io
    from PIL import Image
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_phase5_2_lines import _browser
    df = _table()
    for tag, extra, pieces in (("nan", {}, 2), ("joined", {"filter": "entries >= 3"}, 1)):
        figure = [["bin_center_0"], ["quantile_0", "quantile_2"],
                  {"source": "hp_1", "upper": "quantile_4", "visualization_type": ["band", "line"],
                   "band_style": {"color": "#00ff00", "alpha": 1.0}, "line_style": {"color": "black", "width": 3},
                   **extra}]
        try:
            ctx, page, errors = _browser(headless_session, tmp_path, f"band_{tag}", df, figure, "band",
                                         weighted=True, histograms=[_hist(weights="weights")])
        except AssertionError as exc:
            regression(any(m in str(exc) for m in MISSING_BAND), "FR-RI-2: the band page cannot be built")
            raise
        try:
            page.evaluate("""() => { const w = window.__riH.models().find(m => 'value' in m && m.js_property_callbacks &&
                                       (m.name === 'weights' || (m.name == null && m.title === 'weights'))); w.value = ['w0'];
                                     for (const m of window.__riH.models()) if (m.type === 'Legend') m.visible = false; }""")
            page.wait_for_function("() => Bokeh.documents[0].is_idle")
            page.wait_for_timeout(200)
            png = page.locator("canvas").first.screenshot()
            (tmp_path / f"band_{tag}.png").write_bytes(png)
            a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
            green = (a[..., 1] > 200) & (a[..., 0] < 60) & (a[..., 2] < 60)
            black = (a.sum(axis=2) < 120)
            cols = np.flatnonzero(green.any(axis=0))
            span = cols[-1] - cols[0] if len(cols) else 0
            gap = int(np.diff(cols).max()) - 1 if len(cols) > 1 else 0          # the widest run of columns without green
            # two pieces: a gap of the NaN bin and its two half-bins (2 of the 5 bin steps of the span); one piece: none
            if pieces == 2:
                assert gap > 0.3 * span, (tag, gap, span)
            else:
                assert gap < 0.02 * span, (tag, gap, span)
            assert green.sum() > 500, (tag, int(green.sum()))
            # the line on top: black pixels with green neighbours above and below, inside the band's columns
            inside = black[:, cols].any(axis=0).sum() if len(cols) else 0
            assert inside > 0.5 * len(cols), (tag, int(inside), len(cols))
            assert not errors, errors
        finally:
            ctx.close()
