"""PHASE_5_2 stage 4: vertical lines at the exact quantiles of the selected rows of a histogram (FR-RI-3).

In plain words.
- A 1-D histogram figure with the option "quantile_lines": {"levels": [...], ...} draws a vertical line at each
  level's quantile of the histogram's column over ALL its selected finite rows: not only those inside the histogram's
  range, and not from its bins. The feature request's example: A ~ Gaussian(0.5, 0.4), 5000 rows, A >= 0.2 selected,
  histogram [0, 1] in 10 bins: the median is 0.6038 (dfdraw), not 0.5526 or any other binned value, and 10 % is 0.2899;
  90 % is 1.06, outside the shown range, so no line.
- The estimators are dfdraw's (MathUtils.quantile_of_sorted, shared with the profiles of stage 3): without weights
  NumPy's linear; with weights "inverted_cdf" by default and "linear" as an option. A level asked twice is computed
  once; an empty selection, a negative or non-finite weight, or no positive weight gives no line.
- The drawing: a line only when the value is inside the shown range; the median solid, the other levels dashed, in
  the curve's colour; a legend entry per level ("median", "q=10%"); an optional text at the top ("10%: 0.2899");
  a horizontal line with RootInteractive's flipped histogram axes.
- Stage 4 drew one curve and refused several; since stage 6 a histogram of several curves (a list of weights) has
  the lines of each curve (test_headless_phase5_2_quantile_lines_curves.py). A 2-D histogram, a wrong key, level,
  method or text are refused by name.
- Before the fix (commit 1 of stage 4, the tests only) the option was silently ignored: the page was built without
  any quantile line. Strict expected failures (KnownDefect); since the fix, regressions that name it.
Author: Claude10.RI, 2026-11-22.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_phase5_2_band import ref_quantile


def regression(condition: bool, message: str) -> None:
    """After the fix, the exact signature of a fixed defect is a regression: an ordinary failure that names it."""
    if condition:
        raise AssertionError("REGRESSION, a fixed defect is back: " + message)


UNIT = pytest.mark.headless_unit
LEVELS = [0.1, 0.5, 0.9]


def _node(f):
    """The target, backend and layer marks of a Node test. The feature marks are written out on each test, because
    scripts/generate_capability_matrix.py reads only the literal form @pytest.mark.feature("ID")."""
    for m in reversed([pytest.mark.node_display, pytest.mark.backend("node"), pytest.mark.layer("integration")]):
        f = m(f)
    return f


# ------------------------------------------------------------------ data and pages
def _gauss(n=5000) -> pd.DataFrame:
    """The feature request's example: A ~ Gaussian(0.5, 0.4), seed 1; a weight column wr (zeros included)."""
    from RootInteractive.Tools.pandaTools import initMetadata
    d = pd.DataFrame({"A": np.random.default_rng(1).normal(0.5, 0.4, n)})
    d["wr"] = np.round(np.random.default_rng(2).random(n) * 3, 1)
    initMetadata(d)
    return d


def _hist(nbins=10, range_=(0, 1), **extra):
    return {"name": "hA", "variables": ["A"], "nbins": nbins, "range": list(range_), **extra}


def _figure(**quantile_lines):
    q = {"levels": list(LEVELS), "text": "both"}
    q.update(quantile_lines)
    return [["A"], ["hA"], {"quantile_lines": q}]


def _from_array(df, figures, histograms, widgets=(("range", ["A"]),), params=(), **kw):
    import RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA as sa
    before = sa.show
    sa.show = lambda *a, **k: None
    try:
        return sa.bokehDrawSA.fromArray(df, None, list(figures), [list(w) for w in widgets],
                                        histogramArray=list(histograms), parameterArray=list(params),
                                        layout=[list(range(len(figures)))], widgetLayout=[list(range(len(widgets)))], **kw)
    finally:
        sa.show = before


def _build_error(*a, **kw):
    try:
        _from_array(*a, **kw)
        return None
    except Exception as exc:                                    # noqa: BLE001 - the signature names the exception
        last = traceback.extract_tb(exc.__traceback__)[-1]
        return type(exc).__name__, str(exc), f"{Path(last.filename).name}:{last.name}"


def _has_quantile_lines(page) -> bool:
    from bokeh.models import GlyphRenderer
    return any(isinstance(m, GlyphRenderer) and type(m.glyph).__name__ in ("VSpan", "HSpan")
               for m in page.pAll.references())


def _ignored(df, figures, histograms, **kw) -> bool:
    """The snapshot's signature: the page is built and holds no quantile line."""
    if _build_error(df, figures, histograms, **kw) is not None:
        return False
    return not _has_quantile_lines(_from_array(df, figures, histograms, **kw))


def _page(tmp_path, tag, df, figures, histograms, **kw):
    from bokeh.document import Document
    regression(_ignored(df, figures, histograms, **kw), "FR-RI-3: quantile_lines is ignored")
    doc = Document()
    doc.add_root(_from_array(df, figures, histograms, **kw).pAll)
    out = tmp_path / f"qlines_{tag}.json"
    out.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")
    return out


# ------------------------------------------------------------------ Node helpers
STATE = r"""() => {
  const all = window.__riH.models();
  const q = all.find(m => m.type === 'HistogramQuantiles');
  const col = (c) => Array.from(q.get_column(c) ?? []);
  const r = all.find(m => m.type === 'GlyphRenderer' && (m.glyph.type === 'VSpan' || m.glyph.type === 'HSpan'));
  const x = col('x').map(Number);                     // reading a column computes the values first
  return {values: [...q.values.entries()], x, level: col('level').map(Number), label: col('label'),
          dash: col('dash'), text: col('text'), glyph: r.glyph.type};
}"""

RANGE = r"""(a) => {
  // the shown X range, as a drawn page computes it (Node draws nothing, so its automatic range is never computed)
  const r = window.__riH.models().find(m => m.type === 'HistogramQuantiles').x_range;
  r.start = a[0]; r.end = a[1];
  return true;
}"""

SET = r"""async (a) => {
  const w = window.__riH.models().filter(m => 'value' in m && m.js_property_callbacks && (m.name === a.name || (m.name == null && m.title === a.name)));
  if (w.length !== 1) throw Error('widget identity: ' + a.name + ': ' + w.length);
  w[0].value = a.value;
  await new Promise(resolve => setTimeout(resolve, 0));
  return true;
}"""


def _open(session, path):
    r = session.request("new_doc", path=str(path))
    assert r.get("ok") and not r.get("page_errors") and not r.get("console_errors"), r


def _eval(session, code, arg=None):
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode
    r = session.request("eval", code=code, arg=arg)
    assert r.get("ok") and not r.get("page_errors") and not r.get("console_errors"), r
    return decode(r["result"])


def _values(state):
    return {float(q): v for q, v in state["values"]}


def _close(got, want):
    for q in want:
        if np.isnan(want[q]):
            assert np.isnan(got[q]), (q, got, want)
        else:
            assert got[q] == pytest.approx(want[q], rel=1e-12, abs=1e-14), (q, got, want)


# ------------------------------------------------------------------ Node: the numbers
@_node
@pytest.mark.feature("DISPLAY.quantile_lines")
def test_the_feature_requests_example_all_selected_rows_not_the_bins(node_session, tmp_path):
    """Selected A >= 0.2: 10 % 0.2899 and the median 0.6038 drawn, 90 % (1.06) outside the shown range not drawn;
    the same values whatever the histogram's bins and range (2 or 10 bins, [0, 1] or [-1, 2])."""
    df = _gauss()
    sel = df.A[df.A >= 0.2].to_numpy()
    want = {q: float(np.quantile(sel, q)) for q in LEVELS}
    assert round(want[0.1], 4) == 0.2899 and round(want[0.5], 4) == 0.6038 and round(want[0.9], 2) == 1.06
    for nbins, range_ in ((10, (0, 1)), (2, (0, 1)), (10, (-1, 2))):
        _open(node_session, _page(tmp_path, f"example_{nbins}_{range_[0]}", df, [_figure()], [_hist(nbins, range_)]))
        _eval(node_session, SET, {"name": "A", "value": [0.2, float(df.A.max())]})
        assert _eval(node_session, STATE)["level"] == [], "no line before the shown range is known"
        _eval(node_session, RANGE, list(range_))
        s = _eval(node_session, STATE)
        _close(_values(s), want)
        assert s["level"] == [q for q in LEVELS if range_[0] <= want[q] <= range_[1]], s
        if range_ == (0, 1):
            assert s["level"] == [0.1, 0.5] and s["text"] == ["10%: 0.2899", "median: 0.6038"], s
            assert s["label"] == ["q=10%", "median"] and s["dash"] == ["dashed", "solid"], s


@_node
@pytest.mark.feature("DISPLAY.quantile_lines")
def test_edge_cases_nan_inf_levels_0_and_1_one_row_empty_equal_values_duplicates(node_session, tmp_path):
    """Rows chosen by a slider on the row number s: all rows (NaN and infinite values left out, values outside the
    histogram's range kept), two equal values, one row, only non-finite values (empty: no line); levels 0 and 1 are
    the minimum and the maximum; a level asked twice is computed and drawn once."""
    from RootInteractive.Tools.pandaTools import initMetadata
    a = np.array([0.7, 0.7, 0.3, np.nan, np.inf, -np.inf, 0.1, 5.0, -3.0])
    df = pd.DataFrame({"A": a, "s": np.arange(len(a), dtype=float), "wr": np.ones(len(a))})
    initMetadata(df)
    levels = [0.0, 0.5, 0.5, 1.0, 0.25]
    _open(node_session, _page(tmp_path, "edges", df, [_figure(levels=levels)], [_hist()], widgets=(("range", ["s"]),)))
    _eval(node_session, RANGE, [-10, 10])
    finite = a[np.isfinite(a)]
    s = _eval(node_session, STATE)
    _close(_values(s), {q: float(np.quantile(finite, q)) for q in levels})
    assert _values(s)[0.0] == -3.0 and _values(s)[1.0] == 5.0
    assert len(s["level"]) == len(set(s["level"])) and s["level"].count(0.5) == 1, s
    for rows, want in (([0, 1], 0.7), ([2, 2], 0.3), ([3, 5], float("nan"))):
        _eval(node_session, SET, {"name": "s", "value": rows})
        state = _eval(node_session, STATE)
        got = _values(state)
        assert all((np.isnan(want) and np.isnan(v)) or v == want for v in got.values()), (rows, got)
        assert np.isnan(want) == (state["x"] == []), (rows, state)


@_node
@pytest.mark.feature("DISPLAY.quantile_lines")
@pytest.mark.parametrize("method", ["default", "linear", "inverted_cdf"])
def test_weighted_quantiles_follow_dfdraws_definitions(node_session, tmp_path, method):
    df = _gauss(2000)
    _open(node_session, _page(tmp_path, f"w_{method}", df, [_figure(method=method)], [_hist(weights="wr")]))
    _eval(node_session, SET, {"name": "A", "value": [0.2, float(df.A.max())]})
    sel = df[df.A >= 0.2]
    ref_method = "inverted_cdf" if method == "default" else method
    _close(_values(_eval(node_session, STATE)), {q: ref_quantile(sel.A, sel.wr, q, ref_method) for q in LEVELS})


@_node
@pytest.mark.feature("DISPLAY.quantile_lines")
@pytest.mark.parametrize("bad", [-1.0, float("nan"), "none_positive"])
def test_a_negative_or_non_finite_weight_or_no_positive_weight_draws_no_line(node_session, tmp_path, bad):
    df = _gauss(500)
    if bad == "none_positive":
        df["wr"] = 0.0
    else:
        df.loc[df.index[10], "wr"] = bad
    _open(node_session, _page(tmp_path, f"bad_{bad}", df, [_figure()], [_hist(weights="wr")]))
    s = _eval(node_session, STATE)
    assert all(np.isnan(v) for v in _values(s).values()) and s["x"] == [], s


# ------------------------------------------------------------------ Python: refusals
@UNIT
@pytest.mark.feature("DISPLAY.quantile_lines")
@pytest.mark.parametrize("quantile_lines,named", [
    ({"levels": [0.5, 1.5]}, ["levels", "1.5"]),
    ({"levels": []}, ["levels"]),
    ({"levels": [0.5], "methd": "linear"}, ["'methd'", "'method'"]),
    ({"levels": [0.5], "method": "median"}, ["'median'", "'inverted_cdf'"]),
    ({"levels": [0.5], "text": "all"}, ["'all'", "'both'"])])
def test_a_wrong_option_is_refused_by_name(quantile_lines, named):
    figure = [["A"], ["hA"], {"quantile_lines": quantile_lines}]
    error = _build_error(_gauss(200), [figure], [_hist()])
    regression(error is None and _ignored(_gauss(200), [figure], [_hist()]), "FR-RI-3: quantile_lines is ignored")
    assert error is not None and error[0] == "ValueError" and all(n in error[1] for n in named), error


@UNIT
@pytest.mark.feature("DISPLAY.quantile_lines")
def test_a_2d_histogram_is_refused_by_name():
    """(Stage 4 also refused a histogram of several curves; stage 6 draws their lines, curve by curve.)"""
    df = _gauss(300)
    df["B"] = df.A * 2
    two_d = _build_error(df, [[["A"], ["h2"], {"quantile_lines": {"levels": [0.5]}}]],
                         [{"name": "h2", "variables": ["A", "B"], "nbins": [4, 4], "range": [[0, 1], [0, 2]]}])
    assert two_d is not None and two_d[0] == "ValueError" and "2-D" in two_d[1], two_d


# ------------------------------------------------------------------ Chromium: drawn, followed, flipped
def _browser(headless_session, tmp_path, tag, df, figures, histograms, **kw):
    from bokeh.io import save
    from bokeh.resources import INLINE
    from RootInteractive.InteractiveDrawing.bokeh.headless.test_headless_display import _open_browser_page
    regression(_ignored(df, figures, histograms, **kw), "FR-RI-3: quantile_lines is ignored")
    p = tmp_path / f"qlines_{tag}.html"
    save(_from_array(df, figures, histograms, **kw).pAll, filename=str(p), resources=INLINE, title=tag)
    return _open_browser_page(headless_session, p)


def _red_lines(png: bytes, flip: bool) -> int:
    """The number of separate red lines in the screenshot: runs of pixel columns (rows when flipped) holding at least
    35 % of the red pixels of the fullest one (a dashed line about half of a solid one), so that the texts and the
    legend's samples (red too, but short) do not count."""
    import io
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    red = (a[..., 0] > 180) & (a[..., 1] < 90) & (a[..., 2] < 90)
    counts = red.sum(axis=1 if flip else 0)
    if counts.max() == 0:
        return 0
    long = np.flatnonzero(counts >= 0.35 * counts.max())
    return 1 + int(np.sum(np.diff(long) > 1))


DRAWN = r"""() => {
  const all = window.__riH.models();
  const r = all.find(m => m.type === 'GlyphRenderer' && (m.glyph.type === 'VSpan' || m.glyph.type === 'HSpan'));
  const labels = all.filter(m => m.type === 'LabelSet');
  const items = all.filter(m => m.type === 'LegendItem').flatMap(it => it.renderers.includes(r) ? Array.from(it.get_labels_list_from_label_prop()).map(String) : []);
  const ds = r.data_source;
  return {glyph: r.glyph.type, n: ds.get_length(), x: Array.from(ds.get_column('x')).map(Number),
          text: Array.from(ds.get_column('text')), labels: labels.length, legend: items,
          color: String(r.glyph.line_color.value ?? r.glyph.line_color)};
}"""


@pytest.mark.browser_display
@pytest.mark.feature("DISPLAY.quantile_lines")
@pytest.mark.parametrize("flip", [False, True])
def test_browser_the_lines_texts_legend_follow_the_slider_and_flip(headless_session, tmp_path, flip):
    df = _gauss()
    figure = [["A"], ["hA"], {"quantile_lines": {"levels": list(LEVELS), "text": "both", "color": "red"},
                              "flip_histogram_axes": flip}]
    ctx, page, errors = _browser(headless_session, tmp_path, f"flip_{flip}", df, [figure], [_hist()])
    try:
        before = page.evaluate(DRAWN)
        assert before["glyph"] == ("HSpan" if flip else "VSpan"), before
        assert before["x"] == pytest.approx([float(np.quantile(df.A, 0.5))]) and before["n"] == 1, before
        page.evaluate(SET, {"name": "A", "value": [0.2, float(df.A.max())]})
        page.wait_for_function("() => Bokeh.documents[0].is_idle")
        page.wait_for_timeout(300)
        after = page.evaluate(DRAWN)
        sel = df.A[df.A >= 0.2]
        assert after["x"] == pytest.approx([float(np.quantile(sel, 0.1)), float(np.quantile(sel, 0.5))]), after
        assert after["text"] == ["10%: 0.2899", "median: 0.6038"] and after["labels"] == 1, after
        assert after["legend"] == ["q=10%", "median"] and after["color"] == "red", after
        png = page.locator("canvas").first.screenshot()
        (tmp_path / f"qlines_flip_{flip}.png").write_bytes(png)
        assert _red_lines(png, flip) == 2, "two red lines drawn (10 % and the median)"
        assert not errors, errors
    finally:
        ctx.close()
