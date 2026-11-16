"""RI-HIST-BINEDGE-20261115-01: strict independent edge tests of *production* JS models.

Run RED against installed old HistogramCDS/HistoNdCDS, then GREEN against
candidate TypeScript patch.  No tolerance, no reuse of the RI bin-index oracle.
Source of expected membership: represented numpy.linspace edges, numpy.histogram
and numpy.histogramdd.  Both implementations read the same canonical range.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from bokeh.document import Document
from bokeh.models import ColumnDataSource

from RootInteractive.InteractiveDrawing.bokeh.HistogramCDS import HistogramCDS
from RootInteractive.InteractiveDrawing.bokeh.HistoNdCDS import HistoNdCDS
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.node_session import decode

# The test group of this file (added by Claude21.RI, 2026-11-16; nothing else is changed): every test here belongs
# to the histogram group of the Node tests, so that the run of the test groups selects it.
pytestmark = pytest.mark.node_histograms


GEOMETRIES = [
    pytest.param(-10.0, -2.0, 50, id="negative_offset_50_bins"),
    pytest.param(0.025, 0.975, 8, id="grouped_dashboard_8_bins"),
    pytest.param(0.0, 1.0, 4, id="exact_binary_edges"),
    pytest.param(-1.0, 1.0, 13, id="cross_zero_13_bins"),
    pytest.param(0.1, 0.9, 7, id="nonbinary_7_bins"),
]


def _values(edges):
    """Every internal represented edge and the floats one ULP to either side."""
    x = [edges[0], edges[-1], np.nextafter(edges[0], -np.inf),
         np.nextafter(edges[-1], np.inf)]
    for e in edges[1:-1]:
        x.extend([np.nextafter(e, -np.inf), e, np.nextafter(e, np.inf)])
    return np.asarray(x, dtype=np.float64)


def _data(values):
    # Vary the second-axis bin and weights to detect N-D stride and weighted mistakes.
    n = len(values)
    return {
        "x": list(map(float, values)),
        "y": [0.25 if i % 3 else 1.25 for i in range(n)],
        "w": [float(1 + (i * 7) % 11) for i in range(n)],
    }


def _production_document(data, lo, hi, nbins, path):
    src = ColumnDataSource(data=data, name="edge_source")
    models = [
        HistogramCDS(source=src, sample="x", nbins=nbins,
                     range=[lo, hi], name="edge_hist_1d"),
        HistogramCDS(source=src, sample="x", nbins=nbins,
                     range=[lo, hi], weights="w", name="edge_hist_weighted_1d"),
        HistoNdCDS(source=src, sample_variables=["x"], nbins=[nbins],
                   range=[[lo, hi]], name="edge_hist_nd_1axis"),
        HistoNdCDS(source=src, sample_variables=["x"], nbins=[nbins],
                   range=[[lo, hi]], weights="w", name="edge_hist_weighted_nd_1axis"),
        HistoNdCDS(source=src, sample_variables=["x", "y"], nbins=[nbins, 2],
                   range=[[lo, hi], [0.0, 2.0]], name="edge_hist_nd_2axes"),
        HistoNdCDS(source=src, sample_variables=["x", "y"], nbins=[nbins, 2],
                   range=[[lo, hi], [0.0, 2.0]], weights="w", name="edge_hist_weighted_nd_2axes"),
    ]
    doc = Document()
    for m in models:
        doc.add_root(m)
    path.write_text(json.dumps(doc.to_json(deferred=False)), encoding="utf-8")


def _observed(node_session, path):
    reply = node_session.request("new_doc", path=str(path))
    assert reply.get("ok"), reply
    assert not reply.get("page_errors") and not reply.get("console_errors"), reply
    js = """() => {
      const q = (name, kind) => window.__riH.col(name, 'bin_count', [kind]);
      return {
        one: q('edge_hist_1d', 'HistogramCDS'),
        one_w: q('edge_hist_weighted_1d', 'HistogramCDS'),
        nd_one: q('edge_hist_nd_1axis', 'HistoNdCDS'),
        nd_one_w: q('edge_hist_weighted_nd_1axis', 'HistoNdCDS'),
        nd_two: q('edge_hist_nd_2axes', 'HistoNdCDS'),
        nd_two_w: q('edge_hist_weighted_nd_2axes', 'HistoNdCDS'),
        bottom_1d: window.__riH.col('edge_hist_1d', 'bin_bottom', ['HistogramCDS']),
        top_1d: window.__riH.col('edge_hist_1d', 'bin_top', ['HistogramCDS']),
        bottom_nd: window.__riH.col('edge_hist_nd_1axis', 'bin_bottom_0', ['HistoNdCDS']),
        top_nd: window.__riH.col('edge_hist_nd_1axis', 'bin_top_0', ['HistoNdCDS'])
      };
    }"""
    reply = node_session.request("eval", code=js)
    assert reply.get("ok"), reply
    assert not reply.get("page_errors") and not reply.get("console_errors"), reply
    result = decode(reply["result"])
    assert all(v is not None for v in result.values()), result
    return result


def _expected(data, edges):
    x = np.asarray(data['x'], dtype=float)
    y = np.asarray(data['y'], dtype=float)
    w = np.asarray(data['w'], dtype=float)
    one = np.histogram(x, bins=edges)[0]
    one_w = np.histogram(x, bins=edges, weights=w)[0]
    xy = np.column_stack((x, y))
    nd_two = np.histogramdd(xy, bins=(edges, [0.0, 1.0, 2.0]))[0].ravel(order='F')
    nd_two_w = np.histogramdd(xy, bins=(edges, [0.0, 1.0, 2.0]), weights=w)[0].ravel(order='F')
    return dict(one=one, one_w=one_w, nd_one=one, nd_one_w=one_w,
                nd_two=nd_two, nd_two_w=nd_two_w,
                bottom_1d=edges[:-1], top_1d=edges[1:],
                bottom_nd=edges[:-1], top_nd=edges[1:])


@pytest.mark.parametrize("lo,hi,nbins", GEOMETRIES)
def test_strict_internal_bin_edges_production_models(node_session, tmp_path, lo, hi, nbins):
    """RED on old product for negative/offset examples; must pass only when fixed."""
    edges = np.linspace(lo, hi, nbins + 1, dtype=np.float64)
    values = _values(edges)
    data = _data(values)
    path = tmp_path / "edge_document.json"
    _production_document(data, lo, hi, nbins, path)
    observed = _observed(node_session, path)
    expected = _expected(data, edges)
    for name, truth in expected.items():
        np.testing.assert_array_equal(
            np.asarray(observed[name], dtype=float), truth,
            err_msg=f"{name} range=({lo},{hi}) nbins={nbins}: strict represented-edge contract"
        )


def test_oracle_sensitive_to_one_wrong_bin():
    """Deliberately wrong mutation must fail strict oracle: no tolerant comparator."""
    edges = np.linspace(0.025, 0.975, 9)
    x = np.array([edges[1], edges[7]])
    expected = np.histogram(x, bins=edges)[0]
    wrong = expected.copy()
    wrong[1] -= 1
    wrong[0] += 1
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(wrong, expected)
    assert expected.sum() == wrong.sum() == 2


def test_nonfinite_and_overflow_production_models(node_session, tmp_path):
    """NaN, +-inf, and adjacent out-of-range floats remain excluded."""
    lo, hi, nbins = -10.0, -2.0, 50
    edges = np.linspace(lo, hi, nbins + 1)
    values = np.array([lo, hi, np.nan, np.inf, -np.inf,
                       np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)])
    data = _data(values)
    path = tmp_path / "exceptional_document.json"
    _production_document(data, lo, hi, nbins, path)
    observed = _observed(node_session, path)
    expected = _expected(data, edges)
    for name, truth in expected.items():
        np.testing.assert_array_equal(np.asarray(observed[name], dtype=float), truth,
                                      err_msg=f"{name}: nonfinite/overflow policy")
