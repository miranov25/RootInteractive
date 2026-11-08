"""Unit tests of the oracles (RB-1, RB-2) and of the data generator — hand-computed values, no browser.
Technique: equivalence classes and boundary values."""
import numpy as np
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import data
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import TableSpec
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import filters, histograms

pytestmark = [pytest.mark.layer("unit"), pytest.mark.backend("python"), pytest.mark.headless_unit]   # target (4.2c-2c)


def test_rb2_bin_formula_is_rootinteractive_not_numpy():
    """0.3, 0.6, 0.7 on [0, 1] x 10: RootInteractive's (v*scale + origin)|0 gives 3, 6, 7; numpy gives 2, 5, 6."""
    v = np.array([0.3, 0.6, 0.7])
    assert histograms.bin_index_1d(v, 10, 0.0, 1.0).tolist() == [3, 6, 7]
    numpy_bins = [int(np.histogram([x], bins=10, range=(0, 1))[0].argmax()) for x in v]
    assert numpy_bins == [2, 5, 6]                       # the reason the oracle must not call numpy.histogram


def test_rb2_last_edge_inclusive_and_out_of_range_not_counted():
    v = np.array([0.0, 0.05, 0.999, 1.0, -1e-12, 1.0 + 1e-12, -5.0, 7.0])
    assert histograms.bin_index_1d(v, 10, 0.0, 1.0).tolist() == [0, 0, 9, 9, -1, -1, -1, -1]


def test_rb2_non_finite_not_counted():
    v = np.array([np.nan, np.inf, -np.inf, 0.5])
    assert histograms.bin_index_1d(v, 10, 0.0, 1.0).tolist() == [-1, -1, -1, 5]
    assert histograms.bin_index_nd([v, np.zeros(4)], [10, 4], [(0.0, 1.0), (-3.0, 3.0)]).tolist() == [-1, -1, -1, 25]


def test_rb2_nd_flat_index_first_variable_fastest():
    """x = 0.3 -> bin 3 of 10; y = 2.9 on [-3, 3] x 4: 2.9*(4/6) - (-3*(4/6)) = 3.93 -> bin 3; flat 3 + 3*10 = 33."""
    assert histograms.bin_index_nd([np.array([0.3]), np.array([2.9])], [10, 4], [(0, 1), (-3, 3)]).tolist() == [33]
    # one axis out of range drops the row
    assert histograms.bin_index_nd([np.array([0.3]), np.array([3.1])], [10, 4], [(0, 1), (-3, 3)]).tolist() == [-1]


def test_rb2_1d_and_nd_agree_per_axis():
    rng = np.random.default_rng(0)
    v = rng.uniform(-0.2, 1.2, 50_000)
    v[::97] = np.round(v[::97], 1)                        # many values exactly on bin edges
    one = histograms.bin_index_1d(v, 10, 0.0, 1.0)
    nd = histograms.bin_index_nd([v], [10], [(0.0, 1.0)])
    assert np.array_equal(one, nd)


def test_counts_with_mask_and_weights():
    x = np.array([0.05, 0.15, 0.15, 0.95, np.nan])
    assert histograms.counts([x], [10], [(0, 1)]).tolist() == [1, 2, 0, 0, 0, 0, 0, 0, 0, 1]
    assert histograms.counts([x], [10], [(0, 1)], mask=[True, False, True, True, True]).tolist()[:2] == [1, 1]
    w = np.array([2.0, 0.5, 0.25, 1.0, 9.0])
    assert histograms.counts([x], [10], [(0, 1)], weights=w).tolist()[:2] == [2.0, 0.75]


def test_rb1_range_inclusive_and_nan_excluded():
    v = np.array([-1.0, 1.0, -1.0 - 1e-12, 0.0, np.nan, np.inf])
    assert filters.range_mask(v, -1.0, 1.0).tolist() == [True, True, False, True, False, False]
    assert filters.intersection([[True, False, True], [True, True, False]], 3).tolist() == [True, False, False]
    assert filters.intersection([], 2).tolist() == [True, True]


def test_data_generation_is_seeded_and_columns_independent():
    a = data.generate(TableSpec(1000, 5, {"x": "uniform(0,1)", "y": "normal(0,1) + 10% NaN"}))
    b = data.generate(TableSpec(1000, 5, {"y": "normal(0,1) + 10% NaN", "z": "uniform(2,3)", "x": "uniform(0,1)"}))
    assert np.array_equal(a.x.to_numpy(), b.x.to_numpy())
    assert np.array_equal(a.y.to_numpy(), b.y.to_numpy(), equal_nan=True)
    assert 50 < int(a.y.isna().sum()) < 150
    with pytest.raises(ValueError, match="not understood"):
        data.parse_spec("gauss(0,1)")


def test_case_file_round_trip(tmp_path):
    df = data.generate(TableSpec(100, 1, {"x": "normal(0,1) + 20% NaN"}))
    sha = data.save_case(df, tmp_path / "case.npz")
    assert sha == data.content_sha256(df) == data.content_sha256(data.load_case(tmp_path / "case.npz"))
    assert np.array_equal(data.load_case(tmp_path / "case.npz").x.to_numpy(), df.x.to_numpy(), equal_nan=True)


# ---------------------------------------------------------------------------------------- stage 4.2c-2a
def test_edge_value_column_kind():
    from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import TableSpec
    x = data.generate(TableSpec(4000, 3, {"x": "edges(-2,2.28,6)"})).x.to_numpy()
    lo, hi = -2.0, 2.28
    edges = lo + (hi - lo) * np.arange(7) / 6                           # as the generator computes them
    edges[-1] = hi
    assert (x == 2.28).sum() > 0 and (x == np.nextafter(2.28, -np.inf)).sum() > 0 and np.isinf(x).sum() > 0
    assert 800 < np.isin(x, edges).sum() < 1200                            # about a quarter on exact edges
    with pytest.raises(ValueError):
        data.parse_spec("edges(0,1)")                                      # nbins is required
    with pytest.raises(ValueError):
        data.parse_spec("uniform(0,1,5)")                                  # only edges takes nbins


def test_b1_chain_start():
    from RootInteractive.InteractiveDrawing.bokeh.headless.oracles.filters import b1_first_changed, b1_selection
    n = 64
    full, some = np.ones(n, bool), np.ones(n, bool)
    some[5] = False
    assert b1_first_changed([full, some]) == 1                            # n % 32 == 0: the first excluding filter
    assert b1_first_changed([np.ones(65, bool), np.ones(65, bool)]) == 0  # n % 32 != 0: the first filter always
    assert b1_first_changed([full, full]) == -1                           # nothing changes: no B1
    now = [full, np.ones(n, bool)]                                        # b's widget moved to keep everything
    assert b1_selection([some, now[1]], [some, some]).tolist() == some.tolist()   # first filter itself applies
