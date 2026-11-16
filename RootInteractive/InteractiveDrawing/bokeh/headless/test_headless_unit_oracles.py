"""Unit tests of the oracles (RB-1, RB-2) and of the data generator — hand-computed values, no browser.
Technique: equivalence classes and boundary values."""
import numpy as np
import pytest

from RootInteractive.InteractiveDrawing.bokeh.headless.engine import data
from RootInteractive.InteractiveDrawing.bokeh.headless.engine.scenario import TableSpec
from RootInteractive.InteractiveDrawing.bokeh.headless.oracles import filters, histograms

pytestmark = [pytest.mark.layer("unit"), pytest.mark.backend("python"), pytest.mark.headless_unit]   # target (4.2c-2c)


def test_rb2_explicit_edges_agree_with_numpy_not_old_arithmetic():
    """Existing decimal inputs obey the canonical represented binary64 edges.
    Historical arithmetic put 0.3, 0.6, 0.7 in bins 3,6,7 (INCORRECT)."""
    v = np.array([0.3, 0.6, 0.7])
    edges = histograms.canonical_edges(10, 0.0, 1.0)
    want = [int(np.histogram([x], bins=edges)[0].argmax()) for x in v]
    assert want == [2, 5, 6]
    assert histograms.bin_index_1d(v, 10, 0.0, 1.0).tolist() == want
    assert histograms.bin_index_nd([v], [10], [(0.0, 1.0)]).tolist() == want
    assert int(np.trunc(v[0] * 10)) == 3  # legacy arithmetic is an explicit negative control


def test_rb2_canonical_edges_and_nextafter_exact_membership():
    """Each represented edge belongs on the right; its predecessor on the left."""
    for lo, hi, nb in ((-10.0, -2.0, 50), (0.025, 0.975, 8), (0.1, 0.9, 7), (-1.0, 1.0, 13)):
        edges = histograms.canonical_edges(nb, lo, hi)
        for i in range(1, nb):
            values = [np.nextafter(edges[i], -np.inf), edges[i], np.nextafter(edges[i], np.inf)]
            ref = np.histogram(values, bins=edges)[0]
            got = histograms.counts([values], [nb], [(lo, hi)])
            np.testing.assert_array_equal(got, ref)
            assert histograms.bin_index_1d(values, nb, lo, hi)[1] == i
        assert histograms.bin_index_1d([hi], nb, lo, hi)[0] == nb-1


def test_rb2_independent_rule_rejects_arithmetic_misbin():
    lo, hi, nb = 0.025, 0.975, 8
    values = np.asarray([histograms.canonical_edges(nb, lo, hi)[1]])
    lo_bound, hi_bound, total = histograms.rule_bounds([values], [nb], [(lo, hi)])
    good = histograms.counts([values], [nb], [(lo, hi)])
    assert not histograms.rule_violations(good, lo_bound, hi_bound, total)
    wrong = good.copy()
    wrong[1] -= 1
    wrong[0] += 1
    assert histograms.rule_violations(wrong, lo_bound, hi_bound, total)


def test_rb2_last_edge_inclusive_and_out_of_range_not_counted():
    v = np.array([0.0, 0.05, 0.999, 1.0, -1e-12, 1.0 + 1e-12, -5.0, 7.0])
    assert histograms.bin_index_1d(v, 10, 0.0, 1.0).tolist() == [0, 0, 9, 9, -1, -1, -1, -1]


def test_rb2_non_finite_not_counted():
    v = np.array([np.nan, np.inf, -np.inf, 0.5])
    assert histograms.bin_index_1d(v, 10, 0.0, 1.0).tolist() == [-1, -1, -1, 5]
    assert histograms.bin_index_nd([v, np.zeros(4)], [10, 4], [(0.0, 1.0), (-3.0, 3.0)]).tolist() == [-1, -1, -1, 25]


def test_rb2_nd_flat_index_first_variable_fastest():
    """x = 0.3 belongs to bin 2 under linspace edges; y = 2.9 belongs to bin 3; flat = 2 + 3*10."""
    assert histograms.bin_index_nd([np.array([0.3]), np.array([2.9])], [10, 4], [(0, 1), (-3, 3)]).tolist() == [32]
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
