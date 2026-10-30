"""Independent Stage-A oracles; RI_MIFOREST_SOURCE optionally selects audited RED bytes."""
import importlib.util
import os
import unittest
import warnings
import sys
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel
from scipy.stats import trim_mean
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

if os.environ.get('RI_MIFOREST_SOURCE'):
    spec = importlib.util.spec_from_file_location('ri_red_candidate', os.environ['RI_MIFOREST_SOURCE'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
else:
    from RootInteractive.MLpipeline import MIForestErrPDF as mod


class FixedTree:
    """Row-dependent predictions chosen independently of implementation reductions."""
    def __init__(self, offset):
        self.offset = offset

    def predict(self, X, **kwargs):
        return X[:, 0].astype(float) + self.offset


class StatisticsOracles(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.X = np.arange(30, dtype=np.float32).reshape(15, 2)
        self.offsets = np.array([9., -3., 1., 5.])
        self.rf = SimpleNamespace(estimators_=[FixedTree(x) for x in self.offsets], verbose=0)
        self.request = {'mean': 0, 'median': 0, 'std': 0,
                        'quantile': [0, .25, .5, 1], 'trim_mean': [0, .25]}

    def check_output(self, out, qkey='quantile', X=None, offsets=None):
        X = self.X if X is None else np.asarray(X)
        offsets = self.offsets if offsets is None else offsets
        expected = offsets[:, None] + X[:, 0]
        refs = {'mean': expected.mean(0), 'median': np.median(expected, axis=0),
                'std': expected.std(0)}
        self.assertEqual(set(out), {'mean', 'median', 'std', qkey, 'trim_mean'})
        for key, ref in refs.items():
            np.testing.assert_allclose(out[key], ref, rtol=1e-12, atol=1e-12)
        for q in self.request['quantile']:
            np.testing.assert_allclose(out[qkey][q], np.quantile(expected, q, axis=0))
        for q in self.request['trim_mean']:
            np.testing.assert_allclose(out['trim_mean'][q], trim_mean(expected, q, axis=0))

    def test_A01_hand_computed_even_and_odd_medians(self):
        for offsets, median in [([9., -3., 1., 5.], 3.), ([9., -3., 1.], 1.)]:
            self.rf.estimators_ = [FixedTree(x) for x in offsets]
            out = mod.predictRFStat(self.rf, self.X, {'median': 0}, 1, max_rows=-1)
            np.testing.assert_array_equal(out['median'], self.X[:, 0] + median)

    def test_A02_chunks_nested_outputs_and_jobs(self):
        for chunk in [1, 4, 100, -1]:
            for jobs in [1, 2, None]:
                with self.subTest(chunk=chunk, jobs=jobs):
                    self.check_output(mod.predictRFStat(self.rf, self.X, self.request, jobs, chunk))
        for jobs in [-1, -2]:
            with self.subTest(jobs=jobs):
                self.check_output(mod.predictRFStat(self.rf, self.X, self.request, jobs, -1))

    def test_A04_all_adapters(self):
        self.check_output(mod.predictRFStatNew(self.rf, self.X, self.request, 2), 'quantiles')
        self.check_output(mod.predictForestStat(self.rf.estimators_, self.X, self.request, 2), 'quantiles')
        with Parallel(n_jobs=2, require='sharedmem') as parallel:
            self.check_output(mod.predictRFStatChunk(self.rf, self.X, self.request, parallel, 2))

    def test_A03_dtypes_frames_noncontiguous_and_empty(self):
        for X in [self.X.astype(np.float64), pd.DataFrame(self.X), self.X[::-2]]:
            with self.subTest(kind=type(X).__name__):
                self.check_output(mod.predictRFStat(self.rf, X, self.request, 1, 3), X=X)
        for fn in [lambda: mod.predictRFStat(self.rf, self.X[:0], self.request, 1),
                   lambda: mod.predictRFStatNew(self.rf, self.X[:0], self.request, 1),
                   lambda: mod.predictForestStat(self.rf.estimators_, self.X[:0], self.request, 1)]:
            out = fn()
            self.assertEqual(out['mean'].shape, (0,))
            self.assertEqual(out.get('quantile', out.get('quantiles'))[.5].shape, (0,))

    def test_A03_invalid_requests(self):
        cases = [({'quantile': [1.1]}, 1, 2), ({'quantile': [-.1]}, 1, 2), ({'quantile': [np.nan]}, 1, 2),
                 ({'trim_mean': [.5]}, 1, 2), ({'bogus': 1}, 1, 2),
                 ({'mean': 1}, 0, 2), ({'mean': 1}, 1, 0)]
        for request, jobs, chunk in cases:
            with self.subTest(request=request, jobs=jobs, chunk=chunk):
                with self.assertRaises((ValueError, TypeError)):
                    mod.predictRFStat(self.rf, self.X, request, jobs, chunk)
        for X in [self.X.ravel(), np.array([[np.nan, 1]]), np.array([[np.inf, 1]])]:
            with self.assertRaises(ValueError):
                mod.predictRFStat(self.rf, X, {'mean': 0}, 1)
        with self.assertRaises(ValueError):
            mod.predictForestStat([], self.X, {'mean': 0}, 1)

    def test_A01_scalar_quantile_and_request_not_mutated(self):
        request = {'quantile': .25}
        out = mod.predictRFStat(self.rf, self.X, request, 1, 2)
        self.assertEqual(request, {'quantile': .25})
        np.testing.assert_allclose(out['quantile'][.25], self.X[:, 0])

    def test_A01_std2_formula_and_chunk_invariance(self):
        for offsets in [self.offsets, self.offsets[:3]]:
            self.rf.estimators_ = [FixedTree(x) for x in offsets]
            np.random.seed(103)
            order = np.random.permutation(len(offsets))
            half = len(offsets) // 2
            ref = np.std(offsets[order[:half]] - offsets[order[half:2*half]])
            for chunk in [1, 4, -1]:
                np.random.seed(103)
                out = mod.predictRFStat(self.rf, self.X, {'std2': 0}, 2, chunk)
                np.testing.assert_allclose(out['std2'], ref)
            arr = offsets[None, :] + self.X[:, :1]
            np.random.seed(103)
            dest = np.empty(len(arr))
            mod._std2(arr, dest)
            np.testing.assert_allclose(dest, ref)

    def test_real_sklearn_forest(self):
        rf = RandomForestRegressor(n_estimators=5, random_state=17).fit(self.X, np.sin(self.X[:, 0]))
        ref = np.array([t.predict(self.X) for t in rf.estimators_])
        out = mod.predictRFStat(rf, self.X.astype(float), self.request, 2, 4)
        np.testing.assert_allclose(out['mean'], rf.predict(self.X))
        np.testing.assert_allclose(out['median'], np.median(ref, axis=0))
        np.testing.assert_allclose(out['quantile'][.25], np.quantile(ref, .25, axis=0))

    def test_named_columns_and_warning_free_prediction(self):
        """Named schemas must be checked before conversion, for all adapters."""
        X = pd.DataFrame({'a': np.arange(15.), 'b': np.arange(15.)[::-1]})
        y = 100 * X.a + X.b
        tree = DecisionTreeRegressor(random_state=2).fit(X, y)
        for rf in [SimpleNamespace(estimators_=[tree], verbose=0),
                   RandomForestRegressor(n_estimators=3, random_state=2).fit(X, y)]:
            reference = rf.estimators_[0].predict(X) if rf.estimators_[0] is tree else rf.predict(X)
            for jobs in [1, 2]:
                adapters = [lambda data: mod.predictRFStat(rf, data, {'mean': []}, jobs, 4),
                            lambda data: mod.predictRFStatNew(rf, data, {'mean': []}, jobs)]
                if rf.estimators_[0] is tree:
                    adapters.append(lambda data: mod.predictForestStat(rf.estimators_, data, {'mean': []}, jobs))
                for adapter in adapters:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        for data in [X, X.to_numpy()]:
                            np.testing.assert_allclose(adapter(data)['mean'], reference)
                    self.assertEqual(caught, [])
                    for bad in [X[['b', 'a']], X.rename(columns={'a': 'wrong'}), X.iloc[:0][['b', 'a']]]:
                        with self.assertRaisesRegex(ValueError, 'feature names'):
                            adapter(bad)
        with Parallel(n_jobs=2, require='sharedmem') as parallel:
            with self.assertRaisesRegex(ValueError, 'feature names'):
                mod.predictRFStatChunk(SimpleNamespace(estimators_=[tree]), X[['b', 'a']],
                                       {'mean': []}, parallel, 2)

    def test_reordered_dataframe_refused(self):
        """Refuse silently wrong feature order even before a no-op/empty prediction."""
        X = pd.DataFrame({'a': [0., 1., 2.], 'b': [2., 1., 0.]})
        tree = DecisionTreeRegressor(random_state=2).fit(X, [0., 100., 200.])
        for data in [X[['b', 'a']], X.iloc[:0][['b', 'a']]]:
            with self.assertRaisesRegex(ValueError, 'feature names'):
                mod.predictForestStat([tree], data, {'mean': []}, 2)

    def test_asymmetric_trimmed_mean_oracle(self):
        """Explicit sorted-member sums distinguish trimming from mean and wrong fractions."""
        offsets = [-100., -2., 1., 4., 7., 1000.]
        rf = SimpleNamespace(estimators_=[FixedTree(x) for x in offsets], verbose=0)
        for jobs in [1, 2]:
            out = mod.predictRFStat(rf, self.X, {'trim_mean': [0, .2, .34]}, jobs, 4)
            for fraction, expected in [(0, 910./6), (.2, 2.5), (.34, 2.5)]:
                np.testing.assert_allclose(out['trim_mean'][fraction], self.X[:, 0].astype(float) + expected,
                                           rtol=1e-12, atol=1e-12)
        # A second asymmetric sample distinguishes the two nonzero trim fractions.
        rf.estimators_ = [FixedTree(x) for x in [-100., -20., 1., 4., 7., 1000.]]
        out = mod.predictRFStat(rf, self.X, {'trim_mean': [.2, .34]}, 1, 4)
        np.testing.assert_allclose(out['trim_mean'][.2], self.X[:, 0] - 2.)
        np.testing.assert_allclose(out['trim_mean'][.34], self.X[:, 0] + 2.5)

    def test_bounded_prediction_chunks(self):
        sizes = []

        class RecordingTree:
            def predict(self, X):
                sizes.append(len(X))
                return X[:, 0]

        X = np.ones((10003, 2), dtype=np.float32)
        rf = SimpleNamespace(estimators_=[RecordingTree()], verbose=0)
        out = mod.predictRFStat(rf, X, {'mean': 0}, 1)
        self.assertLessEqual(max(sizes), 10000)
        self.assertEqual(sum(sizes), len(X))
        np.testing.assert_array_equal(out['mean'], np.ones(len(X)))

    def test_prediction_errors_are_not_swallowed(self):
        class BrokenTree:
            def predict(self, X):
                raise RuntimeError('oracle member failure')

        with self.assertRaisesRegex(RuntimeError, 'oracle member failure'):
            mod.predictForestStat([BrokenTree()], self.X, {'mean': 0}, 2)
        with self.assertRaises(ValueError):
            mod.predictRFStat(SimpleNamespace(estimators_=[FixedTree(0)], verbose=0),
                              self.X, {'std2': 0}, 1)

    def test_augmented_consumer_actual_module(self):
        try:
            from RootInteractive.MLpipeline.augmentedForest import AugmentedRandomForestArray
        except ModuleNotFoundError as exc:
            if exc.name in ('xgboost', 'RootInteractive.MLpipeline.augmentedForest'):
                self.skipTest('actual augmentedForest consumer unavailable: ' + str(exc))
            raise
        consumer_module = sys.modules[AugmentedRandomForestArray.__module__]
        # Route the consumer through the same selected source during RED checks.
        routing = patch.object(consumer_module, 'predictRFStat', mod.predictRFStat)
        routing.start()
        self.addCleanup(routing.stop)
        model = AugmentedRandomForestArray(2, 2, 1, 3)
        model.fit(self.X, np.sin(self.X[:, 0]), np.zeros(2))
        ref = np.array([t.predict(self.X) for t in model.estimators_])
        np.testing.assert_allclose(model.predict(self.X), ref.mean(0))
        out = model.predictRFStat(self.X.astype(np.float64), self.request, max_rows=4)
        np.testing.assert_allclose(out['median'], np.median(ref, axis=0))
        np.testing.assert_allclose(out['quantile'][.25], np.quantile(ref, .25, axis=0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
