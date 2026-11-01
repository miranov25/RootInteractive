"""Stage B deterministic contracts; calibration is a separate Stage D task."""
import importlib.util
import itertools
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor

if os.environ.get('RI_MIFOREST_SOURCE'):
    spec = importlib.util.spec_from_file_location('ri_stage_b_candidate', os.environ['RI_MIFOREST_SOURCE'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
else:
    from RootInteractive.MLpipeline import MIForestErrPDF as mod


def model(**options):
    opts = dict(min_tree=6, mean_depth=10, niter_max=1, n_jobs=1,
                random_state=42, verbose=0)
    opts.update(options)
    return mod.MIForestErrPDF('Regressor', opts, min_samples_leaf=2)


class ReducibleContracts(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack([np.linspace(-1, 1, 97), np.sin(np.arange(97))])
        self.y = self.X[:, 0] + .15 * self.X[:, 1]
        self.addCleanup(np.random.set_state, np.random.get_state())

    def test_B02_coverage_counts_remainders_and_defaults(self):
        for n, depth, minimum, count in [(97, 10, 6, 6), (65, 2, 5, 16),
                                          (100, 10, 5, 6), (64, 10, 1, 4)]:
            X = np.arange(n, dtype=float)[:, None]
            m = model(min_tree=minimum, mean_depth=depth).fit(X, X[:, 0])
            all_rows = np.concatenate(m.block_indices_)
            np.testing.assert_array_equal(np.sort(all_rows), np.arange(n))
            self.assertEqual(m.block_count_, count)
            self.assertLessEqual(np.ptp(m.block_sizes_), 1)
            self.assertTrue(np.all(m.block_sizes_ > 0))
            # Adjacent pair-size mismatch is at most one, with maximal equal pairing.
            self.assertLessEqual(np.count_nonzero(m.block_sizes_[::2] != m.block_sizes_[1::2]), 1)
        default = mod.MIForestErrPDF('Regressor', {'random_state': 3}).fit(self.X, self.y)
        self.assertEqual(default.block_count_, 64)
        self.assertEqual(default.n_repeats_, 3)
        self.assertEqual(len(default.estimators_), 192)

    def test_B03_same_members_mean_variance_importance(self):
        m = model(niter_max=3).fit(pd.DataFrame(self.X, columns=['a', 'b']), self.y[:, None])
        block = np.array([np.mean([t.predict(self.X) for t in trees], axis=0)
                          for trees in m.block_trees_])
        np.testing.assert_allclose(m.predict(self.X), block.mean(0), rtol=1e-13, atol=1e-13)
        ref = np.sqrt(np.var(block[1::2] - block[::2], axis=0, ddof=1) / (4 * m.n_pairs_))
        np.testing.assert_allclose(m.predictReducibleStd(self.X), ref, rtol=1e-13, atol=1e-13)
        out = m.predictStat(self.X, {'mean': [], 'std': [], 'median': [], 'quantile': [.25]}, 'all')
        np.testing.assert_allclose(out['mean'], block.mean(0))
        np.testing.assert_allclose(out['std'], block.std(0))
        np.testing.assert_allclose(out['median'], np.median(block, axis=0))
        np.testing.assert_allclose(out['quantiles'][.25], np.quantile(block, .25, axis=0))
        for group in [0, 1]:
            np.testing.assert_allclose(m.predictStat(self.X, {'mean': []}, group)['mean'], block[group::2].mean(0))
        with self.assertWarnsRegex(FutureWarning, 'group 0'):
            old = m.predictStat(self.X)
        np.testing.assert_allclose(old['mean'], block[::2].mean(0))
        np.testing.assert_allclose(m.getImportance(), np.mean([t.feature_importances_ for t in m.estimators_], axis=0))
        for bad in [pd.DataFrame(self.X, columns=['b', 'a']), self.X[:, :1], self.X * np.nan]:
            with self.assertRaises(ValueError):
                m.predict(bad)
        self.assertEqual(m.predict(self.X[:0]).shape, (0,))
        self.assertEqual(m.predictReducibleStd(self.X[:0]).shape, (0,))

    def test_B01_enumerated_subsets_and_permutation_adapter(self):
        m = model(min_tree=8).fit(self.X, self.y)
        differences = np.array([1., -3., 2., 7.])
        class ConstantMember:
            def __init__(self, value): self.value = value
            def predict(self, X): return np.full(len(X), self.value)
        m.members_ = tuple(ConstantMember(x) for d in differences for x in [0, d])
        closed = np.std(differences, ddof=1) / np.sqrt(4 * len(differences))
        np.testing.assert_allclose(m.predictReducibleStd(self.X[:2]), closed)
        for k in [1, 2, 3]:
            means = np.array([np.mean(differences[list(indices)])
                              for indices in itertools.combinations(range(4), k)])
            raw = means.std(ddof=0)
            np.testing.assert_allclose(raw * .5 * np.sqrt((k/4)/(1-k/4)), closed, rtol=1e-14)
        result = {'kept': 'value'}
        returned = m.predictReducibleError(self.X[:2], result, 6000, [.3, .6, .9])
        self.assertIsNone(returned)
        self.assertEqual(result['kept'], 'value')
        for f, k in [(.3, 1), (.6, 2), (.9, 3)]:
            np.testing.assert_allclose(result[f], result[1+f] * .5 * np.sqrt((k/4)/(1-k/4)))
            np.testing.assert_allclose(result[f], closed, rtol=.04)
        for fractions, nper in [([0], 3), ([1], 3), ([.1], 3), ([.5, .5], 3), ([np.nan], 3), ([.5], 1)]:
            with self.assertRaises(ValueError):
                m.predictReducibleError(self.X, {}, nper, fractions)

    def test_B04_reproducibility_repeats_jobs_and_refit(self):
        state = np.random.get_state()
        a = model(niter_max=3).fit(self.X, self.y)
        b = model(niter_max=3, n_jobs=2).fit(self.X, self.y)
        for x, y in zip(a.block_indices_, b.block_indices_): np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(a.tree_seeds_, b.tree_seeds_)
        np.testing.assert_allclose(a.predict(self.X), b.predict(self.X))
        one = model(niter_max=1).fit(self.X, self.y)
        np.testing.assert_allclose(a.predict(self.X), one.predict(self.X))
        np.testing.assert_allclose(a.predictReducibleStd(self.X), one.predictReducibleStd(self.X))
        a.predictReducibleError(self.X[:2], {}, 8, [.5])
        final = np.random.get_state()
        self.assertEqual(state[0], final[0]); np.testing.assert_array_equal(state[1], final[1])
        self.assertEqual(state[2:], final[2:])
        self.assertIs(a.fitWithParam(self.X, self.y, 4, 1), a)
        self.assertEqual(a.n_repeats_, 1)
        self.assertEqual(len(a.estimators_), a.block_count_)
        a.noise_model_ = object(); a.oof_mean_ = np.zeros(1)
        a.fit(self.X, self.y)
        self.assertFalse(hasattr(a, 'noise_model_')); self.assertFalse(hasattr(a, 'oof_mean_'))
        a.optionsErrPDF['fit_noise'] = True
        a.optionsErrPDF['noise_options'] = {'criterion': 'absolute_error'}
        with self.assertRaisesRegex(ValueError, 'squared_error'): a.fit(self.X, self.y)
        with self.assertRaises(NotFittedError): a.predict(self.X)
        a.optionsErrPDF['fit_noise'] = False
        a.fit(self.X, self.y)
        with self.assertRaises(ValueError): a.fit(self.X, self.y[:-1])
        with self.assertRaises(NotFittedError): a.predictReducibleStd(self.X)

    def test_randomized_repeats_forward_tree_kwargs(self):
        opts = dict(min_tree=6, mean_depth=10, niter_max=3, random_state=31, n_jobs=2)
        a = mod.MIForestErrPDF('Regressor', opts, splitter='random', max_depth=3).fit(self.X, self.y)
        b = mod.MIForestErrPDF('Regressor', opts, splitter='random', max_depth=3).fit(self.X, self.y)
        np.testing.assert_array_equal(a.predict(self.X), b.predict(self.X))
        self.assertTrue(any(not np.array_equal(trees[0].predict(self.X), trees[1].predict(self.X)) for trees in a.block_trees_))
        self.assertEqual(a.block_trees_[0][0].get_params()['max_depth'], 3)

    def test_group_weights_mean_refusal_and_opt_in(self):
        groups = np.arange(len(self.y)) // 3
        weights = np.linspace(1, 2, len(self.y)); weights[:3] = 0
        m = model().fit(self.X, self.y, weights, groups=groups)
        np.testing.assert_array_equal(np.sort(np.concatenate(m.block_indices_)), np.arange(3, 97))
        self.assertEqual(m.n_zero_weight_, 3)
        seen = {}
        for b, idx in enumerate(m.block_indices_):
            for g in groups[idx]:
                self.assertEqual(seen.setdefault(g, b), b)
            ref = DecisionTreeRegressor(min_samples_leaf=2, random_state=int(m.tree_seeds_[b, 0])).fit(self.X[idx].astype(np.float32), self.y[idx], sample_weight=weights[idx])
            np.testing.assert_allclose(m.block_trees_[b][0].predict(self.X), ref.predict(self.X))
        self.assertTrue(np.isfinite(m.predict(self.X)).all())
        with self.assertRaisesRegex(ValueError, 'experimental=True'): m.predictReducibleStd(self.X)
        with self.assertRaisesRegex(ValueError, 'experimental=True'): m.predictReducibleError(self.X, {}, 5, [.5])
        with self.assertWarnsRegex(RuntimeWarning, 'validated=False'):
            sigma = m.predictReducibleStd(self.X, experimental=True)
        self.assertTrue(np.isfinite(sigma).all()); self.assertFalse(m.uncertainty_status_['validated'])
        for kwargs in [dict(groups=groups), dict(sample_weight=np.linspace(1, 2, len(self.y)))]:
            fit = model().fit(self.X, self.y, **kwargs)
            with self.assertRaises(ValueError): fit.predictReducibleStd(self.X)
        plain = model().fit(self.X, self.y)
        equal = model().fit(self.X, self.y, np.full(len(self.y), 2.))
        np.testing.assert_allclose(plain.predict(self.X), equal.predict(self.X))
        np.testing.assert_allclose(plain.predictReducibleStd(self.X), equal.predictReducibleStd(self.X))

    def test_invalid_fit_contracts_and_noise_unavailable(self):
        for weights in [np.zeros(97), -np.ones(97), np.full(97, np.nan), np.ones(96)]:
            with self.assertRaises(ValueError): model().fit(self.X, self.y, weights)
        for kwargs in [dict(groups=np.zeros(97)), dict(groups=np.zeros(96)), dict(groups=np.full(97, np.nan))]:
            with self.assertRaises(ValueError): model().fit(self.X, self.y, **kwargs)
        for options in [dict(min_tree=98), dict(min_tree=0), dict(niter_max=0), dict(mean_depth=0),
                        dict(mean_depth=np.nan), dict(random_state=-1), dict(n_jobs=0), dict(fit_noise='yes')]:
            with self.assertRaises(ValueError): model(**options).fit(self.X, self.y)
        for X,y in [(self.X[:0], self.y[:0]), (self.X, np.column_stack([self.y,self.y])), (self.X, self.y*np.nan)]:
            with self.assertRaises(ValueError): model().fit(X,y)
        with self.assertRaises(ValueError): mod.MIForestErrPDF('Regressor', {'invented': 1})
        with self.assertRaises(TypeError): mod.MIForestErrPDF('Regressor', {}, invented=1)
        with self.assertRaises(NotImplementedError): mod.MIForestErrPDF('Classifier', {})
        m = model().fit(self.X, self.y)
        with self.assertRaisesRegex(RuntimeError, 'noise model unavailable'): m.predictUncertainty(self.X)
        with self.assertRaises(TypeError): m.predict(self.X, invented=True)
        for selector in [0, 1, -1]:
            with self.assertRaisesRegex(TypeError, 'historical predictStat'):
                m.predictStat(self.X, selector, {'mean': 0}, slice(0, len(self.X)))

    def test_options_not_mutated_and_refit_failure_drops_named_schema(self):
        options = dict(min_tree=4, mean_depth=10, niter_max=1, random_state=5)
        original = dict(options)
        m = mod.MIForestErrPDF('Regressor', options)
        m.fit(pd.DataFrame(self.X, columns=['a','b']), self.y)
        self.assertEqual(options, original)
        m.fit(self.X, self.y)
        self.assertFalse(hasattr(m, 'feature_names_in_'))

    def test_chunk_invariance_and_empty_permutation_outputs(self):
        m = model(niter_max=3).fit(self.X, self.y)
        ref = m.predictReducibleStd(self.X)
        normal = {}; m.predictReducibleError(self.X, normal, 7, [.5])
        with patch.object(mod, '_STAT_CHUNK_ROWS', 7):
            np.testing.assert_allclose(m.predictReducibleStd(self.X), ref)
            other = {}; m.predictReducibleError(self.X, other, 7, [.5])
        for k in normal: np.testing.assert_allclose(normal[k], other[k])
        empty = {}; m.predictReducibleError(self.X[:0], empty, 7, [.5])
        self.assertTrue(all(v.shape == (0,) for v in empty.values()))

    def test_B05_actual_fitter_class_public_methods(self):
        """Exercise the actual imported Fitter module, without replacement methods."""
        from RootInteractive.MLpipeline.NDFunctionInterface import Fitter
        data = SimpleNamespace(X_train=self.X, y_train=self.y[:, None], X_values=['a','b'])
        fitter = Fitter(data)
        m = model()
        fitter.Register_Model('paired', m)
        fitter.Fit()
        np.testing.assert_allclose(fitter.Predict(self.X, 'paired'), m.predict(self.X))
        frame = pd.DataFrame(self.X, columns=['a','b'])
        out = fitter.AppendOtherPandas('paired', frame)
        np.testing.assert_allclose(out['paired'], m.predict(self.X))
        stats = fitter.AppendStatPandas('paired', frame.copy(), prefix='qa_',
                                        statDictionary={'mean': [], 'std': []}, treeType='all')
        np.testing.assert_allclose(stats['qa_pairedmean'], m.predict(self.X))
        np.testing.assert_allclose(stats['qa_pairedstd'], m.predictStat(self.X, treeType='all')['std'])


if __name__ == '__main__': unittest.main(verbosity=2)
