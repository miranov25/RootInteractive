"""Augmentation behavior oracles; stdlib unittest or pytest collection."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class AugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ.get('RI_PHASE22_SOURCE', Path(__file__).resolve().parents[1]/'MIForestErrPDF.py'))
        cls.mod = load(cls.source, 'augmentation_candidate')
        cls.X = np.random.default_rng(16).random((1024, 3)).astype('float32')
        cls.y = cls.X[:, 0]+np.random.default_rng(17).normal(0, .2, 1024)
        cls.Q = np.random.default_rng(18).random((81, 3)).astype('float32')

    def model(self, mode='historical', sigma=(.02, .01, 0), jobs=1, chunk=23):
        return self.mod.MIForestErrPDF('Regressor', dict(n_blocks=8, niter_max=3,
            random_state=41, n_jobs=jobs, verbose=0, block_sampling=mode,
            prediction_chunk_rows=chunk, smear_sigma=sigma), min_samples_leaf=2)

    def test_disabled_and_zero_width_exact_previous_source(self):
        base = os.environ.get('RI_PHASE22_AUGMENT_BASELINE')
        if not base:
            self.skipTest('supply RI_PHASE22_AUGMENT_BASELINE for previous-source equivalence')
        old = load(base, 'augmentation_baseline')
        for mode in ('historical', 'fixed'):
            a = self.model(mode)
            opts = dict(a.optionsErrPDF)
            opts.pop('smear_sigma')
            opts.pop('store_leaf_stats', None)
            ref = old.MIForestErrPDF('Regressor', opts, min_samples_leaf=2).fit(self.X, self.y)
            for sigma, augment in [((.02, .01, 0), False), ((0, 0, 0), True), (None, False)]:
                a = self.model(mode, sigma=sigma).fit(self.X, self.y, augment=augment)
                for fn in ('predict', 'predictReducibleVariance', 'predictReducibleStd'):
                    self.assertEqual(getattr(a, fn)(self.Q, augment=augment).tobytes(),
                                     getattr(ref, fn)(self.Q).tobytes())

    def test_training_inputs_targets_memberships_and_tree_seed_oracle(self):
        from sklearn.tree import DecisionTreeRegressor
        for mode in ('historical', 'fixed'):
            X, y = self.X.copy(), self.y.copy()
            a = self.model(mode).fit(X, y, augment=True)
            plain = self.model(mode).fit(X, y)
            np.testing.assert_array_equal(a.tree_seeds_, plain.tree_seeds_)
            for b in range(a.block_count_):
                for l in range(a.n_repeats_):
                    idx = a.iteration_block_indices_[l][b] if mode == 'historical' else a.block_indices_[b]
                    ref_idx = plain.iteration_block_indices_[l][b] if mode == 'historical' else plain.block_indices_[b]
                    np.testing.assert_array_equal(idx, ref_idx)
                    rng = np.random.default_rng(int(a.fit_smear_seeds_[b, l]))
                    train = (X[idx]+rng.standard_normal(X[idx].shape)*np.array([.02, .01, 0])).astype('float32')
                    ref = DecisionTreeRegressor(random_state=int(a.tree_seeds_[b, l]), min_samples_leaf=2)
                    ref.fit(train, y[idx])
                    np.testing.assert_array_equal(ref.predict(self.Q), a.block_trees_[b][l].predict(self.Q))
                    np.testing.assert_array_equal(train[:, 2], X[idx, 2])
            np.testing.assert_array_equal(X, self.X)
            np.testing.assert_array_equal(y, self.y)
            self.assertGreater(np.max(abs(a.predict(self.Q)-plain.predict(self.Q))), 0)

    def test_prediction_draw_oracle_mean_variance_and_std(self):
        for mode in ('historical', 'fixed'):
            a = self.model(mode).fit(self.X, self.y, augment=True)
            seeds = np.random.default_rng(123).integers(2**63-1, size=(8, 3))
            T = []
            for b in range(8):
                row = []
                for l in range(3):
                    rng = np.random.default_rng(int(seeds[b, l]))
                    query = (self.Q+rng.standard_normal(self.Q.shape)*np.array([.02, .01, 0])).astype('float32')
                    row.append(a.block_trees_[b][l].predict(query))
                T.append(row)
            T = np.asarray(T)
            np.testing.assert_allclose(a.predict(self.Q, augment=True, random_state=123), T.mean(1).mean(0), atol=0, rtol=1e-14)
            if mode == 'historical':
                v1 = np.var(T[1::2, 0]-T[::2, 0], axis=0, ddof=1)/16
                raw = v1-(1-1/3)*np.var(T.mean(0), axis=0, ddof=1)
            else:
                members = T.mean(1)
                raw = np.var(members[1::2]-members[::2], axis=0, ddof=1)/16
            np.testing.assert_array_equal(a.predictReducibleVariance(self.Q, augment=True, random_state=123), raw)
            np.testing.assert_allclose(a.predictReducibleStd(self.Q, augment=True, random_state=123), np.sqrt(np.maximum(raw, 0)), atol=0, rtol=1e-14)

    def test_independent_per_tree_not_one_shared_jitter(self):
        # Identity trees expose query draws directly, independent of split thresholds.
        a = self.model(sigma=(.5, 0, 0)).fit(self.X, self.y)
        class Identity:
            def predict(self, X):
                return X[:, 0]
        a.block_trees_ = tuple(tuple(Identity() for _ in range(3)) for _ in range(8))
        a.members_ = tuple(self.mod._BlockMember(row) for row in a.block_trees_)
        seeds = np.random.default_rng(9).integers(2**63-1, size=(8, 3))
        draws = [(self.Q+np.random.default_rng(int(seed)).standard_normal(self.Q.shape)*[.5, 0, 0]).astype('float32')[:, 0]
                 for seed in seeds.ravel()]
        result = a.predict(self.Q, augment=True, random_state=9)
        np.testing.assert_allclose(result, np.asarray(draws).reshape(8, 3, -1).mean(1).mean(0), rtol=1e-14)
        self.assertFalse(np.array_equal(draws[0], draws[1]))
        self.assertFalse(np.array_equal(result, draws[0]))

    def test_replay_fresh_calls_threads_chunks_and_global_rng(self):
        np.random.seed(71)
        before = np.random.get_state()
        for mode in ('historical', 'fixed'):
            a = self.model(mode).fit(self.X, self.y, augment=True)
            b = self.model(mode, jobs=2, chunk=1000).fit(self.X, self.y, augment=True)
            for fn in ('predict', 'predictReducibleVariance'):
                left = getattr(a, fn)(self.Q, augment=True, random_state=19)
                np.testing.assert_array_equal(left, getattr(a, fn)(self.Q, augment=True, random_state=19))
                np.testing.assert_array_equal(left, getattr(b, fn)(self.Q, augment=True, random_state=19))
            first = a.predict(self.Q, augment=True)
            np.testing.assert_array_equal(first, b.predict(self.Q, augment=True))
            self.assertFalse(np.array_equal(first, a.predict(self.Q, augment=True)))
            # Explicit replay calls above do not advance the default stream.
            np.testing.assert_array_equal(a.predict(self.Q), b.predict(self.Q))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_input_schema_empty_and_no_mutation(self):
        import pandas as pd
        X = pd.DataFrame(self.X, columns=['a', 'b', 'c'])
        Q = pd.DataFrame(self.Q, columns=X.columns)
        a = self.model().fit(X, self.y, augment=True)
        before = Q.copy()
        self.assertEqual(a.predict(Q.iloc[:0], augment=True, random_state=1).shape, (0,))
        self.assertEqual(a.predictReducibleVariance(Q.iloc[:0], augment=True).shape, (0,))
        a.predict(Q, augment=True)
        pd.testing.assert_frame_equal(Q, before)
        with self.assertRaises(ValueError):
            a.predict(Q[['b', 'a', 'c']], augment=True)

    def test_invalid_widths_flags_seeds_and_failed_refit(self):
        from sklearn.exceptions import NotFittedError
        for sigma in ([-1, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0], [.1], .1,
                      [[.1, .1, .1]], [lambda x: .1]*3):
            with self.assertRaises(ValueError):
                self.model(sigma=sigma).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            self.model(sigma=None).fit(self.X, self.y, augment=True)
        for flag in (1, None, 'yes'):
            with self.assertRaises(ValueError): self.model().fit(self.X, self.y, augment=flag)
        a = self.model().fit(self.X, self.y)
        for seed in (True, -1, 2**32, '3', 1.5):
            with self.assertRaises(ValueError): a.predict(self.Q, augment=True, random_state=seed)
        with self.assertRaises(ValueError): a.predict(self.Q, augment=1)
        with self.assertRaises(ValueError): a.predict(self.Q, random_state=1)
        with self.assertRaises(ValueError): self.model(sigma=None).fit(self.X, self.y).predict(self.Q, augment=True)
        with self.assertRaises(ValueError): a.fit(self.X, self.y, augment='yes')
        with self.assertRaises(NotFittedError): a.predict(self.Q)

    def test_unsupported_augmented_noise_refused(self):
        a = self.model(mode='fixed')
        a.optionsErrPDF['fit_noise'] = True
        with self.assertRaises(NotImplementedError): a.fit(self.X, self.y, augment=True)

    def test_pickle_replays_private_stream(self):
        import pickle
        a = self.model().fit(self.X, self.y, augment=True)
        a.predict(self.Q, augment=True)
        b = pickle.loads(pickle.dumps(a))
        np.testing.assert_array_equal(a.predict(self.Q, augment=True), b.predict(self.Q, augment=True))


class BenchmarkTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        default = root/'benchmark_phase_2_2_augmentation.py'
        if not default.exists():
            default = root/'benchmarks'/'benchmark_phase_2_2_augmentation.py'
        cls.bench = load(Path(os.environ.get('RI_PHASE22_AUGMENT_BENCHMARK', default)), 'augmentation_truth_benchmark')

    def test_no_input_noise_reduces_to_latent_truth(self):
        self.assertAlmostEqual(float(self.bench.latent_function(np.array([[0., 1., .25, .5]]))[0]), 1.)
        Q = np.random.default_rng(33).random((19, 4))
        mean, var = self.bench.conditional_truth(Q, np.zeros(4), 1.)
        np.testing.assert_allclose(mean, self.bench.latent_function(Q), atol=1e-14)
        np.testing.assert_allclose(var, 1., atol=1e-13)

    def test_uniform_posterior_boundary_against_truncated_normal(self):
        from scipy.stats import truncnorm
        Q = np.array([[a, .4, .3, .2] for a in [-.03, 0, .5, 1., 1.03]])
        mean, var = self.bench.conditional_truth(Q, [.02, 0, 0, 0], 0.)
        expected_mean, expected_var = truncnorm.stats(-Q[:, 0]/.02, (1-Q[:, 0])/.02,
                                                     loc=Q[:, 0], scale=.02, moments='mv')
        other = .4*np.sin(2*np.pi*.3)
        np.testing.assert_allclose(mean, 3*expected_mean+other, atol=2e-13)
        np.testing.assert_allclose(var, 9*expected_var, atol=2e-13)

    def test_zero_sampling_variance_reports_absolute_not_ratio(self):
        out = self.bench.calibration(np.ones((4, 9)), np.full((4, 9), -.1), 10, np.random.default_rng(1))
        self.assertIsNone(out['ratio'])
        self.assertIsNone(out['ci95'])
        self.assertAlmostEqual(out['raw_variance'], -.1)
        self.assertEqual(out['refit_variance'], 0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source')
    p.add_argument('--baseline')
    args, rest = p.parse_known_args()
    if args.source: os.environ['RI_PHASE22_SOURCE'] = args.source
    if args.baseline: os.environ['RI_PHASE22_AUGMENT_BASELINE'] = args.baseline
    unittest.main(argv=[__file__]+rest)
