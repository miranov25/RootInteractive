"""E1 engineering contracts; no E2/backend or scientific-calibration claim."""
import copy
from unittest.mock import patch
import unittest
import numpy as np
import pandas as pd
from RootInteractive.MLpipeline import MIForestErrPDF as mod


class FastPathContracts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(510)
        self.X = rng.uniform(-1, 1, (257, 3))
        self.y = self.X[:, 0] + rng.normal(0, .2, len(self.X))

    def model(self, **changes):
        options = dict(min_tree=8, mean_depth=12, n_jobs=1, niter_max=1,
                       random_state=51, fit_noise=True,
                       noise_options=dict(n_estimators=8, min_samples_leaf=4))
        options.update(changes)
        return mod.MIForestErrPDF('Regressor', options, min_samples_leaf=2)

    def test_exact_public_prediction_oracle_and_one_member_pass(self):
        m = self.model(prediction_chunk_rows=13).fit(self.X, self.y)
        X = self.X.astype(np.float32)
        expected = {key: np.empty(len(X)) for key in ('mean', 'reducible_std', 'irreducible_var_raw')}
        for first in range(0, len(X), 13):
            chunk = X[first:first+13]
            # Independent public per-tree prediction with original reduction order.
            members = np.asarray([np.mean([t.predict(chunk) for t in block], axis=0)
                                  for block in m.block_trees_])
            sl = slice(first, first + len(chunk))
            expected['mean'][sl] = members.mean(axis=0)
            expected['reducible_std'][sl] = np.sqrt(np.var(members[1::2]-members[::2],axis=0,ddof=1)/(4*m.n_pairs_))
            expected['irreducible_var_raw'][sl] = m.noise_model_.predict(chunk)
        original = m._validated_member_prediction
        with patch.object(m, '_validated_member_prediction', wraps=original) as count:
            out = m.predictUncertainty(self.X)
        self.assertEqual(count.call_count, m.block_count_*int(np.ceil(len(X)/13)))
        for key, value in expected.items(): self.assertEqual(out[key].tobytes(), value.tobytes(), key)
        self.assertEqual(out['mean'].tobytes(), m.predict(self.X).tobytes())
        self.assertEqual(out['reducible_std'].tobytes(), m.predictReducibleStd(self.X).tobytes())
        self.assertEqual(out['mean'].tobytes(), m.predictStat(self.X, {'mean':True}, treeType='all')['mean'].tobytes())

    def test_fixed_blocks_equal_legacy_equivalent_and_override_growth(self):
        old = self.model(fit_noise=False).fit(self.X, self.y)
        fixed = self.model(fit_noise=False, n_blocks=8, mean_depth=1).fit(self.X, self.y)
        self.assertEqual(fixed.block_count_, 8)
        self.assertEqual(fixed.requested_n_blocks_, 8)
        for a,b in zip(old.block_indices_, fixed.block_indices_): np.testing.assert_array_equal(a,b)
        self.assertEqual(old.predict(self.X).tobytes(), fixed.predict(self.X).tobytes())
        self.assertEqual(old.predictReducibleStd(self.X).tobytes(), fixed.predictReducibleStd(self.X).tobytes())
        np.testing.assert_array_equal(np.sort(np.concatenate(fixed.block_indices_)), np.arange(len(self.X)))

    def test_named_invalid_empty_and_trusted_tree_boundary(self):
        frame = pd.DataFrame(self.X, columns=['a','b','c'])
        m = self.model().fit(frame, self.y)
        for method in (m.predict, m.predictReducibleStd, m.predictUncertainty):
            for bad in (frame[['b','a','c']], self.X[:,:2], np.full((2,3),np.nan), np.full((2,3),np.inf)):
                with self.assertRaises(ValueError): method(bad)
        with patch.object(mod.DecisionTreeRegressor, 'predict', autospec=True,
                          side_effect=RuntimeError('tree failure')):
            with self.assertRaisesRegex(RuntimeError, 'tree failure'): m.predict(self.X)
        out = m.predictUncertainty(frame.iloc[:0])
        for key,value in out.items():
            if key!='metadata': self.assertEqual(value.shape,(0,))
        with self.assertRaises(ValueError):m.predictUncertainty(frame[['b','a','c']].iloc[:0])

    def test_chunk_and_jobs_preserve_mean_reducible_and_bounded_calls(self):
        m = self.model(fit_noise=False,n_blocks=8,niter_max=3).fit(self.X,self.y)
        before=(m.predict(self.X),m.predictReducibleStd(self.X))
        for size in (1,17,100000):
            m.prediction_chunk_rows_=size
            m.optionsErrPDF['n_jobs']=2
            original=m._validated_member_prediction
            sizes=[]
            def counted(member, chunk):
                sizes.append(len(chunk));return original(member,chunk)
            with patch.object(m,'_validated_member_prediction',side_effect=counted):
                actual=m.predict(self.X)
            self.assertLessEqual(max(sizes),size)
            # Changing chunk shapes can change NumPy's reduction rounding for
            # L>1 (not tree predictions). Disclose this configuration delta;
            # byte identity is separately tested at identical chunk settings.
            np.testing.assert_allclose(actual, before[0], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(m.predictReducibleStd(self.X), before[1], rtol=1e-12, atol=1e-12)

    def test_fixed_oof_blocks_and_refit_invalidations(self):
        m=self.model(n_blocks=8).fit(self.X,self.y)
        self.assertTrue(all(f['block_count']==8 for f in m.oof_folds_))
        m.optionsErrPDF['n_blocks']=7
        with self.assertRaises(ValueError):m.fit(self.X,self.y)
        self.assertFalse(hasattr(m,'noise_model_'))
        self.assertFalse(hasattr(m,'prediction_chunk_rows_'))
        for key,bad in [('n_blocks',True),('n_blocks',3),('n_blocks',8.),('n_blocks',258),
                        ('prediction_chunk_rows',0),('prediction_chunk_rows',False),('prediction_chunk_rows',1.5)]:
            with self.assertRaises(ValueError,msg=str((key,bad))):self.model(**{key:bad}).fit(self.X,self.y)

    def test_groups_weights_feasibility_and_options_copy(self):
        weights=np.ones(len(self.X));weights[::11]=0
        groups=np.arange(len(self.X))//3
        m=self.model(n_blocks=8,fit_noise=False).fit(self.X,self.y,weights,groups=groups)
        for group in np.unique(groups[weights>0]):
            self.assertEqual(len(np.unique(m.block_membership_[(groups==group)&(weights>0)])),1)
        np.testing.assert_array_equal(m.block_membership_[weights==0],-1)
        with self.assertRaises(ValueError):m.predictReducibleStd(self.X)
        with self.assertWarns(RuntimeWarning):m.predictReducibleStd(self.X,experimental=True)
        with self.assertRaises(ValueError):self.model(n_blocks=8).fit(self.X,self.y,groups=np.zeros(len(self.X)))
        options={'n_blocks':8,'prediction_chunk_rows':17};before=copy.deepcopy(options)
        mod.MIForestErrPDF('Regressor',options)
        self.assertEqual(options,before)


if __name__ == '__main__':unittest.main(verbosity=2)
