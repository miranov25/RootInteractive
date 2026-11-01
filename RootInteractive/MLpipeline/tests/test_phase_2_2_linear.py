"""Local-linear and pandas-export oracles. Run with --source or pytest."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import numpy as np


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


class LinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod=load(os.environ.get('RI_PHASE22_SOURCE',str(Path(__file__).resolve().parents[1]/'MIForestErrPDF.py')),'linear_candidate')
        cls.X=np.random.default_rng(13).random((1600,4)).astype('float32')
        cls.Q=np.random.default_rng(14).random((71,4)).astype('float32')
        cls.coeff=np.array([1.,2.,3.,0.])
        cls.y=cls.X.astype(float)@cls.coeff

    def model(self, mode='historical', **kw):
        opts=dict(n_blocks=8,niter_max=3,n_jobs=1,random_state=41,verbose=0,
                  block_sampling=mode,store_leaf_stats=True,smear_sigma=[.05,.05,.05,.3],prediction_chunk_rows=19)
        opts.update(kw)
        return self.mod.MIForestErrPDF('Regressor',opts,min_samples_leaf=3)

    def test_affine_reproduction_gradient_and_prediction_dispatch(self):
        for mode in ['fixed','historical']:
            m=self.model(mode).fit(self.X,self.y)
            r=m.predictLinear(self.Q,return_diagnostics=True)
            self.assertFalse(r['fallback'].any())
            np.testing.assert_allclose(r['mean'],self.Q.astype(float)@self.coeff,atol=2e-12)
            np.testing.assert_allclose(r['gradient'],np.broadcast_to(self.coeff,(len(self.Q),4)),atol=2e-11)
            np.testing.assert_array_equal(m.predict(self.Q,prediction='linear'),r['mean'])

    def test_constant_target_with_full_rank_metadata(self):
        # Constant targets normally produce a single leaf per tree; different tree
        # centroids can still identify a full-rank constant plane across trees.
        m=self.model().fit(self.X,np.full(len(self.X),2.7))
        np.testing.assert_allclose(m.predictLinear(self.Q),2.7,atol=1e-11)

    def test_selected_leaf_sums_and_smearing_centering_oracle(self):
        m=self.model().fit(self.X,self.y,augment=True)
        for aug in [False,True]:
            kw=dict(augment=True,random_state=123) if aug else {}
            table=m.leafQueryFrame(self.Q,**kw)
            expected=[]
            for i in range(len(self.Q)):
                part=table[table.query_position==i]
                Z=np.column_stack([np.ones(len(part))]+[part['delta_'+str(j)].to_numpy() for j in range(4)])
                expected.append(np.linalg.lstsq(Z,part.leaf_value.to_numpy(),rcond=None)[0][0])
                for j in range(4):
                    np.testing.assert_allclose(part['delta_'+str(j)],part['centroid_'+str(j)]-self.Q[i,j],atol=0)
            np.testing.assert_allclose(m.predictLinear(self.Q,**kw),expected,rtol=1e-10,atol=1e-11)

    def test_training_augmented_centroid_and_scatter_oracle(self):
        m=self.model().fit(self.X,self.y,augment=True)
        table=m.leafStatsFrame()
        for b,l in [(0,0),(2,1)]:
            idx=m.iteration_block_indices_[l][b]
            jitter=(self.X[idx]+np.random.default_rng(int(m.fit_smear_seeds_[b,l])).standard_normal(self.X[idx].shape)*m.smear_sigma_).astype('float32')
            tree=m.block_trees_[b][l];leaf=tree.apply(jitter)
            part=table[(table.block==b)&(table.iteration==l)]
            for row in part.itertuples():
                selected=leaf==row.leaf_node
                self.assertEqual(selected.sum(),row.count)
                np.testing.assert_allclose([getattr(row,'centroid_'+str(j)) for j in range(4)],jitter[selected].astype(float).mean(0),atol=1e-14)
                self.assertAlmostEqual(row.target_mean,self.y[idx][selected].mean(),places=12)
                if row.count>1:self.assertAlmostEqual(row.target_scatter_var,self.y[idx][selected].var(ddof=1),places=12)

    def test_singletons_scatter_missing_and_partition_bounds(self):
        m=self.mod.MIForestErrPDF('Regressor',dict(n_blocks=8,niter_max=1,n_jobs=1,random_state=41,
            block_sampling='historical',store_leaf_stats=True),min_samples_leaf=1).fit(self.X,self.y)
        t=m.leafStatsFrame()
        self.assertTrue((t['count']==1).all())
        self.assertTrue(t.target_scatter_var.isna().all())
        for j in range(4):
            self.assertTrue((t['centroid_'+str(j)]>=t['cell_lower_'+str(j)]).all())
            self.assertTrue((t['centroid_'+str(j)]<=t['cell_upper_'+str(j)]).all())

    def test_weighted_centroid_matches_weighted_tree_means(self):
        m=self.model('fixed').fit(self.X,self.y,sample_weight=1+self.X[:,0])
        t=m.leafStatsFrame()
        np.testing.assert_allclose(t.target_mean,t.leaf_value,atol=1e-13)
        np.testing.assert_allclose(m.predictLinear(self.Q),self.Q.astype(float)@self.coeff,atol=2e-12)

    def test_singular_fallback_and_ridge(self):
        X=self.X.copy();X[:,3]=0
        Q=self.Q.copy();Q[:,3]=0
        m=self.model().fit(X,self.y)
        r=m.predictLinear(Q,return_diagnostics=True)
        self.assertTrue(r['fallback'].all())
        self.assertTrue(np.isnan(r['gradient']).all())
        np.testing.assert_allclose(r['mean'],m.predict(Q),atol=1e-14)
        a=self.model().fit(self.X,self.y)
        r=a.predictLinear(self.Q,ridge=.1,return_diagnostics=True)
        self.assertTrue(np.isfinite(r['mean']).all())
        self.assertGreater(np.max(abs(r['mean']-self.Q.astype(float)@self.coeff)),1e-5)

    def test_chunks_threads_replay(self):
        a=self.model().fit(self.X,self.y,augment=True)
        b=self.model(n_jobs=2,prediction_chunk_rows=500).fit(self.X,self.y,augment=True)
        for aug in [False,True]:
            kw=dict(augment=True,random_state=19) if aug else {}
            np.testing.assert_array_equal(a.predictLinear(self.Q,**kw),b.predictLinear(self.Q,**kw))
        self.assertFalse(np.array_equal(a.predictLinear(self.Q,augment=True),a.predictLinear(self.Q,augment=True)))

    def test_pandas_schemas_copies_and_empty(self):
        import pandas as pd
        X=pd.DataFrame(self.X,columns=list('ABCD'));Q=pd.DataFrame(self.Q,columns=X.columns,index=np.arange(71)*2)
        m=self.model().fit(X,self.y)
        leaf=m.leafStatsFrame();frame=m.predictLinearFrame(Q);hits=m.leafQueryFrame(Q.iloc[:3])
        self.assertEqual(frame.index.tolist(),Q.index.tolist())
        self.assertEqual(leaf.attrs['feature_names'],list('ABCD'))
        self.assertEqual(len(hits),3*len(m.estimators_))
        expected=m.predictLinear(Q);leaf.loc[:,'centroid_0']=0
        np.testing.assert_array_equal(m.predictLinear(Q),expected)
        self.assertEqual(m.predictLinear(Q.iloc[:0]).shape,(0,))
        self.assertEqual(len(m.predictLinearFrame(Q.iloc[:0])),0)
        self.assertEqual(len(m.leafQueryFrame(Q.iloc[:0])),0)
        with self.assertRaises(ValueError):m.predictLinear(Q[['B','A','C','D']])
        with self.assertRaises(ValueError):m.leafQueryFrame(Q,max_rows=2)

    def test_metadata_opt_in_and_invalid_arguments(self):
        m=self.model(store_leaf_stats=False).fit(self.X,self.y)
        for fn in ['predictLinear','predictLinearFrame','leafQueryFrame']:
            with self.assertRaises(RuntimeError):getattr(m,fn)(self.Q)
        self.assertFalse(hasattr(m.estimators_[0],'_ri_leaf_stats'))
        a=self.model().fit(self.X,self.y)
        for kw in [dict(ridge=-1),dict(ridge=True),dict(condition_limit=0),dict(condition_limit=np.inf)]:
            with self.assertRaises(ValueError):a.predictLinear(self.Q,**kw)
        with self.assertRaises(ValueError):a.predict(self.Q,prediction='bad')
        with self.assertRaises(ValueError):self.model(store_leaf_stats=1).fit(self.X,self.y)
        with self.assertRaises(TypeError):a.predictReducibleVariance(self.Q,prediction='linear')

    def test_mean_predictions_unchanged_with_metadata(self):
        for mode in ['fixed','historical']:
            a=self.model(mode).fit(self.X,self.y)
            b=self.model(mode,store_leaf_stats=False).fit(self.X,self.y)
            for fn in ['predict','predictReducibleVariance']:
                np.testing.assert_array_equal(getattr(a,fn)(self.Q),getattr(b,fn)(self.Q))
            np.testing.assert_array_equal(a.predict(self.Q,augment=True,random_state=1),b.predict(self.Q,augment=True,random_state=1))


class LinearBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        path=root/'benchmark_phase_2_2_linear.py'
        if not path.exists():path=root/'benchmarks'/'benchmark_phase_2_2_linear.py'
        cls.bench=load(os.environ.get('RI_PHASE22_LINEAR_BENCHMARK',str(path)),'linear_benchmark_oracle')

    def test_zero_noise_conditional_truth_and_derivatives(self):
        Q=np.random.default_rng(44).random((21,4))
        for f in ['linear','quadratic','sinus']:
            np.testing.assert_allclose(self.bench.conditional_mean(Q,np.zeros(4),f),self.bench.response(Q,f),atol=1e-14)
            for j in range(4):
                left=Q.copy();right=Q.copy();left[:,j]-=1e-6;right[:,j]+=1e-6
                derivative=(self.bench.response(right,f)-self.bench.response(left,f))/2e-6
                np.testing.assert_allclose(derivative,self.bench.slopes(Q,f)[:,j],atol=1e-8)

    def test_noisy_linear_truth_against_truncated_normal(self):
        from scipy.stats import truncnorm
        Q=np.array([[0.,.5,1.,.3],[-.01,1.01,.2,.4]])
        sigma=np.array([.02,.02,.01,0])
        expected=np.zeros(len(Q))
        for j,c in enumerate([1,2,3]):
            expected+=c*truncnorm.mean(-Q[:,j]/sigma[j],(1-Q[:,j])/sigma[j],loc=Q[:,j],scale=sigma[j])
        np.testing.assert_allclose(self.bench.conditional_mean(Q,sigma,'linear'),expected,atol=2e-13)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source');args,rest=p.parse_known_args()
    if args.source:os.environ['RI_PHASE22_SOURCE']=args.source
    unittest.main(argv=[__file__]+rest)
