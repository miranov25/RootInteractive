"""PHASE_2_2 source-derived reshuffling oracles. Run directly or collect with pytest."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import numpy as np


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    return mod


class ReshufflingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=Path(os.environ.get('RI_PHASE22_SOURCE',Path(__file__).resolve().parents[1]/'MIForestErrPDF.py'))
        cls.mod=load(cls.source,'phase22_test_candidate')
        cls.X=np.random.default_rng(16).random((1024,3)).astype('float32')
        cls.y=cls.X[:,0]+np.random.default_rng(17).normal(0,.2,1024)
        cls.Q=np.random.default_rng(18).random((81,3)).astype('float32')

    def model(self, L=4, jobs=1, **options):
        opts=dict(n_blocks=8,niter_max=L,random_state=41,n_jobs=jobs,verbose=0,
                  block_sampling='historical',prediction_chunk_rows=23)
        opts.update(options)
        return self.mod.MIForestErrPDF('Regressor',opts,min_samples_leaf=1)

    def test_source_partition_oracle(self):
        m=self.model().fit(self.X,self.y)
        rng=np.random.RandomState(41)
        for l in range(4):
            a=rng.permutation(512); b=rng.permutation(512)+512
            # Original source uses global bucket index also within the second half.
            expected=[a[0:64],b[256:320],a[64:128],b[320:384],
                      a[128:192],b[384:448],a[192:256],b[448:512]]
            for got,want in zip(m.iteration_block_indices_[l],expected): np.testing.assert_array_equal(got,want)
            counts=np.bincount(np.concatenate(m.iteration_block_indices_[l]),minlength=1024)
            self.assertEqual(int((counts>0).sum()),512); self.assertEqual(counts.max(),1)
        self.assertEqual(len(m.estimators_),32)
        self.assertEqual(m.actual_tree_fraction_,1/16)
        self.assertEqual(m.participation_fraction_,.5)
        self.assertFalse(np.array_equal(m.iteration_membership_[:,0],m.iteration_membership_[:,1]))

    def test_fraction_controls_tree_count_and_rounding(self):
        m=self.model(n_blocks=None,tree_fraction=1/128).fit(self.X,self.y)
        self.assertEqual(m.block_count_,64); self.assertEqual(len(m.estimators_),256)
        m=self.model(n_blocks=None,tree_fraction=1/128).fit(self.X[:1000],self.y[:1000])
        self.assertEqual(m.block_sizes_[0],7)
        self.assertEqual(m.actual_tree_fraction_,.007)
        self.assertEqual(m.requested_tree_fraction_,1/128)
        self.assertEqual(m.participation_fraction_,448/1000)

    def test_mean_and_raw_variance_direct_tree_oracle(self):
        m=self.model().fit(self.X,self.y)
        T=np.array([[t.predict(self.Q) for t in it] for it in m.iteration_trees_])
        expected=T.mean((0,1))
        raw=np.var(T[0,1::2]-T[0,::2],axis=0,ddof=1)/16 - .75*np.var(T.mean(1),axis=0,ddof=1)
        np.testing.assert_allclose(m.predict(self.Q),expected,rtol=1e-14,atol=1e-15)
        np.testing.assert_array_equal(m.predictReducibleVariance(self.Q),raw)
        np.testing.assert_array_equal(m.predictReducibleStd(self.Q),np.sqrt(np.maximum(raw,0)))

    def test_L1_raw_formula(self):
        m=self.model(L=1).fit(self.X,self.y)
        T=np.array([t.predict(self.Q) for t in m.iteration_trees_[0]])
        np.testing.assert_array_equal(m.predictReducibleVariance(self.Q),
            np.var(T[1::2]-T[::2],axis=0,ddof=1)/16)

    def test_negative_raw_not_silently_clipped(self):
        m=self.model(L=4).fit(self.X,self.y)
        class Constant:
            def __init__(self,value): self.value=value
            def predict(self,X): return np.full(len(X),self.value)
        m.iteration_trees_=tuple(tuple(Constant(l) for _ in range(8)) for l in range(4))
        raw=m.predictReducibleVariance(self.Q)
        np.testing.assert_allclose(raw,-1.25)
        np.testing.assert_array_equal(m.predictReducibleStd(self.Q),np.zeros(len(self.Q)))

    def test_original_diagnostic_is_first_iteration_only(self):
        m=self.model().fit(self.X,self.y); out={}
        m.predictReducibleError(self.Q,out,32,[.3,.6,.9])
        T=np.array([t.predict(self.Q) for t in m.iteration_trees_[0]])
        rng=np.random.RandomState(m.permutation_seed_)
        for f in [.3,.6,.9]:
            samples=[]
            for _ in range(32):
                idx=rng.permutation(4)[:int(f*4)]
                samples.append(T[1::2][idx].mean(0)-T[::2][idx].mean(0))
            raw=np.std(samples,axis=0)
            np.testing.assert_allclose(out[f],raw*.5*np.sqrt(f/(1-f)),rtol=1e-14,atol=1e-16)
            np.testing.assert_allclose(out[1+f],raw,rtol=1e-14,atol=1e-16)
        old=out[.3].copy()
        m.iteration_trees_=(m.iteration_trees_[0],)*4
        m.predictReducibleError(self.Q,out,32,[.3])
        np.testing.assert_array_equal(out[.3],old)

    def test_threads_chunks_rng_and_repeatability(self):
        np.random.seed(71); before=np.random.get_state()
        a=self.model().fit(self.X,self.y)
        b=self.model(jobs=2,prediction_chunk_rows=1000).fit(self.X,self.y)
        np.testing.assert_allclose(a.predict(self.Q),b.predict(self.Q),rtol=1e-14,atol=1e-15)
        np.testing.assert_array_equal(a.predictReducibleVariance(self.Q),b.predictReducibleVariance(self.Q))
        x={};y={};a.predictReducibleError(self.Q,x,32,[.3]);b.predictReducibleError(self.Q,y,32,[.3])
        np.testing.assert_array_equal(x[.3],y[.3])
        after=np.random.get_state(); np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])

    def test_refusals_have_supported_control(self):
        for kwargs in [dict(groups=np.arange(1024)),dict(sample_weight=np.ones(1024))]:
            with self.assertRaises(NotImplementedError): self.model().fit(self.X,self.y,**kwargs)
        with self.assertRaises(NotImplementedError): self.model(fit_noise=True).fit(self.X,self.y)
        self.assertEqual(self.model().fit(self.X,self.y).predict(self.Q).shape,(81,))
        for opts in [dict(block_sampling='bad'),dict(tree_fraction=-1),dict(tree_fraction=True),
                     dict(tree_fraction=float('nan')),dict(tree_fraction=1/14,n_blocks=None),
                     dict(tree_fraction=1/128,n_blocks=8),dict(tree_fraction=1e-9,n_blocks=None),
                     dict(block_sampling='fixed',tree_fraction=1/128)]:
            with self.assertRaises(ValueError): self.model(**opts).fit(self.X,self.y)

    def test_failed_refit_invalidates_model(self):
        from sklearn.exceptions import NotFittedError
        m=self.model().fit(self.X,self.y)
        with self.assertRaises(ValueError): m.fit(self.X,np.full(1024,np.nan))
        with self.assertRaises(NotFittedError): m.predict(self.Q)
        self.assertFalse(hasattr(m,'iteration_trees_'))

    def test_feature_schema_and_empty_queries(self):
        import pandas as pd
        X=pd.DataFrame(self.X,columns=['a','b','c']);Q=pd.DataFrame(self.Q,columns=X.columns)
        m=self.model().fit(X,self.y)
        self.assertTrue(np.isfinite(m.predict(Q)).all())
        with self.assertRaises(ValueError): m.predict(Q[['b','a','c']])
        self.assertEqual(m.predict(Q.iloc[:0]).shape,(0,))
        self.assertEqual(m.predictReducibleVariance(Q.iloc[:0]).shape,(0,))
        out={};m.predictReducibleError(Q.iloc[:0],out,4,[.3]);self.assertEqual(out[.3].shape,(0,))

    def test_odd_population_and_leaf_forwarding(self):
        m=self.mod.MIForestErrPDF('Regressor',dict(block_sampling='historical',n_blocks=8,
            niter_max=2,n_jobs=1,random_state=41),min_samples_leaf=5).fit(self.X[:1001],self.y[:1001])
        self.assertEqual(m.block_sizes_[0],62)
        self.assertTrue(all(t.min_samples_leaf==5 for t in m.estimators_))
        self.assertEqual(m.iteration_membership_.shape,(1001,2))

    def test_fixed_noise_and_weighted_grouped_reference(self):
        baseline=os.environ.get('RI_PHASE22_BASELINE')
        if not baseline: self.skipTest('supply RI_PHASE22_BASELINE for prepatch regression')
        old=load(baseline,'phase22_previous_noise')
        for grouped in (False,True):
            opts=dict(n_blocks=8,niter_max=1,n_jobs=1,random_state=43,verbose=0,
                      fit_noise=True,noise_options=dict(n_estimators=4,max_depth=2))
            kw={} if not grouped else dict(groups=np.arange(1024)//4,sample_weight=1+self.X[:,0])
            a=old.MIForestErrPDF('Regressor',opts,min_samples_leaf=2).fit(self.X,self.y,**kw)
            z=self.mod.MIForestErrPDF('Regressor',dict(opts,block_sampling='fixed'),
                                    min_samples_leaf=2).fit(self.X,self.y,**kw)
            left=a.predictUncertainty(self.Q,experimental=True)
            right=z.predictUncertainty(self.Q,experimental=True)
            for k in left:
                if k!='metadata': self.assertEqual(left[k].tobytes(),right[k].tobytes())

    def test_fixed_mode_matches_prepatch_bytes(self):
        baseline=os.environ.get('RI_PHASE22_BASELINE')
        if not baseline: self.skipTest('supply RI_PHASE22_BASELINE for exact prepatch source invariance')
        old=load(baseline,'phase22_old_reference')
        opts=dict(n_blocks=8,niter_max=3,n_jobs=1,random_state=43,verbose=0)
        a=old.MIForestErrPDF('Regressor',opts,min_samples_leaf=2).fit(self.X,self.y)
        z=self.mod.MIForestErrPDF('Regressor',opts,min_samples_leaf=2).fit(self.X,self.y)
        for fn in ['predict','predictReducibleStd']:
            self.assertEqual(getattr(a,fn)(self.Q).tobytes(),getattr(z,fn)(self.Q).tobytes())
        # sqrt then squaring can round by a few ulps; existing SD bytes above stay exact.
        np.testing.assert_allclose(z.predictReducibleVariance(self.Q),a.predictReducibleStd(self.Q)**2,
                                   rtol=4*np.finfo(float).eps,atol=0)
        left={};right={}
        a.predictReducibleError(self.Q,left,32,[.3,.6]);z.predictReducibleError(self.Q,right,32,[.3,.6])
        for k in left: self.assertEqual(left[k].tobytes(),right[k].tobytes())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--baseline')
    args,rest=p.parse_known_args()
    if args.source: os.environ['RI_PHASE22_SOURCE']=args.source
    if args.baseline: os.environ['RI_PHASE22_BASELINE']=args.baseline
    unittest.main(argv=[__file__]+rest)
