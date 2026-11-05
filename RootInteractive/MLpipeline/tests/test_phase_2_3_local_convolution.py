"""Always-run contracts for the opt-in Phase 2.3 candidate, including oracles."""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
from scipy.special import ndtr


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

_here=Path(__file__).resolve().parent
_source=(_here/'MIForestErrPDF.py') if (_here/'MIForestErrPDF.py').exists() else _here.parent/'MIForestErrPDF.py'
mod=None


class LocalConvolutionContracts(unittest.TestCase):
    def test_combined_single_traversal_and_local_parity(self):
        m=self.model();q=self.Q
        expected=(m.predict(q),m.predictLinear(q,return_diagnostics=True),m.predictReducibleVariance(q),
            m.predictLocalConvolution(q,degrees=[1,1,1,0],sigma=[.2,.2,.2,0]))
        with ExitStack() as stack:
            calls=[stack.enter_context(patch.object(t,'apply',wraps=t.apply)) for block in m.block_trees_ for t in block]
            chunks=list(m._combined_prediction_chunks(q,linear=True,variance=True,members=True,
                chunk_rows=7,local_options=dict(degrees=[1,1,1,0],sigma=[.2,.2,.2,0])))
            self.assertTrue(all(c.call_count==3 for c in calls))
        np.testing.assert_array_equal(np.concatenate([v['mean'] for _,_,v in chunks]),expected[0])
        np.testing.assert_allclose(np.concatenate([v['variance'] for _,_,v in chunks]),expected[2],rtol=1e-12,atol=1e-14)
        for key in expected[1]:
            np.testing.assert_allclose(np.concatenate([v['linear'][key] for _,_,v in chunks]),expected[1][key],rtol=1e-10,atol=1e-12,equal_nan=True)
        for key in expected[3]:
            np.testing.assert_allclose(np.concatenate([v['local'][key] for _,_,v in chunks]),expected[3][key],rtol=1e-10,atol=1e-12,equal_nan=True)
        for first,last,value in chunks:
            np.testing.assert_allclose(value['members']['Mean'].mean(1),expected[0][first:last],rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(value['members']['Linear'].mean(1),expected[1]['mean'][first:last],rtol=1e-10,atol=1e-12)

    def test_combined_two_iteration_seed_replay_and_requests(self):
        for layout in ('historical','fixed'):
            options=dict(block_sampling=layout,n_blocks=8,min_tree=8,mean_depth=12,niter_max=2,
                n_jobs=1,random_state=19,verbose=0,store_leaf_stats=True,smear_sigma=[.02,.02,.01,0])
            if layout=='historical':options.update(tree_fraction=.5/8,n_blocks=None)
            m=mod.MIForestErrPDF('Regressor',options).fit(self.X,self.X[:,0]+self.X[:,1])
            kw=dict(augment=True,random_state=51)
            expected=(m.predict(self.Q,**kw),m.predictLinear(self.Q,**kw),m.predictReducibleVariance(self.Q,**kw))
            chunks=list(m._combined_prediction_chunks(self.Q,linear=True,variance=True,chunk_rows=7,**kw))
            for key,value in zip(('mean','linear','variance'),expected):
                actual=np.concatenate([v[key]['mean'] if key=='linear' else v[key] for _,_,v in chunks])
                np.testing.assert_allclose(actual,value,rtol=1e-10,atol=1e-12)
            self.assertNotIn('members',chunks[0][2]);self.assertNotIn('local',chunks[0][2])
        with self.assertRaises(ValueError):
            list(m._combined_prediction_chunks(self.Q,augment=True,local_options={}))
    @classmethod
    def setUpClass(cls):
        global mod
        if mod is None:mod=load(_source,'phase23_candidate')
        cls.X=np.random.default_rng(731).uniform(-1,1,(1200,4)).astype('float32')
        cls.Q=np.random.default_rng(943).uniform(-1.3,1.3,(21,4)).astype('float32')

    def model(self, degrees=False, **tree_options):
        y=self.X[:,0].astype(float)+2*self.X[:,1]+3*self.X[:,2]
        if degrees:y=self.X[:,0].astype(float)**2+2*self.X[:,1]
        options=dict(block_sampling='historical',tree_fraction=.5/8,niter_max=1,
            n_jobs=1,random_state=17,verbose=0,store_leaf_stats=True,
            smear_sigma=[.2,.2,.2,0])
        return mod.MIForestErrPDF('Regressor',options,**tree_options).fit(self.X,y)

    def test_gaussian_mass_analytic_tails_and_zero_width(self):
        mass=mod._ri_gaussian_leaf_mass(np.array([[0.],[0.],[0.]]),
            np.array([[-np.inf],[0.],[8.]]),np.array([[0.],[np.inf],[9.]]),np.array([1.]))
        np.testing.assert_allclose(mass[:2],[.5,.5],atol=0,rtol=0)
        self.assertGreater(mass[2],0)
        np.testing.assert_allclose(mass[2],ndtr(-8)-ndtr(-9),rtol=1e-14)
        z=mod._ri_gaussian_leaf_mass(np.array([[0.],[0.]]),
            np.array([[-np.inf],[0.]]),np.array([[0.],[np.inf]]),np.array([0.]))
        np.testing.assert_array_equal(z,[1.,0.])

    def test_parent_zero_and_zero_sigma_preserve_unsmeared_tree_mean(self):
        m=self.model();expected=m.predict(self.Q)
        for levels,sigma in [(0,[.2,.2,.2,0]),(1,[0,0,0,0])]:
            out=m.predictLocalConvolution(self.Q,degrees=[1,1,1,0],sigma=sigma,parent_levels=levels)
            np.testing.assert_allclose(out['mean_parent_convolved'],expected,rtol=1e-14,atol=1e-14)
            np.testing.assert_allclose(out['parent_variance_raw'],m.predictReducibleVariance(self.Q),rtol=1e-13,atol=1e-14)

    def test_affine_recovery_active_coordinates_and_independent_halves(self):
        m=self.model();o=m.predictLocalConvolution(self.Q,degrees=[1,1,1,0],parent_levels=1)
        self.assertFalse(o['fallback'].any())
        expected=self.Q[:,0].astype(float)+2*self.Q[:,1]+3*self.Q[:,2]
        np.testing.assert_allclose(o['mean_local'],expected,atol=2e-6,rtol=1e-6)
        np.testing.assert_allclose(o['gradient'],np.tile([1,2,3,0],(len(self.Q),1)),atol=3e-6,rtol=1e-6)
        np.testing.assert_array_equal(o['mean_local'],o['mean_local_convolved'])
        # Half outputs use separate normal equations, not copied full-fit means.
        noisy=m.predictLocalConvolution(self.Q,degrees=[1,1,1,0],ridge=.1)
        self.assertGreater(np.max(abs(noisy['local_half_A']-noisy['local_half_B'])),1e-7)

    def test_quadratic_degree_and_analytic_polynomial_convolution(self):
        m=self.model(degrees=True)
        o=m.predictLocalConvolution(self.Q,degrees=[2,1,0,0],parent_levels=1)
        self.assertFalse(o['fallback'].any())
        expected=self.Q[:,0].astype(float)**2+2*self.Q[:,1]
        np.testing.assert_allclose(o['mean_local'],expected,atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(o['mean_local_convolved']-o['mean_local'],.2**2,atol=1e-10)
        np.testing.assert_allclose(o['curvature'][:,0],2.,atol=1e-9)

    def test_chunk_work_budget_determinism_and_baseline_unchanged(self):
        m=self.model();before=m.predict(self.Q).copy();linear=m.predictLinear(self.Q).copy()
        a=m.predictLocalConvolution(self.Q,degrees=[1,1,1,0],chunk_rows=21,max_working_mb=2)
        b=m.predictLocalConvolution(self.Q,degrees=[1,1,1,0],chunk_rows=3,max_working_mb=.001)
        for key in a:np.testing.assert_allclose(a[key],b[key],rtol=2e-8,atol=2e-9,equal_nan=True,err_msg=key)
        np.testing.assert_array_equal(before,m.predict(self.Q))
        np.testing.assert_array_equal(linear,m.predictLinear(self.Q))

    def test_full_root_convolution_against_independent_recursive_oracle(self):
        m=self.model();q=self.Q[:3];sigma=np.array([.2,.2,.2,0])
        o=m.predictLocalConvolution(q,degrees=[0,0,0,0],parent_levels=100000)
        results=[]
        for block in m.block_trees_:
            tree=block[0].tree_
            def visit(node,x,lo,hi):
                left,right=tree.children_left[node],tree.children_right[node]
                if left==-1:
                    factors=[]
                    for j,s in enumerate(sigma):
                        if s==0:factors.append(float(lo[j]<x[j]<=hi[j]))
                        else:
                            aa=(lo[j]-x[j])/s;bb=(hi[j]-x[j])/s
                            factors.append(ndtr(-aa)-ndtr(-bb) if aa>=0 else ndtr(bb)-ndtr(aa))
                    return np.prod(factors)*tree.value[node,0,0]
                j=tree.feature[node];cut=tree.threshold[node]
                high=hi.copy();high[j]=min(high[j],cut)
                low=lo.copy();low[j]=max(low[j],cut)
                return visit(left,x,lo,high)+visit(right,x,low,hi)
            results.append([visit(0,x,np.full(4,-np.inf),np.full(4,np.inf)) for x in q])
        np.testing.assert_allclose(o['mean_parent_convolved'],np.mean(results,axis=0),atol=1e-13)
        np.testing.assert_allclose(o['retained_mass'],1.,atol=1e-14)
        np.testing.assert_allclose(o['mean_local'],o['mean_parent_convolved'],atol=1e-13)

    def test_noisy_weighted_fit_against_independent_lstsq(self):
        m=self.model().fit(self.X,self.X[:,0]+np.random.default_rng(41).normal(size=len(self.X)))
        q=self.Q[:2];sigma=np.array([.2,.2,.2,0])
        actual=m.predictLocalConvolution(q,sigma=sigma,degrees=[1,1,1,0],parent_levels=1)
        for row,x in enumerate(q):
            designs=[];targets=[];weights=[];means=[]
            for block in m.block_trees_:
                fitted=block[0];tree=fitted.tree_;size=tree.node_count
                parent=np.zeros(size,dtype=int);lo=np.full((size,4),-np.inf);hi=-lo.copy()
                for node in range(size):
                    left,right=tree.children_left[node],tree.children_right[node]
                    if left==-1:continue
                    parent[left]=parent[right]=node
                    lo[left]=lo[right]=lo[node];hi[left]=hi[right]=hi[node]
                    j=tree.feature[node];hi[left,j]=tree.threshold[node];lo[right,j]=tree.threshold[node]
                ancestor=parent[fitted.apply(x[None,:])[0]]
                ids=[];mass=[]
                for leaf in np.flatnonzero(tree.children_left==-1):
                    cur=leaf
                    while cur!=ancestor and cur!=0:cur=parent[cur]
                    if cur!=ancestor:continue
                    factors=[]
                    for j,width in enumerate(sigma):
                        if width==0:factors.append(float(lo[leaf,j]<x[j]<=hi[leaf,j]))
                        else:
                            a=(lo[leaf,j]-x[j])/width;b=(hi[leaf,j]-x[j])/width
                            factors.append(ndtr(-a)-ndtr(-b) if a>=0 else ndtr(b)-ndtr(a))
                    ids.append(leaf);mass.append(np.prod(factors))
                weight=np.array(mass)/sum(mass);y=tree.value[ids,0,0]
                delta=fitted._ri_leaf_stats['centroid'][ids,:3]-x[:3]
                designs.append(np.column_stack([np.ones(len(ids)),delta]))
                targets.append(y);weights.append(weight);means.append(weight@y)
            X=np.vstack(designs);y=np.concatenate(targets);w=np.sqrt(np.concatenate(weights))
            beta=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]
            np.testing.assert_allclose(actual['mean_parent_convolved'][row],np.mean(means),atol=1e-12)
            np.testing.assert_allclose(actual['mean_local'][row],beta[0],atol=1e-10)
            np.testing.assert_allclose(actual['gradient'][row,:3],beta[1:],atol=1e-9)

    def test_fixed_and_historical_multi_iteration_variance_contract(self):
        for layout in ('fixed','historical'):
            options=dict(block_sampling=layout,n_blocks=8,min_tree=8,mean_depth=12,
                niter_max=2,n_jobs=1,random_state=19,verbose=0,store_leaf_stats=True)
            if layout=='historical':options.update(tree_fraction=.5/8,n_blocks=None)
            m=mod.MIForestErrPDF('Regressor',options).fit(self.X,self.X[:,0]+self.X[:,1])
            result=m.predictLocalConvolution(self.Q,sigma=[0,0,0,0],degrees=[1,1,1,0])
            np.testing.assert_allclose(result['parent_variance_raw'],m.predictReducibleVariance(self.Q),atol=1e-14,rtol=1e-12)

    def test_ridge_condition_and_coefficients_against_augmented_lstsq(self):
        # Root support makes every terminal region independently enumerable.
        # SVD of an augmented design supplies a solver/condition oracle distinct
        # from production normal-equation eigendecomposition and solve.
        m=self.model();q=self.Q[:3];sigma=np.array([.6,.6,.6,.6])
        for ridge in (0., .2, 2.):
            actual=m.predictLocalConvolution(q,sigma=sigma,degrees=[1,1,1,0],
                parent_levels=100000,ridge=ridge)
            for row,x in enumerate(q):
                designs=[];targets=[];weights=[];sides=[]
                for b,block in enumerate(m.block_trees_):
                    tree=block[0];stat=tree._ri_leaf_stats;ids=stat['leaf']
                    lo=(stat['cell_lower'][ids]-x)/sigma
                    hi=(stat['cell_upper'][ids]-x)/sigma
                    factors=np.where(lo>=0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo))
                    mass=factors.prod(axis=1);weight=mass/mass.sum()
                    delta=stat['centroid'][ids,:3]-x[:3]
                    designs.append(np.column_stack([np.ones(len(ids)),delta]))
                    targets.append(tree.tree_.value[ids,0,0]);weights.append(weight)
                    sides.append(np.full(len(ids),b%2))
                X=np.vstack(designs);y=np.concatenate(targets)
                w=np.concatenate(weights);side=np.concatenate(sides)
                for label,select,trees in [('full',np.ones(len(y),dtype=bool),len(m.block_trees_)),
                    ('A',side==0,len(m.block_trees_)//2),('B',side==1,len(m.block_trees_)//2)]:
                    design=X[select];weight=w[select]/trees
                    scales=np.sqrt(np.sum(weight[:,None]*design**2,axis=0))
                    weighted=(design/scales)*np.sqrt(weight[:,None])
                    penalty=np.diag([0.,np.sqrt(ridge),np.sqrt(ridge),np.sqrt(ridge)])
                    augmented=np.vstack([weighted,penalty])
                    target=np.r_[y[select]*np.sqrt(weight),np.zeros(4)]
                    beta=np.linalg.lstsq(augmented,target,rcond=None)[0]/scales
                    singular=np.linalg.svd(augmented,compute_uv=False)
                    condition=(singular[0]/singular[-1])**2
                    cond_key='condition' if label=='full' else 'local_half_'+label+'_condition'
                    mean_key='mean_local' if label=='full' else 'local_half_'+label
                    np.testing.assert_allclose(actual[cond_key][row],condition,rtol=1e-11,atol=1e-12)
                    np.testing.assert_allclose(actual[mean_key][row],beta[0],rtol=1e-10,atol=1e-12)
                    if label=='full':np.testing.assert_allclose(actual['gradient'][row,:3],beta[1:],rtol=1e-10,atol=1e-12)

    def test_one_singular_half_is_reported_without_hiding_full_fit(self):
        m=self.model()
        # Controlled geometry-statistic fixture: half A has no identifiable A
        # slope; half B and the full fit retain independent varying centroids.
        for b,block in enumerate(m.block_trees_):
            if b%2==0:
                for tree in block:tree._ri_leaf_stats['centroid'][:,0]=0.
        q=self.Q[:5]
        actual=m.predictLocalConvolution(q,sigma=[.6]*4,degrees=[1,1,1,0],parent_levels=100000)
        self.assertTrue(actual['local_half_A_fallback'].all())
        self.assertTrue((actual['local_half_A_condition']>1e12).all())
        self.assertFalse(actual['local_half_B_fallback'].any())
        self.assertFalse(actual['fallback'].any())
        np.testing.assert_array_equal(actual['local_half_A'],actual['parent_half_A'])
        self.assertEqual(actual['local_half_A_fallback'].dtype,np.dtype(bool))

    def test_empty_half_diagnostics_are_shaped(self):
        out=self.model().predictLocalConvolution(self.Q[:0],degrees=[1,1,1,0])
        for label in ('A','B'):
            self.assertEqual(out['local_half_'+label+'_fallback'].shape,(0,))
            self.assertEqual(out['local_half_'+label+'_condition'].shape,(0,))
            self.assertEqual(out['local_half_'+label+'_fallback'].dtype,np.dtype(bool))

    def test_condition_gate_uses_unpenalized_intercept_at_large_ridge(self):
        m=self.model()
        for block in m.block_trees_:
            for tree in block:tree._ri_leaf_stats['centroid'][:,0]=0.
        q=self.Q[:1].copy();q[:,0]=1.
        # Scaled design [1,-1]: actual matrix [[1,-1],[-1,1+ridge]].
        # The old (eigen+ridge) ratio approaches 1 and wrongly admits both.
        for ridge,expected_fallback in ((1e11,False),(2e12,True)):
            out=m.predictLocalConvolution(q,sigma=[.6]*4,degrees=[1,0,0,0],
                parent_levels=100000,ridge=ridge)
            matrix=np.array([[1.,-1.],[-1.,1.+ridge]])
            expected=np.linalg.cond(matrix)
            np.testing.assert_allclose(out['condition'],expected,rtol=1e-12)
            self.assertEqual(bool(out['fallback'][0]),expected_fallback)
            for label in ('A','B'):
                self.assertEqual(bool(out['local_half_'+label+'_fallback'][0]),expected_fallback)
            if expected_fallback:
                np.testing.assert_array_equal(out['mean_local'],out['mean_parent_convolved'])

    def test_batched_cdf_matches_coordinate_reference_exactly(self):
        rng=np.random.default_rng(519)
        q=rng.normal(size=(1001,4));lo=q+rng.normal(size=q.shape)
        hi=lo+rng.uniform(0,2,size=q.shape)
        lo[::11]=-np.inf;hi[::13]=np.inf
        for sigma in (np.array([.02,.02,.01,0]),np.ones(4),np.zeros(4)):
            expected=np.ones(len(q))
            for j,width in enumerate(sigma):
                if width==0:expected*=(q[:,j]>lo[:,j])&(q[:,j]<=hi[:,j])
                else:
                    a=(lo[:,j]-q[:,j])/width;b=(hi[:,j]-q[:,j])/width
                    expected*=np.maximum(0.,np.where(a>=0,ndtr(-a)-ndtr(-b),ndtr(b)-ndtr(a)))
            np.testing.assert_array_equal(mod._ri_gaussian_leaf_mass(q,lo,hi,sigma),expected)

    def test_ragged_pair_order_and_work_bound(self):
        for counts in (np.array([0,1,9,0,2,33]),np.zeros(5,dtype=int),np.array([101])):
            starts=np.arange(len(counts))*1000
            cumulative=np.r_[0,np.cumsum(counts)]
            qi=np.searchsorted(cumulative[1:],np.arange(cumulative[-1]),side='right')
            li=starts[qi]+np.arange(cumulative[-1])-cumulative[qi]
            for limit in (1,7,10000):
                chunks=list(mod._ri_descendant_chunks(starts,counts,limit))
                self.assertTrue(all(len(a)<=limit for a,b in chunks))
                actual_q=np.concatenate([a for a,b in chunks]) if chunks else np.array([],dtype=int)
                actual_l=np.concatenate([b for a,b in chunks]) if chunks else np.array([],dtype=int)
                np.testing.assert_array_equal(actual_q,qi);np.testing.assert_array_equal(actual_l,li)

    def test_geometry_cache_reuse_level_replacement_and_refit(self):
        m=self.model();tree=m.block_trees_[0][0]
        first=mod._ri_parent_geometry(tree,1)
        self.assertIs(first,mod._ri_parent_geometry(tree,1))
        changed=mod._ri_parent_geometry(tree,0)
        self.assertIsNot(first,changed)
        self.assertIs(changed,mod._ri_parent_geometry(tree,0))
        np.testing.assert_array_equal(changed['ancestor'],np.arange(tree.tree_.node_count))
        leaf=tree._ri_leaf_stats['leaf']
        np.testing.assert_array_equal(np.sort(changed['value']),np.sort(tree.tree_.value[leaf,0,0]))
        m.fit(self.X,self.X[:,0])
        replacement=m.block_trees_[0][0]
        self.assertIsNot(tree,replacement)
        self.assertIsNot(first,mod._ri_parent_geometry(replacement,1))

    def test_thread_reduction_order_and_single_job_dispatch(self):
        m=self.model();q=self.Q
        one=m.predictLocalConvolution(q,degrees=[1,1,1,0],max_working_mb=2.)
        for jobs in (2,4):
            m.optionsErrPDF['n_jobs']=jobs
            actual=m.predictLocalConvolution(q,degrees=[1,1,1,0],max_working_mb=2.)
            for key in one:np.testing.assert_array_equal(one[key],actual[key],err_msg=key)
        m.optionsErrPDF['n_jobs']=1
        # Fitting is outside this patch. Prediction must not instantiate joblib
        # Parallel at jobs=1, including augmented streams and statistics.
        with patch.object(mod,'Parallel',side_effect=AssertionError('serial dispatch')):
            m.predict(q);m.predictLinear(q);m.predictReducibleVariance(q)
            m.predict(q,augment=True,random_state=4)
            m.predictStat(q,{'mean':True,'quantile':[.1,.5,.9]},treeType='all')

    def test_singular_design_falls_back_without_inventing_gradient(self):
        X=self.X.copy();X[:,3]=0
        m=self.model().fit(X,X[:,0]+2*X[:,1])
        Q=self.Q.copy();Q[:,3]=0
        o=m.predictLocalConvolution(Q,degrees=[1,1,1,1])
        self.assertTrue(o['fallback'].all())
        self.assertTrue(np.isnan(o['gradient']).all())
        np.testing.assert_array_equal(o['mean_local'],o['mean_parent_convolved'])

    def test_invalid_options_growth_empty_and_fallback(self):
        m=self.model()
        for kw in [dict(degrees=[True,1,1,0]),dict(degrees=[1,1]),dict(degrees=[3,1,1,0]),
                   dict(parent_levels=-1),dict(parent_levels=True),dict(chunk_rows=0),
                   dict(sigma=[-1,0,0,0]),dict(max_working_mb=0),dict(ridge=-1)]:
            with self.assertRaises(ValueError):m.predictLocalConvolution(self.Q,**kw)
        with self.assertRaisesRegex(ValueError,'unrestricted'):
            self.model(min_samples_leaf=2).predictLocalConvolution(self.Q)
        out=m.predictLocalConvolution(self.Q[:0],degrees=[1,1,1,0])
        self.assertEqual(out['gradient'].shape,(0,4))
        degenerate=m.predictLocalConvolution(self.Q,degrees=[1,1,1,1],sigma=[0,0,0,0],parent_levels=0)
        # D is retained only if requested; no hardwired removal in model API.
        self.assertEqual(degenerate['gradient'].shape,(len(self.Q),4))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=_source)
    p.add_argument('--serial-only',action='store_true',help='exclude the pre-existing multiworker test')
    a,remaining=p.parse_known_args();mod=load(a.source,'phase23_candidate')
    if a.serial_only:
        remaining += ['LocalConvolutionContracts.'+name for name in unittest.defaultTestLoader.getTestCaseNames(LocalConvolutionContracts)
            if name!='test_thread_reduction_order_and_single_job_dispatch']
    unittest.main(argv=[sys.argv[0]]+remaining,verbosity=2)
