"""Stage C engineering oracles; these do not replace Stage D calibration."""
import copy
import os
import importlib.util
import itertools
import unittest
from unittest.mock import patch
import numpy as np
if os.environ.get('RI_MIFOREST_SOURCE'):
    spec=importlib.util.spec_from_file_location('ri_stage_c_candidate',os.environ['RI_MIFOREST_SOURCE'])
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
else:
    from RootInteractive.MLpipeline import MIForestErrPDF as mod


def model(**options):
    opts = dict(min_tree=6, mean_depth=10, niter_max=1, n_jobs=1,
                random_state=42, verbose=0, fit_noise=True,
                noise_options=dict(n_estimators=12, min_samples_leaf=5))
    opts.update(options)
    return mod.MIForestErrPDF('Regressor', opts, min_samples_leaf=2)


class NoiseContracts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.X = rng.uniform(-1, 1, (180, 2))
        self.y = self.X[:, 0] + rng.normal(0, .2, len(self.X))

    def test_C01_actual_row_group_exclusion_and_fold_variance(self):
        groups = np.arange(len(self.X)) // 3
        m = model(mean_depth=3).fit(self.X, self.y, groups=groups)
        np.testing.assert_array_equal(np.sort(np.concatenate([f['heldout_indices'] for f in m.oof_folds_])), np.arange(180))
        for f in m.oof_folds_:
            train, held = f['train_indices'], f['heldout_indices']
            self.assertEqual(f['block_count'],18)
            self.assertEqual(m.block_count_,22)
            self.assertFalse(set(groups[train]) & set(groups[held]))
            np.testing.assert_array_equal(np.sort(np.concatenate(f['block_indices'])), train)
            # Independently refit the fold from public constructor/fit and saved seed.
            child = model(fit_noise=False, mean_depth=m.mean_depth_, random_state=f['seed']).fit(self.X[train], self.y[train], groups=groups[train])
            members = np.array([np.mean([t.predict(self.X[held]) for t in block], axis=0) for block in child.block_trees_])
            np.testing.assert_allclose(m.oof_mean_[held], members.mean(0))
            np.testing.assert_allclose(m.oof_reducible_var_[held], np.var(members[1::2]-members[::2], axis=0, ddof=1)/(4*child.n_pairs_))
        with self.assertRaises(ValueError): m.predictUncertainty(self.X)
        with self.assertWarns(RuntimeWarning): out = m.predictUncertainty(self.X, experimental=True)
        self.assertFalse(out['metadata']['validated'])
        with self.assertWarns(RuntimeWarning): ref = m.predictReducibleStd(self.X, experimental=True)
        np.testing.assert_array_equal(out['reducible_std'], ref)

    def test_C02_signed_subtraction_before_clipping_and_outputs(self):
        def members(owner, values):
            predictions = np.tile(np.array([0.,-2.,0.,0.,0.,2.])[:, None], (1, len(values)))
            yield 0, predictions
        with patch.object(mod.MIForestErrPDF, 'predict', lambda owner, X: np.zeros(len(X))), patch.object(mod.MIForestErrPDF, '_member_chunks', members):
            m = model().fit(self.X, np.zeros(len(self.X)))
        np.testing.assert_allclose(m.oof_signed_target_, -1/3)
        out = m.predictUncertainty(self.X)
        np.testing.assert_allclose(out['irreducible_var_raw'], -1/3)
        np.testing.assert_array_equal(out['irreducible_std'], 0.)
        np.testing.assert_array_equal(out['mean'], m.predict(self.X))
        np.testing.assert_array_equal(out['reducible_std'], m.predictReducibleStd(self.X))
        np.testing.assert_allclose(out['predictive_std']**2, out['reducible_std']**2 + np.maximum(out['irreducible_var_raw'],0))
        for k,v in m.predictUncertainty(self.X[:0]).items():
            if k != 'metadata': self.assertEqual(v.shape, (0,))
        out['metadata']['validated'] = True
        self.assertFalse(m.uncertainty_status_['validated'])

    def test_C03_noise_stress_diagnostics_and_algebra(self):
        rng = np.random.default_rng(88)
        for sigma, underfit, outliers in [(0,False,False),(.2,False,False),(.1+.2*(self.X[:,0]+1)/2,False,False),(.2,True,True)]:
            y = np.sin(3*self.X[:,0]) + rng.normal(size=len(self.X))*sigma
            if outliers: y[::30] += 8
            m = model()
            if underfit: m.options['max_depth'] = 1
            m.fit(self.X,y)
            out = m.predictUncertainty(self.X[:20])
            self.assertTrue(all(np.isfinite(v).all() for k,v in out.items() if k!='metadata'))
            self.assertFalse(out['metadata']['validated'])
            np.testing.assert_allclose(m.oof_signed_target_, (y-m.oof_mean_)**2-m.oof_reducible_var_)
            self.assertTrue(np.all(out['predictive_std'] >= out['reducible_std']))

    def test_noise_refits_failure_seeds_and_weights(self):
        m = model().fit(self.X,self.y)
        a = model().fit(self.X,self.y)
        for k,v in m.predictUncertainty(self.X).items():
            if k!='metadata': np.testing.assert_array_equal(v,a.predictUncertainty(self.X)[k])
        m.optionsErrPDF['fit_noise']=False; m.fit(self.X,self.y)
        self.assertFalse(hasattr(m,'noise_model_')); self.assertFalse(hasattr(m,'oof_mean_'))
        with self.assertRaises(RuntimeError): m.predictUncertainty(self.X)
        m.optionsErrPDF['fit_noise']=True; m.fit(self.X,self.y)
        m.optionsErrPDF['noise_options']={'criterion':'absolute_error'}
        with self.assertRaises(ValueError): m.fit(self.X,self.y)
        self.assertFalse(hasattr(m,'is_fitted_')); self.assertFalse(hasattr(m,'noise_model_'))
        small = model(min_tree=6)
        with self.assertRaises(ValueError): small.fit(self.X[:6],self.y[:6])
        w=np.ones(len(self.y));w[:4]=0
        z=model().fit(self.X,self.y,w)
        self.assertTrue(np.isnan(z.oof_mean_[:4]).all())
        self.assertTrue(np.all(z.oof_fold_ids_[:4]==-1))
        plain=model().fit(self.X,self.y)
        uniform=model().fit(self.X,self.y,np.full(len(self.y),3.))
        for k,v in plain.predictUncertainty(self.X).items():
            if k!='metadata': np.testing.assert_array_equal(v,uniform.predictUncertainty(self.X)[k])

    def test_B_review_hardening_and_group_optimization_semantics(self):
        X=self.X; y=self.y
        a=model(fit_noise=False).fit(X,y)
        for weight in [.1,3.,7.3]:
            b=model(fit_noise=False).fit(X,y,np.full(len(y),weight))
            np.testing.assert_array_equal(a.predict(X),b.predict(X))
            np.testing.assert_array_equal(a.predictReducibleStd(X),b.predictReducibleStd(X))
        r=mod.MIForestErrPDF('Regressor',dict(min_tree=6,niter_max=3,random_state=31,n_jobs=1),splitter='random').fit(X,y)
        members=np.array([np.mean([t.predict(X) for t in block],axis=0) for block in r.block_trees_])
        np.testing.assert_allclose(r.predict(X),members.mean(0))
        np.testing.assert_allclose(r.predictReducibleStd(X),np.sqrt(np.var(members[1::2]-members[::2],axis=0,ddof=1)/(4*r.n_pairs_)))
        with self.assertRaises(ValueError): r.predictStat(X,treeType=True)
        before=np.random.get_state(); r.predictStat(X,{'std2':0},treeType='all'); after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1]); self.assertEqual(before[2:],after[2:])
        # Compare optimized grouping against the original slow exact algorithm.
        codes=np.repeat(np.arange(20),np.arange(1,21)); active=np.arange(len(codes))
        rng=np.random.default_rng(73)
        by=[active[codes==k] for k in range(20)]
        order=sorted(rng.permutation(20),key=lambda k:-len(by[k]))
        buckets=[[] for _ in range(6)];sizes=np.zeros(6,int);ties=rng.permutation(6)
        for k in order:
            b=min(ties,key=lambda i:sizes[i]);buckets[b].append(by[k]);sizes[b]+=len(by[k])
        actual,_=mod.MIForestErrPDF._group_blocks(active,codes,6,np.random.default_rng(73))
        for expected,got in zip(buckets,actual):np.testing.assert_array_equal(np.concatenate(expected),got)
        grouped=model(fit_noise=False).fit(X,y,groups=np.arange(180)//3)
        ordered=np.sort(grouped.block_sizes_)
        self.assertEqual(sorted(abs(grouped.block_sizes_[::2]-grouped.block_sizes_[1::2])),sorted(abs(ordered[::2]-ordered[1::2])))
        with self.assertWarns(RuntimeWarning): sig=grouped.predictReducibleStd(X,experimental=True)
        block=np.array([member.predict(X) for member in grouped.members_])
        np.testing.assert_allclose(sig,np.sqrt(np.var(block[1::2]-block[::2],axis=0,ddof=1)/(4*grouped.n_pairs_)))

    def test_weighted_noise_reference_and_private_rng(self):
        from sklearn.ensemble import RandomForestRegressor
        weights=np.linspace(1,3,len(self.y));weights[:4]=0
        before=np.random.get_state()
        m=model().fit(self.X,self.y,weights)
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])
        active=m.active_indices_
        reference=RandomForestRegressor(**m.noise_options_).fit(self.X[active].astype(np.float32),m.oof_signed_target_[active],sample_weight=weights[active])
        with self.assertRaises(ValueError):m.predictUncertainty(self.X)
        with self.assertWarns(RuntimeWarning):out=m.predictUncertainty(self.X,experimental=True)
        np.testing.assert_array_equal(out['irreducible_var_raw'],reference.predict(self.X.astype(np.float32)))
        off=model(fit_noise=False).fit(self.X,self.y,weights)
        np.testing.assert_array_equal(out['mean'],off.predict(self.X))

    def test_migrated_actual_append_prediction_function(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace
        import pandas as pd
        from RootInteractive.MLpipeline.NDFunctionInterface import Fitter
        path=Path(__file__).resolve().parents[1]/'test_RandomForestErrPDF.py'
        tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='appendPrediction')
        fitted=model().fit(self.X,self.y)
        fitter=Fitter(SimpleNamespace(X_values=['a','b']))
        fitter.Register_Model('noise',fitted)
        df=pd.DataFrame(self.X,columns=['a','b']);df['value']=self.y;df['valueOrig']=self.X[:,0]
        namespace=dict(np=np,doFitErrRF=lambda *args:(fitter,df))
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        _,out=namespace['appendPrediction'](len(df))
        expected=fitted.predictUncertainty(self.X)
        np.testing.assert_allclose(out['irrError'],expected['irreducible_std'])
        np.testing.assert_allclose(out['stdPoint'],expected['predictive_std'])
        np.testing.assert_allclose(out['irreducibleVarRaw'],expected['irreducible_var_raw'])

if __name__=='__main__':unittest.main(verbosity=2)
