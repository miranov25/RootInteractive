"""Step 1: compare candidate with independently loaded frozen source files."""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class QuantileBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global mod,old,bench,old_bench
        if 'mod' not in globals():
            names=['RI_QUANTILE_SOURCE','RI_QUANTILE_BASELINE_SOURCE',
                   'RI_QUANTILE_BENCHMARK','RI_QUANTILE_BASELINE_BENCHMARK']
            if not all(os.environ.get(name) for name in names):
                raise unittest.SkipTest('baseline-comparison test requires four RI_QUANTILE_* paths or its command-line entry point')
            mod,old,bench,old_bench=[load(os.environ[name],name) for name in names]

    def test_distribution_legacy_bytes(self):
        rng=np.random.default_rng(392)
        for members in (1,2,3,8,9,512):
            for values in (rng.normal(size=(41,members)),np.zeros((7,members)),
                           np.full((5,members),-0.),np.full((4,members),2.)):
                before=values.copy()
                for label in ('Mean','Linear','RF'):
                    for statistics in ([],['sigma'],['quantiles'],['sigma','quantiles']):
                        expected=old_bench.distribution(values,label,statistics)
                        actual=bench.distribution(values,label,statistics)
                        self.assertEqual(list(expected),list(actual))
                        for key in expected:self.assertEqual(expected[key].tobytes(),actual[key].tobytes(),key)
                self.assertEqual(before.tobytes(),values.tobytes())

    def test_one_quantile_call_and_no_median(self):
        values=np.random.default_rng(91).normal(size=(101,32))
        with patch.object(np,'quantile',wraps=np.quantile) as q, patch.object(np,'median',side_effect=AssertionError('extra median')):
            bench.distribution(values,'Linear',['sigma','quantiles'])
        self.assertEqual(q.call_count,1)

    def test_probability_names_and_order(self):
        for i in range(100):self.assertEqual(bench.quantile_suffix(i/100),str(i))
        self.assertEqual(bench.quantile_suffix(.025),'2.5')
        self.assertEqual(bench.quantile_suffix(1.),'100')
        values=np.random.default_rng(2).normal(size=(23,12))
        probabilities=[1.,.55,.29,0.,.55,.025]
        saved=probabilities.copy()
        result=bench.distribution(values,'Mean',['quantiles'],probabilities)
        self.assertEqual(probabilities,saved)
        for p in probabilities:
            np.testing.assert_array_equal(result['predict_quantile'+bench.quantile_suffix(p)+'_Mean'],np.quantile(values,p,axis=1))
        for p in (True,np.bool_(False),-.1,1.1,np.nan,np.inf):
            with self.assertRaises(ValueError):bench.quantile_suffix(p)
        for p in ([],[True],[float('nan')]):
            with self.assertRaises(ValueError):bench.distribution(values,'Mean',['quantiles'],p)

    def test_helpers_legacy_bytes_chunks_jobs(self):
        from sklearn.tree import DecisionTreeRegressor
        rng=np.random.default_rng(5);X=rng.normal(size=(49,3));y=rng.normal(size=49)
        trees=[DecisionTreeRegressor(max_depth=3,random_state=i).fit(X,y+rng.normal(size=49)) for i in range(8)]
        request={'mean':True,'std':True,'std2':True,'median':True,'quantile':[1.,.3,.5,0.,.3,.55],'trim_mean':[.1]}
        saved=copy.deepcopy(request)
        for count in (7,8):
            for chunk in (1,13,10000):
                for jobs in (1,2):
                    np.random.seed(71)
                    expected=old._forest_statistics(trees[:count],X,request,jobs,chunk,'quantiles')
                    np.random.seed(71)
                    actual=mod._forest_statistics(trees[:count],X,request,jobs,chunk,'quantiles')
                    for key in expected:
                        if isinstance(expected[key],dict):
                            for p in expected[key]:self.assertEqual(expected[key][p].tobytes(),actual[key][p].tobytes(),str((key,p,chunk,jobs)))
                        else:self.assertEqual(expected[key].tobytes(),actual[key].tobytes(),key)
        self.assertEqual(request,saved)
        for request in ({'median':True},{'quantile':[]},{'quantile':.5},{'quantile':1.0},{}):
            for query in (X,X[:0]):
                expected=old._forest_statistics(trees,query,request,1,13,'quantile')
                with patch.object(np,'quantile',wraps=np.quantile) as calls:
                    actual=mod._forest_statistics(trees,query,request,1,13,'quantile')
                self.assertEqual(calls.call_count,0 if not len(query) or not request or request=={'quantile':[]} else 4)
                for key in expected:
                    if isinstance(expected[key],dict):
                        for p in expected[key]:self.assertEqual(expected[key][p].tobytes(),actual[key][p].tobytes())
                    else:self.assertEqual(expected[key].tobytes(),actual[key].tobytes())

    def test_boolean_refusal_before_prediction(self):
        class NeverTree:
            def predict(self,X):raise AssertionError('prediction happened')
        for value in (True,False,np.bool_(True),[.1,True],np.array([True])):
            with self.assertRaisesRegex(ValueError,'Boolean'):
                mod._forest_statistics([NeverTree()],np.ones((2,1)),{'quantile':value},1,10,'quantile')
        self.assertEqual(mod._stat_requests({'quantile':1})['quantile'],[1.])

    def test_model_public_outputs(self):
        X=np.random.default_rng(92).normal(size=(200,3));y=X[:,0]+X[:,1]**2
        options=dict(min_tree=8,n_blocks=8,niter_max=2,n_jobs=1,random_state=12,fit_noise=False)
        left=old.MIForestErrPDF('Regressor',options,min_samples_leaf=1).fit(X,y)
        right=mod.MIForestErrPDF('Regressor',options,min_samples_leaf=1).fit(X,y)
        self.assertEqual(left.predict(X).tobytes(),right.predict(X).tobytes())
        for group in (0,1,'all'):
            req={'mean':True,'median':True,'std':True,'quantile':[.1,.3,.5,.7,.9]}
            a=left.predictStat(X,req,treeType=group);b=right.predictStat(X,req,treeType=group)
            for key in a:
                if isinstance(a[key],dict):
                    for p in a[key]:self.assertEqual(a[key][p].tobytes(),b[key][p].tobytes())
                else:self.assertEqual(a[key].tobytes(),b[key].tobytes())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--baseline-source',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--baseline-benchmark',type=Path,required=True)
    a,remaining=p.parse_known_args()
    mod=load(a.source,'candidate');old=load(a.baseline_source,'baseline')
    bench=load(a.benchmark,'bench');old_bench=load(a.baseline_benchmark,'old_bench')
    unittest.main(argv=['test_quantile_batch']+remaining,verbosity=1)
