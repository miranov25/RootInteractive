"""Three-function local-linear benchmark, pandas diagnostics and mean reference.

Zero-response reference also disables input noise, isolating approximation on
finite sampled designs. Noisy case uses response N(0,1) and configured input errors.
Smearing is independently selectable; default off to isolate linear prediction.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import zipfile
import numpy as np


def response(X, function):
    if function=='linear':return X[:,0]+2*X[:,1]+3*X[:,2]
    if function=='quadratic':return X[:,0]+2*X[:,1]+3*X[:,2]**2
    return 3*X[:,0]+X[:,1]*np.sin(2*np.pi*X[:,2])


def slopes(X, function):
    out=np.zeros_like(X,dtype=float)
    if function in ('linear','quadratic'):
        out[:,0]=1;out[:,1]=2;out[:,2]=3 if function=='linear' else 6*X[:,2]
    else:
        out[:,0]=3;out[:,1]=np.sin(2*np.pi*X[:,2]);out[:,2]=2*np.pi*X[:,1]*np.cos(2*np.pi*X[:,2])
    return out


def conditional_mean(Q, sigma, function, nodes=256):
    Q=np.asarray(Q,dtype=float)
    z,w=np.polynomial.legendre.leggauss(nodes);z=(z+1)/2;w=w/2
    transforms=[lambda x:x,lambda x:x,
                (lambda x:x*x) if function=='quadratic' else ((lambda x:np.sin(2*np.pi*x)) if function=='sinus' else (lambda x:x))]
    moments=[]
    for j,fun in enumerate(transforms):
        if sigma[j]==0:moments.append(fun(Q[:,j]));continue
        log=-.5*((z[None,:]-Q[:,j,None])/sigma[j])**2
        weights=np.exp(log-log.max(1,keepdims=True))*w
        weights/=weights.sum(1,keepdims=True)
        moments.append(weights@fun(z))
    a,b,c=moments
    return 3*a+b*c if function=='sinus' else a+2*b+3*c


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path)
    p.add_argument('--n-train',type=int,default=50000);p.add_argument('--n-eval',type=int,default=512)
    p.add_argument('--replicas',type=int,default=4);p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--iterations',type=int,default=4);p.add_argument('--fraction',type=float,default=1/128)
    p.add_argument('--leaf',type=int,default=1);p.add_argument('--ridge',type=float,default=0.)
    p.add_argument('--condition-limit',type=float,default=1e10)
    p.add_argument('--input-sigma',type=float,nargs=4,default=[.02,.02,.01,0])
    p.add_argument('--smear-sigma',type=float,nargs=4,default=[.05,.05,.05,.3])
    p.add_argument('--fit-augment',action='store_true');p.add_argument('--predict-augment',action='store_true')
    p.add_argument('--functions',nargs='+',choices=['linear','quadratic','sinus'],default=['linear','quadratic','sinus'])
    p.add_argument('--cases',nargs='+',choices=['zero','noisy'],default=['zero','noisy'])
    p.add_argument('--seed',type=int,default=1731);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if not 1<=a.n_train<=100000 or min(a.n_eval,a.replicas,a.jobs,a.iterations,a.leaf)<1 or a.replicas<2:p.error('positive counts, replicas>=2, n-train<=100000 required')
    for widths in [a.input_sigma,a.smear_sigma]:
        if np.any(~np.isfinite(widths)) or np.any(np.asarray(widths)<0):p.error('invalid sigma')
    source=(a.source or Path(__file__).resolve().parents[1]/'MIForestErrPDF.py').resolve()
    spec=importlib.util.spec_from_file_location('phase22_linear_bench',source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'candidate_MIForestErrPDF.py').write_bytes(source.read_bytes())
    (a.output/'benchmark_executed.py').write_bytes(Path(__file__).read_bytes())
    from sklearn import __version__ as sklearn_version
    import pandas as pd
    from threadpoolctl import threadpool_limits
    def dump(name, value):(a.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    dump('invocation.json',dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),python=platform.python_version(),numpy=np.__version__,sklearn=sklearn_version))
    records=[];summaries=[];arrays={};tick_all=time.perf_counter()
    for function in a.functions:
        for case in a.cases:
            sigma=np.zeros(4) if case=='zero' else np.asarray(a.input_sigma)
            qrng=np.random.default_rng(a.seed+500000)
            latent_q=qrng.random((a.n_eval,4));Q=(latent_q+qrng.normal(size=latent_q.shape)*sigma).astype('float32')
            truth=conditional_mean(Q,sigma,function)
            fine=conditional_mean(Q,sigma,function,512)
            if abs(truth-fine).max()>1e-7:raise ValueError('conditional truth quadrature did not converge')
            outputs={key:[] for key in ['mean','linear']};grad=[];fallback=[]
            for r in range(a.replicas):
                rng=np.random.default_rng(a.seed+r)
                latent=rng.random((a.n_train,4)).astype('float32')
                y=response(latent.astype(float),function)+(rng.normal(size=a.n_train) if case=='noisy' else 0.)
                X=(latent+rng.normal(size=latent.shape)*sigma).astype('float32')
                model=mod.MIForestErrPDF('Regressor',dict(block_sampling='historical',tree_fraction=a.fraction,
                    niter_max=a.iterations,n_jobs=a.jobs,random_state=(a.seed+r)%2**32,
                    fit_noise=False,verbose=0,store_leaf_stats=True,smear_sigma=a.smear_sigma),min_samples_leaf=a.leaf)
                kw=dict(augment=True,random_state=(a.seed+r+900000)%2**32) if a.predict_augment else {}
                with threadpool_limits(limits=a.jobs):
                    t=time.perf_counter();model.fit(X,y,augment=a.fit_augment);fit_time=time.perf_counter()-t
                    t=time.perf_counter();mean=model.predict(Q,**kw);mean_time=time.perf_counter()-t
                    t=time.perf_counter();result=model.predictLinear(Q,ridge=a.ridge,condition_limit=a.condition_limit,return_diagnostics=True,**kw);linear_time=time.perf_counter()-t
                outputs['mean'].append(mean);outputs['linear'].append(result['mean'])
                grad.append(result['gradient']);fallback.append(result['fallback'])
                records.append(dict(function=function,case=case,replica=r,status='EXECUTED',fit_seconds=fit_time,
                    mean_seconds=mean_time,linear_seconds=linear_time,fallback_fraction=float(result['fallback'].mean()),trees=len(model.estimators_),
                    leaf_metadata_bytes=sum(v.nbytes for t in model.estimators_ for v in t._ri_leaf_stats.values())))
                dump('records.json',records)
                # First-refit dataframe examples; all leaf rows exported for one model.
                if r==0:
                    frame=model.predictLinearFrame(pd.DataFrame(Q,columns=list('ABCD')),ridge=a.ridge,condition_limit=a.condition_limit,**kw)
                    frame['truth']=truth;frame['reference_mean']=mean
                    for j in range(4):
                        frame['input_sigma_'+str(j)]=sigma[j];frame['smear_sigma_'+str(j)]=a.smear_sigma[j]
                    frame.to_csv(a.output/(function+'_'+case+'_queries.csv'),index=True,index_label='query_position')
                if r==0 and function==a.functions[0] and case==a.cases[0]:
                    model.leafStatsFrame().to_csv(a.output/'leaf_stats.csv',index=False)
                    model.leafQueryFrame(Q[:8],**kw).to_csv(a.output/'selected_leaves.csv',index=False)
                print(function,case,r,'fit=%.3fs mean=%.3fs linear=%.3fs fallback=%.3f'%(fit_time,mean_time,linear_time,result['fallback'].mean()),flush=True)
            for mode in ['mean','linear']:
                pred=np.asarray(outputs[mode]);bias=pred.mean(0)-truth
                row=dict(function=function,case=case,mode=mode,rmse=float(np.sqrt(np.mean((pred-truth)**2))),
                    squared_bias_of_replica_mean=float(np.mean(bias**2)),variance_ddof0=float(np.var(pred,axis=0).mean()),
                    refit_variance_ddof1=float(np.var(pred,axis=0,ddof=1).mean()),
                    fallback_fraction=float(np.mean(fallback)) if mode=='linear' else None)
                if mode=='linear' and case=='zero':
                    gradients=np.asarray(grad);valid=np.isfinite(gradients).all(2)
                    target=slopes(Q.astype(float),function)
                    row['gradient_rmse']=float(np.sqrt(np.mean((gradients-np.broadcast_to(target,gradients.shape))[valid]**2))) if valid.any() else None
                summaries.append(row);arrays[function+'_'+case+'_'+mode]=pred
            arrays[function+'_'+case+'_truth']=truth;arrays[function+'_'+case+'_query']=Q
    dump('metrics.json',summaries);np.savez_compressed(a.output/'predictions.npz',**arrays)
    lines=['# Three-function local-linear benchmark','','**Status:** development evidence, no independent approval or calibrated local-linear uncertainty.','',
        f'N={a.n_train}, evaluation rows={a.n_eval}, independent refits={a.replicas}, fraction={a.fraction:g}, L={a.iterations}, leaf={a.leaf}, ridge={a.ridge:g}.',
        f'Fit augmentation={a.fit_augment}; prediction augmentation={a.predict_augment}; actual additional widths={a.smear_sigma}.','',
        'Functions: A+2B+3C; A+2B+3C²; 3A+B*sin(2πC). Latent coordinates independent Uniform[0,1).',
        'Reference zero: response and input noise both disabled. It retains finite-design sampling effects. Noisy: response N(0,1), independent input measurement errors '+str(a.input_sigma)+'.',
        'Primary truth is E[y|observed X], checked by 256/512-node posterior quadrature. Slopes are compared with known function derivatives only in the input-noise-free reference.','',
        '| Function | Case | Predictor | RMSE | Squared bias of replica mean | Variance (ddof=0) | Linear fallback |','|---|---|---|---:|---:|---:|---:|']
    for m in summaries:lines.append('| %s | %s | %s | %.6g | %.6g | %.6g | %s |'%(m['function'],m['case'],m['mode'],m['rmse'],m['squared_bias_of_replica_mean'],m['variance_ddof0'],'N/A' if m['fallback_fraction'] is None else '%.3f'%m['fallback_fraction']))
    lines+=['','RMSE² equals the displayed sample bias² plus ddof=0 variance. Sample-mean bias² still includes finite-refit uncertainty; it is not an unbiased population-bias estimator. Four refits are a development screen, not acceptance.','',
        'Pandas/ADF exports: leaf_stats.csv (one model, all tree leaves), selected_leaves.csv (eight queries, selected leaf per tree), and one query table per function/case with predictions, gradients, truth, flags and condition diagnostics. Numeric feature suffixes 0/1/2/3 map to A/B/C/D. Singleton-leaf scatter is missing, not zero noise. Unbounded partition edges appear as infinity.','',
        'No standard errors or coverage claim for local-linear predictions. Existing reducible-variance methods describe the original mean predictor only. Compare local-linear refit variance separately. Metadata collection is included in fit times.','',
        'Median seconds by function/case (fit, mean, linear):']
    for function in a.functions:
        for case in a.cases:
            rr=[r for r in records if (r['function'],r['case'])==(function,case)]
            lines.append('- %s/%s: %s'%(function,case,', '.join('%.4f'%np.median([r[k] for r in rr]) for k in ['fit_seconds','mean_seconds','linear_seconds'])))
    lines+=['',f'Benchmark wall time after setup: {time.perf_counter()-tick_all:.3f}s.','']
    (a.output/'REPORT.md').write_text('\n'.join(lines))
    with zipfile.ZipFile(str(a.output)+'.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.iterdir()):z.write(f,a.output.name+'/'+f.name)
    print('REPORT:',a.output/'REPORT.md');print('EVIDENCE:',str(a.output)+'.zip')


if __name__=='__main__':main()
