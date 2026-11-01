"""PHASE_2_2 fixed-sigma fit/predict augmentation benchmark, independent refits.

Standalone, no optional comparison packages. Default 50000 rows, 8 refits,
4D latent Uniform[0,1) inputs, independent observation errors .02,.02,.01,0.
Tests all four fit/predict flags, including zero RESPONSE noise (input noise remains).
Reports accuracy against E[y|observed X], raw variance calibration and runtime.
Development evidence, not formal acceptance or per-point coverage certification.
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


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def load(path):
    spec = importlib.util.spec_from_file_location('phase22_augmentation_bench', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def latent_function(X):
    return 3*X[:, 0]+X[:, 1]*np.sin(2*np.pi*X[:, 2])


def conditional_truth(observed, sigma, response_sigma, nodes=256):
    """Deterministic quadrature for independent uniform latent priors + Gaussian errors.

    Normalized posterior weights avoid Gaussian tail cancellation. Returns the
    conditional mean and conditional response variance at the observed query.
    """
    z, w = np.polynomial.legendre.leggauss(nodes)
    z, w = (z+1)/2, w/2
    transforms = [(lambda x: x, lambda x: x*x),
                  (lambda x: x, lambda x: x*x),
                  (lambda x: np.sin(2*np.pi*x), lambda x: np.sin(2*np.pi*x)**2)]
    moments = []
    for j, (f, f2) in enumerate(transforms):
        if sigma[j] == 0:
            moments.append((f(observed[:, j]), f2(observed[:, j])))
        else:
            logw = -.5*((z[None, :]-observed[:, j, None])/sigma[j])**2
            logw -= logw.max(1, keepdims=True)
            weights = np.exp(logw)*w[None, :]
            weights /= weights.sum(1, keepdims=True)
            moments.append((weights@f(z), weights@f2(z)))
    (a, a2), (b, b2), (c, c2) = moments
    mean = 3*a+b*c
    variance = np.maximum(9*a2+b2*c2+6*a*b*c-mean**2, 0)+response_sigma**2
    return mean, variance


def calibration(pred, raw, bootstrap, rng):
    """Paired bootstrap over independent training replicas; query rows stay fixed."""
    R = len(pred)
    truth_var = float(np.var(pred, axis=0, ddof=1).mean())
    estimate = float(raw.mean())
    if truth_var <= 0:
        return dict(refit_variance=truth_var, raw_variance=estimate, ratio=None, ci95=None)
    ratios = []
    for _ in range(bootstrap):
        idx = rng.integers(R, size=R)
        # Resampling a finite empirical distribution reduces its expected variance
        # by (R-1)/R; correct that bootstrap denominator only, not the point estimate.
        denom = np.var(pred[idx], axis=0, ddof=1).mean()*R/(R-1)
        if denom > 0:
            ratios.append(raw[idx].mean()/denom)
    return dict(refit_variance=truth_var, raw_variance=estimate, ratio=estimate/truth_var,
                ci95=np.quantile(ratios, [.025, .975]).tolist() if ratios else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--n-train', type=int, default=50000)
    p.add_argument('--n-eval', type=int, default=2048)
    p.add_argument('--replicas', type=int, default=8)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--fraction', type=float, default=1/128)
    p.add_argument('--leaf', type=int, default=1)
    p.add_argument('--chunk-rows', type=int, default=4096)
    p.add_argument('--seed', type=int, default=1731)
    p.add_argument('--input-sigma', type=float, nargs=4, default=[.02, .02, .01, 0])
    p.add_argument('--smear-sigma', type=float, nargs=4, default=[.01, .01, .005, 0])
    p.add_argument('--cases', nargs='+', choices=['four_d', 'zero_response'], default=['four_d', 'zero_response'])
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if not 1 <= a.n_train <= 100000 or min(a.n_eval, a.jobs, a.iterations, a.leaf, a.chunk_rows, a.bootstrap)<1 or a.replicas<3:
        p.error('positive counts, at least 3 replicas, and n-train <=100000 required')
    input_sigma, smear_sigma = np.asarray(a.input_sigma), np.asarray(a.smear_sigma)
    if np.any(~np.isfinite(input_sigma)) or np.any(input_sigma<0) or np.any(~np.isfinite(smear_sigma)) or np.any(smear_sigma<0):
        p.error('sigma values must be finite and nonnegative')
    if not np.isfinite(a.fraction) or a.fraction<=0:
        p.error('fraction must be finite and positive')
    count = .5/a.fraction
    if not np.isfinite(count) or count<4 or count>a.n_train//2 or not np.isclose(count, round(count), rtol=0, atol=1e-10) or round(count)%2:
        p.error('historical fraction requires even integer 1/(2*f)>=4 and nonempty buckets')
    source = (a.source or Path(__file__).resolve().parents[1]/'MIForestErrPDF.py').resolve()
    module = load(source)
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'candidate_MIForestErrPDF.py').write_bytes(source.read_bytes())
    (a.output/'benchmark_executed.py').write_bytes(Path(__file__).read_bytes())
    from sklearn import __version__ as sklearn_version
    from threadpoolctl import threadpool_limits
    dump(a.output/'invocation.json', dict(configuration={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
         python=platform.python_version(), numpy=np.__version__, sklearn=sklearn_version,
         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), status='DEVELOPMENT_NOT_ACCEPTANCE'))
    rngq = np.random.default_rng(a.seed+500000)
    latent_q = rngq.random((a.n_eval, 4))
    Q = (latent_q+rngq.normal(size=latent_q.shape)*input_sigma).astype('float32')
    mean_truth, noise_truth = conditional_truth(Q, input_sigma, 0)
    fine, fine_var = conditional_truth(Q, input_sigma, 0, nodes=512)
    delta = max(float(abs(mean_truth-fine).max()), float(abs(noise_truth-fine_var).max()))
    if delta>1e-7:
        raise ValueError('conditional-truth quadrature did not converge; max discrepancy=%g'%delta)
    bins = np.clip((Q[:, 0]*4).astype(int), 0, 3)
    records, metrics, saved = [], [], {}
    start = time.perf_counter()
    for case in a.cases:
        response_sigma = 1. if case=='four_d' else 0.
        predictions = {f'{fit}:{pred}': [] for fit in (False, True) for pred in (False, True)}
        variances = {key: [] for key in predictions}
        for r in range(a.replicas):
            rng = np.random.default_rng(a.seed+r)
            latent = rng.random((a.n_train, 4))
            y = latent_function(latent)+response_sigma*rng.normal(size=a.n_train)
            X = (latent+rng.normal(size=latent.shape)*input_sigma).astype('float32')
            fit_seed = (a.seed+r) % 2**32
            for fit_aug in (False, True):
                model = module.MIForestErrPDF('Regressor', dict(block_sampling='historical',
                    tree_fraction=a.fraction, niter_max=a.iterations, n_jobs=a.jobs,
                    random_state=fit_seed, fit_noise=False, verbose=0,
                    prediction_chunk_rows=a.chunk_rows, smear_sigma=smear_sigma.tolist()), min_samples_leaf=a.leaf)
                with threadpool_limits(limits=a.jobs):
                    tick = time.perf_counter()
                    model.fit(X, y, augment=fit_aug)
                    fit_seconds = time.perf_counter()-tick
                    for pred_aug in (False, True):
                        key = f'{fit_aug}:{pred_aug}'
                        kw = dict(augment=True, random_state=(a.seed+r+900000)%2**32) if pred_aug else {}
                        tick = time.perf_counter()
                        mean = model.predict(Q, **kw)
                        mean_seconds = time.perf_counter()-tick
                        tick = time.perf_counter()
                        raw = model.predictReducibleVariance(Q, **kw)
                        raw_seconds = time.perf_counter()-tick
                        predictions[key].append(mean)
                        variances[key].append(raw)
                        records.append(dict(case=case, replica=r, fit_augment=fit_aug, predict_augment=pred_aug,
                            status='EXECUTED', fit_seconds=fit_seconds, mean_seconds=mean_seconds,
                            raw_variance_seconds=raw_seconds, fit_timing_shared_with_other_predict_flag=True,
                            trees=len(model.estimators_), actual_tree_fraction=model.actual_tree_fraction_,
                            participation_fraction=model.participation_fraction_))
                        print(case, r, key, 'fit=%.3fs mean=%.3fs raw=%.3fs'%(fit_seconds, mean_seconds, raw_seconds), flush=True)
                dump(a.output/'records.json', records)
        for key in predictions:
            fit_aug, pred_aug = (v=='True' for v in key.split(':'))
            pred, raw = np.asarray(predictions[key]), np.asarray(variances[key])
            saved[case+'_'+key+'_mean'] = pred
            saved[case+'_'+key+'_raw'] = raw
            for b in range(4):
                selected = bins==b
                if not selected.any():
                    continue
                stat = calibration(pred[:, selected], raw[:, selected], a.bootstrap, np.random.default_rng(a.seed+b))
                metrics.append(dict(case=case, fit_augment=fit_aug, predict_augment=pred_aug, bin=b,
                    n_queries=int(selected.sum()), rmse_conditional_mean=float(np.sqrt(np.mean((pred[:, selected]-mean_truth[selected])**2))),
                    mean_bias=float((pred[:, selected]-mean_truth[selected]).mean()),
                    rmse_latent_diagnostic=float(np.sqrt(np.mean((pred[:, selected]-latent_function(latent_q)[selected])**2))),
                    negative_raw_fraction=float(np.mean(raw[:, selected]<0)),
                    conditional_response_variance_truth=float((noise_truth[selected]+response_sigma**2).mean()), **stat))
    np.savez_compressed(a.output/'predictions.npz', observed_query=Q, latent_query=latent_q,
                        conditional_mean_truth=mean_truth, input_induced_variance_truth=noise_truth, bins=bins, **saved)
    dump(a.output/'metrics.json', metrics)
    lines = ['# PHASE_2_2 augmentation benchmark', '', '**Status:** executed development benchmark; no acceptance or coverage certification.', '',
        f'{a.n_train} training rows; {a.n_eval} fixed observed queries; {a.replicas} independent refits; L={a.iterations}; requested fraction={a.fraction:g}; leaf={a.leaf}; jobs={a.jobs}.', '',
        f'Input observation sigma A/B/C/D: {a.input_sigma}. Additional per-tree fit/predict smear widths: {a.smear_sigma}. No clipping at [0,1] boundaries.', '',
        'Latent A/B/C/D are independent Uniform[0,1). y=3A+B*sin(2πC)+ε. Response ε sigma is 1 in four_d and 0 in zero_response. Input observation errors remain in both cases. Targets are formed before input noise. Query observations are fixed across all refits and options.', '',
        f'Primary accuracy target: conditional mean E[y|observed X], computed by posterior quadrature; max 256/512-node discrepancy {delta:.3g}. Latent-truth RMSE is a separate diagnostic in metrics.json.', '',
        'Raw sampling variance is compared with variance across independent refits at each query, averaged within four observed-A bins. Ratio target is 1. Intervals are approximate 95% paired bootstrap over training replicas (fixed query grid); few replicas give weak intervals. Raw negatives are retained. This does not test per-query interval coverage.', '',
        'Prediction augmentation uses one draw per tree, fresh seed per replica; mean and raw variance use the same explicit seed. Thus refit variance includes prediction Monte Carlo variation. No response-noise estimator is fitted or certified here. Zero response noise still has input-induced conditional variance; no zero-noise ratio is claimed.', '',
        '| Case | Fit smear | Predict smear | A bin | RMSE vs conditional mean | Raw/refit variance | 95% bootstrap CI | Negative raw |',
        '|---|---|---|---|---:|---:|---|---:|']
    for m in metrics:
        ci = 'undefined' if m['ci95'] is None else '[%.3f, %.3f]'%tuple(m['ci95'])
        ratio = 'undefined' if m['ratio'] is None else '%.3f'%m['ratio']
        lines.append('| %s | %s | %s | %d | %.5f | %s | %s | %.1f%% |'%(m['case'], m['fit_augment'], m['predict_augment'], m['bin'], m['rmse_conditional_mean'], ratio, ci, 100*m['negative_raw_fraction']))
    lines += ['', '## Runtime', '', 'Fit timings are shared by the two prediction flags; do not add duplicated fit timings. Mean and raw-variance evaluation are timed separately. First-use costs are retained. Ratios are descriptive, not speed claims from repeated timing trials.', '',
              '| Case | Fit smear | Predict smear | Median fit seconds | Median mean seconds | Median raw variance seconds |', '|---|---|---|---:|---:|---:|']
    for case in a.cases:
        for fa in (False, True):
            for pa in (False, True):
                rr = [r for r in records if (r['case'],r['fit_augment'],r['predict_augment'])==(case,fa,pa)]
                lines.append('| %s | %s | %s | %.4f | %.4f | %.4f |'%(case,fa,pa,*[np.median([r[k] for r in rr]) for k in ('fit_seconds','mean_seconds','raw_variance_seconds')]))
    lines += ['', f'Benchmark loop/report elapsed: {time.perf_counter()-start:.3f} seconds (truth quadrature precedes this timer).', '',
              'Source snapshot, executed script, environment/configuration, all records, metrics and prediction arrays are included in the evidence ZIP. No automatic scale-up. Compare settings at 50000 rows first, then validate at 100000 before considering larger samples.', '']
    (a.output/'REPORT.md').write_text('\n'.join(lines))
    with zipfile.ZipFile(str(a.output)+'.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.iterdir()):
            z.write(f, a.output.name+'/'+f.name)
    print('REPORT:', a.output/'REPORT.md')
    print('EVIDENCE:', str(a.output)+'.zip')


if __name__ == '__main__':
    main()
