"""Stage B/C example: dimensionless toy target, experimental sampling-error estimate.

Run from the repository environment with:
python RootInteractive/MLpipeline/examples/phase_2_1_reducible_example.py
Noise is a signed OOF variance estimate; detector calibration and PDFs are not established.
"""
import numpy as np
from RootInteractive.MLpipeline.MIForestErrPDF import MIForestErrPDF


def main():
    rng = np.random.default_rng(20261030)
    X = rng.uniform(-1, 1, (4096, 1))
    y = X[:, 0] + rng.normal(0, .2, len(X))
    grid = np.linspace(-.95, .95, 201)[:, None]
    model = MIForestErrPDF('Regressor', dict(random_state=73, min_tree=64,
        mean_depth=10, niter_max=1, n_jobs=2, fit_noise=False),
        max_depth=None, min_samples_leaf=2, max_features=1.)
    model.fit(X, y)
    mean = model.predict(grid)
    sigma = model.predictReducibleStd(grid)
    # Independent public-tree reference, averaging within the saved blocks.
    members = np.asarray([np.mean([tree.predict(grid) for tree in block], axis=0)
                          for block in model.block_trees_])
    expected = np.sqrt(np.var(members[1::2] - members[::2], axis=0, ddof=1)
                       / (4 * model.n_pairs_))
    assert mean.shape == sigma.shape == (201,)
    assert np.isfinite(mean).all() and np.isfinite(sigma).all() and (sigma >= 0).all()
    np.testing.assert_allclose(mean, members.mean(axis=0), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(sigma, expected, rtol=1e-13, atol=1e-13)
    print('Stage B example PASS: target/sigma units=dimensionless; calibration status=unvalidated')
    model.optionsErrPDF['fit_noise'] = True
    model.fit(X, y)
    uncertainty = model.predictUncertainty(grid)
    raw = uncertainty['irreducible_var_raw']
    np.testing.assert_allclose(uncertainty['irreducible_std']**2, np.maximum(raw, 0.))
    np.testing.assert_allclose(uncertainty['predictive_std']**2, uncertainty['reducible_std']**2 + np.maximum(raw, 0.))
    print('Stage C signed-noise example PASS; mean raw variance=%.6g; clipped fraction=%.6g; validated=False' % (raw.mean(), (raw < 0).mean()))
    print('blocks=%d pairs=%d mean_sigma=%.6g' % (model.block_count_, model.n_pairs_, sigma.mean()))


if __name__ == '__main__':
    main()
