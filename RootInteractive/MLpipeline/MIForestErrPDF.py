"""Disjoint-block regression with out-of-fold signed noise-variance estimation.

Tree/member spread and quantiles are not calibrated response-noise estimates.
"""
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
#
from joblib import Parallel, delayed
import threading
from scipy import stats

def _accumulate_prediction(predict, X, out,col, lock):
    """
    inspired by https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/ensemble/_forest.py#L1410
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.

    """
    prediction = predict(X, check_input=False)
    with lock:
        out[col] += prediction

def _accumulate_predictionNL(predict, X, out,col):
    """
    inspired by https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/ensemble/_forest.py#L1410
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.

    """
    prediction = predict(X, check_input=False)
    out[col] += prediction

def simple_predict(predict, X, out, col):
    out[col] = predict(X, check_input=False)

def _std2(X, out):
    """Write paired-difference population spread after a global-RNG column permutation.

    X has shape (rows, members). For odd member counts the last permuted
    member is unused. This legacy diagnostic is not a fitted-mean standard
    error or a response-noise estimate. The caller can seed NumPy for replay.
    """
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError('std2 requires a 2-D array with at least two members')
    order = np.random.permutation(values.shape[1])
    half = values.shape[1] // 2
    np.std(values[:, order[:half]] - values[:, order[half:2 * half]], axis=1, out=out)


_STAT_CHUNK_ROWS = 10000


def _stat_requests(statDictionary):
    """Validate requests without changing the caller's dictionary."""
    if not isinstance(statDictionary, dict):
        raise TypeError('statDictionary must be a dictionary')
    unknown = set(statDictionary) - {'mean', 'median', 'std', 'std2', 'quantile', 'trim_mean'}
    if unknown:
        raise ValueError('unknown forest statistics: ' + ', '.join(map(str, unknown)))
    request = dict(statDictionary)
    for name, upper, inclusive in [('quantile', 1., True), ('trim_mean', .5, False)]:
        if name not in request:
            continue
        if name == 'quantile' and any(isinstance(v, (bool, np.bool_))
                                      for v in np.asarray(request[name], dtype=object).flat):
            raise ValueError('quantile requires numeric probabilities, not Boolean values')
        try:
            values = np.asarray(request[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(name + ' must contain numeric fractions') from exc
        if values.ndim > 1:
            raise ValueError(name + ' must be a scalar or one-dimensional sequence')
        values = np.atleast_1d(values)
        invalid = values > upper if inclusive else values >= upper
        if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(invalid):
            raise ValueError(name + ' fraction outside its permitted interval')
        request[name] = list(values)
    return request


def _stat_input(X, schemas=()):
    """Convert dense finite prediction inputs to sklearn tree precision."""
    # Validate labels before conversion, including empty frames and no-op requests.
    if hasattr(X, 'columns'):
        columns = np.asarray(X.columns)
        for names in schemas:
            if names is not None and not np.array_equal(columns, np.asarray(names)):
                raise ValueError('X feature names and order must match fitted feature names')
    try:
        values = np.asarray(X, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError('X must be a dense numeric matrix') from exc
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError('X must have shape (rows, features) and finite values')
    return values


def _stat_tree_prediction(tree, X):
    """Use the public prediction method; propagate validation and fitted-state errors."""
    names = getattr(tree, 'feature_names_in_', None)
    if names is not None:
        # Public predict validates its fitted schema without per-tree warnings.
        # ndarray callers retain positional semantics; never suppress warnings.
        import pandas as pd
        X = pd.DataFrame(X, columns=names, copy=False)
    prediction = np.asarray(tree.predict(X))
    if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
        raise ValueError('forest statistics require finite single-output member predictions')
    return prediction


def _forest_statistics(trees, X, statDictionary, n_jobs, max_rows, quantile_key, parallel=None, verbose=0, feature_names=None):
    """Canonical tree-statistics owner with bounded member-prediction chunks.

    Population std (ddof=0), NumPy linear quantiles and SciPy trimmed means
    reduce over explicit tree predictions, not response distributions. Nested
    result arrays are preallocated once. Temporary predictions scale with
    members * min(max_rows, 10000) rows, plus public prediction validation.
    A single std2 permutation per call makes that diagnostic chunk/job invariant.
    """
    if n_jobs is not None and (isinstance(n_jobs, (bool, np.bool_)) or
                              not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0):
        raise ValueError('n_jobs must be None or a nonzero integer')
    if isinstance(max_rows, (bool, np.bool_)) or not isinstance(max_rows, (int, np.integer)) or max_rows == 0:
        raise ValueError('max_rows must be a nonzero integer')
    trees = list(trees)
    if not trees:
        raise ValueError('forest statistics require at least one fitted member')
    request = _stat_requests(statDictionary)
    values = _stat_input(X, [feature_names] +
                         [getattr(tree, 'feature_names_in_', None) for tree in trees])
    size = len(values)
    chunk = max(size, 1) if max_rows < 0 else min(max_rows, _STAT_CHUNK_ROWS)
    out = {}
    for name, fractions in request.items():
        if name in ('quantile', 'trim_mean'):
            out[quantile_key if name == 'quantile' else name] = {q: np.empty(size) for q in fractions}
        else:
            out[name] = np.empty(size)
    order = None
    if 'std2' in request:
        if len(trees) < 2:
            raise ValueError('std2 requires at least two members')
        order = np.random.permutation(len(trees))
    if not size or not request:
        return out
    if parallel is None and n_jobs == 1:
        return _fill_forest_statistics(trees, values, request, out, chunk, quantile_key, None, order)
    if parallel is None:
        with Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem') as runner:
            return _fill_forest_statistics(trees, values, request, out, chunk, quantile_key, runner, order)
    return _fill_forest_statistics(trees, values, request, out, chunk, quantile_key, parallel, order)


def _fill_forest_statistics(trees, X, request, out, chunk, quantile_key, parallel, order):
    """Fill canonical outputs; each chunk is released before predicting the next."""
    for first in range(0, len(X), chunk):
        last = min(first + chunk, len(X))
        predictions = np.asarray([_stat_tree_prediction(tree, X[first:last]) for tree in trees]
            if parallel is None else parallel(delayed(_stat_tree_prediction)(tree, X[first:last]) for tree in trees))
        for name in ('mean', 'std'):
            if name in request:
                out[name][first:last] = getattr(np, name)(predictions, axis=0)
        if order is not None:
            half = len(trees) // 2
            out['std2'][first:last] = np.std(predictions[order[:half]] - predictions[order[half:2 * half]], axis=0)
        probabilities = list(dict.fromkeys(request.get('quantile', [])))
        if 'median' in request and .5 not in probabilities:
            probabilities.append(.5)
        if probabilities:
            work = predictions.copy()
            quantiles = np.quantile(work, probabilities, axis=0, method='linear', overwrite_input=True)
            for index, q in enumerate(probabilities):
                if q in out.get(quantile_key, {}):
                    out[quantile_key][q][first:last] = quantiles[index]
            if 'median' in request:
                # Q50 partitions both middle ranks. Preserve np.median's mean
                # reduction (including its rounding), not quantile's lerp.
                middle = len(work) // 2
                out['median'][first:last] = np.mean(work[(len(work)-1)//2:middle+1], axis=0)
        for q in request.get('trim_mean', []):
            out['trim_mean'][q][first:last] = stats.trim_mean(predictions, q, axis=0)
    return out


def predictRFStatChunk(rf, X, statDictionary, parallel, n_jobs):
    """Compatible helper adapter; nested quantiles use the singular 'quantile' key."""
    return _forest_statistics(rf.estimators_, X, statDictionary, n_jobs, _STAT_CHUNK_ROWS,
                              'quantile', parallel, getattr(rf, 'verbose', 0),
                              getattr(rf, 'feature_names_in_', None))


def predictRFStat(rf, X, statDictionary, n_jobs, max_rows=1000000):
    """Return requested tree statistics, with nested singular 'quantile' results.

    X is a finite dense ndarray/DataFrame. Named columns must match fitted
    feature names and order when available; ndarrays remain positional.
    Empty rows produce shaped empty
    results. max_rows < 0 disables chunking; zero is invalid. Positive chunks
    are capped at 10000 rows. Negative job counts follow joblib conventions.
    'std2' preserves the legacy RNG-based diagnostic; it is not uncertainty.
    """
    return _forest_statistics(rf.estimators_, X, statDictionary, n_jobs, max_rows,
                              'quantile', verbose=getattr(rf, 'verbose', 0),
                              feature_names=getattr(rf, 'feature_names_in_', None))


def predictRFStatNew(rf, X, statDictionary, n_jobs):
    """Compatible forest adapter; preserve historical plural 'quantiles' output."""
    return _forest_statistics(rf.estimators_, X, statDictionary, n_jobs, _STAT_CHUNK_ROWS,
                              'quantiles', verbose=getattr(rf, 'verbose', 0),
                              feature_names=getattr(rf, 'feature_names_in_', None))


def predictForestStat(treeArray, X, statDictionary, n_jobs, verbose=False):
    """Statistics over supplied trees, with plural 'quantiles' and bounded chunks."""
    return _forest_statistics(treeArray, X, statDictionary, n_jobs, _STAT_CHUNK_ROWS,
                              'quantiles', verbose=verbose)


class _BlockMember:
    """One statistical member: the mean of trees trained on the same block."""
    def __init__(self, trees):
        self.trees = tuple(trees)

    def predict(self, X):
        return np.mean([_stat_tree_prediction(tree, X) for tree in self.trees], axis=0)


class MIForestErrPDF:
    """Disjoint-block regression mean and experimental paired sampling error.

    Block-member spread/quantiles are descriptive, not a response PDF. The
    paired error formula still requires Stage D scientific calibration.
    Optional noise fitting uses held-out signed residual-variance targets.
    block_sampling='historical' restores fixed input-order A/B halves with
    per-iteration reshuffling and approximately half-row participation; 'fixed'
    remains the default. tree_fraction optionally sets historical rows/tree/N
    through B=1/(2*f); fitted actual fractions expose integer rounding.
    Historical groups/weights/noise are refused pending their validation.
    n_blocks=None retains the depth-derived block count; an even integer >=4
    requests an exact count and overrides that derivation. Tree leaf settings
    remain public DecisionTreeRegressor kwargs. prediction_chunk_rows bounds
    mean/reducible/combined prediction temporaries (default 10000); generic
    descriptive predictStat helpers retain their separate historical chunk cap.
    smear_sigma=None disables configuration; otherwise supply one actual Gaussian
    width per feature. fit(..., augment=True) jitters each tree's training inputs
    independently, with unchanged targets/memberships. predict(..., augment=True)
    draws fresh input jitter per tree and call. Explicit prediction random_state
    replays a draw; the same seed aligns mean and raw variance evaluations.
    Smearing never clips to the observed feature range. Augmented uncertainty
    remains an uncalibrated approximation; callable widths/local linear fits are
    separate future options. predictStat and legacy subset diagnostics stay unsmeared.
    """
    def __init__(self, switch, optionsErrPDF0, **kwargs):
        import copy
        if switch == 'Classifier':
            raise NotImplementedError('classification repair remains OPEN; this phase supports regression only')
        if switch != 'Regressor':
            raise ValueError("switch must be 'Regressor'")
        defaults = dict(mean_depth=10, min_tree=64, niter_max=3,
                        test_fraction=.025, n_jobs=4, nPermutation_RedErr=128,
                        dump_progress=False, verbose=1, fit_noise=False, random_state=None,
                        noise_options={}, n_blocks=None, prediction_chunk_rows=10000,
                        block_sampling='fixed', tree_fraction=None, smear_sigma=None, store_leaf_stats=False)
        unknown = set(optionsErrPDF0) - set(defaults)
        if unknown:
            raise ValueError('unknown optionsErrPDF: ' + ', '.join(sorted(unknown)))
        self.optionsErrPDF = copy.deepcopy(defaults)
        self.optionsErrPDF.update(copy.deepcopy(optionsErrPDF0))
        self.options = copy.deepcopy(kwargs)
        # Constructor validation and public sklearn keyword forwarding.
        DecisionTreeRegressor(**self.options)
        self.type = switch
        self.method = 'MIForestErrPDF'
        self.model_name = ''
        self._reset_fit()

    def _reset_fit(self):
        for name in list(vars(self)):
            if name.endswith('_'):
                delattr(self, name)
        self.trees = {0: [], 1: []}
        self.deltaTrees = []
        self.nTrees = 0

    @staticmethod
    def _positive_int(value, name, minimum=1):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(name + ' must be an integer >= ' + str(minimum))
        return int(value)

    def _configure_smearing(self, features, augment):
        """Fixed actual Gaussian widths; no scale or probability parameter."""
        if not isinstance(augment, (bool, np.bool_)):
            raise ValueError('augment must be Boolean')
        supplied = self.optionsErrPDF['smear_sigma']
        if supplied is None:
            if augment:
                raise ValueError('augment=True requires smear_sigma, one actual width per feature')
            sigma = np.zeros(features)
        else:
            try:
                sigma = np.asarray(supplied, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError('smear_sigma must contain fixed numeric widths; functions are not supported yet') from exc
            if sigma.shape != (features,) or not np.isfinite(sigma).all() or np.any(sigma < 0):
                raise ValueError('smear_sigma must contain one finite nonnegative width per feature')
        if augment and self.optionsErrPDF['fit_noise']:
            raise NotImplementedError('augmented noise fitting requires separate validation')
        self.smear_sigma_ = sigma.copy()
        self.smear_configured_ = supplied is not None
        self.fit_augment_ = bool(augment)

    def _smear_values(self, values, rng):
        if not np.any(self.smear_sigma_):
            return values
        # Each call owns its bounded temporary. Original inputs and y stay unchanged.
        with np.errstate(over='ignore', invalid='ignore'):
            jittered = np.asarray(values + rng.standard_normal(values.shape)*self.smear_sigma_, dtype=np.float32)
        if not np.isfinite(jittered).all():
            raise ValueError('smearing produced nonfinite tree inputs')
        return jittered

    def _smear_streams(self, augment, random_state):
        if not isinstance(augment, (bool, np.bool_)):
            raise ValueError('augment must be Boolean')
        if random_state is not None and (isinstance(random_state, (bool, np.bool_)) or
                not isinstance(random_state, (int, np.integer)) or not 0 <= random_state < 2**32):
            raise ValueError('prediction random_state must be None or a nonnegative 32-bit integer')
        if not augment:
            if random_state is not None:
                raise ValueError('prediction random_state requires augment=True')
            return None
        if not self.smear_configured_:
            raise ValueError('augment=True requires smear_sigma at fit time')
        if not np.any(self.smear_sigma_):
            return None
        # Allocate streams before threading: scheduling and chunk boundaries do not
        # change draws. Omitted seeds advance a private per-model stream per call.
        rng = self._prediction_smear_rng_ if random_state is None else np.random.default_rng(random_state)
        seeds = rng.integers(2**63-1, size=(self.block_count_, self.n_repeats_))
        return [[np.random.default_rng(int(seed)) for seed in row] for row in seeds]

    def _tree_prediction(self, tree, chunk, rng=None):
        query = chunk if rng is None else self._smear_values(chunk, rng)
        return self._validated_member_prediction(_BlockMember((tree,)), query)

    def _smeared_member_prediction(self, member, chunk, streams):
        return np.mean([self._tree_prediction(tree, chunk, rng)
                        for tree, rng in zip(member.trees, streams)], axis=0)

    def fit(self, xIn, yIn, sample_weight=None, *, groups=None, augment=False):
        """Fit mean trees; augment=True adds independent fixed-sigma input jitter per tree."""
        return self.fitWithParam(xIn, yIn, self.optionsErrPDF['mean_depth'],
                                 self.optionsErrPDF['niter_max'], sample_weight, groups=groups, augment=augment)

    def fitWithParam(self, x0, y0, meanDepth, nIterMax, sample_weight=None, *, groups=None, augment=False):
        """Fit with explicit depth/repetition overrides; failed refits invalidate old state."""
        import pandas as pd
        self._reset_fit()
        try:
            X = _stat_input(x0)
            self._configure_smearing(X.shape[1], augment)
            retain = self.optionsErrPDF['store_leaf_stats']
            if not isinstance(retain, (bool, np.bool_)):
                raise ValueError('store_leaf_stats must be Boolean')
            if retain and self.options.get('criterion') == 'absolute_error':
                raise ValueError('local centroid fits require mean-valued leaves; absolute_error uses medians')
            self.store_leaf_stats_ = bool(retain)
            y = np.asarray(y0, dtype=float)
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            if y.ndim != 1 or len(y) != len(X) or not len(y) or not np.isfinite(y).all():
                raise ValueError('y must be finite, nonempty, and single-output with one value per row')
            if isinstance(meanDepth, (bool, np.bool_)) or not isinstance(meanDepth, (int, float, np.integer, np.floating)) or not np.isfinite(meanDepth) or meanDepth <= 0:
                raise ValueError('meanDepth must be finite and positive')
            repeats = self._positive_int(nIterMax, 'nIterMax')
            sampling = self.optionsErrPDF['block_sampling']
            if sampling not in ('fixed', 'historical'):
                raise ValueError("block_sampling must be 'fixed' or 'historical'")
            fraction = self.optionsErrPDF['tree_fraction']
            if sampling == 'fixed' and fraction is not None:
                raise ValueError('tree_fraction is supported only with historical sampling')
            if sampling == 'historical' and (groups is not None or sample_weight is not None):
                raise NotImplementedError('historical sampling: groups and sample weights require separate validation')
            if sampling == 'historical' and self.optionsErrPDF['fit_noise']:
                raise NotImplementedError('historical sampling: noise fit remains a separate validation checkpoint')
            minimum = self._positive_int(self.optionsErrPDF['min_tree'], 'min_tree')
            requested_blocks = self.optionsErrPDF['n_blocks']
            if requested_blocks is not None:
                requested_blocks = self._positive_int(requested_blocks, 'n_blocks', 4)
                if requested_blocks % 2:
                    raise ValueError('n_blocks must be even')
            chunk_rows = self._positive_int(self.optionsErrPDF['prediction_chunk_rows'],
                                            'prediction_chunk_rows')
            jobs = self.optionsErrPDF['n_jobs']
            if jobs is not None and (isinstance(jobs, (bool, np.bool_)) or not isinstance(jobs, (int, np.integer)) or jobs == 0):
                raise ValueError('n_jobs must be None or a nonzero integer')
            noise = self.optionsErrPDF['fit_noise']
            if not isinstance(noise, (bool, np.bool_)):
                raise ValueError('fit_noise must be Boolean')
            seed = self.optionsErrPDF['random_state']
            tree_seed = self.options.get('random_state')
            if seed is None:
                seed = tree_seed
            elif tree_seed is not None and tree_seed != seed:
                raise ValueError('conflicting random_state in tree kwargs and optionsErrPDF')
            if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32):
                raise ValueError('random_state must be None or a nonnegative 32-bit integer')
            self._prediction_smear_rng_ = np.random.default_rng(
                None if seed is None else np.random.SeedSequence([int(seed), 71427]))
            smear_rng = np.random.default_rng(
                None if seed is None else np.random.SeedSequence([int(seed), 71426]))
            rng = np.random.default_rng(seed)
            weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
            if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError('sample_weight must be finite nonnegative values, one per row')
            active = np.flatnonzero(weights > 0)
            if not len(active):
                raise ValueError('sample_weight must contain positive weight')
            fit_weights = np.ones(len(y)) if np.all(weights[active] == weights[active[0]]) else weights
            codes = None
            if groups is not None:
                group_values = np.asarray(groups)
                if group_values.ndim != 1 or len(group_values) != len(y) or pd.isna(group_values).any():
                    raise ValueError('groups must contain one nonmissing group ID per row')
                codes, labels = pd.factorize(group_values[active], sort=False)
            N = len(active)
            count = max(4, 2 * int(math.ceil(max(minimum, math.floor(N / 2.**meanDepth) if meanDepth < 1024 else 0) / 2)))
            if requested_blocks is not None:
                count = requested_blocks
            if sampling == 'historical':
                if fraction is not None:
                    if isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, (int, float, np.integer, np.floating)):
                        raise ValueError('tree_fraction must be a finite positive number')
                    if not np.isfinite(fraction) or fraction <= 0:
                        raise ValueError('tree_fraction must be a finite positive number')
                    exact_count = .5 / float(fraction)
                    if not np.isfinite(exact_count) or exact_count < 4 or exact_count > len(y)//2:
                        raise ValueError('tree_fraction requests too few trees or empty historical buckets')
                    fraction_count = int(round(exact_count))
                    if fraction_count % 2 or not np.isclose(exact_count, fraction_count, rtol=0, atol=1e-10):
                        raise ValueError('historical tree_fraction requires 1/(2*f) to be an even integer >=4')
                    if requested_blocks is not None and requested_blocks != fraction_count:
                        raise ValueError('n_blocks conflicts with historical tree_fraction')
                    count = fraction_count
                return self._fit_historical(X, y, x0, count, repeats, meanDepth,
                                            seed, jobs, chunk_rows, fraction, smear_rng)
            units = N if codes is None else len(labels)
            if count > units:
                raise ValueError('requested block count exceeds independent rows/groups')
            if codes is None:
                blocks = [a.copy() for a in np.array_split(rng.permutation(active), count)]
                group_counts = np.array([len(a) for a in blocks])
            else:
                blocks, group_counts = self._group_blocks(active, codes, count, rng)
            # Pair adjacent sizes, then seed pair order/orientation.
            order = sorted(range(count), key=lambda i: -len(blocks[i]))
            pairs = np.asarray(order).reshape(-1, 2)[rng.permutation(count // 2)]
            for pair in pairs:
                if rng.integers(2):
                    pair[:] = pair[::-1]
            order = pairs.ravel()
            blocks = [blocks[i] for i in order]
            group_counts = group_counts[order]
            seeds = rng.choice(2**31 - 1, size=(count, repeats), replace=False)
            permutation_seed = int(rng.integers(0, 2**31 - 1))
            self.fit_smear_seeds_ = smear_rng.integers(2**63-1, size=(count, repeats))
            tree_options = dict(self.options)
            tree_options.pop('random_state', None)
            def fit_block(b):
                idx = blocks[b]
                if weights[idx].sum() <= 0:
                    raise ValueError('each block requires positive weight')
                result = []
                for iteration in range(repeats):
                    tree = DecisionTreeRegressor(random_state=int(seeds[b, iteration]), **tree_options)
                    train = self._smear_values(X[idx], np.random.default_rng(self.fit_smear_seeds_[b, iteration])) if self.fit_augment_ else X[idx]
                    tree.fit(train, y[idx], sample_weight=fit_weights[idx])
                    if self.store_leaf_stats_:
                        self._retain_leaf_stats(tree, train, y[idx], fit_weights[idx])
                    result.append(tree)
                return result
            fitted = Parallel(n_jobs=jobs, require='sharedmem')(
                delayed(fit_block)(b) for b in range(count))
            # Publish only after every member is fitted successfully.
            self.block_trees_ = tuple(tuple(a) for a in fitted)
            self.members_ = tuple(_BlockMember(a) for a in self.block_trees_)
            self.estimators_ = tuple(t for a in self.block_trees_ for t in a)
            self.block_indices_ = tuple(a.copy() for a in blocks)
            self.block_membership_ = np.full(len(y), -1, dtype=int)
            for b, indices in enumerate(blocks):
                self.block_membership_[indices] = b
            self.active_indices_ = active.copy()
            self.n_zero_weight_ = len(y) - N
            self.block_sizes_ = np.array([len(a) for a in blocks])
            self.block_group_counts_ = group_counts
            self.block_count_ = count
            self.requested_n_blocks_ = requested_blocks
            self.prediction_chunk_rows_ = chunk_rows
            self.n_pairs_ = count // 2
            self.n_repeats_ = repeats
            self.mean_depth_ = meanDepth
            self.tree_seeds_ = seeds
            self.permutation_seed_ = permutation_seed
            self.fit_seed_ = seed
            self.n_features_in_ = X.shape[1]
            if hasattr(x0, 'columns'):
                self.feature_names_in_ = np.asarray(x0.columns).copy()
            self.fit_noise_ = False
            nonuniform = not np.all(weights[active] == weights[active[0]])
            self.uncertainty_status_ = dict(validated=False, grouped=groups is not None,
                nonuniform_weights=bool(nonuniform), unequal_block_sizes=bool(np.ptp(self.block_sizes_)),
                limitations=['paired sampling-error interpretation awaits Stage D calibration',
                             'unequal blocks, groups and weighting can violate comparable-block assumptions'])
            self.trees = {0: [t for a in self.block_trees_[::2] for t in a],
                          1: [t for a in self.block_trees_[1::2] for t in a]}
            self.nTrees = count
            if noise:
                self._fit_noise(X, y, weights, groups, meanDepth, repeats,
                                int(rng.integers(0, 2**31 - 1)))
                self.fit_noise_ = True
            self.is_fitted_ = True
            if self.optionsErrPDF['dump_progress']:
                print('MIForestErrPDF: fitted %d blocks, %d trees per block, %d active rows' % (count, repeats, N))
            return self
        except Exception:
            self._reset_fit()
            raise

    def _fit_historical(self, X, y, original_X, count, repeats, depth, seed, jobs, chunk, fraction, smear_rng):
        """Restore original input-order halves, within-half reshuffling and partial participation.

        count is TOTAL trees per iteration. Each tree uses floor(N/(2*count))
        rows, so a pass uses approximately N/2 rows. tree_fraction is rows/tree
        divided by total N; B=1/(2*f), not 1/f for this historical layout.
        Private seeded streams reproduce the instrumented historical source adapter.
        Groups, weights and noise fitting are deliberately refused in this mode.
        """
        bucket = len(y)//(2*count)
        if count < 4 or count % 2 or bucket < 1:
            raise ValueError('historical sampling requires even tree count >=4 and nonempty buckets')
        halves = (np.arange(len(y)//2), np.arange(len(y)//2, len(y)))
        partition_rng = np.random.RandomState(seed)
        tree_rng = np.random.default_rng(None if seed is None else seed+1781)
        m = count//2
        order = np.column_stack((np.arange(m), np.arange(m, count))).ravel()
        all_indices, all_seeds = [], []
        for iteration in range(repeats):
            shuffled = [half[partition_rng.permutation(len(half))] for half in halves]
            chronological = [shuffled[2*b//count][b*bucket:(b+1)*bucket].copy() for b in range(count)]
            seeds = tree_rng.integers(2**31-1, size=count)
            all_indices.append(tuple(chronological[b] for b in order))
            all_seeds.append(seeds[order])
        self.fit_smear_seeds_ = smear_rng.integers(2**63-1, size=(count, repeats))
        tree_options = dict(self.options)
        tree_options.pop('random_state', None)
        def fit_one(iteration, b):
            idx = all_indices[iteration][b]
            train = self._smear_values(X[idx], np.random.default_rng(self.fit_smear_seeds_[b, iteration])) if self.fit_augment_ else X[idx]
            tree = DecisionTreeRegressor(random_state=int(all_seeds[iteration][b]), **tree_options).fit(train, y[idx])
            if self.store_leaf_stats_:
                self._retain_leaf_stats(tree, train, y[idx], np.ones(len(idx)))
            return tree
        trees = Parallel(n_jobs=jobs, require='sharedmem')(
            delayed(fit_one)(l, b) for l in range(repeats) for b in range(count))
        iteration_trees = tuple(tuple(trees[l*count:(l+1)*count]) for l in range(repeats))
        self.iteration_trees_ = iteration_trees
        self.iteration_block_indices_ = tuple(all_indices)
        self.iteration_membership_ = np.full((len(y), repeats), -1, dtype=np.int32)
        for l, blocks in enumerate(all_indices):
            for b, idx in enumerate(blocks):
                self.iteration_membership_[idx, l] = b
        self.block_trees_ = tuple(tuple(iteration_trees[l][b] for l in range(repeats)) for b in range(count))
        self.members_ = tuple(_BlockMember(block) for block in self.block_trees_)
        self.estimators_ = tuple(t for iteration in iteration_trees for t in iteration)
        self.block_indices_ = tuple(idx.copy() for idx in all_indices[0])
        self.block_membership_ = self.iteration_membership_[:, 0].copy()
        self.block_sizes_ = np.full(count, bucket)
        self.block_group_counts_ = self.block_sizes_.copy()
        self.active_indices_ = np.arange(len(y))
        self.n_zero_weight_ = 0
        self.block_count_, self.nTrees = count, count
        self.n_pairs_, self.n_repeats_ = m, repeats
        self.mean_depth_, self.fit_seed_ = depth, seed
        self.requested_n_blocks_ = self.optionsErrPDF['n_blocks']
        self.block_sampling_ = 'historical'
        self.requested_tree_fraction_ = fraction
        self.actual_tree_fraction_ = bucket/len(y)
        self.participation_fraction_ = count*bucket/len(y)
        self.tree_seeds_ = np.asarray(all_seeds).T
        self.permutation_seed_ = int(np.random.default_rng(seed).integers(2**31-1))
        self.prediction_chunk_rows_ = chunk
        self.n_features_in_ = X.shape[1]
        if hasattr(original_X, 'columns'):
            self.feature_names_in_ = np.asarray(original_X.columns).copy()
        self.trees = {h: [t for iteration in iteration_trees for t in iteration[h::2]] for h in (0, 1)}
        self.fit_noise_ = False
        self.uncertainty_status_ = dict(validated=False, grouped=False, nonuniform_weights=False,
            unequal_block_sizes=False, block_sampling='historical',
            raw_variance='first-iteration paired variance minus (1-1/L)*iteration-mean variance',
            displayed_std='sqrt(max(raw_variance,0)); not a calibrated per-track interval',
            legacy_diagnostic='first-iteration subset permutations, requested-fraction correction',
            limitations=['historical partial participation and input-order halves',
                         'raw variance formula for this layout awaits independent-refit calibration',
                         'tree kwargs now forwarded; source comparison uses original leaf default 1',
                         'no groups, weights or noise fit in historical mode'])
        self.is_fitted_ = True
        return self

    def _historical_chunks(self, values, smear_streams=None):
        """Predict each historical tree once per bounded query chunk, in iteration/pair order."""
        size = self._chunk_rows()
        if self.optionsErrPDF['n_jobs'] == 1:
            for first in range(0, len(values), size):
                q = values[first:first+size]
                T = np.asarray([self._tree_prediction(tree, q,
                    None if smear_streams is None else smear_streams[b][l])
                    for l, iteration in enumerate(self.iteration_trees_) for b, tree in enumerate(iteration)])
                yield first, T.reshape(self.n_repeats_, self.block_count_, len(q))
            return
        with Parallel(n_jobs=self.optionsErrPDF['n_jobs'], require='sharedmem') as runner:
            for first in range(0, len(values), size):
                q = values[first:first+size]
                T = np.asarray(runner(delayed(self._tree_prediction)(
                    tree, q, None if smear_streams is None else smear_streams[b][l])
                    for l, iteration in enumerate(self.iteration_trees_) for b, tree in enumerate(iteration)))
                yield first, T.reshape(self.n_repeats_, self.block_count_, len(q))

    def predictReducibleVariance(self, data, *, experimental=False, augment=False, random_state=None):
        """Signed sampling-variance candidate; historical correction includes iteration covariance.

        No clipping before regional averaging. This does not include model bias.
        Scientific calibration of the historical half-layout remains open.
        """
        values = self._prediction_input(data)
        self._uncertainty_policy(experimental)
        streams = self._smear_streams(augment, random_state)
        out = np.empty(len(values))
        if getattr(self, 'block_sampling_', 'fixed') == 'historical':
            for first, T in self._historical_chunks(values, streams):
                v1 = np.var(T[0, 1::2]-T[0, ::2], axis=0, ddof=1)/(4*self.n_pairs_)
                vi = np.var(T.mean(1), axis=0, ddof=1) if self.n_repeats_ > 1 else np.zeros(T.shape[2])
                out[first:first+T.shape[2]] = v1-(1-1/self.n_repeats_)*vi
        else:
            for first, members in self._member_chunks(values, streams):
                out[first:first+members.shape[1]] = np.var(members[1::2]-members[::2], axis=0, ddof=1)/(4*self.n_pairs_)
        return out

    def _historical_subset_error(self, data, result, permutations, fractions):
        """Original first-iteration diagnostic; the fraction here selects pairs, not training rows."""
        values = self._prediction_input(data)
        if not isinstance(result, dict):
            raise TypeError('statDictionary must be a dictionary')
        permutations = self._positive_int(permutations, 'nPermutation', 2)
        fractions = np.asarray(fractions, dtype=float)
        if fractions.ndim != 1 or not np.isfinite(fractions).all() or np.any((fractions<=0)|(fractions>=1)):
            raise ValueError('fractions must be finite and strictly between zero and one')
        counts = np.floor(fractions*self.n_pairs_).astype(int)
        if np.any(counts<1) or np.any(counts>=self.n_pairs_):
            raise ValueError('fraction must select between one and m-1 first-iteration pairs')
        keys = list(fractions)+list(1+fractions)
        if len(set(keys)) != len(keys):
            raise ValueError('fraction output keys must be distinct')
        outputs = {key: np.empty(len(values)) for key in keys}
        size = self._chunk_rows()
        with Parallel(n_jobs=self.optionsErrPDF['n_jobs'], require='sharedmem') as runner:
            for first in range(0, len(values), size):
                q = values[first:first+size]
                T = np.asarray(runner(delayed(self._validated_member_prediction)(
                    _BlockMember((tree,)), q) for tree in self.iteration_trees_[0]))
                differences = T[1::2]-T[::2]
                rng = np.random.RandomState(self.permutation_seed_)
                for f, k in zip(fractions, counts):
                    means = np.array([differences[rng.permutation(self.n_pairs_)[:k]].mean(0)
                                      for _ in range(permutations)])
                    raw = np.std(means, axis=0)
                    outputs[f][first:first+len(q)] = raw*.5*np.sqrt(f/(1-f))
                    outputs[1+f][first:first+len(q)] = raw
        result.update(outputs)
        return None

    @staticmethod
    def _group_blocks(active, codes, count, rng):
        """Stable O(N log N + G log B) gathering/assignment; preserve seeded ties."""
        import heapq
        units = int(codes.max()) + 1
        order = np.argsort(codes, kind='stable')
        by_group = np.split(active[order], np.cumsum(np.bincount(codes))[:-1])
        shuffled = rng.permutation(units)
        ordered = sorted(shuffled, key=lambda k: -len(by_group[k]))
        buckets = [[] for _ in range(count)]
        group_counts = np.zeros(count, dtype=int)
        tie_order = rng.permutation(count)
        heap = [(0, rank, int(b)) for rank, b in enumerate(tie_order)]
        heapq.heapify(heap)
        for k in ordered:
            size, rank, b = heapq.heappop(heap)
            buckets[b].append(by_group[k])
            group_counts[b] += 1
            heapq.heappush(heap, (size + len(by_group[k]), rank, b))
        if any(not bucket for bucket in buckets):
            raise ValueError('not enough groups for nonempty blocks/folds')
        return [np.concatenate(bucket) for bucket in buckets], group_counts

    def _fit_noise(self, X, y, weights, groups, depth, repeats, seed):
        """Five OOF mean ensembles, signed subtraction before noise-model fitting."""
        import copy
        import pandas as pd
        rng = np.random.default_rng(seed)
        active = self.active_indices_
        if groups is None:
            if len(active) < 5:
                raise ValueError('noise fitting requires five nonempty independent folds')
            folds = [a.copy() for a in np.array_split(rng.permutation(active), 5)]
        else:
            codes, labels = pd.factorize(np.asarray(groups)[active], sort=False)
            if len(labels) < 5:
                raise ValueError('noise fitting requires at least five independent groups')
            folds, _ = self._group_blocks(active, codes, 5, rng)
        fold_seeds = rng.choice(2**31 - 1, size=6, replace=False)
        oof_mean = np.full(len(y), np.nan)
        oof_var = np.full(len(y), np.nan)
        fold_ids = np.full(len(y), -1, dtype=int)
        records = []
        for f, held in enumerate(folds):
            train = active[~np.isin(active, held)]
            opts = copy.deepcopy(self.optionsErrPDF)
            opts.update(fit_noise=False, random_state=int(fold_seeds[f]),
                        mean_depth=depth, niter_max=repeats, dump_progress=False)
            kwargs = dict(self.options); kwargs.pop('random_state', None)
            child = type(self)('Regressor', opts, **kwargs)
            child.fit(X[train], y[train], weights[train],
                      groups=None if groups is None else np.asarray(groups)[train])
            oof_mean[held] = child.predict(X[held])
            # Private arithmetic path: no calibration claim or experimental warning per fold.
            variance = np.empty(len(held))
            for first, members in child._member_chunks(X[held]):
                variance[first:first + members.shape[1]] = np.var(
                    members[1::2] - members[::2], axis=0, ddof=1) / (4 * child.n_pairs_)
            oof_var[held] = variance
            fold_ids[held] = f
            records.append(dict(train_indices=train.copy(), heldout_indices=held.copy(),
                block_indices=tuple(train[idx] for idx in child.block_indices_),
                block_count=child.block_count_, seed=int(fold_seeds[f])))
        signed = (y[active] - oof_mean[active])**2 - oof_var[active]
        if not np.isfinite(signed).all():
            raise ValueError('OOF corrected targets must be finite')
        requested = self.optionsErrPDF['noise_options']
        if not isinstance(requested, dict):
            raise ValueError('noise_options must be a dictionary of RandomForestRegressor options')
        options = dict(n_estimators=128, min_samples_leaf=20, criterion='squared_error',
                       random_state=int(fold_seeds[-1]), n_jobs=self.optionsErrPDF['n_jobs'])
        options.update(copy.deepcopy(requested))
        if options.get('criterion') != 'squared_error':
            raise ValueError('noise model requires squared_error to average signed corrections')
        noise_model = RandomForestRegressor(**options)
        w = weights[active]
        fit_weights = np.ones(len(w)) if np.all(w == w[0]) else w
        noise_model.fit(X[active], signed, sample_weight=fit_weights)
        self.noise_model_ = noise_model
        self.noise_options_ = copy.deepcopy(options)
        self.oof_mean_ = oof_mean
        self.oof_reducible_var_ = oof_var
        self.oof_signed_target_ = np.full(len(y), np.nan)
        self.oof_signed_target_[active] = signed
        self.oof_fold_ids_ = fold_ids
        self.oof_folds_ = tuple(records)
        self.noise_seed_ = seed

    def _prediction_input(self, data):
        from sklearn.exceptions import NotFittedError
        if not getattr(self, 'is_fitted_', False):
            raise NotFittedError('MIForestErrPDF must be successfully fitted first')
        values = _stat_input(data, [getattr(self, 'feature_names_in_', None)])
        if values.shape[1] != self.n_features_in_:
            raise ValueError('X feature count differs from fitted model')
        return values

    def _chunk_rows(self):
        # Persist the fitted option, retaining the helper cap when at its default.
        size = getattr(self, 'prediction_chunk_rows_', _STAT_CHUNK_ROWS)
        return min(size, _STAT_CHUNK_ROWS) if size == 10000 else size

    @staticmethod
    def _validated_member_prediction(member, chunk):
        # Only our exact, owned, unnamed sklearn regression trees may bypass
        # their repeated X validation. The public boundary already checked the
        # schema, feature count, finite values and float32 representation.
        if type(member) is not _BlockMember:
            return _stat_tree_prediction(member, chunk)
        predictions = []
        for tree in member.trees:
            if type(tree) is DecisionTreeRegressor and not hasattr(tree, 'feature_names_in_'):
                prediction = np.asarray(tree.predict(chunk, check_input=False))
                if prediction.shape != (len(chunk),) or not np.isfinite(prediction).all():
                    raise ValueError('forest statistics require finite single-output member predictions')
            else:
                prediction = _stat_tree_prediction(tree, chunk)
            predictions.append(prediction)
        return np.mean(predictions, axis=0)

    def _member_chunks(self, values, smear_streams=None):
        size = self._chunk_rows()
        if self.optionsErrPDF['n_jobs'] == 1:
            for first in range(0, len(values), size):
                chunk = values[first:first+size]
                prediction = np.asarray([self._validated_member_prediction(member, chunk)
                    for member in self.members_] if smear_streams is None else
                    [self._smeared_member_prediction(member, chunk, smear_streams[b])
                     for b, member in enumerate(self.members_)])
                yield first, prediction
            return
        with Parallel(n_jobs=self.optionsErrPDF['n_jobs'], require='sharedmem') as runner:
            for first in range(0, len(values), size):
                chunk = values[first:first + size]
                if smear_streams is None:
                    prediction = np.asarray(runner(delayed(self._validated_member_prediction)(
                        member, chunk) for member in self.members_))
                else:
                    prediction = np.asarray(runner(delayed(self._smeared_member_prediction)(
                        member, chunk, smear_streams[b]) for b, member in enumerate(self.members_)))
                yield first, prediction

    def _uncertainty_policy(self, experimental=False):
        import warnings
        if not isinstance(experimental, (bool, np.bool_)):
            raise ValueError('experimental must be Boolean')
        status = self.uncertainty_status_
        if status['grouped'] or status['nonuniform_weights']:
            if not experimental:
                raise ValueError('grouped/nonuniform-weight uncertainty is not calibrated; use predictReducibleStd(experimental=True) explicitly')
            warnings.warn('experimental grouped/weighted uncertainty: validated=False', RuntimeWarning, stacklevel=3)

    def predict(self, data, *, augment=False, random_state=None, prediction="mean", **options):
        """Equal-block mean; augment=True independently smears queries for each tree.

        No uncertainty implied. random_state replays jitter without advancing
        the model stream; without it each augmented call draws fresh jitter.
        """
        if prediction == 'linear':
            return self.predictLinear(data, augment=augment, random_state=random_state, **options)
        if prediction != 'mean':
            raise ValueError("prediction must be 'mean' or 'linear'")
        if options:
            raise TypeError('unknown prediction options: ' + ', '.join(options))
        values = self._prediction_input(data)
        streams = self._smear_streams(augment, random_state)
        out = np.empty(len(values))
        for first, members in self._member_chunks(values, streams):
            out[first:first + members.shape[1]] = np.mean(members, axis=0)
        return out


    def predictLocalConvolution(self, data, *, sigma=None, degrees=None,
                                parent_levels=1, ridge=0., chunk_rows=512,
                                max_working_mb=64., progress=None):
        """Experimental deterministic parent-kernel regression; no calibrated SE.

        degrees[j]=0 excludes coordinate j; 1/2 includes its linear/quadratic
        terms, without interactions. Descendant leaf centroids/values supply
        observations. Leaf Gaussian masses, normalized within each selected
        parent, weight these observations equally by tree. See the accompanying
        CRR for the distinction between local and full-tree convolution.
        Existing prediction, response-spread and quantile APIs are unchanged.
        local_half_A/B_fallback and local_half_A/B_condition describe each
        independent half solve; half means remain uncalibrated diagnostics.
        """
        self._require_leaf_stats()
        values = self._prediction_input(data)
        return _ri_local_convolution(self, values, sigma, degrees, parent_levels,
                                     ridge, chunk_rows, max_working_mb, progress)

    @staticmethod
    def _retain_leaf_stats(tree, X, y, weights):
        """Leaf sums in actual fitting coordinates, including per-tree fit jitter."""
        ids = tree.apply(X, check_input=False)
        size, dim = tree.tree_.node_count, X.shape[1]
        count = np.bincount(ids, minlength=size)
        sw = np.bincount(ids, weights=weights, minlength=size)
        sw2 = np.bincount(ids, weights=weights**2, minlength=size)
        leaf = np.flatnonzero(count)
        centroid = np.full((size, dim), np.nan)
        low, high = np.full((size, dim), np.inf), np.full((size, dim), -np.inf)
        for j in range(dim):
            centroid[leaf, j] = np.bincount(ids, weights=weights*X[:, j], minlength=size)[leaf]/sw[leaf]
            np.minimum.at(low[:, j], ids, X[:, j])
            np.maximum.at(high[:, j], ids, X[:, j])
        mean = np.full(size, np.nan)
        mean[leaf] = np.bincount(ids, weights=weights*y, minlength=size)[leaf]/sw[leaf]
        ss = np.bincount(ids, weights=weights*(y-mean[ids])**2, minlength=size)
        denom = sw[leaf]-sw2[leaf]/sw[leaf]
        scatter = np.full(size, np.nan)
        valid = leaf[denom > 0]
        scatter[valid] = ss[valid]/(sw[valid]-sw2[valid]/sw[valid])
        # Partition bounds differ from observed sample extents (singleton extent=0).
        lower, upper = np.full((size, dim), -np.inf), np.full((size, dim), np.inf)
        for node in range(size):
            left, right = tree.tree_.children_left[node], tree.tree_.children_right[node]
            if left < 0:
                continue
            j, threshold = tree.tree_.feature[node], tree.tree_.threshold[node]
            lower[left], upper[left] = lower[node], upper[node]
            lower[right], upper[right] = lower[node], upper[node]
            upper[left, j], lower[right, j] = threshold, threshold
        tree._ri_leaf_stats = dict(leaf=leaf, count=count, sum_weight=sw, sum_weight2=sw2,
            centroid=centroid, sample_min=low, sample_max=high, cell_lower=lower,
            cell_upper=upper, target_mean=mean, target_scatter_var=scatter)

    def _require_leaf_stats(self):
        if not getattr(self, 'store_leaf_stats_', False):
            raise RuntimeError('fit with store_leaf_stats=True to use linear prediction or leaf exports')

    def _linear_selections(self, values, streams):
        # Each tree owns its RNG; ordered joblib results preserve seeded jitter
        # and the original accumulation order for every worker count.
        def select(b, l, tree):
            q = values if streams is None else self._smear_values(values, streams[b][l])
            ids = tree.apply(q, check_input=False)
            return b, l, tree, ids, tree._ri_leaf_stats, q
        if self.optionsErrPDF['n_jobs'] == 1:
            for b, trees in enumerate(self.block_trees_):
                for l, tree in enumerate(trees):
                    yield select(b, l, tree)
            return
        with Parallel(n_jobs=self.optionsErrPDF['n_jobs'], require='sharedmem') as runner:
            selected = runner(delayed(select)(b, l, tree)
                for b, trees in enumerate(self.block_trees_) for l, tree in enumerate(trees))
        yield from selected

    def _linear_from_selected(self, q, selected, ridge=0., condition_limit=1e10):
        """Shared arithmetic owner; selected leaves already belong to this batch."""
        d = q.shape[1]
        sz, sy = np.zeros_like(q, dtype=float), np.zeros(len(q))
        szz, szy = np.zeros((len(q), d, d)), np.zeros_like(q, dtype=float)
        count = 0
        for _, _, tree, ids, stat, _ in selected:
            delta = stat['centroid'][ids]-q  # original query, not smeared traversal coordinate
            y = tree.tree_.value[ids, 0, 0]
            sz += delta; sy += y
            szz += np.einsum('ni,nj->nij', delta, delta)
            szy += delta*y[:, None]
            count += 1
        zbar, ybar = sz/count, sy/count
        covariance = szz/count-np.einsum('ni,nj->nij', zbar, zbar)
        cross = szy/count-zbar*ybar[:, None]
        scales = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0))
        usable = (scales > np.finfo(float).eps).all(1)
        safe_scales = np.where(scales > np.finfo(float).eps, scales, 1.)
        matrix = covariance/(safe_scales[:, :, None]*safe_scales[:, None, :])
        eigen = np.linalg.eigvalsh(matrix)
        # A ridge may regularize a singular design, but zero-spread axes remain
        # unidentified; with ridge=0 enforce the original design condition.
        denom = eigen[:, 0]+ridge
        cond = np.full(len(q), np.inf)
        np.divide(eigen[:, -1]+ridge, denom, out=cond, where=denom > 0)
        usable &= np.isfinite(cond) & (cond <= condition_limit)
        result = ybar.copy()
        good = np.flatnonzero(usable)
        slopes = np.full_like(zbar, np.nan)
        if len(good):
            solved = np.linalg.solve(matrix[good]+ridge*np.eye(d),
                                     (cross/safe_scales)[good, :, None])[:, :, 0]
            slopes[good] = solved/safe_scales[good]
            result[good] -= np.sum(zbar[good]*slopes[good], axis=1)
        return dict(mean=result, gradient=slopes, tree_mean=ybar, fallback=~usable, condition=cond)

    def _combined_prediction_chunks(self, data, *, linear=False, variance=False,
            members=False, augment=False, random_state=None, ridge=0.,
            chunk_rows=None, max_working_mb=64., local_options=None):
        """Private request engine: one traversal, scoped to bounded query batches."""
        values = self._prediction_input(data)
        if linear or local_options is not None:
            self._require_leaf_stats()
        if augment and random_state is None:
            raise ValueError('combined augmentation requires an explicit replay seed')
        if local_options is not None and augment:
            raise ValueError('deterministic local convolution cannot reuse jittered selections')
        if isinstance(ridge, (bool, np.bool_)) or not np.isscalar(ridge) or not np.isfinite(ridge) or ridge < 0:
            raise ValueError('ridge must be finite and nonnegative')
        if not np.isfinite(max_working_mb) or max_working_mb <= 0:
            raise ValueError('max_working_mb must be positive')
        if variance:
            self._uncertainty_policy(False)
        streams = self._smear_streams(augment, random_state)
        B = len(self.block_trees_); L = len(self.block_trees_[0])
        if any(len(block) != L for block in self.block_trees_):
            raise ValueError('combined predictions require rectangular iterations')
        requested = self._chunk_rows() if chunk_rows is None else self._positive_int(chunk_rows, 'chunk_rows')
        # Member values, optional offsets, adjusted values and IDs are bounded.
        per_row = 8 * B * L * (values.shape[1] + 5)
        size = min(requested, self._chunk_rows(), max(1, int(max_working_mb*1024**2/per_row)))
        for first in range(0, len(values), size):
            q = values[first:first+size]; n = len(q)
            if self.store_leaf_stats_:
                selected = list(self._linear_selections(q, streams))
            else:
                selected = []
                for b, block in enumerate(self.block_trees_):
                    for l, tree in enumerate(block):
                        query = q if streams is None else self._smear_values(q, streams[b][l])
                        selected.append((b,l,tree,tree.apply(query,check_input=False),None,query))
            raw = np.column_stack([tree.tree_.value[ids,0,0] for _,_,tree,ids,_,_ in selected])
            # Preserve the public block/iteration reduction order, not a flat
            # mean across trees with a potentially different rounding order.
            block_means = np.asarray([np.mean([raw[:,b*L+l] for l in range(L)],axis=0) for b in range(B)])
            result = dict(mean=block_means.mean(axis=0))
            if variance:
                if getattr(self,'block_sampling_','fixed') == 'historical':
                    T = np.asarray([[raw[:,b*L+l] for b in range(B)] for l in range(L)])
                    v = np.var(T[0,1::2]-T[0,::2],axis=0,ddof=1)/(4*(B//2))
                    if L>1:v -= (1-1/L)*np.var(T.mean(axis=1),axis=0,ddof=1)
                else:
                    v = np.var(block_means[1::2]-block_means[::2],axis=0,ddof=1)/(4*self.n_pairs_)
                result['variance'] = v
            if linear:
                result['linear'] = self._linear_from_selected(q, selected, ridge)
            if members:
                result['members'] = dict(Mean=raw)
                if linear:
                    gradient = np.where(result['linear']['fallback'][:,None],0.,result['linear']['gradient'])
                    delta = np.stack([stat['centroid'][ids]-q for _,_,_,ids,stat,_ in selected],axis=1)
                    result['members']['Linear'] = raw-np.einsum('ntd,nd->nt',delta,gradient)
                    del delta
            if local_options is not None:
                options = dict(local_options)
                unknown = set(options)-{'sigma','degrees','parent_levels','ridge','max_working_mb'}
                if unknown:raise TypeError('unknown private local options: '+', '.join(sorted(unknown)))
                ids = np.column_stack([item[3] for item in selected])
                result['local'] = _ri_local_convolution(self,q,options.get('sigma'),options.get('degrees'),
                    options.get('parent_levels',1),options.get('ridge',ridge),n,
                    options.get('max_working_mb',max_working_mb),None,selected_leaves=ids)
            yield first, first+n, result

    def predictLinear(self, data, *, augment=False, random_state=None, ridge=0.,
                      condition_limit=1e10, return_diagnostics=False):
        """Equal-weight affine fit to selected (leaf centroid, leaf mean) pairs.

        Slopes use per-query centroid-spread scaling; ridge is dimensionless and
        penalizes slopes only. Singular/ill-conditioned designs fall back to the
        selected-tree mean with NaN gradients. This is NOT a calibrated interval.
        """
        values = self._prediction_input(data)
        self._require_leaf_stats()
        for value, name, minimum in [(ridge, 'ridge', 0.), (condition_limit, 'condition_limit', 1.)]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value < minimum:
                raise ValueError(name+' must be finite and >= '+str(minimum))
        if not isinstance(return_diagnostics, (bool, np.bool_)):
            raise ValueError('return_diagnostics must be Boolean')
        streams = self._smear_streams(augment, random_state)
        n, d = values.shape
        pred, tree_mean = np.empty(n), np.empty(n)
        gradient = np.full((n, d), np.nan)
        fallback, condition = np.ones(n, dtype=bool), np.full(n, np.inf)
        for first in range(0, n, self._chunk_rows()):
            q = values[first:first+self._chunk_rows()]
            batch = self._linear_from_selected(q, self._linear_selections(q, streams), ridge, condition_limit)
            result, ybar = batch['mean'], batch['tree_mean']
            slopes, usable, cond = batch['gradient'], ~batch['fallback'], batch['condition']
            sl = slice(first, first+len(q))
            pred[sl], tree_mean[sl] = result, ybar
            gradient[sl], fallback[sl], condition[sl] = slopes, ~usable, cond
        if return_diagnostics:
            return dict(mean=pred, gradient=gradient, tree_mean=tree_mean,
                        fallback=fallback, condition=condition)
        return pred

    def leafStatsFrame(self):
        """Pandas table: one row per fitted tree/leaf; fit arrays are never exposed by reference."""
        import pandas as pd
        self._prediction_input(np.empty((0, getattr(self, 'n_features_in_', 0))))
        self._require_leaf_stats()
        frames = []
        for b, trees in enumerate(self.block_trees_):
            for l, tree in enumerate(trees):
                stat = tree._ri_leaf_stats; ids = stat['leaf']
                row = dict(block=b, iteration=l, leaf_node=ids.copy(),
                    leaf_value=tree.tree_.value[ids, 0, 0].copy())
                for key in ['count', 'sum_weight', 'sum_weight2', 'target_mean', 'target_scatter_var']:
                    row[key] = stat[key][ids].copy()
                for j in range(self.n_features_in_):
                    for key in ['centroid', 'sample_min', 'sample_max', 'cell_lower', 'cell_upper']:
                        row[key+'_'+str(j)] = stat[key][ids, j].copy()
                frames.append(pd.DataFrame(row))
        frame = pd.concat(frames, ignore_index=True)
        frame.attrs['feature_names'] = list(getattr(self, 'feature_names_in_', range(self.n_features_in_)))
        frame.attrs['coordinates'] = 'actual tree fitting inputs, including fit augmentation'
        frame.attrs['scatter'] = 'within-leaf sample scatter, unavailable for singleton leaves; not noise calibration'
        return frame

    def predictLinearFrame(self, data, **options):
        """Pandas table: query coordinates, linear/mean predictions, slopes, fallback and condition."""
        import pandas as pd
        values = self._prediction_input(data)
        if 'return_diagnostics' in options:
            raise TypeError('predictLinearFrame controls return_diagnostics')
        result = self.predictLinear(data, return_diagnostics=True, **options)
        row = dict(prediction=result['mean'], tree_mean=result['tree_mean'],
                   fallback=result['fallback'], condition=result['condition'])
        for j in range(self.n_features_in_):
            row['x_'+str(j)] = values[:, j].copy()
            row['gradient_'+str(j)] = result['gradient'][:, j]
        frame = pd.DataFrame(row, index=getattr(data, 'index', None) if hasattr(data, 'columns') else None)
        frame.attrs['feature_names'] = list(getattr(self, 'feature_names_in_', range(self.n_features_in_)))
        frame.attrs['uncertainty'] = 'no local-linear uncertainty estimate supplied'
        return frame

    def leafQueryFrame(self, data, *, augment=False, random_state=None, max_rows=1000000):
        """Pandas debug table: selected leaf per query/tree, including jittered traversal coordinates.

        Pass a small query subset; this explicit export has query*tree rows.
        Original query row positions are used even for duplicate pandas indexes.
        """
        import pandas as pd
        values = self._prediction_input(data)
        self._require_leaf_stats()
        limit = self._positive_int(max_rows, 'max_rows')
        if len(values)*len(self.estimators_) > limit:
            raise ValueError('leafQueryFrame exceeds max_rows; select fewer queries or raise max_rows')
        streams = self._smear_streams(augment, random_state)
        tables = []
        for b, l, tree, ids, stat, traversal in self._linear_selections(values, streams):
            row = dict(query_position=np.arange(len(values)), block=b, iteration=l,
                       leaf_node=ids.copy(), leaf_value=tree.tree_.value[ids, 0, 0].copy(), count=stat['count'][ids])
            for j in range(self.n_features_in_):
                row['x_'+str(j)] = values[:, j].copy()
                row['traversal_'+str(j)] = traversal[:, j].copy()
                row['centroid_'+str(j)] = stat['centroid'][ids, j].copy()
                row['delta_'+str(j)] = stat['centroid'][ids, j]-values[:, j]
            tables.append(pd.DataFrame(row))
        frame = pd.concat(tables, ignore_index=True)
        frame.attrs['feature_names'] = list(getattr(self, 'feature_names_in_', range(self.n_features_in_)))
        return frame

    def predictStat(self, data, statDictionary=None, treeType=-1, *legacy):
        """Statistics of block averages; -1 warns and retains group-0 fallback.

        Historical four-argument residual-tree calls are unsupported. Quantiles
        describe block-member predictions, not response distributions.
        """
        import warnings
        if legacy or (statDictionary is not None and not isinstance(statDictionary, dict)):
            raise TypeError("historical predictStat(data, treeType, statDictionary, treeSlice) is unsupported; use predictStat(data, statDictionary, treeType=0/1/'all')")
        values = self._prediction_input(data)
        if isinstance(treeType, (bool, np.bool_)):
            raise ValueError('Boolean treeType selectors are unsupported')
        if treeType == -1:
            warnings.warn("predictStat treeType=-1 selects group 0; specify 0, 1 or 'all'", FutureWarning, stacklevel=2)
            treeType = 0
        if treeType == 'all':
            selected = self.members_
        elif treeType in (0, 1):
            selected = self.members_[treeType::2]
        else:
            raise ValueError("treeType must be -1, 0, 1 or 'all'")
        request = _stat_requests({'mean': True, 'std': True} if statDictionary is None else statDictionary)
        want_std2 = 'std2' in request
        if want_std2:
            request.pop('std2')
            if len(selected) < 2:
                raise ValueError('std2 requires at least two block members')
        result = predictForestStat(selected, values, request, self.optionsErrPDF['n_jobs'])
        if want_std2:
            order = np.random.default_rng(self.permutation_seed_).permutation(len(selected))
            half = len(selected) // 2
            result['std2'] = np.empty(len(values))
            for first in range(0, len(values), _STAT_CHUNK_ROWS):
                members = np.asarray([member.predict(values[first:first + _STAT_CHUNK_ROWS]) for member in selected])
                result['std2'][first:first + members.shape[1]] = np.std(
                    members[order[:half]] - members[order[half:2*half]], axis=0)
        return result

    def predictReducibleStd(self, data, *, experimental=False, augment=False, random_state=None):
        """sqrt(var(T_1j-T_0j, ddof=1)/(4*m)); same members as predict.

        Target units. Finite-array identity only; scientific calibration awaits
        Stage D. Repeated trees are averaged, never counted as independent blocks.
        """
        if getattr(self, 'block_sampling_', 'fixed') == 'historical':
            return np.sqrt(np.maximum(self.predictReducibleVariance(data, experimental=experimental,
                augment=augment, random_state=random_state), 0.))
        values = self._prediction_input(data)
        self._uncertainty_policy(experimental)
        streams = self._smear_streams(augment, random_state)
        out = np.empty(len(values))
        for first, members in self._member_chunks(values, streams):
            differences = members[1::2] - members[::2]
            out[first:first + members.shape[1]] = np.sqrt(np.var(differences, axis=0, ddof=1) / (4 * self.n_pairs_))
        return out

    def predictReducibleError(self, data, statDictionary, nPermutation, fractionArray):
        """Legacy in-place paired subset adapter (None return).

        Uses actual k/m fractions and population spread across permutations;
        finite Monte Carlo budgets have downward variance bias. Prefer the
        authoritative closed form predictReducibleStd. No global RNG mutation.
        """
        if getattr(self, 'block_sampling_', 'fixed') == 'historical':
            return self._historical_subset_error(data, statDictionary, nPermutation, fractionArray)
        values = self._prediction_input(data)
        self._uncertainty_policy()
        if not isinstance(statDictionary, dict):
            raise TypeError('statDictionary must be a dictionary')
        permutations = self._positive_int(nPermutation, 'nPermutation', 2)
        fractions = np.asarray(fractionArray, dtype=float)
        if fractions.ndim != 1 or not np.isfinite(fractions).all() or np.any((fractions <= 0) | (fractions >= 1)):
            raise ValueError('fractions must be a finite sequence strictly between zero and one')
        keys = list(fractions) + list(1 + fractions)
        if len(set(keys)) != len(keys):
            raise ValueError('fraction output keys must be distinct and noncolliding')
        counts = np.floor(fractions * self.n_pairs_).astype(int)
        if np.any(counts < 1) or np.any(counts >= self.n_pairs_):
            raise ValueError('actual subset count must satisfy 1 <= k < pair count')
        result = {key: np.empty(len(values)) for key in keys}
        for first, members in self._member_chunks(values):
            differences = members[1::2] - members[::2]
            for fraction, k in zip(fractions, counts):
                rng = np.random.default_rng(self.permutation_seed_)
                mean = np.zeros(members.shape[1])
                m2 = np.zeros(members.shape[1])
                for iteration in range(1, permutations + 1):
                    delta = differences[rng.permutation(self.n_pairs_)[:k]].mean(axis=0)
                    step = delta - mean
                    mean += step / iteration
                    m2 += step * (delta - mean)
                raw = np.sqrt(m2 / permutations)
                actual = k / self.n_pairs_
                sl = slice(first, first + len(raw))
                result[fraction][sl] = raw * .5 * np.sqrt(actual / (1 - actual))
                result[1 + fraction][sl] = raw
        statDictionary.update(result)
        return None

    def predictUncertainty(self, data, *, experimental=False):
        """Mean, paired sigma and raw/clipped conditional-noise variance diagnostics.

        All sigmas have target units; raw variance has squared target units.
        Bias contamination and clipping remain visible; no PDF/coverage guarantee.
        """
        import copy
        values = self._prediction_input(data)
        if not self.fit_noise_:
            raise RuntimeError('noise model unavailable; fit with fit_noise=True or use predict and predictReducibleStd')
        self._uncertainty_policy(experimental)
        raw = np.empty(len(values))
        size = self._chunk_rows()
        for first in range(0, len(values), size):
            chunk = values[first:first + size]
            raw[first:first + len(chunk)] = self.noise_model_.predict(chunk)
        if not np.isfinite(raw).all():
            raise ValueError('noise model returned nonfinite variance')
        mean = np.empty(len(values))
        reducible = np.empty(len(values))
        for first, members in self._member_chunks(values):
            mean[first:first + members.shape[1]] = np.mean(members, axis=0)
            reducible[first:first + members.shape[1]] = np.sqrt(np.var(
                members[1::2] - members[::2], axis=0, ddof=1) / (4 * self.n_pairs_))
        clipped = np.maximum(raw, 0.)
        metadata = copy.deepcopy(self.uncertainty_status_)
        metadata.update(noise_fit=True, noise_folds=5,
                        noise_interpretation='signed OOF residual variance minus fitted-mean variance; may contain squared bias')
        return dict(mean=mean, reducible_std=reducible,
            irreducible_std=np.sqrt(clipped), irreducible_var_raw=raw,
            predictive_std=np.sqrt(reducible**2 + clipped), metadata=metadata)

    def getImportance(self):
        """Mean feature importance with the same block/within-block weights as predict."""
        self._prediction_input(np.empty((0, self.n_features_in_)) if getattr(self, 'is_fitted_', False) else np.empty((0, 1)))
        return np.mean([np.mean([t.feature_importances_ for t in block], axis=0)
                        for block in self.block_trees_], axis=0)


# PHASE_2_3 experimental, opt-in path. Legacy estimator definitions are preserved.
def _ri_parent_geometry(tree, levels):
    """One cached parent level per fitted tree, replaced on level/refit change."""
    t = tree.tree_; size = t.node_count; dim = tree.n_features_in_
    stats = tree._ri_leaf_stats
    level = min(levels, t.max_depth)
    cached = getattr(tree, '_ri_parent_geometry_cache', None)
    if cached is not None and cached[0] is t and cached[1] is stats and cached[2] == level:
        return cached[3]
    parent = np.zeros(size, dtype=np.intp)
    start = np.zeros(size, dtype=np.intp); stop = start.copy()
    leaf_ids = []
    stack = [(0, False)]
    while stack:
        node, done = stack.pop()
        if done:
            stop[node] = len(leaf_ids); continue
        start[node] = len(leaf_ids)
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:
            leaf_ids.append(node)
            stop[node] = len(leaf_ids); continue
        parent[left] = parent[right] = node
        stack.extend([(node, True), (right, False), (left, False)])
    ancestor = np.arange(size)
    # Saturate at the root; arbitrarily large levels need no long loop.
    for _ in range(level):
        ancestor = parent[ancestor]
    leaves = np.asarray(leaf_ids)
    geometry = dict(start=start, stop=stop, ancestor=ancestor,
                lower=stats['cell_lower'][leaves], upper=stats['cell_upper'][leaves],
                value=t.value[leaves, 0, 0],
                centroid=stats['centroid'][leaves])
    tree._ri_parent_geometry_cache = (t, stats, level, geometry)
    return geometry


def _ri_gaussian_leaf_mass(query, lower, upper, sigma):
    """Matched rows, stable positive-tail CDF subtraction; no random draws."""
    from scipy.special import ndtr
    active = np.flatnonzero(sigma > 0)
    factors = np.empty_like(query, dtype=float)
    if len(active):
        a = (lower[:, active] - query[:, active]) / sigma[active]
        b = (upper[:, active] - query[:, active]) / sigma[active]
        reflected = a >= 0
        # One CDF call for both limits and all active coordinates. Reflection
        # retains the existing stable positive-tail subtraction exactly.
        limits = np.stack((np.where(reflected, -a, b), np.where(reflected, -b, a)))
        cdf = ndtr(limits)
        factors[:, active] = np.maximum(0., cdf[0] - cdf[1])
    inactive = np.flatnonzero(sigma == 0)
    factors[:, inactive] = ((query[:, inactive] > lower[:, inactive]) &
                            (query[:, inactive] <= upper[:, inactive]))
    mass = np.ones(len(query))
    for j in range(query.shape[1]):
        mass *= factors[:, j]
    return mass


def _ri_descendant_chunks(starts, counts, max_pairs):
    """Bounded ragged expansion; only two scalar searches per work slice."""
    cumulative = np.r_[0, np.cumsum(counts)]
    for begin in range(0, int(cumulative[-1]), max_pairs):
        end = min(begin + max_pairs, int(cumulative[-1]))
        first = np.searchsorted(cumulative[1:], begin, side='right')
        last = np.searchsorted(cumulative[1:], end - 1, side='right') + 1
        owners = np.arange(first, last)
        lengths = np.minimum(cumulative[owners + 1], end) - np.maximum(cumulative[owners], begin)
        qi = np.repeat(owners, lengths)
        li = starts[qi] + np.arange(begin, end) - cumulative[qi]
        yield qi, li


def _ri_local_tree_moments(item, q, sigma, terms, P, max_pairs, selected=None):
    (b, l, tree), geo = item
    n = len(q)
    if selected is None:selected = tree.apply(q, check_input=False)
    nodes = geo['ancestor'][selected]
    starts = geo['start'][nodes]; counts = geo['stop'][nodes] - starts
    # Subdivide by actual number of descendant observations, not just
    # query count. One enormous subtree is also split across leaf slices.
    weighted_y = np.zeros(n); total_mass = np.zeros(n)
    tree_G = np.zeros((n, P, P)); tree_h = np.zeros((n, P))
    # Collect unnormalized moments and mass together. Normalize each
    # tree's complete contribution once, preserving equal tree weight.
    for qi, li in _ri_descendant_chunks(starts, counts, max_pairs):
        mass = _ri_gaussian_leaf_mass(q[qi], geo['lower'][li], geo['upper'][li], sigma)
        total_mass += np.bincount(qi, mass, minlength=n)
        weighted_y += np.bincount(qi, mass * geo['value'][li], minlength=n)
        delta = geo['centroid'][li] - q[qi]
        design = np.column_stack([np.ones(len(qi))] + [delta[:, j]**power for j, power in terms])
        weighted_design = mass[:, None] * design
        # Intercept moments are the same mass/value sums already accumulated.
        for k in range(1, P):
            tree_h[:, k] += np.bincount(qi, weighted_design[:, k] * geo['value'][li], minlength=n)
            for m in range(k + 1):
                value = np.bincount(qi, weighted_design[:, k] * design[:, m], minlength=n)
                tree_G[:, k, m] += value
                if m != k: tree_G[:, m, k] += value
    if np.any(total_mass <= 0):
        raise FloatingPointError('parent Gaussian mass underflow; no silent renormalization')
    tree_h[:, 0] = weighted_y
    tree_G[:, 0, 0] = total_mass
    return b, l, weighted_y / total_mass, total_mass, counts, tree_h / total_mass[:, None], tree_G / total_mass[:, None, None]


def _ri_local_convolution(model, values, sigma, degrees, levels, ridge,
                          chunk_rows, max_working_mb, progress, selected_leaves=None):
    dim = values.shape[1]
    sigma = model.smear_sigma_ if sigma is None else sigma
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (dim,) or not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError('sigma must contain one finite nonnegative width per coordinate')
    degrees = [1] * dim if degrees is None else list(degrees)
    if len(degrees) != dim or any(isinstance(d, (bool, np.bool_)) or
            not isinstance(d, (int, np.integer)) or d not in (0, 1, 2) for d in degrees):
        raise ValueError('degrees must be coordinate-wise integers 0, 1 or 2')
    for name, value, minimum in [('parent_levels', levels, 0), ('chunk_rows', chunk_rows, 1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(name + ' must be an integer >= ' + str(minimum))
    if not np.isfinite(ridge) or ridge < 0 or not np.isfinite(max_working_mb) or max_working_mb <= 0:
        raise ValueError('ridge must be nonnegative and max_working_mb positive')
    blocks = model.block_trees_; B = len(blocks); L = len(blocks[0])
    if B < 4 or B % 2 or any(len(b) != L for b in blocks):
        raise ValueError('at least four paired blocks and rectangular iterations required')
    trees = [(b, l, tree) for b, block in enumerate(blocks) for l, tree in enumerate(block)]
    for _, _, tree in trees:
        p = tree.get_params()
        expected = dict(min_samples_leaf=1, min_samples_split=2, max_depth=None,
                        max_leaf_nodes=None, min_impurity_decrease=0., ccp_alpha=0.,
                        min_weight_fraction_leaf=0.)
        if any(p[k] != v for k, v in expected.items()):
            raise ValueError('local convolution requires unrestricted mean-tree growth')
    geometry = [_ri_parent_geometry(tree, levels) for _, _, tree in trees]
    terms = [(j, power) for j, degree in enumerate(degrees) for power in range(1, degree + 1)]
    P = 1 + len(terms); N = len(values)
    # Conservative bound on temporaries per expanded query/descendant pair.
    # This is a work-array budget, not a total-process RSS guarantee.
    from joblib import effective_n_jobs
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    jobs = min(len(trees), effective_n_jobs(model.optionsErrPDF['n_jobs']))
    max_pairs = max(1, int(max_working_mb * 1024**2 / (jobs * 8 * (12*dim + 3*P + 24))))
    output = {k: np.empty(N) for k in ('mean_parent_convolved', 'mean_local',
        'mean_local_convolved', 'parent_half_A', 'parent_half_B',
        'parent_variance_raw', 'retained_mass', 'minimum_retained_mass',
        'condition', 'local_half_A', 'local_half_B',
        'local_half_A_condition', 'local_half_B_condition')}
    output.update(gradient=np.full((N, dim), np.nan),
                  curvature=np.full((N, dim), np.nan),
                  fallback=np.zeros(N, dtype=bool),
                  local_half_A_fallback=np.zeros(N, dtype=bool),
                  local_half_B_fallback=np.zeros(N, dtype=bool),
                  descendant_pairs=np.zeros(N, dtype=np.int64))
    def solve(G, h, mean):
        scales = np.sqrt(np.maximum(np.diagonal(G, axis1=1, axis2=2), 0.))
        valid = (scales > np.finfo(float).eps).all(1)
        safe = np.where(scales > np.finfo(float).eps, scales, 1.)
        matrix = G / safe[:, :, None] / safe[:, None, :]
        # Condition exactly the matrix solved below: ridge excludes intercept.
        penalty = np.eye(P) * ridge; penalty[0, 0] = 0.
        penalized = matrix + penalty
        eigen = np.linalg.eigvalsh(penalized)
        condition = np.full(len(G), np.inf)
        np.divide(eigen[:, -1], eigen[:, 0],
                  out=condition, where=eigen[:, 0] > 0)
        valid &= np.isfinite(condition) & (condition <= 1e12)
        coef = np.zeros_like(h); coef[:, 0] = mean
        if valid.any():
            # Use individual fallback if a numerically singular batch slips
            # through the condition test; never silently solve a pseudo-inverse.
            try:
                coef[valid] = np.linalg.solve(penalized[valid],
                    (h[valid] / safe[valid])[..., None])[..., 0] / safe[valid]
            except np.linalg.LinAlgError:
                for i in np.flatnonzero(valid):
                    try: coef[i] = np.linalg.solve(penalized[i], h[i]/safe[i]) / safe[i]
                    except np.linalg.LinAlgError: valid[i] = False
        return coef, ~valid, condition
    with ThreadPoolExecutor(max_workers=jobs) if jobs > 1 else nullcontext() as executor:
        for first in range(0, N, chunk_rows):
            last = min(first + chunk_rows, N); q = values[first:last]; n = len(q)
            G = np.zeros((2, n, P, P)); h = np.zeros((2, n, P))
            T = np.empty((L, B, n)); retained = np.zeros(n); minimum = np.ones(n)
            pairs_count = np.zeros(n, dtype=np.int64)
            def evaluate(indexed):
                index, item = indexed
                selected = None if selected_leaves is None else selected_leaves[first:last,index]
                return _ri_local_tree_moments(item, q, sigma, terms, P, max_pairs, selected)
            for wave in range(0, len(trees), jobs):
                items = list(enumerate(zip(trees[wave:wave+jobs], geometry[wave:wave+jobs]),start=wave))
                results = map(evaluate, items) if executor is None else executor.map(evaluate, items)
                # Executor.map preserves tree order. At most jobs results can be
                # pending; never retain a full forest of query normal equations.
                for b, l, tree_mean, total_mass, counts, tree_h, tree_G in results:
                    T[l, b] = tree_mean
                    retained += total_mass; minimum = np.minimum(minimum, total_mass)
                    pairs_count += counts
                    h[b % 2] += tree_h
                    G[b % 2] += tree_G
            G /= B*L/2; h /= B*L/2
            mean = T.mean(0).mean(0)
            coef, fallback, condition = solve(G.mean(0), h.mean(0), mean)
            gradient = np.zeros((n, dim)); curvature = np.zeros_like(gradient)
            for k, (j, power) in enumerate(terms, 1):
                if power == 1: gradient[:, j] = coef[:, k]
                else: curvature[:, j] = 2 * coef[:, k]
            gradient[fallback] = np.nan; curvature[fallback] = np.nan
            convolved_local = coef[:, 0].copy()
            for k, (j, power) in enumerate(terms, 1):
                if power == 2: convolved_local += coef[:, k] * sigma[j]**2
            sl = slice(first, last)
            output['mean_parent_convolved'][sl] = mean
            output['mean_local'][sl] = coef[:, 0]
            output['mean_local_convolved'][sl] = convolved_local
            output['parent_half_A'][sl] = T[:, ::2].mean(0).mean(0)
            output['parent_half_B'][sl] = T[:, 1::2].mean(0).mean(0)
            v1 = np.var(T[0, 1::2] - T[0, ::2], axis=0, ddof=1) / (4 * (B//2))
            vi = np.var(T.mean(1), axis=0, ddof=1) if L > 1 else np.zeros(n)
            if getattr(model, 'block_sampling_', 'fixed') == 'historical':
                output['parent_variance_raw'][sl] = v1 - (1 - 1/L)*vi
            else:
                members = T.mean(0)
                output['parent_variance_raw'][sl] = np.var(members[1::2]-members[::2], axis=0, ddof=1)/(4*(B//2))
            # Half fits are genuinely refitted with their own observations. Their
            # difference is a diagnostic, not a claimed calibrated standard error.
            for side, label in enumerate(('A', 'B')):
                half_coef, half_fallback, half_condition = solve(
                    G[side], h[side], output['parent_half_' + label][sl])
                output['local_half_' + label][sl] = half_coef[:, 0]
                output['local_half_' + label + '_fallback'][sl] = half_fallback
                output['local_half_' + label + '_condition'][sl] = half_condition
            output['retained_mass'][sl] = retained / (B*L)
            output['minimum_retained_mass'][sl] = minimum
            output['gradient'][sl] = gradient; output['curvature'][sl] = curvature
            output['fallback'][sl] = fallback; output['condition'][sl] = condition
            output['descendant_pairs'][sl] = pairs_count
            if progress: progress(first, last)
    return output
