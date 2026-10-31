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
    if parallel is None:
        with Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem') as runner:
            return _fill_forest_statistics(trees, values, request, out, chunk, quantile_key, runner, order)
    return _fill_forest_statistics(trees, values, request, out, chunk, quantile_key, parallel, order)


def _fill_forest_statistics(trees, X, request, out, chunk, quantile_key, parallel, order):
    """Fill canonical outputs; each chunk is released before predicting the next."""
    for first in range(0, len(X), chunk):
        last = min(first + chunk, len(X))
        predictions = np.asarray(parallel(delayed(_stat_tree_prediction)(tree, X[first:last]) for tree in trees))
        for name in ('mean', 'median', 'std'):
            if name in request:
                out[name][first:last] = getattr(np, name)(predictions, axis=0)
        if order is not None:
            half = len(trees) // 2
            out['std2'][first:last] = np.std(predictions[order[:half]] - predictions[order[half:2 * half]], axis=0)
        for q in request.get('quantile', []):
            out[quantile_key][q][first:last] = np.quantile(predictions, q, axis=0)
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
                        noise_options={})
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

    def fit(self, xIn, yIn, sample_weight=None, *, groups=None):
        """Fit mean members on disjoint original-data blocks; return self."""
        return self.fitWithParam(xIn, yIn, self.optionsErrPDF['mean_depth'],
                                 self.optionsErrPDF['niter_max'], sample_weight, groups=groups)

    def fitWithParam(self, x0, y0, meanDepth, nIterMax, sample_weight=None, *, groups=None):
        """Fit with explicit depth/repetition overrides; failed refits invalidate old state."""
        import pandas as pd
        self._reset_fit()
        try:
            X = _stat_input(x0)
            y = np.asarray(y0, dtype=float)
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            if y.ndim != 1 or len(y) != len(X) or not len(y) or not np.isfinite(y).all():
                raise ValueError('y must be finite, nonempty, and single-output with one value per row')
            if isinstance(meanDepth, (bool, np.bool_)) or not isinstance(meanDepth, (int, float, np.integer, np.floating)) or not np.isfinite(meanDepth) or meanDepth <= 0:
                raise ValueError('meanDepth must be finite and positive')
            repeats = self._positive_int(nIterMax, 'nIterMax')
            minimum = self._positive_int(self.optionsErrPDF['min_tree'], 'min_tree')
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
            tree_options = dict(self.options)
            tree_options.pop('random_state', None)
            def fit_block(b):
                idx = blocks[b]
                if weights[idx].sum() <= 0:
                    raise ValueError('each block requires positive weight')
                result = []
                for iteration in range(repeats):
                    tree = DecisionTreeRegressor(random_state=int(seeds[b, iteration]), **tree_options)
                    tree.fit(X[idx], y[idx], sample_weight=fit_weights[idx])
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

    def _member_chunks(self, values):
        for first in range(0, len(values), _STAT_CHUNK_ROWS):
            chunk = values[first:first + _STAT_CHUNK_ROWS]
            with Parallel(n_jobs=self.optionsErrPDF['n_jobs'], require='sharedmem') as runner:
                prediction = np.asarray(runner(delayed(member.predict)(chunk) for member in self.members_))
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

    def predict(self, data, **options):
        """Full equal-block ensemble mean in target units; no uncertainty implied."""
        if options:
            raise TypeError('unknown prediction options: ' + ', '.join(options))
        return self.predictStat(data, {'mean': True}, treeType='all')['mean']

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

    def predictReducibleStd(self, data, *, experimental=False):
        """sqrt(var(T_1j-T_0j, ddof=1)/(4*m)); same members as predict.

        Target units. Finite-array identity only; scientific calibration awaits
        Stage D. Repeated trees are averaged, never counted as independent blocks.
        """
        values = self._prediction_input(data)
        self._uncertainty_policy(experimental)
        out = np.empty(len(values))
        for first, members in self._member_chunks(values):
            differences = members[1::2] - members[::2]
            out[first:first + members.shape[1]] = np.sqrt(np.var(differences, axis=0, ddof=1) / (4 * self.n_pairs_))
        return out

    def predictReducibleError(self, data, statDictionary, nPermutation, fractionArray):
        """Legacy in-place paired subset adapter (None return).

        Uses actual k/m fractions and population spread across permutations;
        finite Monte Carlo budgets have downward variance bias. Prefer the
        authoritative closed form predictReducibleStd. No global RNG mutation.
        """
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
        for first in range(0, len(values), _STAT_CHUNK_ROWS):
            chunk = values[first:first + _STAT_CHUNK_ROWS]
            raw[first:first + len(chunk)] = self.noise_model_.predict(chunk)
        if not np.isfinite(raw).all():
            raise ValueError('noise model returned nonfinite variance')
        reducible = np.empty(len(values))
        for first, members in self._member_chunks(values):
            reducible[first:first + members.shape[1]] = np.sqrt(np.var(
                members[1::2] - members[::2], axis=0, ddof=1) / (4 * self.n_pairs_))
        clipped = np.maximum(raw, 0.)
        metadata = copy.deepcopy(self.uncertainty_status_)
        metadata.update(noise_fit=True, noise_folds=5,
                        noise_interpretation='signed OOF residual variance minus fitted-mean variance; may contain squared bias')
        return dict(mean=self.predict(values), reducible_std=reducible,
            irreducible_std=np.sqrt(clipped), irreducible_var_raw=raw,
            predictive_std=np.sqrt(reducible**2 + clipped), metadata=metadata)

    def getImportance(self):
        """Mean feature importance with the same block/within-block weights as predict."""
        self._prediction_input(np.empty((0, self.n_features_in_)) if getattr(self, 'is_fitted_', False) else np.empty((0, 1)))
        return np.mean([np.mean([t.feature_importances_ for t in block], axis=0)
                        for block in self.block_trees_], axis=0)
