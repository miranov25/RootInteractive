"""Stage-A statistics repair; the legacy MIForestErrPDF estimator awaits B/C repairs.

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


class MIForestErrPDF:
    """
    Forest with reducible and irreducible error estimation and PDF estimation
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """

    def __init__(self, switch, optionsErrPDF0, **kwargs):
        """
        :param switch: It can be chosen between 'Classifier' or 'Regressor' with the corresponding string.
        :param options ErrPDF options - see default PDF options
        :param kwargs: All options from sklearn can be used. For instance
        """
        defaultErrPDFOptions = {
            "mean_depth": 10,              # mean depth of the tree
            "min_tree": 64,                # minimal number of trees for fit
            "niter_max": 3,                # number of iterations
            "test_fraction": 0.025,         # test fraction of points to use to evaluate progress
            "n_jobs": 4,                   # nJobs - not yet implemented
            "nPermutation_RedErr":128,     # number of itteration to evaluate reducible error
            "dump_progress": False,        # switch to follow progress -slow donw by factor 10
            "verbose": 1
        }
        self.optionsErrPDF = defaultErrPDFOptions
        self.optionsErrPDF.update(optionsErrPDF0)
        self.method = "MIForestErrPDF"
        self.trees = {}
        self.trees[0] = []  # trees for data
        self.trees[1] = []  # trees for reference data
        self.deltaTrees = []  # trees for delta residuals
        self.type = switch
        self.options = kwargs
        self.model_name = ''
        self.nTrees=0

    def fit(self, xIn, yIn, sample_weight=None):
        self.fitWithParam(xIn, yIn, self.optionsErrPDF["mean_depth"], self.optionsErrPDF["niter_max"])

    def fitWithParam(self, x0, y0, meanDepth, nIterMax, sample_weight=None):
        """
        make fits in iteration
        :param X_train:         input data set
        :param y_train:         input data set
        :param meanDepth        mean depth of te tree
        :return:
        """
        n_jobs=self.optionsErrPDF["n_jobs"]
        fractions = [0.2, 0.3, 0.4,0.5, 0.6,0.7, 0.8,0.9]  # fracions to estimate reducible error
        assert len(x0) == len(y0)
        nRows = x0.shape[0]
        nSampleTest = int(nRows * self.optionsErrPDF["test_fraction"])
        nTrees = nRows / (2 ** meanDepth)
        nTrees = np.exp2(np.log2(nTrees)).astype(int)  # number of independent trees forced to be 2**N
        if nTrees < self.optionsErrPDF["min_tree"]:
            nTrees = self.optionsErrPDF["min_tree"]
        bucketSize = ((nRows / nTrees) // 2)
        self.nTrees=nTrees
        #
        #xIn, _, yIn, _ = train_test_split(x0, y0, test_size=0.0)
        xIn=x0; yIn=y0;
        #  nRows=100; xIn=np.linspace(0.0, 1.0, num=nRows); yIn=np.linspace(0.0, 1.0, num=nRows);
        x = {}
        y = {}
        ypred = {}
        statDict = {"mean": 0, "median": 0, "std": 0,"trim_mean":[0.1]}
        statDict0 = {"mean": 0, "median": 0, "std": 0,"trim_mean":[0.1]}
        statDict1 = {"mean": 0, "median": 0, "std": 0,"trim_mean":[0.1]}
        reducibleErrorEst = {}
        for tIter in range(0, nIterMax):
            # create permutations in 2 groups data and reference data - to be fitted with tree and refernce tree
            permutation = np.random.permutation(nRows // 2)
            # x[0]=xIn[0:nRows//2][permutation]
            x[0] = xIn[0:nRows // 2].iloc[permutation].reset_index(drop=True)
            y[0] = yIn[0:nRows // 2].iloc[permutation]
            permutation = np.random.permutation(nRows - nRows // 2)
            # x[1]=xIn[nRows//2:nRows][permutation]
            x[1] = xIn[nRows // 2:nRows].iloc[permutation].reset_index(drop=True)
            y[1] = yIn[nRows // 2:nRows].iloc[permutation]
            #
            # fit trees
            treeAllNew = []
            for iTree in range(0, nTrees):
                iter = (2 * iTree // nTrees)
                i0 = int(iTree * bucketSize)
                i1 = int((iTree + 1) * bucketSize)
                if self.type == 'Classifier':
                    clf = DecisionTreeClassifier()
                elif self.type == 'Regressor':
                    clf = DecisionTreeRegressor()
                #clf.fit(x[iter][i0:i1], y[iter][i0:i1])
                self.trees[iter].append(clf)
                treeAllNew.append(clf)
            Parallel(n_jobs=n_jobs,require="sharedmem",)(
                delayed(tree.fit)(x[(2 * iTree // nTrees)][int(iTree * bucketSize):int((iTree + 1) * bucketSize)],
                                  y[(2 * iTree // nTrees)][int(iTree * bucketSize):int((iTree + 1) * bucketSize)])
                for iTree,tree in enumerate(treeAllNew)
            )
            treeSlice = slice(0, (nTrees // 2) * (tIter + 1) - 1)
            if (not self.optionsErrPDF["dump_progress"]) & (tIter < nIterMax):
                #yPred0 = self.predictStat(xIn[:nSampleTest], 0, statDict, treeSlice)
                #yPred1 = self.predictStat(xIn[:nSampleTest], 1, statDict, treeSlice)
                #rmsPred = np.std(yPred0 - yPred1)
                statDictOut0=predictForestStat(self.trees[0][treeSlice],xIn[:nSampleTest].to_numpy(dtype=np.float32), statDict0,n_jobs)
                statDictOut1=predictForestStat(self.trees[1][treeSlice],xIn[:nSampleTest].to_numpy(dtype=np.float32), statDict1,n_jobs)
                rmsPred = np.std(statDictOut0["mean"] - statDictOut1["mean"])
                rmsPredMed = np.std(statDictOut0["median"] - statDictOut1["median"])
                keyName=statDict0["trim_mean"][0]
                rmsPredTM = np.std(statDictOut0["trim_mean"][keyName] - statDictOut1["trim_mean"][keyName])
                print(f"{tIter}\t{nTrees}\t{rmsPred:.4}\t{rmsPredMed:.4}\t{rmsPredTM:.4}")
                #self.predictReducibleError(xIn[:nSampleTest], reducibleErrorEst, 64, fractions)
                #for fraction in fractions:
                #    reducibleErrorEst[fraction] = np.mean(reducibleErrorEst[fraction])
                #print(f"{tIter}\t{nTrees}\t{rmsPred:.4f}\t{rmsPredMed:.4f}",reducibleErrorEst)
                continue
            return  0
            # estimate reducible error
            #yPred0 = self.predictStat(xIn, 0, statDict, treeSlice)
            predictForestStat(self.trees[0][treeSlice],xIn.to_numpy(dtype=np.float32), statDict,n_jobs)
            yPred0 =statDict["mean"]
            yPredMed0 = statDict["median"]
            yPredStd0 = statDict["std"]
            #yPred1 = self.predictStat(xIn, 1, statDict, treeSlice)
            predictForestStat(self.trees[1][treeSlice],xIn.to_numpy(dtype=np.float32), statDict,n_jobs)
            yPred1 =statDict["mean"]
            yPredMed1 = statDict["median"]
            yPredStd1 = statDict["std"]
            yPred = 0.5 * (yPred0 + yPred1)
            yDelta=(yIn - yPred)
            # make delta regression trees
            self.deltaTrees.clear()

            for iTree in range(0, 2 * nTrees):
                i0 = int(iTree * bucketSize)
                i1 = int((iTree + 1) * bucketSize)
                if self.type == 'Classifier':
                    clf = DecisionTreeClassifier()
                elif self.type == 'Regressor':
                    clf = DecisionTreeRegressor()
                #clf.fit(xIn[i0:i1], yDelta[i0:i1])
                self.deltaTrees.append(clf)
            Parallel(n_jobs=n_jobs,require="sharedmem",)(
                delayed(tree.fit)(xIn[int(iTree * bucketSize):int((iTree + 1) * bucketSize)],
                                  yDelta[int(iTree * bucketSize):int((iTree + 1) * bucketSize)])
                for iTree,tree in enumerate(self.deltaTrees)
            )

            rmsPred = np.std(yPred0 - yPred1)
            rmsPredMed = np.std(yPredMed0 - yPredMed1)
            meanStd = np.mean(0.5 * (yPredStd0 + yPredStd1))
            meanDeltaRelStd = np.std((yPredStd0 - yPredStd1)) / np.mean(0.5 * (yPredStd0 + yPredStd1))
            rmsPoint = np.std(0.5 * (yPred0 + yPred1) - yIn)
            #
            treeSlice = slice(0, len(self.deltaTrees))
            #deltaPred = self.predictStat(xIn[:nSampleTest], -1, statDict, treeSlice)
            predictForestStat(self.deltaTrees,xIn.to_numpy(dtype=np.float32)[:nSampleTest], statDict,n_jobs)
            meanStdDelta = np.mean(statDict["std"])
            irreducibleStdEst = np.sqrt(meanStdDelta ** 2 - (rmsPred ** 2) / 2.)
            self.predictReducibleError(xIn[:nSampleTest], reducibleErrorEst, 64, fractions)
            meanlocalErrEst = []
            for fraction in fractions:
                meanlocalErrEst.append(np.mean(reducibleErrorEst[fraction]))
            # print(tIter,nTrees, rmsPoint, meanStdDelta, irreducibleStdEst,  meanStd, meanDeltaRelStd,  rmsPred, rmsPredMed)
            print(f"{tIter}\t{nTrees}\t{rmsPoint:.4f}\t{meanStdDelta:.4}\t{irreducibleStdEst:.4f}\t{meanStd:.4f}\t"
                  f"{meanDeltaRelStd:.4f}\t{rmsPred:.4f}\t{rmsPredMed:.4f}",meanlocalErrEst)
        return 0

    def predictStat(self, data, statDictionary,treeType=-1):
        """
        predict value
        :param data:           inout data
        :param statDictonary:  statistics to output
        :param treeType:
        :return:
        """
        # assert(treeType!=0 & treeType!=1)
        if treeType>=0 & treeType<2:
            return predictForestStat(self.trees[treeType], data.to_numpy(dtype=np.float32), statDictionary, self.optionsErrPDF["n_jobs"])
        else:
            return predictForestStat(self.trees[0], data.to_numpy(dtype=np.float32), statDictionary, self.optionsErrPDF["n_jobs"])


    def predictReducibleError(self, data, statDictionary, nPermutation, fractionArray):
        """
        predict local reducible array
        :param data:           input data
        :param statDictionary:  statistics to output
        :param nPermutation:
        :param fractionArray:
        :return:
        """
        # assert(treeType!=0 & treeType!=1)
        forestPredict = {}
        n_jobs=self.optionsErrPDF["n_jobs"]
        for treeType in [0, 1]:
            counter = 0
            for counter, tree in enumerate(self.trees[treeType]): pass
            selTree = np.zeros((counter + 1, data.shape[0]))
            forestPredict[treeType] = selTree
            #for counter, tree in enumerate(self.trees[treeType]):
            #    selTree[counter] = tree.predict(data)
            Parallel(n_jobs=n_jobs,require="sharedmem",)(
                delayed(_accumulate_predictionNL)(e.predict, data.to_numpy(dtype=np.float32), selTree, col)
                for col,e in enumerate(self.trees[treeType][0:self.nTrees//2])
            )
        #

        nTreesAll = self.nTrees
        nIter=2*len(self.trees[0])//(self.nTrees)
        for fraction in fractionArray:
            deltas = np.zeros((nPermutation, data.shape[0]))
            for iPer in range(0, nPermutation):
                permutation = np.random.permutation(self.nTrees//2)
                #permutation2=permutation+((4)*self.nTrees//2)
                nTreesM = int(fraction * self.nTrees//2)
                y0 = np.mean(forestPredict[0][permutation][0:nTreesM], 0)
                y1 = np.mean(forestPredict[1][permutation][0:nTreesM], 0)
                deltas[iPer] = y1 - y0
            statDictionary[fraction] = np.std(deltas, 0) * np.sqrt(fraction/(1. - fraction))*0.5;
            statDictionary[1+fraction] = np.std(deltas, 0)
        return



    def predict(self, data, **options):
        """
        TODO
        Returns the output of the RF to the provided data.
        :param data: array of the features to get the prediction for.
        :return: array of the predicted values.
        """

    def getImportance(self):
        """
        print sorted importance - TODO
        :param varNames:
        :kwargs:
        :return:
        """
        impTree = np.zeros((len(self.trees[0]), len(self.trees[0][0])))
        for row,tree in enumerate(self.trees[0]):
            impTree[row]=tree.feature_importances_
        return    impTree.mean(axis=0)
