import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import threading
from sklearn.utils.fixes import _joblib_parallel_args

def _accumulate_prediction(predict, X, out,col, lock):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.

    """
    prediction = predict(X, check_input=False)
    with lock:
        out[col] += prediction

def predictRFStat(rf, X, statDictionary,n_jobs):
    """
    predict statistics from random forest
    :param rf:                  - random forest object
    :param X:                   -  input vector
    :param statDictionary:      - dictionary of statistics to predict
    :param n_jobs:              - number of parallel jobs for prediction
    :return:
    """
    allRF = np.zeros((len(rf.estimators_), X.shape[0]))
    lock = threading.Lock()
    Parallel(
        n_jobs=n_jobs,
        verbose=rf.verbose,
        **_joblib_parallel_args(require="sharedmem"),
    )(
            delayed(_accumulate_prediction)(e.predict, X, allRF, col,lock)
            for col,e in enumerate(rf.estimators_)
    )
    #
    if "median" in statDictionary: statDictionary["median"]=np.median(allRF, 0)
    if "mean"  in statDictionary: statDictionary["mean"]=np.mean(allRF, 0)
    if "std"  in statDictionary: statDictionary["std"]=np.std(allRF, 0)
    if "quantile" in   statDictionary:
        statDictionary["quantiles"]={}
        for quant in statDictionary["quantile"]:
            statDictionary["quantiles"][quant]=np.quantile(allRF,quant,axis=0)
    return statDictionary


def predictRFStat0(rf, data, statDictionary):
    """
    predict local reducible array
    :param rf               input random forest
    :param statDictionary:  statistics to output
    :param nPermutation:
    :return:
    """
    # assert(treeType!=0 & treeType!=1)
    allRF = np.zeros((len(rf.estimators_), data.shape[0]))
    for i, tree in enumerate(rf.estimators_):
        allRF[i] = tree.predict(data)
    #
    if "median" in statDictionary: statDictionary["median"]=np.median(allRF, 0)
    if "mean"  in statDictionary: statDictionary["mean"]=np.mean(allRF, 0)
    if "std"  in statDictionary: statDictionary["std"]=np.std(allRF, 0)
    return statDictionary

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
        fractions = [0.4, 0.6, 0.8]  # fracions to estimate reducible error
        assert len(x0) == len(y0)
        nRows = x0.shape[0]
        nSampleTest = int(nRows * self.optionsErrPDF["test_fraction"])
        nTrees = nRows / (2 ** meanDepth)
        nTrees = np.exp2(np.log2(nTrees)).astype(int)  # number of independent trees forced to be 2**N
        if nTrees < self.optionsErrPDF["min_tree"]:
            nTrees = self.optionsErrPDF["min_tree"]
        bucketSize = ((nRows / nTrees) // 2)
        #
        xIn, _, yIn, _ = train_test_split(x0, y0, test_size=0.0)
        #  nRows=100; xIn=np.linspace(0.0, 1.0, num=nRows); yIn=np.linspace(0.0, 1.0, num=nRows);
        x = {}
        y = {}
        ypred = {}
        statDict = {"mean": 0, "median": 0, "std": 0}
        reducibleErrorEst = {}
        for tIter in range(0, nIterMax):
            # create permutations in 2 groups data and reference data - to be fitted with tree and refernce tree
            permutation = np.random.permutation(nRows // 2)
            # x[0]=xIn[0:nRows//2][permutation]
            x[0] = xIn[0:nRows // 2].iloc[permutation].reset_index(drop=True)
            y[0] = yIn[0:nRows // 2][permutation]
            permutation = np.random.permutation(nRows - nRows // 2)
            # x[1]=xIn[nRows//2:nRows][permutation]
            x[1] = xIn[nRows // 2:nRows].iloc[permutation].reset_index(drop=True)
            y[1] = yIn[nRows // 2:nRows][permutation]
            #
            # fit trees
            for iTree in range(0, nTrees):
                iter = (2 * iTree // nTrees)
                i0 = int(iTree * bucketSize)
                i1 = int((iTree + 1) * bucketSize)
                if self.type == 'Classifier':
                    clf = DecisionTreeClassifier()
                elif self.type == 'Regressor':
                    clf = DecisionTreeRegressor()
                clf.fit(x[iter][i0:i1], y[iter][i0:i1])
                self.trees[iter].append(clf)
            treeSlice = slice(0, (nTrees // 2) * (tIter + 1) - 1)
            if (not self.optionsErrPDF["dump_progress"]) & (tIter < nIterMax - 1):
                yPred0 = self.predictStat(xIn[:nSampleTest], 0, statDict, treeSlice)
                yPred1 = self.predictStat(xIn[:nSampleTest], 1, statDict, treeSlice)
                self.predictReducibleError(xIn[:nSampleTest], reducibleErrorEst, 64, fractions)
                for fraction in fractions:
                    reducibleErrorEst[fraction] = np.mean(reducibleErrorEst[fraction])
                rmsPred = np.std(yPred0 - yPred1)
                print(f"{tIter}\t{nTrees}\t{rmsPred:.4f}",reducibleErrorEst)
                continue
            # estimate reducible error

            yPred0 = self.predictStat(xIn, 0, statDict, treeSlice)
            yPredMed0 = statDict["median"]
            yPredStd0 = statDict["std"]
            yPred1 = self.predictStat(xIn, 1, statDict, treeSlice)
            yPredMed1 = statDict["median"]
            yPredStd1 = statDict["std"]
            yPred = 0.5 * (yPred0 + yPred1)
            # make delta regression trees
            self.deltaTrees.clear()
            for iTree in range(0, 2 * nTrees):
                i0 = int(iTree * bucketSize)
                i1 = int((iTree + 1) * bucketSize)
                if self.type == 'Classifier':
                    clf = DecisionTreeClassifier()
                elif self.type == 'Regressor':
                    clf = DecisionTreeRegressor()
                clf.fit(xIn[i0:i1], (yIn - yPred)[i0:i1])
                self.deltaTrees.append(clf)
            rmsPred = np.std(yPred0 - yPred1)
            rmsPredMed = np.std(yPredMed0 - yPredMed1)
            meanStd = np.mean(0.5 * (yPredStd0 + yPredStd1))
            meanDeltaRelStd = np.std((yPredStd0 - yPredStd1)) / np.mean(0.5 * (yPredStd0 + yPredStd1))
            rmsPoint = np.std(0.5 * (yPred0 + yPred1) - yIn)
            #
            treeSlice = slice(0, len(self.deltaTrees))
            deltaPred = self.predictStat(xIn[:nSampleTest], -1, statDict, treeSlice)
            meanStdDelta = np.mean(statDict["std"])
            irreducibleStdEst = np.sqrt(meanStdDelta ** 2 - (rmsPred ** 2) / 2.)
            self.predictReducibleError(xIn[:nSampleTest], reducibleErrorEst, 64, fractions)
            meanlocalErrEst = []
            for fraction in fractions:
                meanlocalErrEst.append(np.mean(reducibleErrorEst[fraction]))
            # print(tIter,nTrees, rmsPoint, meanStdDelta, irreducibleStdEst,  meanStd, meanDeltaRelStd,  rmsPred, rmsPredMed)
            print(f"{tIter}\t{nTrees}\t{rmsPoint:.4f}\t{meanStdDelta:.4}\t{irreducibleStdEst:.4f}\t{meanStd:.4f}\t"
                  f"{meanDeltaRelStd:.4f}\t{rmsPred:.4f}\t{rmsPredMed:.4f}\t{meanlocalErrEst[2]:.4f}")
        return 0

    def predictStat(self, data, treeType, statDictionary, treeSlice):
        """
        predict value
        :param data:           inout data
        :param treeType:
        :param statDictonary:  statistics to output
        :param treeSlice:      slice to use for prediction  e.g. s=slice(0,a)
        :return:
        """
        # assert(treeType!=0 & treeType!=1)
        counter = 0
        if (treeType == 0) | (treeType == 1):
            for counter, tree in enumerate(self.trees[treeType][treeSlice]): pass
            selTree = np.zeros((counter + 1, data.shape[0]))
            for counter, tree in enumerate(self.trees[treeType][treeSlice]):
                selTree[counter] = tree.predict(data)
        else:
            for counter, tree in enumerate(self.deltaTrees[treeSlice]): pass
            selTree = np.zeros((counter + 1, data.shape[0]))
            for counter, tree in enumerate(self.deltaTrees[treeSlice]):
                selTree[counter] = tree.predict(data)

        #
        # stat={"Mean":np.mean(selTree[0:counter], 0), "Median":np.median(selTree[0:counter], 0), "RMS": np.std(selTree[counter], 0)}
        for i, stat in enumerate(statDictionary):
            # print(i,stat)
            if stat == "mean": statDictionary["mean"] = np.mean(selTree[0:counter], 0)
            if stat == "median": statDictionary["median"] = np.median(selTree[0:counter], 0)
            if stat == "std": statDictionary["std"] = np.std(selTree[0:counter], 0)
        return np.mean(selTree[0:counter], 0);

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
        for treeType in [0, 1]:
            counter = 0
            for counter, tree in enumerate(self.trees[treeType]): pass
            selTree = np.zeros((counter + 1, data.shape[0]))
            forestPredict[treeType] = selTree
            for counter, tree in enumerate(self.trees[treeType]):
                selTree[counter] = tree.predict(data)
        #
        nTreesAll = counter

        for fraction in fractionArray:
            deltas = np.zeros((nPermutation, data.shape[0]))
            for iPer in range(0, nPermutation):
                permutation = np.random.permutation(counter)
                nTrees = int(fraction * nTreesAll)
                y0 = np.mean(forestPredict[0][permutation][0:nTrees], 0)
                y1 = np.mean(forestPredict[1][permutation][0:nTrees], 0)
                deltas[iPer] = y1 - y0
            statDictionary[fraction] = np.std(deltas, 0) * np.sqrt(fraction) / (2 * (1. - fraction))
        return

    def predict(self, data, **options):
        """
        TODO
        Returns the output of the RF to the provided data.
        :param data: array of the features to get the prediction for.
        :return: array of the predicted values.
        """

    def printImportance(self, varNames, **kwargs):
        """
        print sorted importance - TODO
        :param varNames:
        :kwargs:
        :return:
        """
