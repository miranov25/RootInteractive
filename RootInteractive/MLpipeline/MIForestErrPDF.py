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
try:
    from sklearn.utils.fixes import _joblib_parallel_args
except:
    pass
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

def predictRFStatChunk(rf, X, statDictionary, parallel, n_jobs):
    """
    inspired by https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/ensemble/_forest.py#L1410
    predict statistics from random forest
    :param rf:                  random forest object
    :param X:                   input vector
    :param statDictionary:      dictionary of statistics to predict
    :param n_jobs:              number of parallel jobs for prediction
    :return:                    dictionary with requested output statistics
    """
    nEstimators = len(rf.estimators_)
    allRF = np.empty((nEstimators, X.shape[0]))
    statOut={}
    parallel(
            delayed(simple_predict)(e.predict, X, allRF, col)
            for col,e in enumerate(rf.estimators_)
    )
    #
    allRFTranspose = allRF.T.copy(order='C')
    blockSize = X.shape[0] // n_jobs + 1
    block_begin = list(range(0, X.shape[0], blockSize))
    block_end = block_begin[1:]
    block_end.append(X.shape[0])
    if "median" in statDictionary:
        parallel(
                delayed(allRF[first:last].partition)(nEstimators // 2)
                for first, last in zip(block_begin, block_end)
                )
        statOut["median"]= allRFTranspose[:,nEstimators//2]
    if "mean"  in statDictionary:
        mean_out = np.empty(X.shape[0])
        parallel(
                delayed(np.mean)(allRFTranspose[first:last], -1, out=mean_out[first:last])
                for first, last in zip(block_begin, block_end)
                )
        statOut["mean"]=mean_out
    if "std"  in statDictionary: 
        std_out = np.empty(X.shape[0])
        parallel(
                delayed(np.std)(allRFTranspose[first:last], -1, out=std_out[first:last])
                for first, last in zip(block_begin, block_end)
                )
        statOut["std"]=std_out
    if "quantile" in statDictionary:
        statOut["quantile"]={}
        quantiles = np.array(statDictionary["quantile"]) * nEstimators
        parallel(
            delayed(allRF[first:last].partition)(quantiles)
            for first, last in zip(block_begin, block_end)
        )
        for iQuant, quant in enumerate(statDictionary["quantile"]):
            statOut["quantile"][quant]=allRFTranspose[:,quantiles[iQuant]]
    if "trim_mean" in   statDictionary:
        statOut["trim_mean"]={}
        for quant in statDictionary["trim_mean"]:
            statOut["trim_mean"][quant]=stats.trim_mean(allRFTranspose,quant,axis=1)
    return statOut

def predictRFStat(rf, X, statDictionary,n_jobs, max_rows=1000000):
    """
    inspired by https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/ensemble/_forest.py#L1410
    predict statistics from random forest
    :param rf:                  random forest object
    :param X:                   input vector
    :param statDictionary:      dictionary of statistics to predict
    :param n_jobs:              number of parallel jobs for prediction
    :param max_rows:
    :return:                    dictionary with requested output statistics
    """
    if(max_rows < 0):
       with Parallel(n_jobs=n_jobs, verbose=rf.verbose, require="sharedmem") as parallel:
           return predictRFStatChunk(rf, X, statDictionary, parallel, n_jobs)
    block_begin = list(range(0, X.shape[0], max_rows))
    block_end = block_begin[1:]
    block_end.append(X.shape[0])    
    answers = []
    with Parallel(n_jobs=n_jobs, verbose=rf.verbose, require="sharedmem") as parallel:
        for (a,b) in zip(block_begin, block_end):
             answers.append(predictRFStatChunk(rf, X[a:b], statDictionary, parallel, n_jobs))
    if not answers:
        return {}
    merged = {}
    for key in answers[0].keys():
        merged[key] = np.concatenate([i[key] for i in answers])
    return merged

def predictRFStatNew(rf, X, statDictionary,n_jobs):
    """
    inspired by https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/ensemble/_forest.py#L1410
    predict statistics from random forest
    :param rf:                  random forest object
    :param X:                   input vector
    :param statDictionary:      dictionary of statistics to predict
    :param n_jobs:              number of parallel jobs for prediction
    :return:                    dictionary with requested output statistics
    """
    allRF = np.zeros((len(rf.estimators_), X.shape[0]))
    lock = threading.Lock()
    statOut={}
    Parallel(n_jobs=n_jobs, verbose=rf.verbose,require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, allRF, col,lock)
            for col,e in enumerate(rf.estimators_)
    )
    #
    if "median" in statDictionary: statOut["median"]=np.median(allRF, 0)
    if "mean"  in statDictionary: statOut["mean"]=np.mean(allRF, 0)
    if "std"  in statDictionary: statOut["std"]=np.std(allRF, 0)
    if "quantile" in   statDictionary:
        statOut["quantiles"]={}
        for quant in statDictionary["quantile"]:
            statOut["quantiles"][quant]=np.quantile(allRF,quant,axis=0)
    if "trim_mean" in   statDictionary:
        statOut["trim_mean"]={}
        for quant in statDictionary["trim_mean"]:
            statOut["trim_mean"][quant]=stats.trim_mean(allRF,quant,axis=0)
    return statOut


def predictForestStat(treeArray, X, statDictionary,n_jobs,verbose=False):
    """
    inspired by https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/ensemble/_forest.py#L1410
    predict statistics from random forest
    :param treeArray:           list of regressors
    :param X:                   input vector
    :param statDictionary:      dictionary of statistics to predict
    :param n_jobs:              number of parallel jobs for prediction
    :return:                    dictionary with requested output statistics
    """
    statOut={}
    allRF = np.zeros((len(treeArray), X.shape[0]))
    #lock = threading.Lock()
    Parallel(n_jobs=n_jobs,**_joblib_parallel_args(require="sharedmem"),)(
            delayed(_accumulate_predictionNL)(e.predict, X, allRF, col)
            for col,e in enumerate(treeArray)
    )

    if "median" in statDictionary: statOut["median"]=np.median(allRF, 0)
    if "mean"  in statDictionary: statOut["mean"]=np.mean(allRF, 0)
    if "std"  in statDictionary: statOut["std"]=np.std(allRF, 0)
    if "quantile" in   statDictionary:
        statOut["quantiles"]={}
        for quant in statDictionary["quantile"]:
            statOut["quantiles"][quant]=np.quantile(allRF,quant,axis=0)
    if "trim_mean" in   statDictionary:
        statOut["trim_mean"]={}
        for quant in statDictionary["trim_mean"]:
            statOut["trim_mean"][quant]=stats.trim_mean(allRF,quant,axis=0)
    return statOut


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
            Parallel(n_jobs=n_jobs,**_joblib_parallel_args(require="sharedmem"),)(
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
            Parallel(n_jobs=n_jobs,**_joblib_parallel_args(require="sharedmem"),)(
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
            Parallel(n_jobs=n_jobs,**_joblib_parallel_args(require="sharedmem"),)(
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
