import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

class MIForestErrPDF:
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """

    def __init__(self, switch, optionsErrPDF0,  **kwargs):
        """
        :param switch: It can be chosen between 'Classifier' or 'Regressor' with the corresponding string.
        :param options ErrPDF options - see default PDF options
        :param kwargs: All options from sklearn can be used. For instance
        """
        defaultErrPDFOptions = {
            "mean_depth":10,                    # mean depth of the tree
            "min_tree":32,                      # minimal number of trees for fit
            "niter_max":3,                      # number of itterations
            "n_jobs":4,                         # nJobs
            "verbose":1
        }
        self.optionsErrPDF=defaultErrPDFOptions
        self.optionsErrPDF.update(optionsErrPDF0)
        self.method="MIForestErrPDF"
        self.trees={}
        self.trees[0]=[]                       # trees for data
        self.trees[1]=[]                       # trees for refernce data
        self.type=switch
        self.options=kwargs
        self.model_name=''

    def fit(self, xIn, yIn,  sample_weight=None):
        self.fitWithParam(xIn,yIn,self.optionsErrPDF["mean_depth"], self.optionsErrPDF["niter_max"])

    def fitWithParam(self, x0, y0, meanDepth, nIterMax, sample_weight=None):
        """
        make fits in iteration
        :param X_train:         input data set
        :param y_train:         input data set
        :param meanDepth        mean depth of te tree
        :return:
        """
        assert len(x0) == len(y0)
        nRows=x0.shape[0]
        nTrees=nRows/(2**meanDepth)
        nTrees=np.exp2(np.log2(nTrees)).astype(int)                  # number of independent trees forced to be 2**N
        if nTrees<self.optionsErrPDF["min_tree"]:
            nTrees=self.optionsErrPDF["min_tree"]
        bucketSize=((nRows/nTrees)//2)
        #
        xIn, _, yIn, _ = train_test_split(x0, y0, test_size=0.0)
        #  nRows=100; xIn=np.linspace(0.0, 1.0, num=nRows); yIn=np.linspace(0.0, 1.0, num=nRows);
        x={}
        y={}
        ypred={}
        statDict={"mean":0,"median":0,"std":0}
        for tIter in range(0,nIterMax):
            #create permutations in 2 groups data and reference data - to be fitted with tree and refernce tree
            permutation= np.random.permutation(nRows//2)
            #x[0]=xIn[0:nRows//2][permutation]
            x[0]=xIn[0:nRows//2].iloc[permutation].reset_index(drop=True)
            y[0]=yIn[0:nRows//2][permutation]
            permutation= np.random.permutation(nRows-nRows//2)
            #x[1]=xIn[nRows//2:nRows][permutation]
            x[1]=xIn[nRows//2:nRows].iloc[permutation].reset_index(drop=True)
            y[1]=yIn[nRows//2:nRows][permutation]
            #
            # fit trees
            for iTree in range(0,nTrees):
                iter=(2*iTree//nTrees)
                i0=int(iTree*bucketSize)
                i1=int((iTree+1)*bucketSize)
                if self.type == 'Classifier':
                    clf = DecisionTreeClassifier()
                elif self.type == 'Regressor':
                    clf = DecisionTreeRegressor()
                clf.fit(x[iter][i0:i1],y[iter][i0:i1])
                self.trees[iter].append(clf)
            # estimate reducible error
            treeSlice=slice(0,(nTrees//2)*(tIter+1)-1)
            yPred0=self.predictStat(xIn,0,statDict,treeSlice)
            yPredMed0=statDict["median"];
            yPredStd0=statDict["std"];
            yPred1=self.predictStat(xIn,1,statDict,treeSlice)
            yPredMed1=statDict["median"];
            yPredStd1=statDict["std"];
            rmsPred=np.std(yPred0-yPred1)
            rmsPredMed=np.std(yPredMed0-yPredMed1)
            meanStd=np.mean(0.5*(yPredStd0+yPredStd1))
            meanDeltaRelStd=np.std((yPredStd0-yPredStd1))/np.mean(0.5*(yPredStd0+yPredStd1))
            rmsPoint=np.std(0.5*(yPred0+yPred1)-yIn)
            print(tIter,nTrees, rmsPoint, meanStd, meanDeltaRelStd,  rmsPred,rmsPredMed )

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
        #assert(treeType!=0 & treeType!=1)
        counter=0
        for counter, tree in enumerate(self.trees[treeType][treeSlice]): pass
        selTree = np.zeros((counter+1, data.shape[0]))

        for counter, tree in enumerate(self.trees[treeType][treeSlice]) :
            selTree[counter] = tree.predict(data)
        #
        #stat={"Mean":np.mean(selTree[0:counter], 0), "Median":np.median(selTree[0:counter], 0), "RMS": np.std(selTree[counter], 0)}
        for i,stat in enumerate(statDictionary):
            #print(i,stat)
            if stat=="mean": statDictionary["mean"]= np.mean(selTree[0:counter], 0)
            if stat=="median": statDictionary["median"]= np.median(selTree[0:counter], 0)
            if stat=="std": statDictionary["std"]= np.std(selTree[0:counter], 0)
        return np.mean(selTree[0:counter], 0);



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

