import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

class RandomForestErrPDF:
    """
    Random Forest (RF) from sklearn. This summarizes the RF for classification and regression.
    It allows a fast application and adjusts the notation fur further application.
    """

    def __init__(self, switch, optionsErrPDF0,  **kwargs):
        """
        The initilialization includes registration and fitting of the random forest
        In the training data are divided to nSplit pairs

        :param switch: It can be chosen between 'Classifier' or 'Regressor' with the corresponding string.
        :param options ErrPDF options - see default PDF options
        :param kwargs: All options from sklearn can be used. For instance
        """
        defaultErrPDFOptions = {
            "nSplit": 8,                        # number of splits of data samples  nSplit to 2 independents sets - creates nSplit*2 instances of the Random forests
            "max_depthBegin":-1,                # starting depth used for regression - default 128 points per bucket (log2(nPoints/256))
            "max_depthEnd":-1,                  # end  depth used for regression - default 128 points per bucket (log2(nPoints/4))
            "n_estimatorsBegin":50,             # starting number of trees
            "n_estimatorsEnd":200,              # starting number of trees
            "depthFractionEnd": 0.002,            # fraction to stop depth scan
            "n_estimatorsFractionEnd": 0.002,     # fraction to stop n_estimators scan
            "max_samples":0.2,                  # max_samples
            "n_jobs":4,                         # nJobs
            "verbose":1
        }
        self.optionsErrPDF=defaultErrPDFOptions
        self.optionsErrPDF.update(optionsErrPDF0)
        self.models=[]
        self.type=switch
        self.options=kwargs
        self.model_name=''
        self.max_depth=self.optionsErrPDF["max_depthBegin"]

    def fit(self, x, y, sample_weight=None):
        """
        1.) Split data to nSplit partitions
        2.) Fit data in each partition  auto-adjusting the deepnees
        3.) Estimation of reducible error
        :param X_train:         input data set
        :param y_train:
        :param sample_weight:
        :return:
        """
        time0=time.perf_counter()
        # 0.) split data sets to nSplit
        dataSet=[]
        for i in range(0,self.optionsErrPDF['nSplit']):
            x0, x1, y0, y1 = train_test_split(x, y)
            dataSet.append([x0,y0])
            dataSet.append([x1,y1])
        # 1.) define initial value of the  max_depth
        maxDepth= int(np.log2(x.shape[0]+1))
        if self.optionsErrPDF['max_depthBegin'] <1:
            self.optionsErrPDF['max_depthBegin'] =  int(np.log2(x.shape[0]/256)+1)
        if self.optionsErrPDF['max_depthEnd'] <1:
            self.optionsErrPDF['max_depthEnd'] =  int(np.log2(x.shape[0]/4)+1)
        self.max_depth=self.optionsErrPDF["max_depthBegin"]
        stdPredBestDepth=0
        stdPointBestDepth=0
        # 2.) make fits  in loop increasing the deepness and n_estimators until precission still improved
        for depth in range(self.optionsErrPDF['max_depthBegin'], self.optionsErrPDF['max_depthEnd']):
            currentModels=[]
            for i in range(0, 2 * self.optionsErrPDF['nSplit']):
                if self.type == 'Classifier':
                    clf = RandomForestClassifier(**(self.options), warm_start=True, max_depth=depth,n_jobs=self.optionsErrPDF['n_jobs'],
                                                 n_estimators=self.optionsErrPDF['n_estimatorsBegin'],max_samples=self.optionsErrPDF['max_samples'])
                elif self.type == 'Regressor':
                    clf = RandomForestRegressor(**(self.options), warm_start=True, max_depth=depth,n_jobs=self.optionsErrPDF['n_jobs'],
                                                n_estimators=self.optionsErrPDF['n_estimatorsBegin'],max_samples=self.optionsErrPDF['max_samples'])
                currentModels.append(clf)
            rmsSum=0
            stdPredBestN=0
            stdPointBestN=0
            while currentModels[0].n_estimators < self.optionsErrPDF['n_estimatorsEnd']:
                for i in range(0,2*self.optionsErrPDF['nSplit']):
                    currentModels[i].fit(dataSet[i][0],dataSet[i][1])

                # caclulate prcession rms
                stdPred=0
                stdPoint=0
                for i in range(0,self.optionsErrPDF['nSplit']):
                    dy0=currentModels[2*i].predict(dataSet[2*i][0])-currentModels[2*i+1].predict(dataSet[2*i][0])
                    dy1=currentModels[2*i].predict(dataSet[2*i+1][0])-currentModels[2*i+1].predict(dataSet[2*i+1][0])
                    stdPoint+=(dataSet[2*i][1]- currentModels[2*i+1].predict(dataSet[2*i][0])).std()
                    stdPoint+=(dataSet[2*i+1][1]- currentModels[2*i].predict(dataSet[2*i+1][0])).std()
                    stdPred+=dy0.std()
                    stdPred+=dy1.std()

                stdPred/=(2*self.optionsErrPDF['nSplit'])
                stdPoint/=(2*self.optionsErrPDF['nSplit'])
                stdIrr=np.sqrt(max(stdPoint*stdPoint-stdPred*stdPred/2.,0))
                nBucketMean=(x.shape[0]/(2**depth))
                stdRedStat=stdIrr/np.sqrt(nBucketMean)
                print(depth, currentModels[i].n_estimators, stdPred, stdRedStat, stdPoint, stdIrr,   nBucketMean,time.perf_counter()-time0)
                if (stdPointBestN>0) and (stdPred/stdPredBestN>(1.-self.optionsErrPDF["n_estimatorsFractionEnd"])) and (stdPred/stdPredOld>(1.-self.optionsErrPDF["n_estimatorsFractionEnd"])):
                    print("I will stop now", stdPred/stdPredBestN, stdPoint/stdPointBestN, self.optionsErrPDF["n_estimatorsFractionEnd"],time.perf_counter()-time0)
                    break
                stdPredOld=stdPred
                for i in range(0,2*self.optionsErrPDF['nSplit']):
                    deltaJobs= 1+((self.optionsErrPDF['n_estimatorsEnd'] - self.optionsErrPDF['n_estimatorsBegin'])//10)//self.optionsErrPDF['n_jobs']
                    deltaJobs*=self.optionsErrPDF['n_jobs']
                    currentModels[i].n_estimators += deltaJobs

            if (stdPointBestDepth>0) and (stdPointBestN/stdPointBestDepth>(1.-self.optionsErrPDF["depthFractionEnd"])):
                print("Deepnes scan stops now", stdPointBestN/stdPointBestDepth, self.optionsErrPDF["depthFractionEnd"],time.perf_counter()-time0)
                break
            else:
                self.models.clear()
                for i in range(0, 2 * self.optionsErrPDF['nSplit']):
                    self.models.append(currentModels[i])
            stdPointBestDepth=stdPointBestN







    def predict(self, data, **options):
        """
        TODO
        Returns the output of the RF to the provided data.
        :param data: array of the features to get the prediction for.
        :return: array of the predicted values.
        """

    def predictStat(self, data, **kwargs):
        """
        TODO
        :param data     - input matrix
        :param kwargs   - optional arguments
        :return: predict statistic mean, median, rms over trees
        """
        defaultOptions = {
            "group": -1,
            "n_groups":-1
        }
        options=defaultOptions
        options.update(kwargs)


    def printImportance(self, varNames, **kwargs):
        """
        print sorted importance - TODO
        :param varNames:
        :kwargs:
        :return:
        """

