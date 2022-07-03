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
#from sklearn.utils.fixes import _joblib_parallel_args
from scipy import stats
import xgboost as xgb


def predictXGBStat(df, regXGB, iteration_range0, variablesXGB):
    statOut = {}
    nIteration = iteration_range0[1] - iteration_range0[0]
    xgbPred = np.zeros((df.shape[0], nIteration))
    print(xgbPred.shape)
    for i in range(0, 99):
        delta = regXGB.predict(df[variablesXGB], iteration_range=(i, i + 1))
        # print(delta.shape,xgbPred[:,i].shape)
        xgbPred[:, i] = delta
    statOut["mean"]   = np.mean(xgbPred, 1)
    statOut["median"] = np.median(xgbPred, 1)
    statOut["std"]    = np.std(xgbPred, 1)
    return statOut

def predictXGBStat(Xin, xgbErrPDF, iteration_range0, input=0):
    statOut = {}
    nIteration = iteration_range0[1] - iteration_range0[0]
    xgbPred = np.zeros((Xin.shape[0], nIteration))
    print(xgbPred.shape)
    for i in range(iteration_range0[input], iteration_range0[1]-3):
        delta = xgbErrPDF.regXGBFac[input].predict(Xin, iteration_range=(i, i + 1))
        # print(delta.shape,xgbPred[:,i].shape)
        xgbPred[:, i-iteration_range0[0]] = delta
    statOut["mean"]   = np.mean(xgbPred, 1)
    statOut["median"] = np.median(xgbPred, 1)
    statOut["std"]    = np.std(xgbPred, 1)
    return statOut


class MIxgboostErrPDF:
    """
    Forest with reducible and irreducible error estimation and PDF estimation
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """

    def __init__(self, optionsErrPDF0):
        """
        :param optionsErrPDF options - see default PDF options
        """
        # {'learning_rate':0.01, 'max_depth':1000 ,"n_estimators":10,"subsample":0.05}
        defaultErrPDFOptions = {
            "learning_rate": 0.01,  # learning rate
            "max_depth": 1000,  # minimal number of trees for fit
            "n_estimators": 100,  # number of estimators
            "subsample": 0.5,  # subsample used
            "dump_progress": 5,  # switch to follow progress
            "step_size":5,       # step size to evaluate
            "min_learning_rate":0.01,
            "max_learning_rate":0.3,
            "coeff_learning_rate":0.5,
            "verbose": 1
        }
        self.optionsErrPDF = defaultErrPDFOptions
        self.optionsErrPDF.update(optionsErrPDF0)
        self.paramXGBFactor=optionsErrPDF0
        self.paramXGBFactor.pop("dump_progress",None)
        self.paramXGBFactor.pop("step_size",None)
        self.paramXGBFactor.pop("min_learning_rate",None)
        self.paramXGBFactor.pop("max_learning_rate",None)
        self.paramXGBFactor.pop("coeff_learning_rate",None)
        self.method = "MIxgboostErrPDF"
        self.regXGBFac={}
        self.regXGBFacRed={}
        for iSample in [0,1 ,2]:
            self.regXGBFac[iSample] = xgb.XGBRegressor(**self.paramXGBFactor)
            self.regXGBFacRed[iSample] = xgb.XGBRegressor(**self.paramXGBFactor)

        # backup the original data for later prediction
        self.Xin={}
        self.Yin={}
        self.iterationStatArray=[]
        self.earlyStop=-1

    def learningRate_fraction(self,stdRatio):
        """
        return
        :param stdRatio:
        :param coeff:
        :param minLearningRate:
        :param maxLearningRate:
        :return:
        """
        min_learning_rate=self.optionsErrPDF["min_learning_rate"]
        max_learning_rate=self.optionsErrPDF["max_learning_rate"]
        coeff_learning_rate=self.optionsErrPDF["coeff_learning_rate"]
        return min(max((coeff_learning_rate*stdRatio),min_learning_rate),max_learning_rate)

    def fit(self,xIn, yIn, sample_weight):
        self.fit0(self, xIn, yIn, sample_weight)

    def fit0(self, xIn, yIn, sample_weight):
        nRows = xIn.shape[0]
        permutation = np.random.permutation(nRows)
        x0 = xIn[permutation[0:nRows // 2]]
        x1 = xIn[permutation[nRows // 2:nRows]]
        y0 = yIn[permutation[0:nRows // 2]]
        y1 = yIn[permutation[nRows // 2:nRows]]
        self.Xin[0]=x0
        self.Xin[1]=x1
        self.Yin[0]=y0
        self.Yin[1]=y1
        step=self.optionsErrPDF["dump_progress"]
        if step<=0: step=1
        #
        print(f"iter\tstdInput\tstd0\tstd1\tstd00\tstd11\ttstd0\ttstd1\tstd01\ttstd01")
        for iter in range(0, self.paramXGBFactor["n_estimators"],step):
            if iter > 0:
                self.regXGBFac[0].fit(x0, y0, xgb_model=self.regXGBFac[0])
                self.regXGBFac[1].fit(x1, y1, xgb_model=self.regXGBFac[1])
            else:
                self.regXGBFac[0].fit(x0, y0)
                self.regXGBFac[1].fit(x1, y1)
            stdInput = yIn.std()
            std00 = (y0 - self.regXGBFac[0].predict(x0)).std()
            std11 = (y1 - self.regXGBFac[1].predict(x1)).std()
            std0 = (y1 - self.regXGBFac[0].predict(x1)).std()
            std1 = (y0 - self.regXGBFac[1].predict(x0)).std()
            tstd0 = stats.mstats.trimmed_std(y1 - self.regXGBFac[0].predict(x1), limits=(0.05, 0.05))
            tstd1 = stats.mstats.trimmed_std(y0 - self.regXGBFac[1].predict(x0), limits=(0.05, 0.05))
            std01 = (self.regXGBFac[0].predict(x0) - self.regXGBFac[1].predict(x0)).std()
            tstd01 = stats.mstats.trimmed_std((self.regXGBFac[0].predict(x0) - self.regXGBFac[1].predict(x0)),
                                              limits=(0.05, 0.05))
            print(
                f"{iter}\t{stdInput:3.5}\t{std0:3.5}\t{std1:3.5}\t{std00:3.5}\t{std11:3.5}\t{tstd0:3.5}\t{tstd1:3.5}\t{std01:3.5}\t{tstd01:3.5}")


    def fit3Fold(self, xIn, yIn, sample_weight):
        step_size=self.optionsErrPDF["step_size"]
        if step_size<=0: step_size=1
        nRows = xIn.shape[0]
        permutation = np.random.permutation(nRows)
        for iSample in [0,1,2]:
            self.Xin[iSample]=xIn[permutation[iSample *nRows // 3: -1+(iSample+1) *nRows // 3]]
            self.Yin[iSample]=yIn[permutation[iSample *nRows // 3: -1+(iSample+1) *nRows // 3]]
            self.regXGBFac[iSample].n_estimators= step_size
            self.regXGBFac[iSample].learnig_rate=self.optionsErrPDF["max_learning_rate"]
        #
        n_steps=self.paramXGBFactor["n_estimators"]//step_size
        print(f"iter\titerAll\tlearning_rate\tstdPoint\tstdPointMean\tstdPredict\tcorel\treducibleErr")
        stdCurrent=yIn.std()
        learning_rate=self.optionsErrPDF["max_learning_rate"]
        for iter in range(0, n_steps):
            iterationStat={}
            iterationStat["iter"]=iter
            predict=[]
            for iSample in [0,1,2]:
                if iter > 0:
                    self.regXGBFac[iSample].fit(self.Xin[iSample], self.Yin[iSample], xgb_model=self.regXGBFac[iSample])
                else:
                    self.regXGBFac[iSample].fit(self.Xin[iSample], self.Yin[iSample])
            stdPoint={}
            stdPointMean={}
            stdPredict={}
            stdIter={}
            for iSample in [0,1,2]:
                iSampleRef0=(iSample+1)%3
                iSampleRef1=(iSample+2)%3
                stdPoint[iSample]=(self.regXGBFac[iSample].predict(self.Xin[iSampleRef0])-self.Yin[iSampleRef0]).std()
                stdPredict[iSample]=(self.regXGBFac[iSample].predict(self.Xin[iSampleRef1])-self.regXGBFac[iSampleRef0].predict(self.Xin[iSampleRef1])).std()
                stdPointMean[iSample]=((self.regXGBFac[iSample].predict(self.Xin[iSampleRef1])+self.regXGBFac[iSampleRef0].predict(self.Xin[iSampleRef1]))*0.5
                              -self.Yin[iSampleRef1]).std()
                ir = (0, 0)
                ir = self.regXGBFac[iSample]._get_iteration_range(ir)
                stdIter[iSample]= self.regXGBFac[iSample].predict(self.Xin[iSampleRef0],iteration_range=(ir[1]-1,ir[1])).std()
                #
                s0=stdPoint[iSample]**2
                s1=stdPointMean[iSample]**2
                s2=stdPredict[iSample]**2
                sStat2=s0-s1+s2/4.
                k=s2/(2*sStat2)
                #print(f"{iter}\t{stdPoint[iSample]:3.4}\t{stdPointMean[iSample]:3.4}\t{stdPredict[iSample]:3.4}",k,np.sqrt(sStat2))
            stdPoint[3]=(stdPoint[0]+stdPoint[1]+stdPoint[2])/3
            stdPointMean[3]=(stdPointMean[0]+stdPointMean[1]+stdPointMean[2])/3
            stdPredict[3]=(stdPredict[0]+stdPredict[1]+stdPredict[2])/3
            stdIter[3] = (stdIter[0] + stdIter[1] + stdIter[2]) / 3
            s0=stdPoint[3]**2
            s1=stdPointMean[3]**2
            s2=stdPredict[3]**2
            sStat2=s0-s1+s2/4.
            k=s2/(2*sStat2)
            ir=(0,0)
            ir=self.regXGBFac[0]._get_iteration_range(ir)
            print(f"{iter}\t{ir[1]}\t{learning_rate:3.5}\t{stdPoint[3]:3.5}\t{stdPointMean[3]:3.5}\t{stdPredict[3]:3.5}\t{k:3.5}\t{np.sqrt(sStat2):3.5}\t{stdIter[3]/stdPoint[3]/learning_rate}")
            iterationStat["stdPoint"]=stdPoint
            iterationStat["stdPointMean"]=stdPoint
            iterationStat["stdPredict"]=stdPredict
            iterationStat["stdIter"]=stdIter
            iterationStat["learning_rate"]=learning_rate
            learning_rate=(learning_rate+self.learningRate_fraction((stdIter[3]/stdPoint[3])/learning_rate))*0.5
            self.iterationStatArray.append(iterationStat)
            for iSample in [0,1,2]: self.regXGBFac[iSample].learning_rate=learning_rate
            if stdPointMean[3]>stdCurrent:
                self.earlyStop=ir[1]
                break
            else:
                stdCurrent =stdPointMean[3]
                self.earlyStop=ir[1]

    def fitReducible(self, learning_rate=0.01, subsample=0.05, n_estimators=100) :
        from sklearn.base import clone
        import copy
        for iSample in [0,1,2]:
            # make copy of model and update for error https://stackoverflow.com/questions/62309466/xgboost-how-to-copy-model
            #self.regXGBFacRed[iSample]=clone(self.regXGBFac[iSample])
            self.regXGBFacRed[iSample]=copy.deepcopy(self.regXGBFac[iSample])
            self.regXGBFacRed[iSample].learning_rate=learning_rate
            self.regXGBFacRed[iSample].subsample=subsample
            self.regXGBFacRed[iSample].n_estimators=n_estimators
            self.regXGBFacRed[iSample].fit(self.Xin[iSample], self.Yin[iSample], xgb_model=self.regXGBFacRed[iSample])



    def predictStatSlow(self, Xin, statDictionary, iSample, predType, iteration_range, YRef=0):
        """
        predict value
        :param data:           inout data
        :param statDictonary:  statistics to output
        :param treeType:
        :return:
        """
        nIteration = iteration_range[1] - iteration_range[0]
        xgbPred  = np.zeros((Xin.shape[0], nIteration))
        xgbPredN  = np.zeros((Xin.shape[0], nIteration))
        xgbPredD = np.zeros((Xin.shape[0], nIteration))
        for i in range(iteration_range[0], iteration_range[1]-1):
            regressor=self.regXGBFac[iSample]
            if predType==1:
                regressor=self.regXGBFacRed[iSample]
            predI=regressor.predict(Xin, iteration_range=(0, i))
            predI1=regressor.predict(Xin, iteration_range=(0, i+1))
            # print(delta.shape,xgbPred[:,i].shape)
            deltaPred=predI1-predI
            xgbPred[:, i-iteration_range[0]]  = deltaPred
            xgbPredN[:, i-iteration_range[0]] = deltaPred/deltaPred.std()
            if YRef : xgbPredD[:, i-iteration_range[0]] = YRef-predI
        statOut={}
        #if YRef
        statOut["mean"]     = np.mean(xgbPred, 1)
        statOut["median"]   = np.median(xgbPred, 1)
        statOut["std"]      = np.std(xgbPred, 1)
        statOut["stdN"]     = np.std(xgbPredN, 1)
        statOut["meanIter"]  = np.mean(xgbPred, 0)
        statOut["stdIter"]  = np.std(xgbPred, 0)
        if YRef:
            statOut["std"]     = np.std(xgbPredD, 1)
            statOut["stdN"]     = np.std(xgbPredN, 1)
            statOut["stdIter"] = np.std(xgbPredD, 0)
        return statOut

    def predictStatIter(self, Xin, statDictionary, iSample, predType, iteration_range, YRef=0):
        """
        predict value
        :param data:           inout data
        :param statDictonary:  statistics to output
        :param treeType:
        :return:
        """
        nIteration = iteration_range[1] - iteration_range[0]
        xgbPred  = np.zeros((Xin.shape[0], nIteration))
        xgbPredN  = np.zeros((Xin.shape[0], nIteration))
        xgbPredD = np.zeros((Xin.shape[0], nIteration))
        pred0=0
        for i in range(iteration_range[0], iteration_range[1]-1):
            regressor=self.regXGBFac[iSample]
            if predType==1:
                regressor=self.regXGBFacRed[iSample]
            if type(pred0)=='int':
                pred0=regressor.predict(Xin, iteration_range=(0, i))
            deltaPred=   regressor.predict(Xin, iteration_range=(i, i+1))-0.5
            #predI=regressor.predict(Xin, iteration_range=(0, i))
            #predI1=regressor.predict(Xin, iteration_range=(0, i+1))
            # print(delta.shape,xgbPred[:,i].shape)
            xgbPred[:, i-iteration_range[0]]  = deltaPred
            xgbPredN[:, i-iteration_range[0]] = deltaPred/deltaPred.std()
            if YRef : xgbPredD[:, i-iteration_range[0]] = YRef-pred0
            pred0+=deltaPred
        statOut={}
        #if YRef
        statOut["mean"]     = np.mean(xgbPred, 1)
        statOut["median"]   = np.median(xgbPred, 1)
        statOut["std"]      = np.std(xgbPred, 1)
        statOut["stdN"]     = np.std(xgbPredN, 1)
        statOut["meanIter"]  = np.mean(xgbPred, 0)
        statOut["stdIter"]  = np.std(xgbPred, 0)
        if YRef:
            statOut["stdRef"]     = np.std(xgbPredD, 1)
            statOut["stdNRef"]     = np.std(xgbPredN, 1)
            statOut["stdIterRef"] = np.std(xgbPredD, 0)

        return statOut

    def predictReducibleError(self,X):
        """
        predict reducible error using 3 fold
        xgbErrPDF.fitReducible() has to be called before
        :param X:
        :return: error estimator
        """
        outStat={}
        stdDiff0={}
        ir=(0,0)
        redStop=self.regXGBFacRed[0]._get_iteration_range(ir)[1]

        for iSample in [0,1,2]:
            print(iSample)
            y=self.predictStatIter(X,{},iSample,1,(self.earlyStop,redStop),0)
            outStat[f"stdNR{iSample}"]=y["stdN"]
            #yP=xgbErrPDF.regXGBFacRed[iSample].predict(df[variableX].to_numpy())

        for i in [0,1,2]:
            stdDiff0[i]=(self.regXGBFacRed[iSample].predict(X)-self.regXGBFacRed[(iSample+1)%3].predict(X)).std()/np.sqrt(2.)
            outStat[f"redErr{i}"]=outStat[f"stdNR{i}"]*stdDiff0[i]
        outStat[f"redErr"]=((outStat[f"redErr0"]+  outStat[f"redErr1"]+  outStat[f"redErr2"])/3.)/np.sqrt(3.)
        outStat[f"stdNR"]=((outStat[f"stdNR0"]+  outStat[f"stdNR1"]+  outStat[f"stdNR2"])/3.)/np.sqrt(3.)
        return outStat

    def predictStat(self,X):
        outStat={}
        for iSample in [0,1,2]:
            print(iSample)
            xP=self.regXGBFac[iSample].predict(X)
            yP=self.regXGBFacRed[iSample].predict(X)
            outStat[f"mean_{iSample}"]=xP
            outStat[f"meanRed_{iSample}"]=yP
            if iSample==0:
                outStat[f"mean"]=outStat[f"mean_{iSample}"]/3.
                outStat[f"meanRed"]=outStat[f"meanRed_{iSample}"]/3.
            else:
                outStat[f"mean"]+=outStat[f"mean_{iSample}"]/3.
                outStat[f"meanRed"]+=outStat[f"meanRed_{iSample}"]/3
        return outStat

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



class MIxgboostErrPDFv2:
    """
    Here I tried to make fits in n itterations using "early stopping" and removing prediction from  early stopping
       * not working well - keeping for documentation
       * more appropriate
    """

    def __init__(self, optionsErrPDF0):
        """
        :param optionsErrPDF options - see default PDF options
        """
        # {'learning_rate':0.01, 'max_depth':1000 ,"n_estimators":10,"subsample":0.05}
        defaultErrPDFOptions = {
            "learning_rate": 0.01,  # learning rate
            "max_depth": 1000,  # minimal number of trees for fit
            "n_estimators": 100,  # number of estimators
            "subsample": 0.05,  # subsample used
            "dump_progress": 5,  # switch to follow progress
            "step_size":5,       # step size to evaluate
            "verbose": 1
        }
        self.optionsErrPDF = defaultErrPDFOptions
        self.optionsErrPDF.update(optionsErrPDF0)
        self.paramXGBFactor=optionsErrPDF0
        self.paramXGBFactor.pop("dump_progress",None)
        self.paramXGBFactor.pop("step_size",None)
        self.method = "MIxgboostErrPDFv2"
        self.regXGBFacIter={}
        # backup the original data for later prediction
        self.Xin={}
        self.Yin={}
        self.regresorArray={}
        self.bestItterArray={}
        self.iterationInfoArray={}

    #
    def fit3Fold(self, xIn, yIn, sample_weight):
        # make regression in itterations
        step_size=self.optionsErrPDF["step_size"]
        if step_size<=0: step_size=1
        nRows = xIn.shape[0]
        permutation = np.random.permutation(nRows)
        for iSample in [0,1,2]:
            self.Xin[iSample]=xIn[permutation[iSample *nRows // 3: -1+(iSample+1) *nRows // 3]]
            self.Yin[iSample]=yIn[permutation[iSample *nRows // 3: -1+(iSample+1) *nRows // 3]]
            #self.regXGBFac[iSample].n_estimators= step_size
        #
        n_steps=self.optionsErrPDF["n_estimators"]//step_size
        #
        n_Regresors = len(self.optionsErrPDF["learning_rate"])
        #
        for iReg0 in range(0,n_Regresors):
            self.regresorArray[iReg0]={}
            self.iterationInfoArray[iReg0]=[]
            self.paramXGBFactor["learning_rate"]=self.optionsErrPDF["learning_rate"][iReg0]
            for iSample in range(0,3):
                self.regresorArray[iReg0][iSample] = xgb.XGBRegressor(**self.paramXGBFactor)
                self.regresorArray[iReg0][iSample].n_estimators= step_size
            # yet y corrected for previous regressions
            yIn={}
            yInDelta={}
            for iSample in range(0,3):
                yIn[iSample]=self.Yin[iSample].copy()
                for iReg1 in range(0,iReg0):
                    yInDelta[iSample]=self.regresorArray[iReg1][iSample].predict(self.Xin[iSample],iteration_range=(0,self.bestItterArray[iReg1]))
                    yIn[iSample]-=self.regresorArray[iReg1][iSample].predict(self.Xin[iSample],iteration_range=(0,self.bestItterArray[iReg1]))
                print(f"Yin\t=\t{iReg0}\t{iSample}\t{yIn[iSample].mean()}\t{np.median(yIn[iSample])}\t\t{yIn[iSample].std()}")
            #
            print(f"iter\titerAll\tstdPoint\tstdPointMean\tstdPredict")
            stdCurrent=yIn[0].std()
            lastIter=0
            for iter in range(0, n_steps):
                iterationStat={}
                iterationStat["iter"]=iter
                predict=[]
                for iSample in [0,1,2]:
                    if iter > 0:
                        self.regresorArray[iReg0][iSample].fit(self.Xin[iSample], yIn[iSample], xgb_model=self.regresorArray[iReg0][iSample])
                    else:
                        self.regresorArray[iReg0][iSample].fit(self.Xin[iSample], yIn[iSample])
                stdPoint={}
                stdPointMean={}
                stdPredict={}
                stdIter={}
                for iSample in [0,1,2]:
                    iSampleRef0=(iSample+1)%3
                    iSampleRef1=(iSample+2)%3
                    stdPoint[iSample]=(self.regresorArray[iReg0][iSample].predict(self.Xin[iSampleRef0])-yIn[iSampleRef0]).std()
                    stdPredict[iSample]=(self.regresorArray[iReg0][iSample].predict(self.Xin[iSampleRef1])-self.regresorArray[iReg0][iSampleRef0].predict(self.Xin[iSampleRef1])).std()
                    stdPointMean[iSample]=((self.regresorArray[iReg0][iSample].predict(self.Xin[iSampleRef1])+self.regresorArray[iReg0][iSampleRef0].predict(self.Xin[iSampleRef1]))*0.5
                                  -yIn[iSampleRef1]).std()
                    ir = (0, 0)
                    ir = self.regresorArray[iReg0][0]._get_iteration_range(ir)
                    stdIterRef=max(ir[1]-step_size,0)
                    stdIter=self.regresorArray[iReg0][iSample].predict(self.Xin[iSampleRef0],iteration_range=(stdIterRef,ir[1])).std()
                    #
                    s0=stdPoint[iSample]**2
                    s1=stdPointMean[iSample]**2
                    s2=stdPredict[iSample]**2
                    sStat2=s0-s1+s2/4.
                    k=s2/(2*sStat2)
                    #print(f"{iter}\t{stdPoint[iSample]:3.4}\t{stdPointMean[iSample]:3.4}\t{stdPredict[iSample]:3.4}",k,np.sqrt(sStat2))
                stdPoint[3]=(stdPoint[0]+stdPoint[1]+stdPoint[2])/3
                stdPointMean[3]=(stdPointMean[0]+stdPointMean[1]+stdPointMean[2])/3
                stdPredict[3]=(stdPredict[0]+stdPredict[1]+stdPredict[2])/3
                stdIter[3]=(stdIter[0]+stdIter[1]+stdIter[2])/3
                s0=stdPoint[3]**2
                s1=stdPointMean[3]**2
                s2=stdPredict[3]**2
                sStat2=s0-s1+s2/4.
                k=s2/(2*sStat2)
                ir=(0,0)
                ir=self.regresorArray[iReg0][0]._get_iteration_range(ir)
                print(f"{iReg0}\t{iter}\t{ir[1]}\t{stdPoint[3]:3.5}\t{stdPointMean[3]:3.5}\t{stdPredict[3]:3.5}\t{k:3.5}\t{np.sqrt(np.abs(sStat2)):3.5}\t{stdIter[3]}")
                iterationStat["stdPoint"]=stdPoint
                iterationStat["stdPointMean"]=stdPoint
                iterationStat["stdPredict"]=stdPredict
                iterationStat["stdIter"]=stdIter
                self.iterationInfoArray[iReg0].append(iterationStat)
                self.bestItterArray[iReg0]=int(ir[1]-1.5*step_size)
                if stdPointMean[3]>stdCurrent:
                    break
                else:
                    stdCurrent =stdPointMean[3]




    #def predictStatFold3(self, data, statDictionary, iteration_range=(0,0), treeType=-1):
    #    predict= (self.regXGBFac[0].predict(data,iteration_range=iteration_range)+
    #            self.regXGBFac[1].predict(data,iteration_range=iteration_range)+
    #            self.regXGBFac[2].predict(data,iteration_range=iteration_range))/3.
    #    return predict

    def predictStat(self, data, statDictionary, treeType=-1):
        """
        predict value
        :param data:           inout data
        :param statDictonary:  statistics to output
        :param treeType:
        :return:
        """
        # assert(treeType!=0 & treeType!=1)


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
