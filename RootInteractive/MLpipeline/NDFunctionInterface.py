import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import regularizers
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from skgarden import RandomForestQuantileRegressor

class RandomForest:
    """
    Random Forest (RF) from sklearn. This summarizes the RF for classification and regression.
    It allows a fast application and adjusts the notation fur further application.
    """

    def __init__(self, switch, X_train, y_train, **RF_params):
        """
        The initilialization includes registration and fitting of the random forest.

        :param switch: It can be chosen between 'Classifier' or 'Regressor' with the corresponding string.
        :param X_train: Features used for training. Can be provided as numpy array or pandas DF (and more?)
        :param y_train: The target for regression or the labels for classification. Also numpy array or pandas DF.
        :param RF_params: All options from sklearn can be used. For instance 
        """
        if switch == 'Classifier':
            clf = RandomForestClassifier(**RF_params)
        elif switch == 'Regressor':
            clf = RandomForestRegressor(**RF_params)
        elif switch == 'RegressorQuantile':
            clf=RandomForestQuantileRegressor(**RF_params)
        else:
            print('specify Classifier or Regressor (first argument)')
            return
        clf.fit(X_train, y_train)
        self.model = clf

    def predict(self, data, **options):
        """
        Returns the output of the RF to the provided data.
        :param data: array of the features to get the prediction for.
        :return: array of the predicted values.
        """
        if  'quantile' in options.keys():
            return self.model.predict(data,quantile=options['quantile'])
        if 'Classifier' in str(self.model):
            return self.model.predict_proba(data)[:, 1]
        else:
            return self.model.predict(data)

    def predictStat(self, data):
        """
        :param data  - input matrix
        :return: predict statistic mean, median, rms over trees
        """
        allRF = np.zeros((len(self.model.estimators_), data.shape[0]))
        for i, tree in enumerate(self.model.estimators_):
            allRF[i] = tree.predict(data)
        return [np.mean(allRF, 0), np.median(allRF, 0), np.std(allRF, 0)]

    def printImportance(self, varNames):
        """
        print sorted importance
        :param varNames:
        :return:
        """
        importance = self.model.feature_importances_
        indices = np.argsort(importance)
        for i in indices:
            print(varNames[i], importance[i])


class KerasModel:
    """
    Allows a fast usage of fully connected neural networks from Keras with predefined options.
    """

    def __init__(self, switch, X_train, y_train, **options):
        """
        Model is created, compiled and trained.
        :param switch: It can be chosen between 'Classifier' or 'Regressor' with the corresponding string.
        :param X_train: Features used for training. Can be provided as numpy array or pandas DF (and more?)
        :param y_train: The target for regression or the labels for classification. Also numpy array or pandas DF.
        :param options: Different options for the design of the neural network. Includes:
                    layout: list with len(list) as the number of hidden layers and the elements integers
                    with the number of neurons.
                    epochs: number of epochs for the training.
                    batchSize: batchSize for each training step.
        """

        if not 'epochs' in options.keys():
            epochs = 3
        else:
            epochs = options['epochs']

        if not 'batchSize' in options.keys():
            batchSize = 50
        else:
            batchSize = options['batchSize']

        if not 'layout' in options.keys():
            layout = [50, 50]
        else:
            layout = options['layout']

        if not 'loss' in options.keys():
            if switch == 'Classifier':
                loss = 'binary_crossentropy'
            else:
                loss = 'mean_absolute_error'
        else:
            loss = options['loss']

        if not 'dropout' in options.keys():
            dropout = 0.2

        else:
            dropout = options['dropout']

        #        if not 'l2' in options.keys():
        #            l2 = 0
        #        else:
        #            l2 = options['l2']
        if not 'l1' in options.keys():
            l1 = 0
        else:
            l1 = options['l1']
        model = Sequential()
        for idx, val in enumerate(layout):
            if idx == 0:
                model.add(Dense(val, input_dim=len(X_train.columns), activation='relu', kernel_regularizer=regularizers.l1(l1)))
                if dropout > 0:
                    model.add(Dropout(dropout))
                #                if l2 > 0:
                #                    model.add(kernel_regularizer=regularizers.l2(l2))
                #                if l1 > 0:
                #                    model.add(kernel_regularizer=regularizers.l1(l1))
                continue
            if idx > 0:
                model.add(Dense(val, activation='relu', kernel_regularizer=regularizers.l1(l1)))
                if dropout > 0:
                    model.add(Dropout(dropout))
        #                if l2 > 0:
        #                    model.add(kernel_regularizer=regularizers.l2(l2))
        if switch == 'Classifier':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss=loss, optimizer='ADAM', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize)
        elif switch == 'Regressor' or switch == 'Compressor':
            model.add(Dense(1, activation='linear'))
            model.compile(loss=loss, optimizer='ADAM', metrics=['mse'])
            if switch == 'Regressor':
                model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize)
            else:
                model.fit(X_train, y_train.predict(X_train), epochs=epochs, batch_size=batchSize)
        else:
            return
        self.model = model

    def predict(self, data):
        """
        Returns the output of the Keras Model to the provided data.
        :param data: array of the features to get the prediction for.
        :return: array of the predicted values.
        """
        return self.model.predict(data)[:, 0]


class KNeighbors:
    """
    KNeighbors from sklearn. This summarizes KNeighbors for classification and regression.
    It allows a fast application and adjusts the notation fur further application.
    """

    def __init__(self, switch, X_train, y_train, **KNN_params):
        """
        The initilialization includes registration and fitting of KNeighbors.
        :param switch: It can be chosen between 'Classifier' or 'Regressor' with the corresponding string.
        :param X_train: Features used for training. Can be provided as numpy array or pandas DF (and more?)
        :param y_train: The target for regression or the labels for classification. Also numpy array or pandas DF.
        :param KNN_params: All options from sklearn can be used.
        """
        if (switch == 'Classifier'):
            clf = KNeighborsClassifier(**KNN_params)
        elif (switch == 'Regressor'):
            clf = KNeighborsRegressor(**KNN_params)
        else:
            print('specify Classifier or Regressor (first argument)')
            return 0
        clf.fit(X_train, y_train)
        self.model = clf

    def predict(self, data):
        """
        Returns the output of the KNeighbors to the provided data.
        :param data: array of the features to get the prediction for.
        :return: array of the predicted values.
        """
        if 'Classifier' in str(self.model):
            return self.model.predict_proba(data)[:, 1]
        else:
            return self.model.predict(data)


class DataContainer:
    """
    The DataContainer allows easy handling of the data (as a pandas DF), by considering the input features, the targets,
    the split in to training and test sample and also selections.
    """

    def __init__(self, data, X, y, split, selection=None):
        """
        DataContainer is created.
        :param data: input data as pandas DF
        :param X: list of features
        :param y: list of targets or labels
        :param split: list. If len(split) = 1: provide a fraction. if len(split) = 2: provide test
                        sample size and training sample size.
        :param selection: selection applied via query on input DF
        """
        if selection is not None:
            data = data.query(selection)
        if (len(split) == 1):
            test_size = None
            if (split[0] < 1 and split[0] >= 0):
                test_size = split[0]
            elif (split[0] >= 1):
                test_size = split[0] / len(X)
            else:
                print('split smaller than 0')
                return 0
        elif (len(split) == 2):
            if (split[0] >= 1 and split[1] < 1 and split[1] >= 0):
                n = split[0]
                test_size = split[1]
            elif (split[0] >= 1 and split[1] >= 1):
                n = split[0] + split[1]
                test_size = split[0] / float(split[0] + split[1])
            else:
                print('wrong split')
            data = data.sample(n=n, frac=None, replace=False, weights=None, random_state=42, axis=None)
        else:
            print('wrong split list length')
            return 0
        X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=test_size, random_state=42)
        Train_sample, Test_sample = train_test_split(data, test_size=test_size, random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.Train_sample = Train_sample
        self.Test_sample = Test_sample
        self.X_values = X
        self.y_values = y


class Bootstrapper:
    """
    Bootstrapper allows bootstrapping of the provided methods. Different random subsampled of the provided data are
    used for training. Mean and standard deviaton can be returned.
    """

    def __init__(self, data, sample_size, iterations):
        """
        Initialize the parameters of the Bootstrapper and define the training and test samples.
        :param data: provided data as pandas DF or numpy array
        :param sample_size: size of each bootstrap sample. If fraction, the fraction of the total data is used.
        If absolute size (>=1) the corresponding number of data points is used for bootstrapping.
        :param iterations: number of random subsamples used for bootstrapping.
        """
        self.Train_samples = []
        self.Test_samples = []
        self.iterations = iterations
        self.X = None
        self.y = None
        if (sample_size < 1 and sample_size > 0):
            frac = sample_size
        elif (sample_size >= 1):
            frac = sample_size / len(data)
        else:
            print('Error: invalid sample size')

        for i in range(iterations):
            Train_tmp, Test_tmp = train_test_split(data, test_size=frac, random_state=i + 2)
            self.Train_samples.append(Train_tmp)
            self.Test_samples.append(Test_tmp)
        self.Model = []

    def Train(self, ClassOrReg, method, X, y, **options):
        """
        Train a certain method on the initialized data.
        :param ClassOrReg: Choose between 'Classifier' or 'Regressor' by providing the string.
        :param method: The method that is bootstrapped. Provide a string. At the moment only 'RandomForest'
                is available.
        :param X: List of input features.
        :param y: Target(s) or labels
        :param options: All options of the corresponding method can be used.
        :return:
        """
        self.X = X
        self.y = y
        for i in range(self.iterations):
            print('Train sample ' + str(i))
            if (method == 'RandomForest'):
                MDL = RandomForest(ClassOrReg, self.Train_samples[i][X], np.ravel(self.Train_samples[i][y]), **options)
                self.Model.append(MDL)

    def predict_std_mean(self, data):
        """
        Returns mean and standard deviation for the trained method on the data set chosen as a parameter.
        :param data: list of features that the standard deviation and mean for the trained method should be
        returned for.
        :return: mean and standard deviation as numpy array
        """
        predictions = []
        for i in range(self.iterations):
            predictions.append(self.Model[i].predict(data))
            print(i)
        mean = np.mean(np.array(predictions), axis=0)
        std = np.std(np.array(predictions), axis=0)
        median = np.median(np.array(predictions), axis=0)
        return mean, std, median

    def predict(self, data):
        """
        Returns mean for the trained method on the data set chosen as a parameter.
        :param data: list of features that the standard deviation and mean for the trained method should be
        returned for.
        :return: mean as numpy array
        """
        predictions = []
        for i in range(self.iterations):
            predictions.append(self.Model[i].predict(data))
            print(i)
        mean = np.mean(np.array(predictions), axis=0)
        return mean

    def OOB(self):
        """
        Out-of-bag error estimation for the data that is subsampled for training. The prediction values for each
        test sample corresponding to a training sample is written to a list and finally the mean and std deviation
        is calculated for the test samples for every data point.
        :return: Output pandas DF with mean and standard deviation added.
        """
        out = []
        ColumnNameList = []
        for idx, element in enumerate(self.Model):
            out.append(self.Test_samples[idx])
            out[idx] = out[idx].assign(method_out=element.predict(out[idx][self.X].values))
            out[idx] = out[idx].rename(columns={'method_out': 'bootstrap_' + str(idx)})
            ColumnNameList.append('bootstrap_' + str(idx))

        for idx, element in enumerate(out):
            if (idx == 0):
                final_out = element
            else:
                final_out = final_out.merge(element, how='outer')

        final_out = final_out.assign(mean_BS=final_out[ColumnNameList].mean(axis=1, skipna=True))
        final_out = final_out.assign(std_BS=final_out[ColumnNameList].std(axis=1, skipna=True))
        final_out = final_out.drop(ColumnNameList, axis=1)
        return final_out


class Fitter:
    """
    The allows simple usage and comparison of different MVA methods.

    Default approach:
    initialize Fitter (by providing the data)
    register methods (Register_Method())
    fit the registered methods (Fit())
    append fitted method to a pandas DF (either to the provided DF by using AppendPandas() or to a different
    DF by using AppendOtherPandas())

    Additional features:
    Compress(): A neural network is trained with the output of a trained method.
    Bootstrap(): Methods are trained using random subsamples. Mean and standard deviation can be returned.
    PlotROCs(): ROCs of the trained methods are plotted.
    RemoveMethod(): Remove methods from the list of methods.
    """

    def __init__(self, data):
        """
        The lists for method management are created and the data for the methods is defined (as DataContainer).
        :param data: data provided as DataContainer.
        """
        self.method_name = []
        self.method = []
        self.options = []
        self.ClassOrReg = []
        self.Models = []
        self.data = data

    def Register_Method(self, method_name, method, ClassOrReg, **options):
        """
        Register a method to be fitted. Can be run several times to register several methods. Every time a different name
        has to be chosen.
        :param method_name: user defined string by which the method can be called in the further course.
        :param method: string that defines the method. It can be chosen between: 'RandomForest', 'KerasModel', 'KNeighbors'
        :param ClassOrReg: string that defines if it is done either regression or classification.
        Choose between: 'Classifier' or 'Regressor'
        :param options: Options for the different models. For more details see the classes of the models.

        TODO: Register input features for single methods.
        """
        self.method_name.append(method_name)
        self.method.append(method)
        self.options.append(options)
        self.ClassOrReg.append(ClassOrReg)

    def Fit(self):
        """
        Fit (train) all registered methods.

        TODO: Register single methods.
        """
        for idx, method in enumerate(self.method):

            if (method == 'RandomForest'):
                MDL = RandomForest(self.ClassOrReg[idx], self.data.X_train, np.ravel(self.data.y_train),
                                   **self.options[idx])

            elif (method == 'KerasModel'):
                MDL = KerasModel(self.ClassOrReg[idx], self.data.X_train, np.ravel(self.data.y_train),
                                 **self.options[idx])

            elif (method == 'KNeighbors'):
                MDL = KNeighbors(self.ClassOrReg[idx], self.data.X_train, np.ravel(self.data.y_train),
                                 **self.options[idx])

            else:
                return
            self.Models.append(MDL)

    def Predict(self, data, method_name):
        """
        Prediction values of a trained method for foreign data. Method selected by registered method name.
        :param data: input features (numpy array or pandas DF)
        :param method_name: registered user selected name as string.
        :return: output values
        """
        i = self.method_name.index(method_name)
        return self.Models[i].predict(data)

    def Compress(self, method_name, **options):
        """
        Trains a neural network with the output of another method chosen by the user defined method
        name. It is added to the list of trained methods and is saved with the method_name:
        method_name + '_Compressed'
        :param method_name: Method that should be compressed chosen by user defined method_name.
        :param options: options for the keras model
        """
        i = self.method_name.index(method_name)
        MDL = KerasModel('Compressor', self.data.X_test, self.Models[i], **options)
        self.method_name.append(self.method_name[i] + '_Compressed')
        self.method.append('Compressor')
        self.options.append(options)
        self.ClassOrReg.append('Compressor')
        self.Models.append(MDL)

    def Bootstrap(self, method_name, method, ClassOrReg, sample_size, iterations, **options):
        """
        Calls class Bootstrapper. See this class for more details.
        :param method_name: user defined string by which the method can be called in the further course.
        :param method: method: string that defines the method. It can be chosen between: 'RandomForest', 'KerasModel', 'KNeighbors'
        :param ClassOrReg: string that defines if it is done either regression or classification.
        Choose between: 'Classifier' or 'Regressor'
        :param sample_size: size of each bootstrap sample. If fraction, the fraction of the total data is used.
        If absolute size (>=1) the corresponding number of data points is used for bootstrapping.
        :param iterations: number of random subsamples used for bootstrapping.
        :param options: options of the chosen method. Look at corresponding class for more details.

        TODO: At the moment the bootstrapping is done with the train sample. This should be replaced by the
        whole sample and the OOB estimate should be returned.
        """
        b = Bootstrapper(self.data.Train_sample, sample_size, iterations)
        b.Train(ClassOrReg, method, self.data.X_values, self.data.y_values, **options)
        self.method_name.append(method_name)
        self.method.append(method + 'Bootstrapped')
        self.options.append(options)
        self.ClassOrReg.append(ClassOrReg)
        self.Models.append(b)

    def printImportance(self):
        for model in self.Models:
            if "printImportance" in dir(model):
                model.printImportance(self.data.X_values)

    def PlotRocs(self):
        """
        Plot ROCs for all trained methods using the test sample.
        """
        fpr = []
        tpr = []
        for model in self.Models:
            fpr_tmp, tpr_tmp, thresholds = metrics.roc_curve(self.data.y_test, model.predict(self.data.X_test))
            fpr.append(fpr_tmp)
            tpr.append(tpr_tmp)
        plt.close
        for i, method in enumerate(self.method_name):
            print('AUC of method ' + method + ' : ', metrics.auc(fpr[i], tpr[i]))
            plt.plot(tpr[i], 1 - fpr[i], label=method)
        plt.legend()
        plt.show()

    def AppendPandas(self, method_name, option):
        """
        A trained method can be chosen via user defined method name and the output can be added to test sample,
        train sample or both together.
        :param method_name: user specified method name for the method that should be added to a pandas DF
        :param option: Select the sample via a string. Options are: 'Test_sample', 'Train_sample' and 'all'.
        :return: the selected pandas DF with the methods prediction values as extra column.
        """

        i = self.method_name.index(method_name)
        out = None
        x_values = None
        if (option == 'Test_sample'):
            out = self.data.Test_sample
            x_values = self.data.X_test
        elif (option == 'Train_sample'):
            out = data.Train_sample
            x_values = self.data.X_train
        elif (option == 'all'):
            out = pd.concat(self.data.Test_sample, self.data.Train_sample)
            x_values = pd.concat(self.data.X_test, self.data.X_train)
        else:
            return 0

        if 'Bootstrapped' not in self.method[i]:
            out = out.assign(method_out=self.Predict(x_values, method_name))
            out = out.rename(columns={'method_out': method_name})

        else:
            out_mu, out_std = self.Models[i].predict_std_mean(x_values)
            out = out.assign(method_outmu=out_mu, method_outstd=out_std)
            out = out.rename(columns={'method_outmu': method_name + '_mu'})
            out = out.rename(columns={'method_outstd': method_name + '_std'})

        return out

    def AppendOtherPandas(self, method_name, data, ):
        """
        The selected method (chosen via user defined method_name) is added to a data frame provided by the user.
        :param method_name: selected method (chosen via user defined method_name)
        :param data: provide a data frame which contains the training features. A column with the prediction values
        is added to this data frame.
        :return: the modified pandas DF
        """

        i = self.method_name.index(method_name)
        out = data
        if 'Bootstrapped' not in self.method[i]:
            out = data.assign(method_out=self.Predict(data[self.data.X_values].values, method_name))
            out = out.rename(columns={'method_out': method_name})

        else:
            out_mu, out_std = self.Models[i].predict_std_mean(data[self.data.X_values].values)
            out = data.assign(method_outmu=out_mu, method_outstd=out_std)
            out = out.rename(columns={'method_outmu': method_name + '_mu'})
            out = out.rename(columns={'method_outstd': method_name + '_std'})

        return out

    def AppendStatPandas(self, method_name, data, ):
        """
        append statisctic colimns from "??? estomators ***" - random forrst or NN with dropout
        :param method_name:
        :param data:
        :return:
        """
        i = self.method_name.index(method_name)
        model = self.Models[i]
        cols = model.predictStat(data[self.data.X_values].values)
        out = data
        out[method_name + 'Mean'] = cols[0]
        out[method_name + 'Median'] = cols[1]
        out[method_name + 'RMS'] = cols[2]
        return out

    def RemoveMethod(self, method_name):
        """
        Remove a selected method from the list of methods.
        :param method_name: user specified name of the method to be removed
        """
        i = self.method_name.index(method_name)
        self.method_name.pop(i)
        self.method.pop(i)
        self.options.pop(i)
        self.ClassOrReg.pop(i)
        self.Models.pop(i)
