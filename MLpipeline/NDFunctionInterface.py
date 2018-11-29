# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


# In[2]:


class RandomForest:

    def __init__(self, switch, X_train, y_train, **RF_params):
        if (switch == 'Classifier'):
            clf = RandomForestClassifier(**RF_params)
        elif (switch == 'Regressor'):
            clf = RandomForestRegressor(**RF_params)
        else:
            print('specify Classifier or Regressor (first argument)')
            return 0
        clf.fit(X_train, y_train)
        self.model = clf

    def predict(self, data):
        if 'Classifier' in str(self.model):
            return self.model.predict_proba(data)[:, 1]
        else:
            return self.model.predict(data)


# In[3]:


class KerasModel:

    def __init__(self, switch, X_train, y_train, **options):

        if not 'epochs' in options.keys():
            epochs = 3
        else:
            epochs = options['epochs']

        if not 'batchsize' in options.keys():
            batchsize = 50
        else:
            batchsize = options['batchsize']

        if not 'layout' in options.keys():
            layout = [50, 50]
        else:
            layout = options['layout']

        model = Sequential()
        for idx, val in enumerate(layout):
            if (idx == 0):
                model.add(Dense(val, input_dim=len(X_train.columns), activation='relu'))
                model.add(Dropout(0.2))
                continue
            if (idx > 0):
                model.add(Dense(val, activation='relu'))
                model.add(Dropout(0.2))
        if (switch == 'Classifier'):
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='ADAM', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize)
        elif (switch == 'Regressor' or switch == 'Compressor'):
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mean_absolute_error', optimizer='ADAM', metrics=['mse'])
            if (switch == 'Regressor'):
                model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize)
            else:
                model.fit(X_train, y_train.predict(X_train), epochs=epochs, batch_size=batchsize)
        else:
            return 0
        self.model = model

    def predict(self, data):
        return self.model.predict(data)[:, 0]


# In[4]:


class KNeighbors:

    def __init__(self, switch, X_train, y_train, **KNN_params):
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
        if 'Classifier' in str(self.model):
            return self.model.predict_proba(data)[:, 1]
        else:
            return self.model.predict(data)


# In[5]:


class PrepareData:

    def __init__(self, data, X, y, split, selection=None):
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


# In[6]:


class Bootstrapper:

    def __init__(self, data, sample_size, iterations):
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
        self.X = X
        self.y = y
        for i in range(self.iterations):
            print('Train sample ' + str(i))
            if (method == 'RandomForest'):
                MDL = RandomForest(ClassOrReg, self.Train_samples[i][X], np.ravel(self.Train_samples[i][y]), **options)
                self.Model.append(MDL)

    def predict_std_mean(self, data):
        predictions = []
        for i in range(self.iterations):
            predictions.append(self.Model[i].predict(data))
            print(i)
        mean = np.mean(np.array(predictions), axis=0)
        std = np.std(np.array(predictions), axis=0)
        return mean, std

    def predict(self, data):
        predictions = []
        for i in range(self.iterations):
            predictions.append(self.Model[i].predict(data))
            print(i)
        mean = np.mean(np.array(predictions), axis=0)
        return mean

    def OOB(self):
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


# In[7]:


class Fitter:

    def __init__(self, data):
        self.method_name = []
        self.method = []
        self.options = []
        self.ClassOrReg = []
        self.Models = []
        self.data = data

    def Register_Method(self, method_name, method, ClassOrReg, **options):
        self.method_name.append(method_name)
        self.method.append(method)
        self.options.append(options)
        self.ClassOrReg.append(ClassOrReg)

    def Fit(self):
        for idx, method in enumerate(self.method):

            if (method == 'RandomForest'):
                MDL = RandomForest(self.ClassOrReg[idx], self.data.X_train, np.ravel(self.data.y_train), **self.options[idx])

            elif (method == 'KerasModel'):
                MDL = KerasModel(self.ClassOrReg[idx], self.data.X_train, np.ravel(self.data.y_train), **self.options[idx])

            elif (method == 'KNeighbors'):
                MDL = KNeighbors(self.ClassOrReg[idx], self.data.X_train, np.ravel(self.data.y_train), **self.options[idx])

            else:
                return 0
            self.Models.append(MDL)

    def Predict(self, data, method_name):
        i = self.method_name.index(method_name)
        return self.Models[i].predict(data)

    def Compress(self, method_name, **options):
        i = self.method_name.index(method_name)
        MDL = KerasModel('Compressor', self.data.X_test, self.Models[i], **options)
        self.method_name.append(self.method_name[i] + '_Compressed')
        self.method.append('Compressor')
        self.options.append(options)
        self.ClassOrReg.append('Compressor')
        self.Models.append(MDL)

    def Bootstrap(self, method_name, method, ClassOrReg, sample_size, iterations, **options):
        b = Bootstrapper(self.data.Train_sample, sample_size, iterations)
        b.Train(ClassOrReg, method, self.data.X_values, self.data.y_values, **options)
        self.method_name.append(method_name)
        self.method.append(method + 'Bootstrapped')
        self.options.append(options)
        self.ClassOrReg.append(ClassOrReg)
        self.Models.append(b)

    def PlotRocs(self):
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

    def AppendOtherPandas(self, method_name, data):

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

    def RemoveMethod(self, method_name):
        i = self.method_name.index(method_name)
        self.method_name.pop(i)
        self.method.pop(i)
        self.options.pop(i)
        self.ClassOrReg.pop(i)
        self.Models.pop(i)
