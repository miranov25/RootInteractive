import numpy as np
from sklearn.ensemble import RandomForestRegressor

def makeTestExp(n):
    X = np.random.uniform(0,5,n)
    noise = np.random.normal(0,.05,[n,2])
    m1 = np.random.uniform(0,2*np.pi,5)
    X_smeared = X+noise[0]
    Y = m1[0]*np.cos(m1[2]*X_smeared+m1[1])*np.exp(m1[3]*X_smeared)+m1[4]+noise[1]
    Y_true = m1[0]*np.cos(m1[2]*X+m1[1])*np.exp(m1[3]*X)+m1[4]
    return X,Y,Y_true

def fit_testExpAnalyticalOLS(X,Y):
    m1 = np.zeros(5)
    m1[4] = np.mean(Y)
    YNorm = Y-m1[4]
    
