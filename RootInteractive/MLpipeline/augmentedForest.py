import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
from RootInteractive.MLpipeline.MIForestErrPDF import predictRFStat
import xgboost as xgb


class AugmentedForestRegressor(RandomForestRegressor):
    def __init__(self,
                 n_estimators='warn',
                 criterion='mse',
                 max_depth=None,
                 max_samples=1.0,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 noise_levels=None):
        super().__init__(n_estimators=n_estimators,
                 criterion=criterion,
                 max_depth=max_depth,
                 max_samples=max_samples,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,
                 bootstrap=bootstrap,
                 oob_score=oob_score,
                 n_jobs=n_jobs,
                 random_state=random_state,
                 verbose=verbose,
                 warm_start=warm_start)

        self.noise_levels = noise_levels

    def _fit_tree(self, tree, X, y, sample_weight, noise_levels):
        tree.fit(X,y,sample_weight)
        return tree

    def fit(self, X, y, sample_weight=None):
        '''
        Override 
        '''
        total_samples = X.shape[0]
        indices = np.random.permutation(total_samples)
        subset_size = total_samples // self.n_estimators
        block_begin = range(0, total_samples, subset_size)
        parallel_runner = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        parallel_runner(
            delayed(self._fit_tree)(
                self._make_estimator(), X, y, sample_weight=sample_weight, noise_levels=self.noise_levels) for x in block_begin
            )
        return self

class AugmentedRandomForestArray:
    # Array of N standard random forests
    # Input parameters include standard options like kernel width
    def __init__(self, n_forests, n_repetitions, n_jobs, max_depth, verbose=0):
        self.n_jobs = n_jobs
        self.forests = [RandomForestRegressor(n_repetitions) for i in range(n_forests)]
        self.estimators_ = []
        self.n_forests = n_forests
        self.n_repetitions = n_repetitions
        self.verbose = verbose
        self.max_depth=max_depth

    def fit(self, X, Y, sigma, sample_weight=None):
        total_samples = X.shape[0]
        indices = np.random.permutation(total_samples)
        subset_size = total_samples // self.n_forests
        if len(sigma) != X.shape[1]:
            raise ValueError("sigma must have the same length as the number of features in X")

        for i in range(self.n_forests):
            start_idx = i * subset_size
            end_idx = (i + 1) * subset_size if i < self.n_forests - 1 else total_samples

            X_train = X[indices[start_idx:end_idx]]
            Y_train = Y[indices[start_idx:end_idx]]

            # Pre-allocate arrays to hold augmented data
            augmented_X = np.empty((self.n_repetitions * X_train.shape[0], X_train.shape[1]), dtype=X.dtype)
            augmented_Y = np.empty(self.n_repetitions * X_train.shape[0], dtype=Y.dtype)

            # Fill the pre-allocated arrays
            for j in range(self.n_repetitions):
                noise = np.random.normal(0, sigma, X_train.shape)
                augmented_X[j * X_train.shape[0]:(j + 1) * X_train.shape[0], :] = X_train + noise
                augmented_Y[j * X_train.shape[0]:(j + 1) * X_train.shape[0]] = Y_train

            # Train the RF model on the augmented data
            self.forests[i].fit(augmented_X, augmented_Y)

        self.estimators_ = [estimator for forest in self.forests for estimator in forest.estimators_]
        # Fit each random forest with augmented input data

    def predict(self, X):
        return predictRFStat(self, X, {"mean":[]}, self.n_jobs)["mean"]
        # Use an ensemble of forests for prediction
        # Unlike standard RFStat which uses a single array of trees, this method uses an array of arrays

    def predictRFStat(self, X, statDictionary, max_rows=10000):
        return predictRFStat(self, X, statDictionary, self.n_jobs, max_rows)
        # Definition for detailed statistical prediction using random forests




def makeAugmentedRF(X, Y, rfArray, nRepetitions, sigmaVec,sigmaVal):
    """
    Augments training data by adding Gaussian noise and trains multiple Random Forest models.
    Parameters:
        X (np.array): Feature matrix.
        Y (np.array): Target vector.
        rfArray (list of RandomForestClassifier): List of RF models to train.
        nRepetitions (int): Number of times to repeat the augmentation.
        sigmaVec (list or np.array): Standard deviations for Gaussian noise.
        sigmaVal  - std gaussian smearing for value
    Returns:
        list of RandomForestClassifier: Trained RF models.
    """
    X=np.array(X)
    Y=np.array(Y)
    # Ensure sigmaVec is correctly sized
    if len(sigmaVec) != X.shape[1]:
        raise ValueError("sigmaVec must have the same length as the number of features in X")

    total_samples = X.shape[0]
    nRF = len(rfArray)
    indices = np.random.permutation(total_samples)
    subset_size = total_samples // nRF  # Use integer division for indexing

    for i in range(nRF):
        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size if i < nRF - 1 else total_samples
        X_train = X[indices[start_idx:end_idx]]
        Y_train = Y[indices[start_idx:end_idx]]
        # Pre-allocate arrays to hold augmented data
        augmented_X = np.zeros((nRepetitions * X_train.shape[0], X_train.shape[1]))
        augmented_Y = np.zeros(nRepetitions * X_train.shape[0], dtype=Y.dtype)
        # Fill the pre-allocated arrays
        for j in range(nRepetitions):
            noise = np.random.normal(0, sigmaVec, X_train.shape)
            noiseV= np.random.normal(0, sigmaVal, Y_train.shape)
            augmented_X[j * X_train.shape[0]:(j + 1) * X_train.shape[0], :] = X_train + noise
            augmented_Y[j * X_train.shape[0]:(j + 1) * X_train.shape[0]] = Y_train+noiseV
        # Train the RF model on the augmented data
        rfArray[i].fit(augmented_X, augmented_Y)
    return rfArray


def makeAugmentXGBoost(X, Y, xgbArray, nRepetitions, sigmaVec, sigmaVal, tolerance=1e-4, maxRounds=10, Ytrue=None):
    """
    Augments training data by adding Gaussian noise and trains multiple XGBoost models.
    Stops when the mean or median RMS of predictions stabilizes across models.

    Parameters:
        X (np.array): Feature matrix.
        Y (np.array): Target vector.
        xgbArray (list): List of XGBoost regressor models to train.
        nRepetitions (int): Number of times to repeat the augmentation.
        sigmaVec (np.array or list): Standard deviations for Gaussian noise (features).
        sigmaVal (float): Standard deviation for Gaussian noise (target values).
        tolerance (float): Minimum decrease in RMS to continue training as a fraction of the previous RMS
        maxRounds (int): Maximum number of training rounds.

    Returns:
        list: List of trained XGBoost models.
    """
    nxgb=len(xgbArray)
    X_augmented = []
    Y_augmented = []
    len_group = int(X.shape[0] / nxgb)  # Integer division to ensure proper slicing

    # Data augmentation with Gaussian noise
    for i in range(nxgb):
        for j in range(nRepetitions):
            X_subset = X[i * len_group:(i + 1) * len_group, :]
            Y_subset = Y[i * len_group:(i + 1) * len_group]
            # Add Gaussian noise to each subset
            if (j!=0):
                X_noisy = X_subset + np.random.normal(0, sigmaVec, X_subset.shape)
                Y_noisy = Y_subset + np.random.normal(0, sigmaVal, Y_subset.shape)
            else:
                X_noisy = X_subset
                Y_noisy = Y_subset
            # Store the augmented data
            X_augmented.append(X_noisy)
            Y_augmented.append(Y_noisy)
    # Stack to create 2D arrays
    X_augmented = np.vstack(X_augmented)
    Y_augmented = np.hstack(Y_augmented)
    #
    trained_models = []
    # Initialize model placeholders
    for model in xgbArray:
        trained_models.append(xgb.XGBRegressor(**model.get_params()))
    # Early stopping tracking
    rms_history = []
    rms_mean_history=[]
    rms_median_history=[]
    pred_std_history=[]
    pred_stdMedian_history=[]
    pred_stdT_history=[]
    nPoints=X_augmented.shape[0]//nxgb
    isEarlyStop = False  # Example value for isEarlyStop variable
    for round in range(maxRounds):
        preds = np.zeros((X.shape[0], len(trained_models)))
        # Train each model in the array and collect predictions
        for i, model in enumerate(trained_models):
            # Train model on its respective slice of augmented data
            X_train_slice = X_augmented[i * nPoints : (i + 1) * nPoints]
            Y_train_slice = Y_augmented[i * nPoints : (i + 1) * nPoints]
            model.fit(X_train_slice, Y_train_slice, xgb_model=model if round > 0 else None, verbose=False)
            # Predict on entire augmented set and store in preds matrix
            preds[:, i] = model.predict(X)
        # Calculate RMS across models for current round
        valPred=np.mean(preds, axis=1)
        valPredMedian=np.median(preds, axis=1)
        pred_std  =  np.std(valPred-Y)
        pred_stdMedian  =  np.std(valPredMedian-Y)
        pred_stdT = np.std(valPred - Ytrue)
        rms_values = np.std(preds, axis=1)    ### rms of the predictions
        rms_mean = np.mean(rms_values)
        rms_median = np.median(rms_values)
        #
        rms_history.append(rms_mean)
        rms_mean_history.append(rms_mean)
        rms_median_history.append(rms_median)
        pred_std_history.append(pred_std)
        pred_stdMedian_history.append(pred_stdMedian)
        pred_stdT_history.append(pred_stdT)
        #
        print(f"Round {round+1} - RMS Mean: {rms_mean:.4f}, RMS Median: {rms_median:.4f}. Pred std: {pred_std:.4f}  Pred stdT: {pred_stdT:.4f}")
        # Early stopping check
        if len(rms_history) > 3:
            avg_recent_rms = np.mean(pred_std_history[-3:])
            print(f"Avg recent RMS: {avg_recent_rms}  {pred_std_history[-1]}  {avg_recent_rms + tolerance*avg_recent_rms}" )
            if pred_std_history[-1] > avg_recent_rms + tolerance*avg_recent_rms:
                print(f"Early stopping at round {round+1} - RMS stabilization. {pred_std_history[-1]} - {avg_recent_rms}")
                isEarlyStop = True
                break

    # Create DataFrame
    dfH = pd.DataFrame({
        'RMS': rms_history,
        'RMS_Mean': rms_mean_history,
        'RMS_Median': rms_median_history,
        'Prediction_Std': pred_std_history,
        'Prediction_Std_Median': pred_stdMedian_history,
        'Prediction_Std_T': pred_stdT_history
    })

    dfH['iter'] = dfH.index  # Add 'iter' as the index
    dfH['isLast'] = dfH.index == len(dfH) - 1  # True for the last row
    dfH['isEarlyStop'] = isEarlyStop  # Add 'isEarlyStop' column with the variable's value
    dfH["avg_recent_rms"]=avg_recent_rms
    return trained_models,valPred,dfH,rms_values,valPredMedian
