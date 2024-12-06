import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed
from RootInteractive.MLpipeline.MIForestErrPDF import predictRFStat

class AugmentedForestRegressor(RandomForestRegressor):
    def __init__(self,
                 n_estimators='warn',
                 criterion='mse',
                 max_depth=None,
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
    def __init__(self, n_forests, n_repetitions, n_jobs, verbose=0):
        self.n_jobs = n_jobs
        self.forests = [RandomForestRegressor(n_repetitions) for i in range(n_forests)]
        self.estimators_ = []
        self.n_forests = n_forests
        self.n_repetitions = n_repetitions
        self.verbose = verbose

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