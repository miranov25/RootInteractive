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
    def __init__(self, n_forests, n_repetitions, n_jobs):
        self.n_jobs = n_jobs

    def fit(self, X, Y, sigma, sample_weight=None):
        pass
        # Fit each random forest with augmented input data

    def predict(self):
        flattened_forest = [estimator for forest in self.estimators for estimator in forest]
        pass
        # Use an ensemble of forests for prediction
        # Unlike standard RFStat which uses a single array of trees, this method uses an array of arrays

    def predictRFStat(self, X, statDictionary, n_jobs):
        pass
        # Definition for detailed statistical prediction using random forests