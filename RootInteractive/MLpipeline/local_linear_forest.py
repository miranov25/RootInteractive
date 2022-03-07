import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV,Ridge

class LocalLinearForestRegressor(RandomForestRegressor):
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
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(n_estimators=n_estimators,
                 criterion=criterion,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,
                 min_impurity_split=min_impurity_split,
                 bootstrap=bootstrap,
                 oob_score=oob_score,
                 n_jobs=n_jobs,
                 random_state=random_state,
                 verbose=verbose,
                 warm_start=warm_start)

        # matrix with leaf node for training observation
        #  in each tree
        self._incidence_matrix = None
        self._X_train = None
        self._Y_train = None

    def _extract_leaf_nodes_ids(self, X):
        '''
        Extract a matrix of dimension (rows, cols), where rows is the number of rows of X and cols is the number of tree in the forest,
        \nthat contains the ids of the leaf node for each observation in each tree.

        Parameters:
        -----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns:
        --------
        numpy.array: 
            (R x C), where R == X.rows and C == number of trees in the forest
        '''
        leafs = []
        leaf_nodes_ids2=np.zeros((len(self.estimators_),X.shape[0]))
        for e in self.estimators_:
            leaf=e.apply(X)
            leafs.append(e.apply(X).reshape(-1, 1))
        leaf_nodes_ids = np.concatenate(leafs, axis=1)


        #for i, e in enumerate(self.estimators_):
        #    leaf_nodes_ids2[i]= e.apply(X)

        # the number of the rows must be the same of the number of observation
        assert leaf_nodes_ids.shape[0] == X.shape[0]
        # the number of the columns must be the same of the number of estimators (trees)
        assert leaf_nodes_ids.shape[1] == len(self.estimators_)

        return leaf_nodes_ids


    def fit(self, X, y, sample_weight=None):
        '''
        Override 
        '''
        super().fit(X, y, sample_weight=sample_weight)
        # save train data
        self._X_train = X
        self._Y_train = y
        # calculate leaf nodes for each observation in each tree
        self._incidence_matrix = self._extract_leaf_nodes_ids(X)
        return self

    def _get_forest_coefficientsOld(self, observation_leaf_ids):
        '''
        1   B   {1 | Xi € Lb(X_actual)}
        -  sum -------------------------
        B  b=1       |Lb(X_actual)|

        Parameters:
        -----------
        observation_leaf_ids: numpy.array [1, n_estimators_]
        '''
        coeffs = []
        coeffNP = np.zeros(self._X_train.shape[0])
        for i in range(0, self._X_train.shape[0]):
            count = 0
            for j in range(0, observation_leaf_ids.shape[1]):
                if self._incidence_matrix[i, j] == observation_leaf_ids[0, j]:  #if obervation leaf ID the same as training leaf in tree j
                    count += 1 / (self._incidence_matrix[:, j] == observation_leaf_ids[0, j]).sum()
            #coeffs.append(1 / self.n_estimators * count)
            coeffNP[i]=(1 / self.n_estimators) * count
        return coeffNP

    def _get_forest_coefficients(self, observation_leaf_ids):
        '''
        1   B   {1 | Xi € Lb(X_actual)}
        -  sum -------------------------
        B  b=1       |Lb(X_actual)|

        Parameters:
        -----------
        observation_leaf_ids: numpy.array [1, n_estimators_]
        '''
        coeffNP = np.zeros(self._X_train.shape[0])
        for j in range(0, observation_leaf_ids.shape[1]):
            oleafId=observation_leaf_ids[0, j]
            indexMatch=np.nonzero(self._incidence_matrix[:,j]==oleafId)
            coeffNP[indexMatch] += 1 / (self._incidence_matrix[:, j] == oleafId).sum()
        coeffNP/=self.n_estimators
        return coeffNP

    def predict(self, X):
        '''
        Override
        '''
        results = []

        X_ = np.array(X)

        # prediction for each observation
        for i in range(0, X_.shape[0]):
            X_actual = X_[i, :].reshape(1, -1)
            # we can calulate the coefficients for one row at a time
            actual_leaf_ids = self._extract_leaf_nodes_ids(X_actual)
            # calculate coefficients weights alpha_i(X_actual)
            alphas = self._get_forest_coefficients(actual_leaf_ids)
            # X_i - X_actual
            X_disc = (self._X_train - X_actual).to_numpy()
            # ridge
            index0=np.nonzero(alphas)
            alpha0=alphas[index0]
            x0=X_disc[index0]
            y0=self._Y_train[index0]
            #ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(x0, y0, alpha0)
            ridge = Ridge().fit(x0, y0, alpha0)
            # ridge predictions
            results.append(ridge.predict(X_actual)[0])

        return np.array(results).reshape(-1)

    def print_tree_structure(self):
        for estimator in self.estimators_:
            # The decision estimator has an attribute called tree_  which stores the entire
            # tree structure and allows access to low level attributes. The binary tree
            # tree_ is represented as a number of parallel arrays. The i-th element of each
            # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
            # Some of the arrays only apply to either leaves or split nodes, resp. In this
            # case the values of nodes of the other type are arbitrary!
            #
            # Among those arrays, we have:
            #   - left_child, id of the left child of the node
            #   - right_child, id of the right child of the node
            #   - feature, feature used for splitting the node
            #   - threshold, threshold value at the node
            #

            # Using those arrays, we can parse the tree structure:
            n_nodes = estimator.tree_.node_count
            children_left = estimator.tree_.children_left
            children_right = estimator.tree_.children_right
            feature = estimator.tree_.feature
            threshold = estimator.tree_.threshold

            # The tree structure can be traversed to compute various properties such
            # as the depth of each node and whether or not it is a leaf.
            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, -1)]  # seed is the root node id and its parent depth
            while len(stack) > 0:
                node_id, parent_depth = stack.pop()
                node_depth[node_id] = parent_depth + 1

                # If we have a test node
                if (children_left[node_id] != children_right[node_id]):
                    stack.append((children_left[node_id], parent_depth + 1))
                    stack.append((children_right[node_id], parent_depth + 1))
                else:
                    is_leaves[node_id] = True

            print('The binary tree structure has %s nodes and has '
                'the following tree structure:'
                % n_nodes)
            for i in range(n_nodes):
                if is_leaves[i]:
                    print('%snode=%s leaf node.' % (node_depth[i] * '\t', i))
                else:
                    print('%snode=%s test node: go to node %s if X[:, %s] <= %s else to '
                        'node %s.'
                        % (node_depth[i] * '\t',
                            i,
                            children_left[i],
                            feature[i],
                            threshold[i],
                            children_right[i],
                            ))
            print()
        

        
    

class TestLiuk(LocalLinearForestRegressor):
    def predict(self, X):
        '''
        Override
        '''
        results = []

        X_ = np.array(X)

        # prediction for each observation
        for i in range(0, X_.shape[0]):
            X_actual = X_[i, :].reshape(1, -1)
            # X_i - X_actual
            X_disc = self._X_train - X_actual
            # ridge
            ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_disc, self._Y_train)
            # ridge predictions
            results.append(ridge.predict(X_actual)[0])

        return np.array(results).reshape(-1)