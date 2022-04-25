# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

from sklearn.decomposition import PCA
from best.feature import augment_features

"""
Modules compatioble with scikitlearn pipeline

"""

class FeatureAugmentorModule:
    """
    Feature augmentation using an 'augment_features' function from the 'PiesUtils' package. See the code for additional details.
    """
    def __init__(self):
        self.mutual_functions = [np.divide, np.multiply]
        self.mutual_function_symbol = ['*', '/']

        self.standalone_functions = []
        self.mutual_function_symbol = []

    def fit(self, X=None, Y=None):
        pass

    def transform(self, X):
        n_orig_features = X.shape[1]
        for idx, func in enumerate(self.mutual_functions):
            X = augment_features(X, operation=func, mutual=True, feature_indexes=np.arange(n_orig_features))

        for idx, func in enumerate(self.standalone_functions):
            X = augment_features(X, operation=func, mutual=False, feature_indexes=np.arange(n_orig_features))
        return X

    def fit_transform(self, X, Y=None):
        return self.transform(X)

    def __call__(self, X):
        return self.transform(X)


class ZScoreModule:
    """
    Z-score normalization compatible with scikit.pipeline.Pipeline
    Enables continuous learning - enabling continuous adaptation.

    Modes
        * Zscore normalization
        * Zscore normalization with fixed mean and std values based on the initial training dataset
            * Possible category-wise normalization with mean and std values estimated from the training dataset - number of features is multiplied by number of categories
        * Zscore normalization with an initial mean and std values trained on the training dataset - adaptation during inference
            https://stats.stackexchange.com/questions/211837/variance-of-subsample



    Attributes
    ----------
    continuous_learning : bool
            If true - An instance updates mean and variance values during each prediction step. Initial outlier filtering is recommended
    trainable : bool
        If false - An instance normalizes inference data based on their current mean value and std
        If true - An instance remembers mean and variance values of training data
    multi_class : bool
        If true - An instance performs normalization for each training class separately
        Number of output features is multiplied by a number of training categories
    mean : numpy ndarray / list
        Trained mean values for each feature. In case multi_class == True -> list of numpy ndarrays for each category
    std : numpy ndarray

    N : int

    """
    def __init__(self, trainable=False, continuous_learning=False, multi_class=False):
        """
        Parameters
        ----------
        continuous_learning : bool
            If true - An instance updates mean and variance values during each prediction step. Initial outlier filtering is recommended
        trainable : bool
            If false - An instance normalizes inference data based on their current mean value and std
            If true - An instance remembers mean and variance values of training data
        multi_class : bool
            If true - An instance performs normalization for each training class separately
            Number of output features is multiplied by a number of training categories
        """


        self.continuous_learning = continuous_learning
        self.trainable = trainable
        self.multi_class = multi_class

        #self.sum_square = None
        self.mean_square = None
        self.mean = None
        self.std = None
        self.N = 0
        self.categories = []

        self._check_modes()

    def fit(self, X=None, Y=None):
        """
        Parameters
        ----------
        X : numpy ndarray
            shape[n_samples, n_features]
        Y : list or numpy array, optional
            category reference for each sample - required only for option with multi_class normalization

        Returns
        --------
        None

        """
        self._check_modes()
        if self.trainable is True:
            if self.multi_class is True:
                self.mean = []
                self.std = []
                self.categories = np.unique(Y)
                Y = np.array(Y)
                for cat_key in self.categories:
                    X_ = X[Y == cat_key]
                    self.mean.append(X_.mean(axis=0).reshape(1, -1))
                    self.std.append(X_.std(axis=0).reshape(1, -1))

            else:
                self.mean = X.mean(axis=0).reshape(1, -1)
                self.std = X.std(axis=0).reshape(1, -1)
                self.N = X.shape[0]
                #self.sum_square = (X**2).sum(axis=0).reshape(1, -1)
                self.mean_square = (X**2).mean(axis=0).reshape(1, -1)


    def transform(self, X=None):
        """
        Parameters
        ----------
        X : numpy ndarray
            shape[n_samples, n_features]

        Returns
        -------
        transformed_data : numpy ndarray
            shape[n_samples, n_features]

        """
        self._check_modes()
        if self.trainable is True:
            if self.multi_class is True:
                temp_list = []
                for cat_idx in range(self.mean.__len__()):
                    mean = self.mean[cat_idx]
                    std = self.std[cat_idx]
                    X_temp = (X - mean) / std
                    temp_list.append(X_temp)
                return np.concatenate(temp_list, axis=1)

            elif self.continuous_learning is True:
                M = X.shape[0]
                N = self.N

                mean_new = X.mean(axis=0).reshape(1, -1)
                mean_old = self.mean

                std_new = X.std(axis=0).reshape(1, -1)
                std_old = self.std

                #sum_square_new = (X**2).sum(axis=0).reshape(1, -1)
                #sum_square_old = self.sum_square

                mean_square_new = (X**2).mean(axis=0).reshape(1, -1)
                mean_square_old = self.mean_square


                c_old = N / (M+N)
                c_new = M / (M+N)
                mean_joint = (mean_old * c_old) + (mean_new * c_new)
                #mean_square_joint = mean_square_old * c_old + mean_square_new * c_new
                mean_square_joint = (mean_square_old * N + mean_square_new * M) / (M + N)

                #sum_square_joint = sum_square_new + sum_square_old

                #std_joint = ((std_old * N) + (std_new * M)) / (M+N) # just average std
                #std_joint = np.sqrt((N * (std_old**2 + (mean_old - mean_joint)**2)) +  (M * (std_new**2 + (mean_new - mean_joint)**2))) / (M + N)
                std_joint = np.sqrt(
                    (
                            N*std_old**2 + M*std_new**2 + N*((mean_old - mean_joint)**2) + M*((mean_new - mean_joint)**2)
                    ) / (N+M)
                )


                #std_joint = np.sqrt(((sum_square_joint) / (N + M)) - mean_joint**2) # average std using sum_square

                #std_joint = np.sqrt((mean_square_joint) - mean_joint**2) # average std using sum_square
                #std_joint = np.sqrt(mean_square_joint - mean_joint**2)

                self.mean = mean_joint
                self.std = std_joint
                self.N = N + M

                return (X - self.mean) / self.std

            else: # trainable
                return (X - self.mean) / self.std

        return (X - X.mean(axis=0).reshape(1, -1)) / X.std(axis=0).reshape(1, -1) # just zscore

    def fit_transform(self, X=None, Y=None):
        """
        Parameters
        ----------
        X : numpy ndarray
            shape[n_samples, n_features]
        Y : list or numpy array, optional
            category reference for each sample - required only for option with multi_class normalization

        Returns
        -------
        transformed_data : numpy ndarray
            shape[n_samples, n_features]

        """
        self.fit(X, Y)
        return self.transform(X)


    def __call__(self, X=None, Y=None):
        return self.transform(X)

    def _check_modes(self):
        if self.trainable is True:
            if self.continuous_learning is True and self.multi_class is True:
                raise AssertionError('[ASSERTION ERROR] - ZScore module - only one of parameters continuous_learning, multi_class can be activated')
        if self.trainable is False:
            if self.continuous_learning is True or self.multi_class is True:
                raise Warning('[WARNING] - ZScore - if trainable is False all other parameters will be considered as False')


class LogModule:
    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X, Y=None):
        return np.log(X)

    def fit_transform(self, X, Y=None):
        return self.transform(X)

    def __call__(self, X):
        return self.transform(X)


class Log10Module:
    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X, Y=None):
        return np.log10(X)

    def fit_transform(self, X, Y=None):
        return self.transform(X)

    def __call__(self, X):
        return self.transform(X)


class PCAModuleSVD:

    def __init__(self, var_threshold = 0.98):
        self.var_threshold = var_threshold

    def fit(self, X, Y=None):
        # Data matrix X, assumes 0-centered
        n, m = X.shape
        #assert np.allclose(X.mean(axis=0), np.zeros(m))
        # Compute covariance matrix
        C = np.dot(X.T, X) / (n-1)
        # Eigen decomposition
        self.eigen_vals, self.eigen_vecs = np.linalg.eig(C)

        norm_eigs = self.eigen_vals / self.eigen_vals.sum()
        self.n = 1
        while norm_eigs[:self.n].sum() < self.var_threshold:
            self.n += 1

    def transform(self, X, Y=None):
        return np.dot(X, self.eigen_vecs[:, :self.n])


    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def __call__(self, X):
        return self.transform(X)


class PCAModule(PCA):
    def __init__(self, var_threshold = 0.98):
        super().__init__()
        self.var_threshold = var_threshold

    def fit(self, X, y=None):
        testPCA = PCA()
        testPCA.fit(X)
        n = 1
        vals = testPCA.singular_values_
        vals = vals / vals.sum()
        while vals[:n].sum() < self.var_threshold:
            n += 1

        super().__init__(n_components=n)
        super().fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

