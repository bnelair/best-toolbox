# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from scipy.stats._multivariate import multivariate_normal_gen, multivariate_normal_frozen

class multivariate_normal_(multivariate_normal_frozen):
    def __init__(self, X, allow_singular=False, seed=None):
        X = X.copy().T
        cov = np.cov(X.T)
        mu = X.mean(axis=0)
        super().__init__(mu, cov, allow_singular, seed)

    def pdf(self, X):
        X = X.copy().T
        return super().pdf(X)

