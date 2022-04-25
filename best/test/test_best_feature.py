# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np

from copy import deepcopy
from shutil import rmtree

import unittest
from unittest import TestCase

from best.feature import *


class TestFeature(TestCase):
    def test_import(self):
        print("Testing import 'from best.feature'")

    def test_zscore(self):
        X = np.random.randn(1000, 4) * 10 - 100

        for v in zscore(X).mean(axis=0):
            self.assertAlmostEqual(v, 0, 1)

        for v in zscore(X).std(axis=0):
            self.assertAlmostEqual(v, 1, 1)


    def test_find_category_outliers(self):
        X = np.random.randn(1000, 4) * 10 - 100
        Y = ['A']*800 + ['B']*200

        X[100] *= 1000
        X[900] *= 1000

        outliers = find_category_outliers(X, Y)
        self.assertIn(100, outliers)
        self.assertIn(900, outliers)


    def test_get_classification_scores(self):
        Y = np.array(['AWAKE']*800 + ['N3']*200)
        YY_1 = np.array(['AWAKE']*800 + ['N3']*200)
        YY_2 = np.array(['N3']*800 + ['AWAKE']*200)

        score1 = get_classification_scores(Y, YY_1)
        score2 = get_classification_scores(Y, YY_2)

        self.assertAlmostEqual(float(score1['kappa_all']), 1.0, places=1)
        self.assertAlmostEqual(float(score1['f1_all']), 1, 2)
        self.assertAlmostEqual(float(score1['accuracy_all']), 1, 2)

        self.assertAlmostEqual(float(score2['kappa_all']), -0.5, 1)
        self.assertAlmostEqual(float(score2['f1_all']), 0, 1)
        self.assertAlmostEqual(float(score2['accuracy_all']), 0, 1)


    def test_augment_features(self):
        X = np.random.randn(1000, 4) * 10 - 100
        Y = ['A']*800 + ['B']*200
        feature_names = ['0', '1', '2', '3']

        def set_zero(v):
            return np.zeros_like(v)

        X_new, feature_names_new = augment_features(x=X, feature_names=feature_names, feature_indexes=np.array([0]), operation=set_zero, operation_str='-', )

        self.assertTrue(feature_names_new[-1] == '-0')
        self.assertTrue(X_new[0, -1] == 0)


    def test_remove_samples(self):
        X = np.random.randn(1000, 4) * 10 - 100
        Y = ['A']*800 + ['B']*200
        feature_names = ['0', '1', '2', '3']

        def set_zero(v):
            return np.zeros_like(v)

        X_new, feature_names_new = remove_features(x=X, feature_names=feature_names, to_del=[0])
        self.assertTrue(X_new.shape[0] == 3)
        self.assertTrue(not '0' in feature_names_new)


    def test_remove_samples(self):
        X = np.random.randn(1000, 4) * 10 - 100
        Y = np.array(['A']*800 + ['B']*200)

        X_new, Y_new = remove_samples(X, Y, to_del=Y=='B')

        self.assertTrue(X_new.shape[0] == 800)
        self.assertTrue(not 'B' in Y_new)


    def test_replace_annotations(self):
        Y = np.array(['A']*800 + ['B']*200)

        Y_new = replace_annotations(Y, 'B', 'C')

        self.assertTrue(not 'B' in Y_new)
        self.assertTrue('C' in Y_new)
        self.assertTrue((Y_new == 'C').sum() == 200)




if __name__ == '__main__':
    unittest.main()
