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

import torch.cuda

from best.deep_learning.seizure_detect import load_trained_model, preprocess_input, infer_seizure_probability
from numpy.random import rand
from best.deep_learning.models_generic import *


class Test_deep_learning(TestCase):
    def test_import(self):
        print("Testing import 'from best.deep_learning'" )


class Test_seizure_detect(TestCase):
    def test_load_trained_model(self):
        modelA = load_trained_model('modelA')
        modelB = load_trained_model('modelB')

    def test_preprocess_input(self):
        modelA = load_trained_model('modelA')
        fs = 500
        x_len = 300
        channels = 1
        x_input = rand(channels, fs * x_len)
        x = preprocess_input(x_input, fs)
        y = infer_seizure_probability(x, modelA)
        self.assertEqual(599, y.shape[1])
        channels = 3
        x_input = rand(channels, fs * x_len)
        x = preprocess_input(x_input, fs)
        y = infer_seizure_probability(x, modelA)
        self.assertEqual(599, y.shape[1])
        if torch.cuda.is_available():
            modelA.cuda(0)
            y = infer_seizure_probability(x, modelA, True, 0)
            self.assertEquals(599, y.shape[1])



if __name__ == '__main__':
    unittest.main()
