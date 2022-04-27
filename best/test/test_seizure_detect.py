from unittest import TestCase
import unittest

import torch.cuda

from best.deep_learning.seizure_detect import load_trained_model, preprocess_input, infer_seizure_probability
from numpy.random import rand


class Test(TestCase):
    def test_load_trained_model(self):
        modelA = load_trained_model('modelA')
        modelB = load_trained_model('modelB')

    def test_preprocess_input(self):
        modelA = load_trained_model('modelA')
        fs = 500
        x_len = 300
        channels = 3
        x_input = rand(channels, fs * x_len)
        x = preprocess_input(x_input, fs)
        y = infer_seizure_probability(x, modelA)
        self.assertEquals(599, y.shape[1])
        if torch.cuda.is_available():
            modelA.cuda(0)
            y = infer_seizure_probability(x, modelA, True, 0)
            self.assertEquals(599, y.shape[1])
        a = 1


if __name__ == '__main__':
    unittest.main()



