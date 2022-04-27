# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.signal import spectrogram
from scipy.stats import zscore
from torch import from_numpy

from best.deep_learning._models import load_trained_model

"""
    Seizure detection module 
    available trained models - 'modelA', 'modelB'
    modelA is the model from the published work.
    modelB had extended training dataset. 
    Optimal input for the model is 300 second. It is recommended to use only middle part of the signal.
    
Example
^^^^^^^^^

.. code-block:: python
    # load model
    modelA =  best.deep_learning.seizure_detect.load_trained_model('modelA')
    # load data
    fs = 500
    x_len = 300
    channels = 3
    # create fake data
    x_input = rand(channels, fs * x_len)
    # preprocess; from raw data to spectrogram
    x = best.deep_learning.seizure_detect.preprocess_input(x_input, fs)
    # get seizure probability; model has 4 output classes, seizure probability is class 4.
    # output is in shape (batch_size, x_len * 2 - 1); probability value for every half-second
    y = best.deep_learning.seizure_detect.infer_seizure_probability(x, modelA)

Sources
^^^^^^^^^^^

The seizure detection and training of the model is described in add website.
"""


def preprocess_input(x, fs):
    """
    This function will calculate a spectrogram. The spectrogram has shape [batch_size, 100, len(x) in seconds * 2 - 1]
    :param x: raw data input in bach form [batch_size, n-samples]
    :type x: iterable
    :param fs: sampling rate of the input signal
    :return: batch of spectrograms from the input
    :rtype: np.array
    """
    x = np.array(x)
    ii, jj = x.shape
    if np.sum(np.isnan(x[0, :])) < 0.75 * jj:
        mu = np.nanmean(x, axis=1)
        std = np.nanstd(x, axis=1)
        x = (x - mu.reshape((ii, 1))) / std.reshape((ii, 1))
        x = np.nan_to_num(x)
        f, t, x = spectrogram(x, fs, nperseg=fs, noverlap=fs / 2, axis=1)
        x2 = x[:, :100, :]
        x = np.empty(x2.shape)
        for kk, xx in enumerate(x2):
            idx = np.sum(xx, axis=0) == 0
            xx[:, ~idx] = zscore(xx[:, ~idx], axis=1)
            x[kk, :, :] = xx
        return x

    else:
        raise ValueError("too many NaN values in input")


def infer_seizure_probability(x, model, use_cuda=False, cuda_number=0):
    """
    infers seizure probability for a given input x; recommended signal len is 300 seconds.
    :param x: output from preprocess_input function, should be in shape [batch_size, 100, time_in_half-seconds]
    :param model: loaded seizure model
    :param use_cuda: if true x is loaded to cuda with cuda_number
    :param cuda_number:
    :return: seizure probability for input x in shape [batch_size, x.shape[2]]
    :rtype: np.array
    """
    x = from_numpy(x)
    batch_size = x.shape[0]
    if use_cuda:
        x = x.float().cuda(cuda_number)
    else:
        x = x.float()
    outputs, probs = model(x)
    probs = probs[:, :, 3].data.cpu().numpy().flatten()
    y = probs.reshape(x.shape[2], batch_size).T
    return y


