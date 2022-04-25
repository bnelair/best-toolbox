# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import unittest
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from scipy.io import loadmat


class SingleDataset_ThreeSeconds:
    def __init__(self, path, categories_keep='', categories_remove=''):
        self.path = path
        self.df = pd.read_csv(os.path.join(path, 'segments.csv'))

        self.categories_keep = categories_keep
        self.categories_remove = categories_remove


        if categories_keep:
            bool_vect = None
            for c in self.categories_keep:
                if isinstance(bool_vect, type(None)):
                    bool_vect = self.df.category_name == c
                else:
                    bool_vect = bool_vect | (self.df.category_name == c)
            self.df = self.df.loc[bool_vect].reset_index(drop=True)

        if categories_remove:
            bool_vect = None
            for c in self.categories_keep:
                if isinstance(bool_vect, type(None)):
                    bool_vect = self.df.category_name != c
                else:
                    bool_vect = (bool_vect) & (self.df.category_name != c)
            self.df = self.df.loc[bool_vect].reset_index(drop=True)


        #self.df = self.df.loc[self.df.category_name == 'pathology'].reset_index(drop=True)

        self.categories = self.df.category_name.unique()

        self.b, self.a = signal.butter(3, 200, analog=False, output='ba', fs=5000)
        self.b_low, self.a_low = signal.butter(3, 0.5, analog=False, output='ba', fs=500, btype='high')


    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, item):
        x = loadmat(os.path.join(self.path, self.df.iloc[item]['segment_id']+'.mat'))['data'].squeeze()

        x[np.isnan(x)] = 0

        x = signal.filtfilt(self.b, self.a, x)
        x = x[::10]
        x = signal.filtfilt(self.b_low, self.a_low, x)

        ystr = self.df.iloc[item].category_name
        y = int(np.where(self.categories==self.df.iloc[item].category_name)[0])
        return x.copy(), y, ystr


class MergeDataset_ThreeSeconds:
    def __init__(self, path, categories_keep='', categories_remove=''):
        if type(path) != list:
            path = [path]

        self.categories_keep = categories_keep
        self.categories_remove = categories_remove

        self.Dats = [SingleDataset_ThreeSeconds(p, self.categories_keep, self.categories_remove) for p in path]
        self.lens = [d.__len__() for d in self.Dats]
        self.categories = np.unique(np.concatenate([d.categories for d in self.Dats]))

    def __len__(self):
        return sum(self.lens)


    def __getitem__(self, item):
        didx = 0 # dataset index

        if item < self.__len__():
            while True:
                if item >= self.lens[didx]:
                    item -= self.lens[didx]
                    didx += 1
                else:
                    x, _, ystr = self.Dats[didx][item]
                    y = int(np.where(self.categories==ystr)[0])
                    return x, y, ystr






