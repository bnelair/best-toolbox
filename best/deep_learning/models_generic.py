# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResLayer1D(nn.Module):
    def __init__(self, n=16, kernel_size=3, padding=1, bias=True):
        super(ResLayer1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n, out_channels=n, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.conv2 = nn.Conv1d(in_channels=n, out_channels=n, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)

    def forward(self, x):
        xe = self.conv1(x)
        xe = F.relu(xe, inplace=False)
        xe = self.conv2(xe)
        xe = F.relu(xe + x)
        return xe


class ResLayer2D(nn.Module):
    def __init__(self, n=16, kernel_size=3, padding=1, bias=True):
        super(ResLayer2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)

    def forward(self, x):
        xe = self.conv1(x)
        xe = F.relu(xe, inplace=False)
        xe = self.conv2(xe)
        xe = F.relu(xe + x, inplace=False)
        return xe


class Attention1D_MIL(nn.Module):
    def __init__(self, inp_size, downsample=2):
        super(Attention1D_MIL, self).__init__()

        self.conv_tanh = nn.Conv1d(in_channels=inp_size, out_channels=inp_size, kernel_size=1)
        self.conv_sigm = nn.Conv1d(in_channels=inp_size, out_channels=inp_size, kernel_size=1)
        self.conv_w = nn.Conv1d(in_channels=inp_size, out_channels=1, kernel_size=1)
        self.conv_final = nn.Conv1d(in_channels=inp_size, out_channels=1, kernel_size=1, stride=1)
        self.conv_donwsample = nn.Conv1d(in_channels=inp_size, out_channels=inp_size, kernel_size=downsample, stride=downsample, groups=inp_size, bias=False)

    def forward(self, x):
        tanh = torch.tanh(self.conv_tanh(x))
        sigm = torch.sigmoid(self.conv_sigm(x))
        z = self.conv_w(tanh * sigm)
        att = self.conv_final(x) * z
        x_weighted = att * x
        return self.conv_donwsample(x_weighted)#, att

