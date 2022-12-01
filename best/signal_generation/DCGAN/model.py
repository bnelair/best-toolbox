# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn


from best.deep_learning import ResLayer1D


class Generator(nn.Module):
    def __init__(self, n_features=32, n_filters=64):
        super(Generator, self).__init__()
        self.n_features = n_features
        self.n_filters = n_filters
        self.dummy_param = nn.Parameter(torch.zeros(1))

        self.main = nn.Sequential(
            # input (batch_size, n_features, n_seconds)
            nn.ConvTranspose1d(in_channels=self.n_features, out_channels=self.n_filters, kernel_size=5, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(self.n_filters),
            nn.ReLU(True),

            # input (batch_size, n_filters, n_seconds*5)
            nn.ConvTranspose1d(in_channels=self.n_filters, out_channels=self.n_filters, kernel_size=20, stride=10, padding=5, bias=False),
            nn.BatchNorm1d(self.n_filters),
            nn.ReLU(True),

            # input (batch_size, n_filters, n_seconds*50)
            nn.ConvTranspose1d(in_channels=self.n_filters, out_channels=self.n_filters, kernel_size=20, stride=10, padding=5, bias=False),
            nn.BatchNorm1d(self.n_filters),
            nn.ReLU(True),

            # input (batch_size, n_filters, n_seconds*500)
            ResLayer1D(self.n_filters, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(self.n_filters),
            nn.ReLU(True),

            # input (batch_size, n_filters, n_seconds*500)
            nn.ConvTranspose1d(in_channels=self.n_filters, out_channels=1, kernel_size=11, stride=1, padding=5, bias=False), # this was formerly 3
            # output (batch_size, 1, n_seconds*500)
            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=21, stride=1, padding=10, bias=False),  # this was formerly 3
        )

    def forward(self, input):
        return self.main(input)

    def generate_signal(self, n_batch=1, n_seconds=1, momentum=0.5):
        noise = self._generate_noise(n_batch, n_seconds, momentum)
        x = self(noise).detach()
        x.detach().requires_grad = False
        return x

    def _generate_noise(self, n_batch=1, n_seconds=3, momentum=0.5):
        noise = [torch.randn(n_batch, self.n_features, 1, device=self.dummy_param.device)]
        # noise = [torch.randn(n_batch, 32, 1, device=self.dummy_param.device) * 0.10 - 5]
        for n in range(n_seconds - 1):
            noise += [
                momentum * noise[-1] + (1-momentum) * torch.randn(n_batch, self.n_features, 1, device=self.dummy_param.device)]
        noise = torch.cat(noise, dim=2)
        return noise


class Discriminator(nn.Module):
    def __init__(self, n_filters):
        super(Discriminator, self).__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.main = nn.Sequential(
            # input (batch_size, 1, n_seconds*500)
            nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*500)
            ResLayer1D(n_filters, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*500)
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=20, stride=10, padding=5, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*50)
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=20, stride=10, padding=5, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*5)
            nn.Conv1d(in_channels=n_filters, out_channels=1, kernel_size=5, stride=5, padding=0, bias=False),
            nn.Sigmoid()
            # output (batch_size, n_filters, n_seconds)
        )

    def forward(self, input):
        return self.main(input)


class DiscriminatorTimeFreq(nn.Module):
    def __init__(self, n_filters):
        super(DiscriminatorTimeFreq, self).__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.main_time = nn.Sequential(
            # input (batch_size, 1, n_seconds*500)
            nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*500)
            ResLayer1D(n_filters, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*500)
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=20, stride=10, padding=5, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*50)
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=20, stride=10, padding=5, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*5)
            nn.Conv1d(in_channels=n_filters, out_channels=1, kernel_size=15, stride=5, padding=0, bias=False),
            nn.Sigmoid()
            # output (batch_size, n_filters, n_seconds)
        )

        self.main_freq = nn.Sequential(
            # input (batch_size, 1, n_seconds*500)
            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=21, stride=1, padding=10, bias=False),
            nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*500)
            ResLayer1D(n_filters, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*500)
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=20, stride=10, padding=5, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*50)
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=20, stride=10, padding=5, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # input (batch_size, n_filters, n_seconds*5)
            nn.Conv1d(in_channels=n_filters, out_channels=1, kernel_size=15, stride=5, padding=0, bias=False),
            nn.Sigmoid()
            # output (batch_size, n_filters, n_seconds)
        )
        #
        # self.main_freq = nn.Sequential(
        #     nn.Linear(750, 100, bias=False),
        #     nn.BatchNorm1d(100),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Linear(100, 100, bias=False),
        #     nn.BatchNorm1d(100),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Linear(100, 1, bias=False),
        #     nn.BatchNorm1d(1),
        #     nn.Sigmoid()
        # )

        # self.main_merged = nn.Sequential(
        #     nn.Linear(200, 100, bias=False),
        #     nn.BatchNorm1d(100),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Linear(100, 1, bias=False),
        #     nn.BatchNorm1d(1),
        #     nn.Sigmoid()
        # )


    def forward(self, input):
        x_time = input[0]
        x_freq = input[1]
        return self.main_time(x_time).squeeze(), self.main_freq(x_freq).squeeze()
        # x_time_ = self.main_time(x_time).squeeze()
        # x_freq_ = self.main_freq(x_freq.squeeze())

        # x_ = torch.cat((x_time_, x_freq_), -1)
        # return self.main_merged(x_)



