import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from best.deep_learning import ResLayer1D


class dbs_artifact_removal_network(nn.Module):
    def __init__(self, n_filters=64, fs=500):
        super().__init__()
        self.dummy_par = nn.Parameter(torch.zeros(1))

        if fs==500:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=11, stride=2, padding=5, bias=False)
        elif fs==250:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=5, stride=1, padding=2, bias=False)

        self.gru = nn.GRU(n_filters, hidden_size=128, num_layers=1, bidirectional=False, bias=False, batch_first=True)
        self.gru_attention = nn.GRU(n_filters, hidden_size=64, num_layers=1, bidirectional=False, bias=False, batch_first=True)

        self.conv2 = nn.Conv1d(in_channels=128+64, out_channels=n_filters, kernel_size=5, stride=1, padding=2, bias=False)
        self.resl2 = ResLayer1D(n_filters, bias=False)

        self.convoutp = nn.ConvTranspose1d(in_channels=n_filters, out_channels=1, kernel_size=12, stride=2, padding=5, bias=False)


        self.convfilter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=11, stride=1, padding=5, bias=False)

        #self.batch_outpatt = nn.BatchNorm1d(1)
        if fs == 500:
            self.convoutp_att = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=12, stride=2, padding=5, bias=False)
        elif fs==250:
            self.convoutp_att = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=6, stride=2, padding=2, bias=False)
    # def eval(self):
    #     super().eval()
    #     #self._opt = optim.Adam(self.parameters(), lr=0)
    #     self._loss = nn.MSELoss()
    #     return self


    def forward(self, x_inp):
        # x_inp = x_art

        x = x_inp
        x = self.conv1(x)
        x_features = F.relu(x)

        x, tmp1 = self.gru(x_features.permute(0, 2, 1), torch.zeros(1, x_inp.shape[0], 128, dtype=torch.float, device=self.device))
        x = x.permute(0, 2, 1)

        x_att, tmp2 = self.gru_attention(x_features.permute(0, 2, 1), torch.zeros(1, x_inp.shape[0], 64, dtype=torch.float, device=self.device))
        x_att = x_att.permute(0, 2, 1)

        x_ = torch.cat((x, x_att), dim=1)


        x_ = self.conv2(x_)
        x_ = F.relu(x_)
        x_ = self.resl2(x_)
        x_outp = self.convoutp(x_)
        x_outp = self.convfilter(x_outp)

        x_att = torch.sigmoid(self.convoutp_att(x_att))

        if not self.training:
            del x, x_features, tmp1, tmp2, x_


        return x_outp, x_att

    @property
    def device(self):
        return self.dummy_par.device


class dbs_artifact_removal_network_light(nn.Module):
    def __init__(self, n_filters=64, fs=500):
        super().__init__()
        self.dummy_par = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=11, stride=2, padding=5, bias=False)

        self.gru = nn.GRU(n_filters, hidden_size=64, num_layers=1, bidirectional=False, bias=False, batch_first=True)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=n_filters, kernel_size=5, stride=1, padding=2, bias=False)
        self.convoutp = nn.ConvTranspose1d(in_channels=n_filters, out_channels=1, kernel_size=12, stride=2, padding=5, bias=False)
        self.convfilter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=11, stride=1, padding=5, bias=False)


    # def eval(self):
    #     super().eval()
    #     #self._opt = optim.Adam(self.parameters(), lr=0)
    #     self._loss = nn.MSELoss()
    #     return self

    def forward(self, x_inp):
        # x_inp = x_art

        x = x_inp
        x = self.conv1(x)
        x_features = F.relu(x)

        x, tmp1 = self.gru(x_features.permute(0, 2, 1), torch.zeros(1, x_inp.shape[0], 64, dtype=torch.float, device=self.device))
        x = x.permute(0, 2, 1)

        x_ = self.conv2(x)
        x_ = F.leaky_relu(x_, 0.2)
        x_outp = self.convoutp(x_)
        x_outp = self.convfilter(x_outp)


        if not self.training:
            del x, x_features, tmp1, x_

        return x_outp

    @property
    def device(self):
        return self.dummy_par.device
