import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from best.deep_learning.models_generic import ResLayer1D

from best.deep_learning import ResLayer1D

class dbs_artifact_removal_network_(nn.Module):
    def __init__(self, n_filters=64, fs=500):
        super().__init__()
        self.dummy_par = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=1, padding=5, bias=False)
        self.gru_1 = nn.GRU(32, hidden_size=64, num_layers=1, bidirectional=False, bias=False, batch_first=True)
        self.conv1_3 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=1, padding=5, bias=False)
        self.gru_encode = nn.GRU(32, hidden_size=64, num_layers=1, bidirectional=True, bias=False, batch_first=True)

        self.gru_decode = nn.GRU(32, hidden_size=64, num_layers=1, bidirectional=True, bias=False, batch_first=True)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.resl3_1 = ResLayer1D(32, bias=False)
        self.convoutp = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=11, stride=1, padding=5, bias=False)


    # def eval(self):
    #     super().eval()
    #     #self._opt = optim.Adam(self.parameters(), lr=0)
    #     self._loss = nn.MSELoss()
    #     return self

    def forward(self, x_inp):
        # x_inp = x_art

        x = x_inp
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x, tmp1 = self.gru_1(x.permute(0, 2, 1), torch.zeros(1, x_inp.shape[0], 64, dtype=torch.float, device=self.device))
        x = x.permute(0, 2, 1)
        x = self.conv1_3(x)
        # xatt =

        x_ = x_inp*(1-torch.relu(torch.sign(x)))
        # x_ = x_inp * (1 - torch.sigmoid(x))
        x_ = self.conv1(x_)
        x_ = F.relu(x_)
        x_ = self.conv1_2(x_)
        x_ = F.relu(x_)
        _, xemb = self.gru_encode(x_.permute(0, 2, 1), torch.zeros(2, x_inp.shape[0], 64, dtype=torch.float, device=self.device))
        x_dec, tmp2 = self.gru_decode(x_.permute(0, 2, 1), xemb)

        x_ = self.conv3(x_dec.permute(0,2,1))
        x_ = F.relu(x_)
        x_ = self.resl3_1(x_)
        x_outp = self.convoutp(x_)
        return x_outp, torch.sigmoid(x)# + x_outp

    @property
    def device(self):
        return self.dummy_par.device

class dbs_artifact_removal_network_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_par = nn.Parameter(torch.zeros(1))

        self.conv1_1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.conv1_2 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=6, dilation=3, bias=False)
        self.conv1_3 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=10, dilation=5, bias=False)
        self.conv1_4 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=20, dilation=10, bias=False)

        self.Res1 = ResLayer1D(64, 11, 5, bias=False)

        # self.gru_attention = nn.GRU(64, hidden_size=64, num_layers=1, bidirectional=False, bias=False, batch_first=True)
        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        # self.linear1 = nn.Linear(64, 64)
        # self.linear2 = nn.Linear(64, 64)
        # self.linear3 = nn.Linear(64, 64)

        self.lin1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.lin2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.lin3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.linear_logvar = nn.Linear(64, 1)
        self.linear_outp = nn.Linear(64, 1)

        # self.gru_decode = nn.GRU(64, hidden_size=64, num_layers=1, bidirectional=True, bias=False, batch_first=True)
        self.ResDecode = ResLayer1D(64, 11, 5, bias=False)

        self.conv3_1 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
        self.conv3_2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=6, dilation=3, bias=False)
        self.conv3_3 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=10, dilation=5, bias=False)
        self.conv3_4 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=20, dilation=10, bias=False)


    def forward(self, x_inp):
        x = x_inp
        x = torch.cat(
            (
                self.conv1_1(x),
                self.conv1_2(x),
                self.conv1_3(x),
                self.conv1_4(x),
            ), 1
        )
        x = F.relu(x)
        x = self.Res1(x)

        # x_att, tmp1 = self.gru_attention(x.permute(0, 2, 1), torch.zeros(1, x_inp.shape[0], 64, dtype=torch.float, device=self.device))
        # x_att = x_att.permute(0, 2, 1)
        x_att = self.conv2_1(x)
        #x_att_outp = torch.sigmoid(self.conv2_2(x))

        logvar = self.linear_logvar(x_att.permute(0,2,1)).permute(0,2,1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(x)
        sample = (eps * std)

        x = \
            F.relu(self.lin1(x))\
            - F.relu(self.lin2((x_att * x))) \
            #+ F.relu(self.lin3((x_att * sample)))

        x_att_outp = torch.sigmoid(self.linear_outp(x_att.permute(0,2,1))).permute(0,2,1)


        # x = x.permute(0, 2, 1)

        # x_dec, tmp2 = self.gru_decode(x.permute(0, 2, 1))
        x_dec = x
        x_dec = self.ResDecode(x)

        x_dec = torch.cat(
            (
                self.conv3_1(x_dec[:, 0:16, :]),
                self.conv3_2(x_dec[:, 16:32, :]),
                self.conv3_3(x_dec[:, 32:48, :]),
                self.conv3_4(x_dec[:, 48:64, :]),
            ), 1
        ).sum(1).unsqueeze(1)

        return x_dec, x_att_outp

    @property
    def device(self):
        return self.dummy_par.device


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



#
# class dbs_artifact_removal_network_(nn.Module):
#     def __init__(self, n_filters=64, fs=500):
#         super().__init__()
#         self.dummy_par = nn.Parameter(torch.zeros(1))
#
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
#         if fs==500:
#             self.conv1_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, stride=2, padding=5, bias=False)
#         elif fs==250:
#             self.conv1_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
#
#         # self.resl1_1 = ResLayer1D(32, kernel_size=21, padding=10, bias=False)
#         # self.conv1_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5, bias=False)
#         self.resl1_2 = ResLayer1D(32, kernel_size=3, padding=1, bias=False)
#         # self.conv1_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=20, bias=False)
#         # self.resl1_3 = ResLayer1D(32, kernel_size=3, padding=1, bias=False)
#
#         self.gru_1 = nn.GRU(32, hidden_size=64, num_layers=1, bidirectional=False, bias=False, batch_first=True)
#         self.gru_2 = nn.GRU(64, hidden_size=64, num_layers=1, bidirectional=True, bias=False, batch_first=True)
#
#         self.gru_attention = nn.GRU(32, hidden_size=64, num_layers=1, bidirectional=False, bias=False, batch_first=True)
#
#         self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
#         # self.resl2_1 = ResLayer1D(32, bias=False)
#         # self.resl2_2 = ResLayer1D(32, bias=False)
#         # self.resl2_3 = ResLayer1D(32, bias=False)
#
#         if fs == 500:
#             self.convoutp = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=12, stride=2, padding=5, bias=False)
#         elif fs==250:
#             self.convoutp = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=11, stride=1, padding=5, bias=False)
#
#
#         # self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
#         # self.resl3_1 = ResLayer1D(64, bias=False)
#         # self.resl3_2 = ResLayer1D(32, bias=False)
#         # self.resl3_3 = ResLayer1D(32, bias=False)
#         # self.convoutp_att = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)
#
#         if fs == 500:
#             self.convoutp_att = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=12, stride=2, padding=5, bias=False)
#         elif fs==250:
#             self.convoutp_att = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=11, stride=1, padding=5, bias=False)
#
#         # self.convfilter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=11, stride=1, padding=5, bias=False)
#
#
#         #self.batch_outpatt = nn.BatchNorm1d(1)
#
#
#     # def eval(self):
#     #     super().eval()
#     #     #self._opt = optim.Adam(self.parameters(), lr=0)
#     #     self._loss = nn.MSELoss()
#     #     return self
#
#
#     def forward(self, x_inp):
#         # x_inp = x_art
#
#         x = x_inp
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv1_2(x)
#         x = F.relu(x)
#
#         # x = self.resl1_1(x)
#         x_features = self.resl1_2(x)
#         # x_features = self.resl1_3(x)
#
#         x, tmp1 = self.gru_1(x_features.permute(0, 2, 1), torch.zeros(1, x_inp.shape[0], 64, dtype=torch.float, device=self.device))
#         x = x.permute(0, 2, 1)
#
#         x_att, tmp2 = self.gru_attention(x_features.permute(0, 2, 1), torch.zeros(1, x_inp.shape[0], 64, dtype=torch.float, device=self.device))
#         x_att = x_att.permute(0, 2, 1)
#         x_att = torch.sigmoid(x_att)
#
#         x_ = x - (x * x_att) + ((1-x_att) * torch.randn_like(x_att))
#
#         x_, tmp1 = self.gru_2(x_.permute(0, 2, 1), torch.zeros(2, x_inp.shape[0], 64, dtype=torch.float, device=self.device))
#         x_ = x_.permute(0, 2, 1)
#         #x_ = x - (x * x_att) + (x_att * torch.randn_like(x))
#         # x_ = x - x_att# + (torch.randn_like(x_att))
#
#         x_ = self.conv2(x_)
#         x_ = F.relu(x_)
#         # x_ = self.resl2_1(x_)
#         x_outp = self.convoutp(x_)
#
#         # x_att = self.conv3(x_att)
#         # x_att = F.relu(x_att)
#         # x_att = self.resl3_1(x_att)
#         x_att = self.convoutp_att(x_att)
#         x_att = torch.sigmoid(x_att)
#         # x_att = self.convoutp_att(x_att)
#
#         if not self.training:
#             del x, x_features, tmp1, tmp2, x_
#
#         return x_outp, x_att# + x_outp
#
#     @property
#     def device(self):
#         return self.dummy_par.device
#


#
# class dbs_artifact_removal_network_light(nn.Module):
#     def __init__(self, n_filters=64, fs=500):
#         super().__init__()
#         self.dummy_par = nn.Parameter(torch.zeros(1))
#
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=11, stride=2, padding=5, bias=False)
#
#         self.gru = nn.GRU(n_filters, hidden_size=64, num_layers=1, bidirectional=False, bias=False, batch_first=True)
#
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=n_filters, kernel_size=5, stride=1, padding=2, bias=False)
#         self.convoutp = nn.ConvTranspose1d(in_channels=n_filters, out_channels=1, kernel_size=12, stride=2, padding=5, bias=False)
#         self.convfilter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=11, stride=1, padding=5, bias=False)
#
#
#     # def eval(self):
#     #     super().eval()
#     #     #self._opt = optim.Adam(self.parameters(), lr=0)
#     #     self._loss = nn.MSELoss()
#     #     return self
#
#     def forward(self, x_inp):
#         # x_inp = x_art
#
#         x = x_inp
#         x = self.conv1(x)
#         x_features = F.relu(x)
#
#         x, tmp1 = self.gru(x_features.permute(0, 2, 1), torch.zeros(1, x_inp.shape[0], 64, dtype=torch.float, device=self.device))
#         x = x.permute(0, 2, 1)
#
#         x_ = self.conv2(x)
#         x_ = F.leaky_relu(x_, 0.2)
#         x_outp = self.convoutp(x_)
#         x_outp = self.convfilter(x_outp)
#
#
#         if not self.training:
#             del x, x_features, tmp1, x_
#
#         return x_outp
#
#     @property
#     def device(self):
#         return self.dummy_par.device
