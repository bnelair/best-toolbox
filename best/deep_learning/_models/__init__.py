# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from best import DELIMITER
from best._config import get_files
from torch import load, nn, zeros, unsqueeze, squeeze
import torch.nn.functional as F


TRAINED_MODELS = {'modelA': 'modelA_paper.pt', 'modelB': 'modelB_full.pt'}


def load_trained_model(model_name):
    if model_name in TRAINED_MODELS.keys():
        return _ModelsSeizureDetect.load_model(TRAINED_MODELS[model_name])
    else:
        raise KeyError(f"unknown trained model {model_name}; available {TRAINED_MODELS.keys()}")


class SeizureDetectModel(nn.Module):

    def __init__(self):
        super().__init__()
        # LSTM layer
        self.lstm_out = 100
        # kernel size
        self.nkernel = 20
        self.conv1 = nn.Conv2d(1, self.nkernel, (5, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(self.nkernel, self.lstm_out*2, (96, 3), padding=(0, 1))
        self.lstm = nn.LSTM(self.lstm_out*2, self.lstm_out, 2, bidirectional=True, dropout=0.5)
        self.fc1 = nn.Linear(2 * self.lstm_out, 4)

    def forward(self, x):
        bs = x.shape[0]
        h0 = zeros(2 * 2, bs, self.lstm_out).to(x.device)
        c0 = zeros(2 * 2, bs, self.lstm_out).to(x.device)
        x = unsqueeze(x, 1)
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = squeeze(x)
        if bs == 1:
            x = unsqueeze(x, 0)
        x = x.permute(2, 0, 1)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        return x, F.softmax(x, dim=2)


class _ModelsSeizureDetect(dict):
    _keys = dict([
        (
            '_'.join(f.split(DELIMITER)[-1].split('_')[:2]),
            f
        )
        for f in get_files(DELIMITER.join(__file__.split(DELIMITER)[:-1]), 'pt')
        if not '.-' in f.split(DELIMITER)[-1]
    ])

    def keys(self):
        return self._keys.keys()

    def get_model(self, item):
        f = self._keys[item]
        state_dict = load(f, map_location='cpu')
        mod = SeizureDetectModel()
        mod.load_state_dict(state_dict, strict=True)
        mod.eval()
        return mod

    @classmethod
    def load_model(cls, model_name):
        f = cls._keys[model_name]
        state_dict = load(f, map_location='cpu')
        mod = SeizureDetectModel()
        mod.load_state_dict(state_dict, strict=True)
        mod.eval()
        return mod

    def __getitem__(self, item):
        return self.get_model(item)

    def __call__(self, item):
        return self[item]

    def __str__(self):
        return 'models ' + str(self._keys.keys())

    def __repr__(self):
        return self.__str__()