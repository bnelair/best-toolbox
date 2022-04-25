import torch
import numpy as np
from best.signal_generation.DCGAN import models_DCGAN
from best.dbs import ArtifactGenerator

class StimArtifactDataset:
    def __init__(self, sig_len=60, use_models=['MultiCenteriEEG_pathology', 'MultiCenteriEEG_physiology'], fs=500, use_artifacts=['RCS'], device='cpu'):
        self._device = device
        print(device)
        print(device)
        print(device)
        print(device)
        self.SigGenerators = [models_DCGAN[k].to(device) for k in use_models]
        self.ArtGenerators = [ArtifactGenerator(k, 500) for k in use_artifacts]
        self._len = 1000000
        self.sig_len = sig_len
        self.fs = fs


    def to(self, device):
        self.SigGenerators = [sg.to(device) for sg in self.SigGenerators]

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self.to(device)

    def __getitem__(self, item):
        idx_sig_gen = item % self.SigGenerators.__len__()
        idx_art_gen = item % self.ArtGenerators.__len__()

        X_orig = self.SigGenerators[idx_sig_gen].generate_signal(n_batch=1, n_seconds=self.sig_len, momentum=0.1).squeeze(0)

        if item % 1 == 0:
            X_art, Y_art = self.ArtGenerators[idx_art_gen].get_signal_random(n_batch=1, n_length=X_orig.shape[-1])
        else:
            f = 10 ** (np.random.rand()*(1.6-0.3)+0.3) # 2 - 40 Hz
            a = 10 ** (np.random.rand()*(1--2)+-2) # 0.01 - 10
            X_art, Y_art = self.ArtGenerators[idx_art_gen].get_signal_periodic(n_batch=1, n_length=X_orig.shape[-1], freq=f, ampl_rel=a)

        X_art = torch.tensor(X_art, dtype=torch.float32).squeeze(0)
        Y_art = torch.tensor(Y_art, dtype=torch.float32).squeeze(0)
        X_art = X_orig + X_art
        if self.fs == 250:
            return X_orig[:, ::2], X_art[:, ::2], Y_art[:, ::2]

        return X_orig, X_art, Y_art

    def __len__(self):
        return self._len




