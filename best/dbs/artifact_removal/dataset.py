import torch
import numpy as np
import scipy.signal as signal
from best.signal_generation.DCGAN import models_DCGAN
from best.dbs import ArtifactGenerator

from mef_tools.io import MefReader
from best.signal import buffer, get_datarate

class RCSDataset:
    def __init__(self, path='/mnt/Helium/filip/Projects/2020_Sleep_Analysis/2022_sleep_architecture_only/M1/M1_nostim_1575446160.0/M1_nostim_1575446160_250Hz_filtered_e0-e3_e12-e13_e4-e5_e8-e11.mefd'):
        Rdr = MefReader(path)
        data = []
        for ch in Rdr.channels:
            if not 'cc' in ch:
                x = Rdr.get_data(ch)
                xb = buffer(x, 250, 3)
                dr = get_datarate(xb)
                xb = xb[dr > 0.95]
                xb[np.isnan(xb)] = np.nanmean(xb)
                xb500 = signal.resample(xb, xb.shape[1]*2, axis=1)
                data += [xb500]

        data = np.concatenate(data, 0)
        self.data = (data - data.mean(1).reshape(-1, 1)) / data.std(1).reshape(-1, 1)

        self._len = 12800000

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        idx = np.random.randint(self.data.shape[0])
        return self.data[idx]

class StimArtifactDataset:
    def __init__(self, sig_len=60, use_models=['MultiCenteriEEG_pathology', 'MultiCenteriEEG_physiology'], fs=500, use_artifacts=['RCS'], device='cpu'):
        self._device = device
        # print(device)
        # print(device)
        # print(device)
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
        # X_orig = X_orig - np.nanmean(X_orig)

        if item % 2 == 0:
            X_art, Y_art = self.ArtGenerators[idx_art_gen].get_signal_random(n_batch=1, n_length=X_orig.shape[-1])
        else:
            f = 10 ** (np.random.rand()*(1.65-0.3)+0.3) # 2 - 45 Hz
            a = 10 ** (np.random.rand()*(1--0.3)+-0.3) # 0.01 - 10
            X_art, Y_art = self.ArtGenerators[idx_art_gen].get_signal_periodic(n_batch=1, n_length=X_orig.shape[-1], freq=f, ampl_rel=a)

        # b, a = signal.butter(np.random.randint(1,3), np.random.rand()*80 + 100, fs=self.fs)
        # X_art = signal.filtfilt(b, a, X_art).copy()

        X_art = torch.tensor(X_art, dtype=torch.float32).squeeze(0)
        Y_art = torch.tensor(Y_art, dtype=torch.float32).squeeze(0)
        X_art = X_orig + X_art

        X_orig = X_orig - np.nanmean(X_art)
        X_art = X_art - np.nanmean(X_art)

        if self.fs == 250:
            return X_orig[::2], X_art[::2], Y_art[::]

        return X_orig, X_art, Y_art

    def __len__(self):
        return self._len

class StimArtifactDataset_RCS:
    def __init__(self, sig_len=60, use_models=[], fs=500, use_artifacts=['RCS'], device='cpu'):
        self._device = device
        # print(device)
        # print(device)
        # print(device)
        # print(device)
        # self.SigGenerators = [models_DCGAN[k].to(device) for k in use_models]
        self.SigDat = RCSDataset()
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
        # idx_sig_gen = item % self.SigGenerators.__len__()
        idx_art_gen = item % self.ArtGenerators.__len__()

        X_orig = torch.tensor(self.SigDat[item].reshape(1, -1).copy(), device=self._device).float()
        #self.SigGenerators[idx_sig_gen].generate_signal(n_batch=1, n_seconds=self.sig_len, momentum=0.1).squeeze(0)
        X_orig = X_orig - np.nanmean(X_orig)


        if item % 2 == 0:
            X_art, Y_art = self.ArtGenerators[idx_art_gen].get_signal_random(n_batch=1, n_length=X_orig.shape[-1])
        else:
            f = 10 ** (np.random.rand()*(1.2-0.3)+0.3) # 2 - 45 Hz
            a = 10 ** (np.random.rand()*(1--0.3)+-0.3) # 0.01 - 10
            X_art, Y_art = self.ArtGenerators[idx_art_gen].get_signal_periodic(n_batch=1, n_length=X_orig.shape[-1], freq=f, ampl_rel=a)

        X_art = torch.tensor(X_art, dtype=torch.float32).squeeze(0)
        Y_art = torch.tensor(Y_art, dtype=torch.float32).squeeze(0)
        X_art = X_orig + X_art
        X_art = X_art - np.nanmean(X_art)
        if self.fs == 250:
            return X_orig[:, ::2], X_art[:, ::2], Y_art[:, ::2]
        #
        # b, a = signal.butter(np.random.randint(1,7), np.random.rand()*80 + 100, fs=self.fs)
        # X_orig = signal.filtfilt(b, a, X_orig)
        # X_art = signal.filtfilt(b, a, X_art)

        return X_orig.copy(), X_art.copy(), Y_art

    def __len__(self):
        return self._len




