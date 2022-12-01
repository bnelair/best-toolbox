# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from tqdm import tqdm

from best.files import create_folder, get_files, get_folders
from best.hypnogram.utils import time_to_timestamp
from best.cloud.mef import MefClient
from best import DELIMITER

from mef_tools.io import MefReader

from copy import deepcopy
from best.cloud.db import SystemStateLoader

from best.feature import augment_features, print_classification_scores, get_classification_scores
from best.feature_extraction.SpectralFeatures import mean_bands, mean_frequency, relative_bands
from best.dbs.artifact_removal import configs_ArtifactEraser, models_ArtifactEraser
from best.cloud.db import SessionFinder
from best.dbs.artifact_removal.trainer import *
from best.signal import get_datarate, buffer
from best.hypnogram.io import load_CyberPSG
import pandas as pd
from best.feature_extraction.FeatureExtractor import SleepSpectralFeatureExtractor_trial
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, roc_curve, f1_score, accuracy_score
from best.modules import ZScoreModule, PCAModule
from best.hypnogram.utils import merge_annotations



class ReturnArrayObject:
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key

    def __getitem__(self, item):
        # for d in self.parent.data:
        #     if d['channel_name'] == item:
        #         return d[self.key]

        if not isinstance(item, list):
            item = [item]

        dat = pd.DataFrame()
        for k, d in self.data.items():
            if d['channel_name'] in item:
                dat[k] = d['channel_name']
        return dat


    def __call__(self):
        return [d[self.key] for d in self.parent.data]


class EEGDataLoader:
    def __init__(self, pid, path_mef_files, path_annotations=None, db_id='UH3'):
        self.pid = pid
        self.db_id = db_id
        self.df_files = self.scan_mef_files(path_mef_files, pid)

        self.SSLoader = SystemStateLoader(self.db_id)

        if path_annotations:
            self.df_annotations = self.scan_hypnogram_annotations(path_annotations)
            self.df_annotations['path_mef'] = self.find_mef_sessions(self.df_annotations)

    def find_mef_sessions(self, df):
        print('Searching for sessions')
        # find a mef session for each annotation
        mef_files = []
        for idx, row in tqdm(list(df.iterrows())):
            s = row['start']
            e = row['end']

            file = self.df_files.loc[(s >= self.df_files.start) & (e <= self.df_files.end)]

            if file.__len__() == 0:
                pth = ''
            elif file.__len__() == 1:
                pth = file.iloc[0]['path']
            else: # if more sessions found, takes the one with more data
                pths = []
                drs = []

                for _, r in file.iterrows():
                    Rdr_ = MefReader(r['path'])
                    ch = [c for c in Rdr_.channels if not 'accl' in c][0]
                    x = Rdr_.get_data(ch, s * 1e6, e * 1e6)
                    pths += [r['path']]
                    drs += [1 - (np.isnan(x).sum() / x.shape[0])]
                    # ref_dr =
                pth = pths[np.argmax(drs)]
            mef_files += [pth]
        return mef_files

    def process_segment(self, s_process, e_process):
        process_segments = pd.DataFrame({'start': np.arange(s_process, e_process, 20 * 60)})
        process_segments['end'] = process_segments['start'] + 20 * 60
        files = self.find_mef_sessions(process_segments)
        process_segments['path_mef'] = files

        start = process_segments.start.min()
        end = process_segments.end.max()

        stim_info = self.SSLoader.load_stimulation_annotations(self.pid, start - 12 * 3600, end + 12 * 3600)
        if stim_info.__len__() == 0:
            stim_info = pd.DataFrame(
                [{'frequency': 0, 'amplitude': 0, 'start': start - 1 * 3600, 'end': end + 1 * 3600}])

        t = np.arange(stim_info.start.min(), stim_info.end.max(), 1)
        stim_ref = np.zeros_like(t) - 1

        for k, row_ in stim_info.iterrows():
            stim_ref[(t >= row_['start']) & (t < row_['end'])] = row_['frequency']



        process_segments_ = []
        for idx, row in process_segments.iterrows():

            # annot saying 0
            # -1 no annot
            # -2 annot saying NaN

            t_ = t[(t > row['start']) & (t < row['end'])]
            subsec = stim_ref[(t > row['start']) & (t < row['end'])]
            subsec[np.isnan(subsec)] = -2

            k_ = 0
            while k_ < subsec.__len__() - 1:
                start_freq = subsec[k_]
                start_time = t_[k_]
                while subsec[k_] == start_freq and k_ < subsec.__len__() - 1:
                    k_ += 1

                if start_freq == -1: start_freq = 0
                if start_freq == -2: start_freq = -1
                process_segments_ += [
                    {
                        'start': start_time,
                        'end': t_[k_],
                        'duration': t_[k_] - start_time,
                        'freq': start_freq,
                        'pid': self.pid,
                        'path_mef': row['path_mef']
                    }
                ]

        return pd.DataFrame(process_segments_)


    @staticmethod
    def scan_mef_files(path, pid=''):
        files_mef = get_files(path, 'mefd')

        df_files = []
        for pth in tqdm(files_mef):
            fid = pth.split(DELIMITER)[-1][:-5].split('_')
            do = True
            if pid:
                if fid[0] != pid:
                    do = False

            if do:
                Rdr = MefReader(pth)
                start = min(Rdr.get_property('start_time')) / 1e6
                end = max(Rdr.get_property('end_time')) / 1e6
                dur = (end - start)



                row = {
                    'pid': fid[0],
                    'fs': int(fid[2]),
                    'mode': fid[4],
                    'amplitude': float(fid[5][:-2]),
                    'start': start,
                    'end': end,
                    'duration': dur,
                    'artifacts_removed': bool(fid[7]),
                    'duration': dur,
                    'path': pth,
                    'channels': Rdr.channels
                }

                df_files += [row]
        df_files = pd.DataFrame(df_files)
        return df_files


    def scan_hypnogram_annotations(self, files_hyp):
        #files_hyp = [f for f in get_files(path, 'pseudogs_hypnogram.xml') if 'data_annotated' in f]

        df_annots = []
        for pth in tqdm(files_hyp):
            fid = pth.split(DELIMITER)[:-1]
            pid = fid[-4]

            files_mef = [f for f in get_files(DELIMITER.join(fid[:-1]), 'mefd') if
                         not 'features' in f.split(DELIMITER)[-1]]

            for fmef in files_mef:
                # print(fmef)
                Rdr = MefReader(fmef)
                fs = max(Rdr.get_property('fsamp'))
                channels = [ch for ch in Rdr.channels if not 'accel' in ch]

                # chans = np.unique(chans)

                mode = fid[8]

                if mode == 'no-stim':
                    mode = '0Hz'
                    amplitude = '0mA'

                elif 'sante' in mode:
                    amplitude = mode[-3:]
                    mode = 'sante145Hz'
                else:
                    amplitude = mode[-3:]
                    mode = mode[:-3]

                amplitude = float(amplitude[:-2])

                hyp = load_CyberPSG(pth)
                hyp = time_to_timestamp(hyp)
                hyp_start = hyp.start.min()
                hyp_end = hyp.end.max()

                row = {
                    'pid': fid[6],
                    'fs': fs,
                    'mode': mode,
                    'amplitude': amplitude,
                    'start': hyp_start,
                    'end': hyp_end,
                    'channels': channels,
                    'path': pth,
                }
                if not row in df_annots:
                    df_annots += [row]

        return pd.DataFrame(df_annots)


    def load_hypnogram_annotations(self):
        hyp_ = []
        for idx, row in self.df_annotations.iterrows():
            path_hyp = row['path']
            if path_hyp:
                hyp = load_CyberPSG(path_hyp)
                hyp = time_to_timestamp(hyp)
                hyp_start = hyp.start.min()
                hyp_end = hyp.end.max()

                stim_info = self.SSLoader.load_stimulation_annotations(self.pid, hyp_start - 12 * 3600, hyp_end + 12 * 3600)
                if stim_info.__len__() == 0:
                    stim_info = pd.DataFrame(
                        [{'frequency': 0, 'amplitude': 0, 'start': hyp_start - 1 * 3600, 'end': hyp_end + 1 * 3600}])

                t = np.arange(stim_info.start.min(), stim_info.end.max(), 1)
                stim_ref = np.zeros_like(t) - 1

                for k, row_ in stim_info.iterrows():
                    stim_ref[(t >= row_['start']) & (t < row_['end'])] = row_['frequency']

                # annot saying 0
                # -1 no annot
                # -2 annot saying NaN
                for k, row_hyp in hyp.iterrows():
                    t_ = t[(t > row_hyp['start']) & (t < row_hyp['end'])]
                    subsec = stim_ref[(t > row_hyp['start']) & (t < row_hyp['end'])]
                    subsec[np.isnan(subsec)] = -2

                    k_ = 0
                    while k_ < subsec.__len__() - 1:
                        start_freq = subsec[k_]
                        start_time = t_[k_]
                        while subsec[k_] == start_freq and k_ < subsec.__len__() - 1:
                            k_ += 1

                        if start_freq == -1: start_freq = 0
                        if start_freq == -2: start_freq = -1
                        hyp_ += [
                            {
                                'annotation': row_hyp['annotation'],
                                'start': start_time,
                                'end': t_[k_],
                                'duration': t_[k_] - start_time,
                                'pid': row['pid'],
                                'freq': start_freq,
                                'amplitude': row['amplitude'],
                                'pid': row['pid'],
                                'fs': row['fs'],
                                'channels': row['channels'],
                                'path_hypnogram': row['path'],
                                'path_mef': row['path_mef'],
                            }
                        ]

        hyp = pd.DataFrame(hyp_)
        return hyp









def compare_channels(x, channels):
    return sum([ch in channels for ch in x['channels']]) == channels.__len__()

def read_vaclav_table(path_csv):
    table = pd.read_csv(path_csv)
    table['DateTimeCompleted'] = table['DateTimeCompleted']

    time = []
    for v in table['DateTimeCompleted']:
        if not type(v) is str:
            v = None
        else:
            v = datetime.strptime(v, "%m/%d/%y %H:%M").timestamp()
        time += [v]
    table['DateTimeCompleted'] = time
    return table




class SignalSegment:
    def __init__(self, x=None, fs=None, channel_name=None, start=None, annotation=None):
        self.data = []
        if not isinstance(x, type(None)):
            self.add_channel(x, fs, channel_name, start, annotation)

        #self.fs = ReturnArrayObject(self, 'fs')
        #self.signal = ReturnArrayObject(self, 'data')
        #self.start = ReturnArrayObject(self, 'start')
        #self.end = ReturnArrayObject(self, 'end')
        #self.annotation = ReturnArrayObject(self, 'end')

    def add_channel(self, signal, fs, channel_name=None, start=None, annotation=''):
        if not channel_name:
            channel_name = 'channel_' + str(self.n_channels).zfill(3)

        if not start:
            start = 0

        end = start + (signal.shape[0] / fs)

        self.data += [{
            'channel_name': channel_name,
            'signal': signal,
            'fs': fs,
            'start': start,
            'end': end,
            'annotation': annotation
        }]


    @property
    def n_channels(self):
        return self.channels.__len__()

    @property
    def channels(self):
        return [d['channel_name'] for d in self.data]

    @property
    def annotation(self):
        kfunc = 'annotation'
        return self._get_attr(kfunc)

    @property
    def fs(self):
        kfunc = 'fs'
        return self._get_attr(kfunc)

    @property
    def start(self):
        kfunc = 'start'
        return self._get_attr(kfunc)

    @property
    def end(self):
        kfunc = 'end'
        return self._get_attr(kfunc)

    @property
    def signal(self):
        kfunc = 'signal'
        return self._get_attr(kfunc)


    def _get_attr(self, kfunc):
        dat = {}
        for d in self.data:
            if kfunc in d.keys():
                dat[d['channel_name']] = d[kfunc]
        return dat



class SignalRecording:
    def __init__(self):
        self.segments = []

    def add_segment(self, segment):
        self.segments += [segment]

    def __len__(self):
        return self.segments.__len__()

    def __getitem__(self, item):
        return self.segments[item]

    def extract_features(self, FtrExtractor):
        FtrRec = FeatureRecording()
        for idx in tqdm(range(self.__len__())):
            seg = self[idx]
            FtrSgm = FtrExtractor.extract_segment(seg)
            FtrRec.add_segment(FtrSgm)
        return FtrRec



class FeatureSegment:
    def __init__(self):
        self.data = {}
        self._datarate = {}
        self._feature_names = []

        self._must_be_keys = ['start', 'segment_duration', 'datarate']

    def add_channel(self, df_features, channel_name, feature_names):
        for k in self._must_be_keys:
            if not k in df_features.keys():
                raise KeyError(f"[KEY MISSING]: all this keys must be present in the df_features {self._must_be_keys}")

        self._feature_names = feature_names
        if not 'annotation' in df_features:
            df_features['annotation'] = ''
        self.data[channel_name] = df_features

    @property
    def datarate(self):
        dr = pd.DataFrame()
        for k, d in self.data.items():
            dr[k] = d['datarate']
        return dr

    @property
    def start(self):
        for k, d in self.data.items():
            return d['start']

    @property
    def segment_duration(self):
        for k, d in self.data.items():
            return d['segment_duration']

    @property
    def annotation(self):
        dr = pd.DataFrame()
        for k, d in self.data.items():
            dr[k] = d['annotation']
        return dr

    @property
    def n_channels(self):
        return self.data.keys().__len__()

    @property
    def channels(self):
        return list(self.data.keys())

    @property
    def feature_names(self):
        return self._feature_names

    def __getitem__(self, item):
        if isinstance(item, type(None)):
            item = self.channels
        if isinstance(item, str):
            item = [item]
        features = pd.DataFrame()
        features['start'] = self.start
        features['segment_duration'] = self.segment_duration
        features['datarate'] = self.datarate.min(axis=1)
        features['annotation'] = np.array([row.unique() for idx, row in self.annotation.iterrows()]).squeeze()

        feature_names = []
        for k, d in self.data.items():
            if k in item:
                for ftr_nm in self.feature_names:
                    fname = f"{k}-{ftr_nm}"
                    features[fname] = d[ftr_nm]
                    feature_names += [fname]
        return features, feature_names

    def select_channels(self, channels):
        new_self = deepcopy(self)
        new_self.data = dict([(ch, self.data[ch]) for ch in channels])
        return new_self


class FeatureRecording:
    def __init__(self):
        self.segments = []

    def add_segment(self, segment):
        self.segments += [segment]

    def __len__(self):
        return self.segments.__len__()

    def __getitem__(self, item):
        return self.segments[item]

    def __setitem__(self, key, value):
        self.segments[key] = value

    def __add__(self, other):
        self.segments += other.segments

    @property
    def channels(self):
        return list(np.unique( [self[k].channels for k in range(self.__len__()) ] ))

    def select_channels(self, channels):
        new_self = deepcopy(self)
        for k in range(self.__len__()):
            new_self[k] = new_self[k].select_channels(channels)
        return new_self


class PIBFeatureExtractor:
    def __init__(self,
                 fbands=[
                     [0.5, 5],  # delta
                     [4, 9],  # theta
                     [8, 14],  # alpha
                     [11, 16],  # spindle
                     [14, 20],
                     [20, 30]
                 ],
                 segment_size=30,
                 fs=250
                 ):

        self.fbands = fbands
        self.segment_size = segment_size
        self.fs = fs

        self.FeatureExtractor_MeanBand = SleepSpectralFeatureExtractor_trial(
            fs=self.fs,
            segm_size=self.segment_size,
            fbands=self.fbands,
            datarate=False
        )

        self.FeatureExtractor_MeanBand._extraction_functions = \
            [
                mean_bands,
            ]

    def extract_segment(self, signal_segment):
        FtrSgm = FeatureSegment()
        for ch in signal_segment.channels:
            x = signal_segment.signal[ch]
            fs = signal_segment.fs[ch]
            s = signal_segment.start[ch]
            annotation = signal_segment.annotation[ch]

            if x.shape[0] / fs >= 30:
                x = buffer(x, fs=fs, segm_size=30, drop=True)
                datarate = get_datarate(x)
                stamps = np.arange(s, s + x.shape[0] * 30, 30)

                mean_bands, feature_names = self.FeatureExtractor_MeanBand(x)
                mean_bands = np.stack(mean_bands).T

                functions = [np.divide]
                symbols = ['/']
                mean_band_derived_features, mean_band_derived_names = mean_bands, feature_names
                for idx in range(functions.__len__()):
                    mean_band_derived_features, mean_band_derived_names = augment_features(
                        mean_band_derived_features,
                        feature_indexes=np.arange(mean_band_derived_features.shape[1]),
                        operation=functions[idx],
                        mutual=True, operation_str=symbols[idx],
                        feature_names=mean_band_derived_names

                    )

                mean_band_derived_features = np.log10(mean_band_derived_features)
                df = pd.DataFrame()
                df['start'] = stamps
                df['segment_duration'] = 30
                df['datarate'] = datarate
                df['annotation'] = annotation
                for f, fn in zip(mean_band_derived_features.T, mean_band_derived_names):
                    df[fn] = f

                FtrSgm.add_channel(
                    df_features=df,
                    channel_name=ch,
                    feature_names=mean_band_derived_names,
                )
        return FtrSgm


class MultiChannelSleepClassifier:
    __name__ = 'MultiChannelSleepClassifier'
    __version__ = '0.0.1'

    def __init__(self,
                 fbands=[
                     [0.5, 5],  # delta
                     [4, 9],  # theta
                     [8, 14],  # alpha
                     [11, 16],  # spindle
                     [14, 20],
                     [20, 30]
                 ],
                 segment_size=30,
                 fs=250,
                 dr_threshold = 0.8
                 ):

        self.fbands = fbands
        self.segment_size = segment_size
        self.fs = fs
        self.dr_threshold = dr_threshold

        self.FtrExtractor = PIBFeatureExtractor(
            fbands=fbands,
            segment_size=segment_size,
            fs=fs
        )

    def fit(self, X, y):
        self.scaler = StandardScaler()
        #self.UMAP = UMAP()
        self.PCA = PCAModule(var_threshold=0.95)
        Xe = self.scaler.fit_transform(X)
        Xe = self.PCA.fit_transform(Xe)

        #Xe = self.UMAP.fit_transform(Xe)
        self.clf = SVC(probability=True)
        self.clf.fit(Xe, y)

    def transform(self, X):
        x = self.scaler.transform(X)
        x = self.PCA.transform(x)
        # self.UMAP.transform(x)
        return x

    def scores(self, X):
        X = self.transform(X)
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(self.transform(X))

    def extract_features(self, SigRecording):
        return SigRecording.extract_features(self.FtrExtractor)

    def train(self, FtrRecording, test_ratio=0.2):
        channels = FtrRecording[0].channels
        self.channels = channels
        dr_threshold = self.dr_threshold

        df = []
        for k in range(FtrRecording.__len__()):
            df_, feature_names = FtrRecording[k][self.channels]
            df += [df_]

        df = pd.concat(df, ignore_index=True)
        df = df.loc[df.datarate >= dr_threshold]

        test_indexes = np.sort(np.concatenate(
            [df.loc[df.annotation == s].index[-int(np.floor(((df.annotation == s).sum() * test_ratio))):].to_numpy() for s in
             df.annotation.unique()]))
        train_indexes = [n for n in np.arange(df.__len__()) if not n in test_indexes and n in df.index]

        x_train = df.loc[train_indexes][feature_names].to_numpy()
        y_train = df.loc[train_indexes]['annotation'].to_numpy()

        x_test = df.loc[test_indexes][feature_names].to_numpy()
        y_test = df.loc[test_indexes]['annotation'].to_numpy()

        self.fit(x_train, y_train)
        yy_test = self.predict(x_test)

        results = {
            'kappa': {},
            'f1': {},
            'accuracy': {},
        }

        results['kappa']['all'] = cohen_kappa_score(y_test, yy_test)
        results['f1']['all'] = f1_score(y_test, yy_test, average='macro')
        results['accuracy']['all'] = accuracy_score(y_test, yy_test)

        for s in np.sort(np.unique(y_train)):
            results['kappa'][s] = cohen_kappa_score(y_test == s, yy_test == s)
            results['f1'][s] = f1_score(y_test == s, yy_test == s)
            results['accuracy'][s] = accuracy_score(y_test == s, yy_test == s)




        ret = {
            'train': {
                'x': x_train,
                'y': y_train
            },
            'test': {
                'x': x_test,
                'y': y_test,
                'yy': yy_test
            }
        }

        return results, ret


    def predict_FeatureRecording(self, FtrRecording):
        dr_threshold = self.dr_threshold

        df = []
        ftr_names = []
        for k in range(FtrRecording.__len__()):
            df_, feature_names = FtrRecording[k][self.channels]
            if feature_names.__len__() > 0: ftr_names = feature_names
            df += [df_]

        if df.__len__() > 0:
            df = pd.concat(df, ignore_index=True)
            df = df.loc[df.datarate >= dr_threshold]

            if df.__len__() == 0:
                return None

            bl = (df.sum(axis=1).isnull()) | (np.isinf(df.sum(axis=1).isnull())) | (np.isnan(df[feature_names]).sum(axis=1) > 0)
            df = df.loc[~bl].reset_index(drop=True)

            if df.__len__() == 0:
                return None

            x_predict = df[ftr_names].to_numpy()

            if x_predict.shape[0] == 0:
                return None

            yy_predict = self.predict(x_predict)

            hyp = df[['annotation', 'start', 'segment_duration']]
            hyp['annotation'] = yy_predict
            hyp['end'] = hyp['start'] + hyp['segment_duration']
            hyp = merge_annotations(hyp)
            return hyp





class CloudEEGDataLoader:
    def __init__(self, pid, db_id='UH3'):
        self.pid = pid
        self.db_id = db_id
        # modes - sante, continuous, nightoff

        self.SSLoader = SystemStateLoader(self.db_id)
        self.MfClient = MefClient()

        #if path_annotations:
            #self.df_annotations = self.scan_hypnogram_annotations(path_annotations)
            #self.df_annotations['path_mef'] = self.find_mef_sessions(self.df_annotations)

    def find_mef_sessions(self, df):
        print('Searching for sessions')
        # find a mef session for each annotation
        mef_files = []
        for idx, row in tqdm(list(df.iterrows())):
            s = row['start']
            e = row['end']

            file = self.df_files.loc[(s >= self.df_files.start) & (e <= self.df_files.end)]

            if file.__len__() == 0:
                pth = ''
            elif file.__len__() == 1:
                pth = file.iloc[0]['path']
            else: # if more sessions found, takes the one with more data
                pths = []
                drs = []

                for _, r in file.iterrows():
                    Rdr_ = MefReader(r['path'])
                    ch = [c for c in Rdr_.channels if not 'accl' in c][0]
                    x = Rdr_.get_data(ch, s * 1e6, e * 1e6)
                    pths += [r['path']]
                    drs += [1 - (np.isnan(x).sum() / x.shape[0])]
                    # ref_dr =
                pth = pths[np.argmax(drs)]
            mef_files += [pth]
        return mef_files

    def process_segment(self, s_process, e_process):

        SFndr = SessionFinder(self.db_id)

        process_segments = pd.DataFrame({'start': np.arange(s_process, e_process, 20 * 60)})
        process_segments['end'] = process_segments['start'] + 20 * 60
        process_segments['path_mef'] = SFndr.find_mef_session_bulk(self.pid, process_segments)
        process_segments = process_segments.loc[[p.__len__() > 0 for p in process_segments['path_mef'].to_list()]].reset_index(drop=True)
        process_segments = self.load_mef_info(process_segments)

        start = process_segments.start.min()
        end = process_segments.end.max()

        stim_info = self.SSLoader.load_stimulation_annotations(self.pid, start - 12 * 3600, end + 12 * 3600)
        if stim_info.__len__() == 0:
            stim_info = pd.DataFrame(
                [{'frequency': 0, 'amplitude': 0, 'start': start - 1 * 3600, 'end': end + 1 * 3600}])

        t = np.arange(stim_info.start.min(), stim_info.end.max(), 1)
        stim_ref = np.zeros_like(t) - 1
        ampl_ref = np.zeros_like(t) - 1
        for k, row_ in stim_info.iterrows():
            stim_ref[(t >= row_['start']) & (t < row_['end'])] = row_['frequency']
            ampl_ref[(t >= row_['start']) & (t < row_['end'])] = row_['amplitude']


        process_segments_ = []
        for idx, row in process_segments.iterrows():
            t_ = t[(t > row['start']) & (t < row['end'])]
            subsec = stim_ref[(t > row['start']) & (t < row['end'])]
            subsec[np.isnan(subsec)] = -2

            subsec_ampl = ampl_ref[(t > row['start']) & (t < row['end'])]
            subsec_ampl[np.isnan(subsec_ampl)] = 0

            k_ = 0
            while k_ < subsec.__len__() - 1:
                k0_ = k_
                start_freq = subsec[k_]
                start_time = t_[k_]
                while subsec[k_] == start_freq and k_ < subsec.__len__() - 1:
                    k_ += 1

                if start_freq == -1: start_freq = 0
                if start_freq == -2: start_freq = -1
                max_ampl = subsec_ampl[k0_:k_]
                max_ampl = np.max(max_ampl[~np.isnan(max_ampl)])
                
                process_segments_ += [
                    {
                        'start': start_time,
                        'end': t_[k_],
                        'duration': t_[k_] - start_time,
                        'freq': start_freq,
                        'ampl': max_ampl,
                        'pid': self.pid,
                        'path_mef': row['path_mef']
                    }
                ]
        return self.load_mef_info(pd.DataFrame(process_segments_))

    def load_mef_info(self, hyp):
        sessions_ = np.unique(hyp.path_mef.unique())
        sessions = pd.DataFrame()
        sessions['path'] = np.unique(sessions_)

        print('Searching for mef session details')
        fs = []
        channels = []
        for idx, row in tqdm(list(sessions.iterrows())):
            success, meta = self.MfClient.request_metadata(row['path'])
            fs_ = np.unique([ch_['fsamp'] for ch_ in meta if not 'acc' in ch_['name']])[0]
            channels_ = [ch_['name'] for ch_ in meta if not 'acc' in ch_['name']]

            fs += [fs_]
            channels += [channels_]

        sessions['channels'] = channels
        sessions['fs'] = fs

        fs = []
        channels = []
        for idx, row in hyp.iterrows():
            sr = sessions.loc[sessions.path == row['path_mef']]
            fs += [sr['fs']]
            channels += [sr['channels']]

        hyp['fs'] = fs
        hyp['channels'] = channels
        return hyp


    def load_stim_info(self, hyp):
        hyp_buffer = []

        hyp = time_to_timestamp(hyp)
        hyp_start = hyp.start.min()
        hyp_end = hyp.end.max()

        # Load stim info
        stim_info = self.SSLoader.load_stimulation_annotations(self.pid, hyp_start - 12 * 3600, hyp_end + 12 * 3600)
        if stim_info.__len__() == 0:
            hyp['freq'] = 0
            hyp['ampl'] = 0
            return hyp

        # prepare for tiling the hypnogram based on stim
        t = np.arange(stim_info.start.min(), stim_info.end.max(), 1)
        stim_ref = np.zeros_like(t) - 1
        ampl_ref = np.zeros_like(t) - 1
        for k, row_ in stim_info.iterrows():
            stim_ref[(t >= row_['start']) & (t <= row_['end'])] = row_['frequency']
            ampl_ref[(t >= row_['start']) & (t <= row_['end'])] = row_['amplitude']

        # tile hypnogram with stim annots
        for k, row_hyp in tqdm(list(hyp.iterrows())):
            t_ = t[(t >= row_hyp['start']) & (t <= row_hyp['end'])]

            subsec_freq = stim_ref[(t >= row_hyp['start']) & (t <= row_hyp['end'])]
            subsec_freq[np.isnan(subsec_freq)] = -2

            subsec_ampl = ampl_ref[(t >= row_hyp['start']) & (t <= row_hyp['end'])]
            subsec_ampl[np.isnan(subsec_ampl)] = 0
            # break

            k_ = 0
            done = False
            while k_ < subsec_freq.__len__() - 1:  # tiling each annotation based on the stim annotations
                done = True
                k0_ = k_
                start_time = t_[k_]
                start_freq = subsec_freq[k_]

                while subsec_freq[k_] == start_freq and k_ < subsec_freq.__len__() - 1:  # move till the stim freq is same
                    k_ += 1

                if start_freq == -1: start_freq = 0
                if start_freq == -2: start_freq = -1

                max_ampl = subsec_ampl[k0_:k_]
                max_ampl = np.max(max_ampl[~np.isnan(max_ampl)])

                r_ = row_hyp.to_dict()
                r_['start'] = start_time
                r_['end'] = t_[k_]
                r_['duration'] = t_[k_] - start_time
                r_['freq'] = start_freq
                r_['ampl'] = max_ampl

                hyp_buffer += [
                    r_
                ]

            if done == False:
                start_freq = 0
                max_ampl = 0
                r_ = row_hyp.to_dict()
                r_['freq'] = start_freq
                r_['ampl'] = max_ampl
                hyp_buffer += [
                    r_
                ]

        return pd.DataFrame(hyp_buffer)


    def load_hypnogram_annotations(self, files_hyp, round_timestamps=True):
        SFndr = SessionFinder(self.db_id)
        hyp = []
        for f in files_hyp:
            # Load hypnogram
            hyp_ = load_CyberPSG(f)
            hyp_ = time_to_timestamp(hyp_)
            if round_timestamps:
                hyp_['start'] = hyp_['start'].round()
                hyp_['end'] = hyp_['end'].round()
            hyp_['pid'] = self.pid
            hyp_['path_hypnogram'] = f
            hyp += [self.load_stim_info(hyp_)]

        hyp = pd.concat(hyp, ignore_index=True).sort_values('start').reset_index(drop=True)
        print('Searching for mef sessions')
        hyp['path_mef'] = SFndr.find_mef_session_bulk(self.pid, hyp)
        return hyp



def compare_channels(x, channels):
    return sum([ch in channels for ch in x['channels'].iloc[0]]) == channels.__len__()


def compare_freq(x, freqs):
    if freqs.__len__() == 0:
        return True
    return x['freq'] in freqs


def compare_ampl(x, freqs):
    if freqs.__len__() == 0:
        return True
    return np.ceil(x['ampl']) in freqs


def remove_artifacts(x, fs, cuda=3):
    model = models_ArtifactEraser[f'RCS_ArtifactEraser_{int(fs)}Hz_64']
    model = model.eval()
    model = model.cuda(cuda)

    mn_x = np.nanmean(x)
    x -= mn_x
    nans = np.isnan(x)

    fs = int(fs)
    shape = x.shape[0]
    x = np.concatenate((np.zeros(1 * fs), x, np.zeros(5 * fs)))

    xb = buffer(x, fs, segm_size=2, overlap=1.0)
    mn = np.nanmean(xb, axis=1)
    mn[np.isnan(mn)] = 0

    std = np.nanstd(xb, axis=1)
    std[np.isnan(std)] = 1

    dr = get_datarate(xb)
    xb[np.isnan(xb)] = 0

    xb -= mn.reshape(-1, 1)  # zscore
    xb /= std.reshape(-1, 1)

    dr_pos = np.where(dr > 0.1)[0]  # only datarate > than 0.1 in 30 minute segment
    xb_proc = xb[dr_pos]
    xb[:] = np.NaN

    # idx_end = 1
    # if xb_proc.shape[0]-512 > 1:
    #     idx_end = xb_proc.shape[0]-512
    idx_end = x.shape[0]


    with torch.no_grad():
        for k in np.arange(0, idx_end, 512):
            tmp = xb_proc[k:k + 512]
            bs = tmp.shape[0]
            if bs > 0:
                tmp_rec, det = model(torch.tensor(tmp).float().view(bs, 1, -1).cuda(cuda))
                tmp_rec = tmp_rec.detach().view(tmp_rec.shape[0], tmp_rec.shape[2]).cpu().numpy()
                xb_proc[k:k + bs] = tmp_rec

    xb[dr_pos] = xb_proc
    xb *= std.reshape(-1, 1)
    xb += mn.reshape(-1, 1)  # denormalize

    xrec = xb[:, int(0.5 * fs):int(1.5 * fs)].flatten()[
           int(0.5 * fs):int(0.5 * fs) + shape]  # crop only valid semgments
    xrec[nans] = np.nan
    x = xrec
    x += mn_x
    return x











