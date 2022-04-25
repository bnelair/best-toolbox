# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from tqdm import tqdm
from copy import deepcopy, copy


import pandas as pd


from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.svm import SVR, LinearSVC
from sklearn import preprocessing
from sklearn.metrics import cohen_kappa_score, roc_curve

from scipy.stats import gaussian_kde
from scipy.signal import filtfilt
from scipy.signal.windows import gaussian
from scipy.optimize import differential_evolution
from scipy.linalg import norm


from best.hypnogram.utils import time_to_timestamp, time_to_utc, merge_annotations
from best.feature import augment_features, balance_classes
from best.signal import unify_sampling_frequency, get_datarate, buffer
from best.stats import kl_divergence_nonparametric
from best.vector import scale, translate, get_mutual_vectors
from best.modules import multivariate_normal_
from best.modules import ZScoreModule, PCAModule
from best.feature_extraction.FeatureExtractor import SleepSpectralFeatureExtractor
from best.feature_extraction.SpectralFeatures import mean_bands, mean_frequency, relative_bands
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class KDEBayesianModel:
    __name__ = "KDEBayesianModel"
    def __init__(self, fbands=[[0.5, 5], # delta
                               [4, 9], # theta
                               [8, 14], # alpha
                               [11, 16], # spindle
                               [14, 20],
                               [20, 30]], segm_size=30, fs=200, bands_to_erase=[], filter_bands = True, nfft=12000,
                 window_smooth_n=3, window_std=1, cat_bias={'AWAKE': 1, 'N2': 1, 'N3': 1, 'REM': 1}, Selector2=True):

        self.fbands = fbands
        self.segm_size = segm_size
        self.fs = fs
        self.bands_to_erase = bands_to_erase
        self.filter_bands = filter_bands
        self.nfft=nfft
        self.filter_order = 100

        self.STATES = []
        self.KDE = []
        self.PipelineClustering = None
        self.FeatureSelector = None

        self.SELECTOR2 = Selector2

        self.FeatureExtractor_MeanBand = SleepSpectralFeatureExtractor(
            fs=self.fs,
            segm_size=self.segm_size,
            fbands=self.fbands,
            bands_to_erase=self.bands_to_erase,
            filter_bands=self.filter_bands,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )


        self.FeatureExtractor_MeanBand = SleepSpectralFeatureExtractor(
            fs=self.fs,
            segm_size=self.segm_size,
            fbands=self.fbands,
            bands_to_erase=self.bands_to_erase,
            filter_bands=self.filter_bands,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )




        self.FeatureExtractor_MeanBand._extraction_functions = \
            [
                mean_bands,
            ]



        self.FeatureExtractor = SleepSpectralFeatureExtractor(
            fs=self.fs,
            fbands=self.fbands,
            bands_to_erase=self.bands_to_erase,
            segm_size=self.segm_size,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )

        self.FeatureExtractor._extraction_functions = \
            [
                mean_frequency,
                #self.FeatureExtractor.MedFreq,
                relative_bands,
                #self.FeatureExtractor.normalized_entropy,
                #self.FeatureExtractor.normalized_entropy_bands
            ]


        self.WINDOW = gaussian(window_smooth_n, window_std)
        self.WINDOW = self.WINDOW / self.WINDOW.sum()
        self.CAT_BIAS = cat_bias
        self.feature_names = None


    def extract_features(self, signal, return_names=False):
        if signal.ndim > 1:
            raise AssertionError('[INPUT ERROR]: Input data has to be of a dimension size 1 - raw signal')
        if signal.shape[0] != self.fs * self.segm_size:
            print('[INPUT WARNING]: input data is not a defined size fs*segm_size ' + str(self.fs*self.segm_size) + '; Signal of a size ' + str(signal.shape[0]) + ' found instead. Extracted features might be inaccurate.')


        ## Mean band-derived features - delta/beta ratio etc
        mean_bands, feature_names = self.FeatureExtractor_MeanBand(signal)
        mean_bands = np.concatenate(mean_bands)

        functions = [np.divide]
        symbols = ['/']
        mean_band_derived_features, mean_band_derived_names = mean_bands, feature_names
        for idx in range(functions.__len__()):
            mean_band_derived_features, mean_band_derived_names = augment_features(

                mean_band_derived_features.reshape(1, -1), feature_indexes=np.arange(mean_band_derived_features.shape[0]), operation=functions[idx], mutual=True,  operation_str=symbols[idx], feature_names=mean_band_derived_names

            )


        mean_band_derived_names = mean_band_derived_names[feature_names.__len__():]
        mean_band_derived_features = mean_band_derived_features[0, feature_names.__len__():]
        #mean_band_derived_names = mean_band_derived_names.squeeze()

        #features = np.log10(np.append(other_features, mean_band_derived_features))
        #feature_names = feature_names + mean_band_derived_names
        features = np.log10(mean_band_derived_features)
        #feature_names = mean_band_derived_names

        ## other features
        other_features, feature_names_other = self.FeatureExtractor(signal)
        other_features = np.concatenate(other_features)
        features = np.append(other_features, features)
        feature_names = list(feature_names_other) + list(mean_band_derived_names)


        self.feature_names = feature_names
        if return_names:
            return features, feature_names
        return features

    def extract_features_bulk(self, list_of_signals, fsamp_list, return_names=False):
        data = list_of_signals
        data, fs = unify_sampling_frequency(data, sampling_frequency=fsamp_list, fs_new=self.fs)
        x = []
        for k in tqdm(range(data.__len__())):
            x += [self.extract_features(data[k])]
        if return_names:
            _, feature_names = self.extract_features(data[k], return_names=True)
            return np.array(x), feature_names
        return np.array(x), fs

    def fit(self, X, y):
        X, y = self._fit(X, y)
        self._fit_kde(X, y)

    def _fit(self, X, y):

        X = deepcopy(X)
        y = deepcopy(y)
        X_, y_ = balance_classes(X, y, std_factor=0.0)

        estimator = SVR(kernel="linear")
        self.SELECTOR = RFECV(estimator, step=5, verbose=True, min_features_to_select=4, n_jobs=10)
        self.PCA = PCAModule(var_threshold=0.98)
        self.ZScore = ZScoreModule(trainable=True, continuous_learning=False, multi_class=False)

        #self.UMAP = UMAP(n_neighbors=30, min_dist=1,
        #                 n_components=2)

        le = preprocessing.LabelEncoder()
        le.fit(y_)
        y__ = le.transform(y_)


        X_ = self.SELECTOR.fit_transform(X_, y__)
        X_ = self.PCA.fit_transform(X_)
        X_ = self.ZScore.fit_transform(X_, y)

        X = self.SELECTOR.transform(X)
        X = self.PCA.transform(X)
        X = self.ZScore.fit_transform(X, y)

        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_, y_)
        if self.SELECTOR2:
            self.SELECTOR2 = SelectFromModel(lsvc, prefit=True, max_features=4)
            #X_ = self.SELECTOR2.transform(X_)
            X = self.SELECTOR2.transform(X)

        #X = self.UMAP.fit_transform(X)
        return X, y


    def _fit_kde(self, X, y):
        self.STATES = np.unique(y)
        self.KDE = []
        for state in self.STATES:
            X_ = X[y==state, :]
            kernel = gaussian_kde(X_.T)
            self.KDE.append(kernel)

    def _likelihood(self, X):
        scores = {}
        for idx, kde in enumerate(self.KDE):
            scores[self.STATES[idx]] = kde.pdf(X.T)
        scores = pd.DataFrame(scores)
        return scores

    def scores(self, X):
        X = self.transform(X)
        scores = self._likelihood(X)


        scores = scores.div(scores.sum(axis=1), axis=0)
        for key in scores.keys():
            scores[key] = filtfilt(self.WINDOW, 1, scores[key])

        for cat in self.CAT_BIAS.keys():
            if cat in scores.keys(): scores[cat] = scores[cat]*self.CAT_BIAS[cat]

        scores = scores.div(scores.sum(axis=1), axis=0)
        return scores

    def transform(self, X):
        X = self.SELECTOR.transform(X)
        X = self.PCA.transform(X)
        X = self.ZScore.transform(X)
        if self.SELECTOR2:
            X = self.SELECTOR2.transform(X)
        #X = self.UMAP.transform(X)
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        return np.array(self.scores(X).idxmax(axis=1))

    def preprocess_signal(self, signal, fs, datarate_threshold=0.85):
        data = buffer(signal, fs, self.segm_size)
        start_time = np.array([k*self.segm_size for k in range(data.__len__())])
        end_time = start_time + self.segm_size
        datarate = np.array(get_datarate(data))

        data = data[datarate >= datarate_threshold]
        start_time = start_time[datarate >= datarate_threshold]
        end_time = end_time[datarate >= datarate_threshold]
        return list(data), start_time, end_time

    def predict_signal(self, signal, fs, datarate_threshold=0.85):
        data, start_time, end_time = self.preprocess_signal(signal, fs, datarate_threshold)
        x, fs = self.extract_features_bulk(data, [fs]*data.__len__())
        scores = self.scores(x)
        df = pd.DataFrame({'annotation': scores.idxmax(axis=1), 'start': start_time, 'end': start_time+30, 'duration':30})
        df = time_to_utc(df)
        df = merge_annotations(df)
        df = time_to_timestamp(df)
        df = df[['annotation', 'start', 'end', 'duration']]
        return df

    def predict_signal_scores(self, signal, fs, datarate_threshold=0.85):
        data, start_time, end_time = self.preprocess_signal(signal, fs, datarate_threshold)
        x, fs = self.extract_features_bulk(data, [fs]*data.__len__())
        scores = self.scores(x)
        return scores


class KDEBayesianCausalModel(KDEBayesianModel):
    __name__ = "KDEBayesianCausalModel"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MarkovFilter = None

    def fit(self, X, y):
        super().fit(X, y)

        scores = super().scores(X)
        self.MarkovFilter = SleepStageProbabilityMarkovChainFilter()
        self.MarkovFilter.fit(scores, y)


    def scores(self, X):
        scores = self._scores(X)
        state = 'AWAKE'
        scores = self.MarkovFilter.predict(scores, state)
        """
        ch_posts = []
        for k in range(scores.__len__()):
            p_likelihood_change = scores.iloc[k][self.MarkovFilter.STATES[self.MarkovFilter.STATES != state]].sum()
            p_prior_change = self.MarkovFilter.get_state_change_prior(state)
            p_post_change = (p_likelihood_change * p_prior_change) / ((p_likelihood_change * p_prior_change) + ((1-p_likelihood_change) * (1-p_prior_change)))
            ch_posts += [[p_post_change, state]]

            p_likelihood = scores.iloc[k][self.MarkovFilter.STATES[self.MarkovFilter.STATES != state]]
            p_prior = self.MarkovFilter.get_changing_state_priors(state)[self.MarkovFilter.STATES != state]
            p_post = p_likelihood*p_prior / sum(p_likelihood*p_prior)

            if p_post_change > 0.5:
                state = p_post.idxmax()
                scores.loc[k, state] = p_post_change
                scores.loc[k, self.MarkovFilter.STATES[self.MarkovFilter.STATES!=state]] = p_post*(1-p_post_change)
            else:
                scores.loc[k, state] = 1-p_post_change
                scores.loc[k, self.MarkovFilter.STATES[self.MarkovFilter.STATES!=state]] = p_post*p_post_change
            #yy[k] = state
        """
        return scores

    def _scores(self, X):
        return super().scores(X)


class MVGaussBayesianModel(KDEBayesianModel):
    __name__ = "MVGaussBayesianModel"

    def _fit_kde(self, X, y):
        self.STATES = np.unique(y)
        self.KDE = []
        for state in self.STATES:
            X_ = X[y==state, :]
            kernel = multivariate_normal_(X_.T)
            self.KDE.append(kernel)


class MVGaussBayesianCausalModel(MVGaussBayesianModel, KDEBayesianCausalModel):
    __name__ = "MVGaussBayesianModel"


class SleepStageProbabilityMarkovChainFilter:
    def __init__(self):
        self.STATES = np.array(['AWAKE', 'N1', 'N2', 'N3', 'REM'])
        self.removed_classes = []
        self.stability = np.ones(self.STATES.__len__())
        self._tmat_orig = np.array(
            [[0.961, 0.038, 0.001, 0.000, 0.000],
             [0.097, 0.215, 0.634, 0.000, 0.054],
             [0.020, 0.001, 0.846, 0.060, 0.073],
             [0.005, 0.001, 0.105, 0.880, 0.009],
             [0.017, 0.003, 0.061, 0.000, 0.918]]
        )

        #self._tmat_orig = np.array(
        #    [
        #        [0.802, 0.19,  0.007, 0.,    0.002],
        #        [0.11,  0.625, 0.24,  0.,    0.026],
        #        [0.018, 0.046, 0.89,  0.036, 0.01 ],
        #        [0.007, 0.006, 0.171, 0.815, 0.001],
        #        [0.017, 0.043, 0.012, 0.,    0.927]
        #    ])


        self.tmat = self._tmat_orig.copy()




        #np.fill_diagonal(self.tmat, self.tmat.diagonal()*stability)
        #diag = self.tmat.diagonal()
        #self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)


        #diag = self.tmat.diagonal() * stability
        #self.tmat = self.tmat * (1-stability)
        #np.fill_diagonal(self.tmat, diag)
        self._get_vars()

    def fit(self, scores, y):
        self.reset_probabilities()
        self.tmat = self._tmat_orig.copy()
        classes = np.unique(y)
        #class_certainty = dict([[state, X[state][y==state].median()] for state in classes])

        for cl in self.STATES:
            if cl not in classes:
                self.remove_class(cl)

        self._tmat_orig = self.tmat.copy()

        #self.correct_certainty(class_certainty)

        stability = []
        for state in self.STATES:
            fpr, tpr, thresholds = roc_curve(np.array(y)==state, np.array(scores[state]))
            #fpr, tpr, thresholds = precision_recall_curve(np.array(y)==state, np.array(scores[state]))
            t = np.sqrt((1-fpr)**2 + tpr**2)
            #t = 2*(fpr*tpr) / (fpr+tpr)
            s = 1 - thresholds[t.argmax()]

            #fpr, tpr, thresholds = precision_recall_curve(np.array(y)==state, np.array(scores[state]))
            #t = np.sqrt(fpr**2 + tpr**2)
            #t = 2 * fpr * tpr / (fpr+tpr)
            #s = thresholds[t.argmax()]
            stability += [s]


            #precision_recall_curve
        #stability = self._optimize(scores, y)
        #stability = [0.51861717, 0.39611577, 1.09778543, 0.32874582]
        #stability = [0.12461134, 0.3359193,  0.18570281, 0.37578364]
        #stability = [0.10092481, 0.30177824, 0.21477364, 0.25693041]
        #stability = [0.06515025678829323, 0.3012126596534704, 0.18660074488436126, 0.2829450179411525]
        #stability = [0.75009757, 0.46948611, 0.53506914, 0.78013241]
        self.reset_probabilities()
        self.weight_probabilities(stability)
        self.stability = stability

    def fit_optimize(self, scores, y, Niter=200, popsize=10):
        self.fit(scores, y)

        stability = self._optimize(scores, y, Niter, popsize)
        self.reset_probabilities()
        self.weight_probabilities(stability)
        self.stability = stability


    def _get_vars(self):
        self.tmat = copy(self.tmat / self.tmat.sum(axis=1).reshape(-1, 1))
        self.change_prob = copy(1 - self.tmat.diagonal())
        self.prob_change_to = copy(self.tmat / (self.tmat.sum(axis=1) - self.tmat.diagonal()).reshape(-1, 1))
        self.prob_change_to[range(self.prob_change_to.shape[0]), range(self.prob_change_to.shape[1])] = 0

    def get_state_idx(self, state):
        return np.where(self.STATES == state)[0][0]

    def get_state_priors(self, state):
        idx = self.get_state_idx(state)
        priors = self.tmat[idx]
        return priors

    def get_state_change_prior(self, state):
        state_idx = self.get_state_idx(state)
        priors = self.get_state_priors(state)
        change_prior = 1 - priors[state_idx]
        return change_prior

    def get_prob_to_change(self, state, x_prob):
        idx = self.get_state_idx(state)
        prob_to_stay = x_prob[idx]
        return 1 - prob_to_stay

    def get_state_change_posterior(self, state, x_prob):
        prior = self.get_state_change_prior(state)
        prob = self.get_prob_to_change(state, x_prob)
        return prior * prob / (prior*prob + (1-prior)*(1-prob))


    def get_changing_state_priors(self, state):
        state_idx = copy(self.get_state_idx(state))
        priors = copy(self.get_state_priors(state))
        priors[state_idx] = 0
        priors = priors / priors.sum()
        return priors


    def get_changing_state_probabilities(self, state, x_prob):
        idx = self.get_state_idx(state)
        x_prob[idx] = 0
        return x_prob / x_prob.sum()


    def get_changing_state_posteriors(self, state, x_prob):
        priors = self.get_changing_state_priors(state)
        probs = self.get_changing_state_probabilities(state, x_prob)
        return probs*priors / np.sum(priors*probs)


    def correct_certainty(self, certainty: dict):
        for state in certainty.keys():
            idx = self.get_state_idx(state)
            cert = certainty[state]
            self.tmat[idx, idx] = cert * self.tmat[idx, idx]

        #diag = self.tmat.diagonal()
        #self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)
        self._get_vars()


    def remove_class(self, class_name):
        idx = self.get_state_idx(class_name)
        self.tmat = np.delete(np.delete(self.tmat, idx, axis=0), idx, axis=1)

        diag = self.tmat.diagonal()
        self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)
        np.fill_diagonal(self.tmat, diag)

        self.STATES = self.STATES[self.STATES != class_name]
        self._get_vars()
        self.removed_classes.append(class_name)


    def reset_probabilities(self):
        self.tmat = self._tmat_orig.copy()


    def weight_probabilities(self, stability: np.ndarray):
        #for idx in range(self.STATES.__len__()):
        #self.tmat[idx, idx] *= stability[idx]
        #diag = self.tmat.diagonal()
        #self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)

        for idx in range(self.STATES.__len__()):
            self.tmat[idx, idx] = stability[idx]

        diag = self.tmat.diagonal()
        self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)

        for idx in range(self.STATES.__len__()):
            self.tmat[idx, idx] = stability[idx]

        self._get_vars()


    def predict(self, scores, state='AWAKE'):
        ch_posts = []
        for k in range(scores.__len__()):
            p_likelihood_change = scores.iloc[k][self.STATES[self.STATES != state]].sum()
            p_prior_change = self.get_state_change_prior(state)
            p_post_change = (p_likelihood_change * p_prior_change) / ((p_likelihood_change * p_prior_change) + ((1-p_likelihood_change) * (1-p_prior_change)))
            ch_posts += [[p_post_change, state]]

            p_likelihood = scores.iloc[k][self.STATES[self.STATES != state]]
            p_prior = self.get_changing_state_priors(state)[self.STATES != state]
            p_post = p_likelihood*p_prior / sum(p_likelihood*p_prior)

            if p_post_change > 0.5:
                state = p_post.idxmax()
                scores.loc[k, state] = p_post_change
                scores.loc[k, self.STATES[self.STATES!=state]] = p_post*(1-p_post_change)
            else:
                scores.loc[k, state] = 1-p_post_change
                scores.loc[k, self.STATES[self.STATES!=state]] = p_post*p_post_change


        for k in np.where(np.isnan(scores))[0]:
            s = np.isnan(scores.iloc[k]).idxmax()
            scores.loc[k, s] = 1 - scores.iloc[k].sum()
        return scores


    def _optimize(self, scores, Y, Niter=100, popsize=20):
        print('Optimizing sleep stage stability using Differential Evolution')
        self.idx = -1
        self.best_kappa = -1
        self.best_stability = np.zeros(self.STATES.__len__())
        self.scores = scores
        self.Y = Y
        self.n_optimize=Niter

        def optimize_(args):
            self.idx += 1
            if self.idx < self.n_optimize:
                #args = [10**i for i in args]
                scores__ = deepcopy(self.scores)
                #self.reset_probabilities()
                self.weight_probabilities(args)
                sc_ = self.predict(scores__)
                y_ = np.array(sc_.idxmax(axis=1))
                kappa = cohen_kappa_score(self.Y, y_)
                sc = 1 - kappa
                #sc = np.sum([1-float(c) for c in list(get_classification_scores(self.Y, y_).values())])
                print(self.idx, kappa, args)

                if kappa > self.best_kappa:
                    self.best_kappa = kappa
                    self.best_stability = args
            else: sc = 0

            return sc




        bounds = [(0.2, 0.90)] * self.STATES.__len__()
        result = differential_evolution(optimize_, bounds, popsize=popsize, seed=1)
        result = self.best_stability

        del self.idx
        del self.best_kappa
        del self.best_stability
        del self.scores
        del self.Y

        return result


class KDEBayesianModelNC:
    __name__ = "KDEBayesianModel"
    def __init__(self, fbands=[[0.5, 5], # delta
                               [4, 9], # theta
                               [8, 14], # alpha
                               [11, 16], # spindle
                               [14, 20],
                               [20, 30]], segm_size=30, fs=200, bands_to_erase=[], filter_bands = True, filter_order=5001, nfft=12000,
                 window_smooth_n=3, window_std=1, cat_bias={'AWAKE': 1, 'N2': 1, 'N3': 1, 'REM': 1}, Selector2=True
                 ):

        self.fbands = fbands
        self.segm_size = segm_size
        self.fs = fs
        self.bands_to_erase = bands_to_erase
        self.filter_bands = filter_bands
        self.filter_order = filter_order
        self.nfft=nfft

        self.STATES = []
        self.KDE = []
        self.PipelineClustering = None
        self.FeatureSelector = None

        self.SELECTOR2 = Selector2

        self.FeatureExtractor_MeanBand = SleepSpectralFeatureExtractor(
            fs=self.fs,
            segm_size=self.segm_size,
            fbands=self.fbands,
            bands_to_erase=self.bands_to_erase,
            filter_bands=self.filter_bands,
            nfiltorder=self.filter_order,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )



        self.FeatureExtractor_MeanBand._extraction_functions = \
            [
                mean_bands,
            ]



        self.FeatureExtractor = SleepSpectralFeatureExtractor(
            fs=self.fs,
            fbands=self.fbands,
            bands_to_erase=self.bands_to_erase,
            segm_size=self.segm_size,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )
        self.FeatureExtractor._extraction_functions = \
            [
                mean_frequency,
                #self.FeatureExtractor.MedFreq,
                relative_bands,
                #self.FeatureExtractor.normalized_entropy,
                #self.FeatureExtractor.normalized_entropy_bands
            ]


        self.WINDOW = gaussian(window_smooth_n, window_std)
        self.WINDOW = self.WINDOW / self.WINDOW.sum()
        self.CAT_BIAS = cat_bias
        self.feature_names = None


    def extract_features(self, signal, return_names=False):
        if signal.ndim > 1:
            raise AssertionError('[INPUT ERROR]: Input data has to be of a dimension size 1 - raw signal')
        if signal.shape[0] != self.fs * self.segm_size:
            print('[INPUT WARNING]: input data is not a defined size fs*segm_size ' + str(self.fs*self.segm_size) + '; Signal of a size ' + str(signal.shape[0]) + ' found instead. Extracted features might be inaccurate.')


        ## Mean band-derived features - delta/beta ratio etc
        mean_bands, feature_names = self.FeatureExtractor_MeanBand(signal)
        mean_bands = np.concatenate(mean_bands)

        functions = [np.divide]
        symbols = ['/']
        mean_band_derived_features, mean_band_derived_names = mean_bands, feature_names
        for idx in range(functions.__len__()):
            mean_band_derived_features, mean_band_derived_names = augment_features(

                mean_band_derived_features.reshape(1, -1), feature_indexes=np.arange(mean_band_derived_features.shape[0]), operation=functions[idx], mutual=True,  operation_str=symbols[idx], feature_names=mean_band_derived_names

            )


        mean_band_derived_names = mean_band_derived_names[feature_names.__len__():]
        mean_band_derived_features = mean_band_derived_features[0, feature_names.__len__():]
        #mean_band_derived_names = mean_band_derived_names.squeeze()

        #features = np.log10(np.append(other_features, mean_band_derived_features))
        #feature_names = feature_names + mean_band_derived_names
        features = np.log10(mean_band_derived_features)
        #feature_names = mean_band_derived_names

        ## other features
        other_features, feature_names_other = self.FeatureExtractor(signal)
        other_features = np.concatenate(other_features)
        features = np.append(other_features, features)
        feature_names = list(feature_names_other) + list(mean_band_derived_names)


        self.feature_names = feature_names
        if return_names:
            return features, feature_names
        return features

    def extract_features_bulk(self, list_of_signals, fsamp_list, return_names=False):
        data = list_of_signals
        data, fs = unify_sampling_frequency(data, sampling_frequency=fsamp_list, fs_new=self.fs)
        x = []
        for k in tqdm(range(data.__len__())):
            x += [self.extract_features(data[k])]
        if return_names:
            _, feature_names = self.extract_features(data[k], return_names=True)
            return np.array(x), feature_names
        return np.array(x), fs

    def fit(self, X, y):
        X, y = self._fit(X, y)
        self._fit_kde(X, y)

    def _fit(self, X, y):

        X = deepcopy(X)
        y = deepcopy(y)
        X_, y_ = balance_classes(X, y, std_factor=0.0)

        estimator = SVR(kernel="linear")
        self.SELECTOR = RFECV(estimator, step=5, verbose=True, min_features_to_select=4, n_jobs=10)
        self.PCA = PCAModule(var_threshold=0.98)
        #self.ZScore = ZScoreModule(trainable=True, continuous_learning=False, multi_class=False)

        #self.UMAP = UMAP(n_neighbors=30, min_dist=1,
        #                 n_components=2)

        le = preprocessing.LabelEncoder()
        le.fit(y_)
        y__ = le.transform(y_)


        X_ = self.SELECTOR.fit_transform(X_, y__)
        X_ = self.PCA.fit_transform(X_)
        #X_ = self.ZScore.fit_transform(X_, y)

        X = self.SELECTOR.transform(X)
        X = self.PCA.transform(X)
       # X = self.ZScore.fit_transform(X, y)

        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_, y_)
        if self.SELECTOR2:
            self.SELECTOR2 = SelectFromModel(lsvc, prefit=True, max_features=4)
            #X_ = self.SELECTOR2.transform(X_)
            X = self.SELECTOR2.transform(X)

        #X = self.UMAP.fit_transform(X)
        return X, y


    def _fit_kde(self, X, y):
        self.STATES = np.unique(y)
        self.KDE = []
        for state in self.STATES:
            X_ = X[y==state, :]
            kernel = gaussian_kde(X_.T)
            self.KDE.append(kernel)

    def _likelihood(self, X):
        scores = {}
        for idx, kde in enumerate(self.KDE):
            scores[self.STATES[idx]] = kde.pdf(X.T)
        scores = pd.DataFrame(scores)
        return scores

    def scores(self, X):
        X = self.transform(X)
        scores = self._likelihood(X)


        scores = scores.div(scores.sum(axis=1), axis=0)
        for key in scores.keys():
            scores[key] = filtfilt(self.WINDOW, 1, scores[key])

        for cat in self.CAT_BIAS.keys():
            if cat in scores.keys(): scores[cat] = scores[cat]*self.CAT_BIAS[cat]

        scores = scores.div(scores.sum(axis=1), axis=0)
        return scores

    def transform(self, X):
        X = self.SELECTOR.transform(X)
        X = self.PCA.transform(X)
        #X = self.ZScore.transform(X)
        if self.SELECTOR2:
            X = self.SELECTOR2.transform(X)
        #X = self.UMAP.transform(X)
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):

        return np.array(self.scores(X).idxmax(axis=1))

    def preprocess_signal(self, signal, fs, datarate_threshold=0.85):
        data = buffer(signal, fs, self.segm_size)
        start_time = np.array([k*self.segm_size for k in range(data.__len__())])
        end_time = start_time + self.segm_size
        datarate = np.array(get_datarate(data))

        data = data[datarate >= datarate_threshold]
        start_time = start_time[datarate >= datarate_threshold]
        end_time = end_time[datarate >= datarate_threshold]
        return list(data), start_time, end_time

    def predict_signal(self, signal, fs, datarate_threshold=0.85):
        data, start_time, end_time = self.preprocess_signal(signal, fs, datarate_threshold)
        x, fs = self.extract_features_bulk(data, [fs]*data.__len__())
        scores = self.scores(x)
        df = pd.DataFrame({'annotation': scores.idxmax(axis=1), 'start': start_time, 'end': start_time+30, 'duration':30})
        df = time_to_utc(df)
        df = merge_annotations(df)
        df = time_to_timestamp(df)
        df = df[['annotation', 'start', 'end', 'duration']]
        return df

    def predict_signal_scores(self, signal, fs, datarate_threshold=0.85):
        data, start_time, end_time = self.preprocess_signal(signal, fs, datarate_threshold)
        x, fs = self.extract_features_bulk(data, [fs]*data.__len__())
        scores = self.scores(x)
        return scores


class Mapper:
    def __init__(self):
        self.TEMPLATE = None
        self.TEMPLATE_NAME = None
        self.CLASSES = None
        self.TEMPLATE_CLASS = None
        self.MAPS = dict()
        self.NDIM = None

        self.N = 1000

        self.SHIFT_RANGE = np.array([-5, 5])
        self.SCALE_RANGE = np.array([0.9, 1.25])
        self.ROTATE_RANGE = np.array([-25, 25])

        self.X_RANGE = []

        self.COREGISTRATION_CLASS_BAN = ['N1']

    def create_template(self, x, y=None, name='Template'):
        self.NDIM = x.shape[1]
        self._get_template(x)
        if not isinstance(y, type(None)):
            self._get_template_class(x, y)

        self.TEMPLATE_NAME = name

    def _get_template(self, x):
        self.X_RANGE = [x.mean() - 20*x.std(), x.mean() + 20*x.std()]
        self.TEMPLATE = self.get_probabilities(x)

    def _get_template_class(self, x, y):
        y = np.array(y)
        self.CLASSES = np.sort(np.unique(y))
        self.TEMPLATE_CLASS = dict([(cl, self.get_probabilities(x[y == cl])) for cl in self.CLASSES])

    def _cost(self, x):
        px = self.get_probabilities(x)
        return kl_divergence_nonparametric(self.TEMPLATE, px)

    def _cost_class(self, x, y, bias={}):
        cost = 0
        classes = list(np.unique(y))
        for cl in self.CLASSES:
            if not cl in self.COREGISTRATION_CLASS_BAN:
                if cl in classes:
                    px = self.get_probabilities(x[y == cl])
                    c_ = kl_divergence_nonparametric(self.TEMPLATE_CLASS[cl], px)
                    if cl in bias.keys():
                        c_ *= bias[cl]
                    cost += c_
        return cost

    def _cost_semi_supervised(self, x, y, bias={}):
        xs_ = x[y!='']
        ys_ = y[y!='']

        return self._cost_class(xs_, ys_, bias=bias) + self._cost(x)

    def fit_map(self, x, y=None, bias={'REM': 2}):
        if not isinstance(y, type(None)):
            x = deepcopy(x)
            y = deepcopy(x)
            x, y = balance_classes(x, y)

        np.random.seed(0)
        transforms = []
        kl = []
        for k in tqdm(range(self.N)):
            x_ = deepcopy(x)
            tr = self._get_random_transform()
            x_ = self._transform(x_, tr)
            if isinstance(y, type(None)):
                c = self._cost(x_)
            else:
                #c = self._cost_class(x_, y)
                c = self._cost_semi_supervised(x_, y, bias=bias)

            kl += [c]
            transforms += [tr]
        return np.array(kl), transforms


    def fit_genetic(self, x, y=None, popsize=15):
        def optimize(args):
            x_ = deepcopy(x)
            tr = {}

            tr['translate'] = np.array([args[i] for i in range(0, 3)])
            tr['scale'] = np.array([args[i] for i in range(3, 6)])
            x_ = self._transform(x_, tr)
            c = self._cost(x_)
            return c

        bounds = [tuple(self.SHIFT_RANGE)] * x.shape[1] + [tuple(self.SCALE_RANGE)] * x.shape[1]
        result = differential_evolution(optimize, bounds, popsize=popsize)

        tr = {}
        tr['translate'] = np.array([result.x[i] for i in range(0, 3)])
        tr['scale'] = np.array([result.x[i] for i in range(3, 6)])
        return tr, self._cost(self._transform(x, tr))

    def fit_genetic_likelihood(self, x, y=None, popsize=15, model=None):
        def optimize(args):
            x_ = deepcopy(x)
            tr = {}

            tr['translate'] = np.array([args[i] for i in range(0, 3)])
            tr['scale'] = np.array([args[i] for i in range(3, 6)])
            x_ = self._transform(x_, tr)
            c = model._likelihood(x_)
            c = 1000 - c.sum().sum()
            return c

        bounds = [tuple(self.SHIFT_RANGE)] * x.shape[1] + [tuple(self.SCALE_RANGE)] * x.shape[1]
        result = differential_evolution(optimize, bounds, popsize=popsize)

        tr = {}
        tr['translate'] = np.array([result.x[i] for i in range(0, 3)])
        tr['scale'] = np.array([result.x[i] for i in range(3, 6)])
        return tr, self._cost(self._transform(x, tr))


    #def fit_map(self, x, name, y=None, model=None):
        #tr, cost = self.fit_genetic(x)
    #    tr, cost = self.fit_genetic_likelihood(x, model=model)
    #    self.MAPS[name] = {'transformation': tr, 'cost': cost}

    def map(self, x, name, model=None):
        if not name in self.MAPS.keys():
            self.fit_map(x, name=name, model=model)

        x = self._transform(x, self.MAPS[name]['transformation'])
        return x


    def _transform(self, x, transform):
        x = deepcopy(x)
        x = scale(x, transform['scale'])
        x = translate(x, transform['translate'])
        # x = rotate(x, transform['rotate'])
        return x

    def _get_random_transform(self):
        transform = {
            'translate': np.random.rand(self.NDIM) * np.diff(self.SHIFT_RANGE) + self.SHIFT_RANGE[0],
            'scale': np.random.rand(self.NDIM) * np.diff(self.SCALE_RANGE) + self.SCALE_RANGE[0]
        }
        #if self.NDIM == 2:
        #    transform['rotate'] = np.random.rand() * np.diff(self.ROTATE_RANGE) + self.ROTATE_RANGE[0]
        #if self.NDIM == 3:
        #    transform['rotate'] = np.random.rand(3) * np.diff(self.ROTATE_RANGE) + self.ROTATE_RANGE[0]

        return transform

    def get_probabilities(self, x):
        rng = self.X_RANGE
        bins = 200
        ps = []
        for k in range(x.shape[1]):
            p, _ = np.histogram(x[:, k], bins, rng)
            ps += [p]

        ps = np.array(ps) / x.shape[0]
        ps[ps == 0] = 1e-9
        ps = ps / ps.sum(axis=1).reshape(-1, 1)
        return ps


class SleepStructureClassifier:
    def __init__(self, states=['WAKE', 'N1', 'N2', 'N3', 'REM']):
        self.STATES = states
        self.norml2 = None
        self.KDE = None
        self.N = None

    def fit(self, x, y):
        xmut, ymut = get_mutual_vectors(x, y)
        norml2 = norm(xmut, axis=1)
        unit_vect = xmut / norml2.reshape(-1, 1)
        xmut = np.concatenate((norml2.reshape(-1, 1), unit_vect), axis=1)
        for s1 in self.STATES:
            self.KDE[s1] = {}
            for s2 in self.STATES:
                s = s1 + '-' + s2
                self.KDE[s1][s2] = gaussian_kde(xmut[ymut==s, :].T)


    def scores(self, x):
        N = x.shape[1]
        xmut, ymut = get_mutual_vectors(x)

        probs = {}
        p = None
        for s1 in self.STATES:
            probs[s1] = {}
            for s2 in self.STATES:
                probs[s1][s2] = self.KDE[s1][s2].pdf(xmut.T)
            probs[s1] = pd.DataFrame(probs[s1])
            if isinstance(p, type(None)):
                p = probs[s1]
            else:
                p = p + probs[s1]

        return pd.DataFrame([p.iloc[k::N].sum() for k in range(N)])


class SleepClassifierWrapper:
    def __init__(self):
        self.MODEL = {}

    def train(self, X, df):
        datarate = get_datarate(X)
        df = df.loc[datarate > 0.85].reset_index(drop=True)
        X = X[datarate > 0.85]


        X = X[df.annotation != 'N1']
        df = df.loc[df.annotation != 'N1'].reset_index(drop=True)

        X = X[df.annotation != 'UNKNOWN']
        df = df.loc[df.annotation != 'UNKNOWN'].reset_index(drop=True)


        fs_downsample = 250
        fbands = [[0.5, 5],  # delta
                  [4, 9],  # theta
                  [8, 14],  # alpha
                  [11, 16],  # spindle
                  [14, 20],
                  [20, 30]]  # (beta3)

        bands_to_erase_2 = [[0, 0.5]] + [[k - 0.5, k + 0.5] for k in np.arange(2, 30, 2)]
        bands_to_erase_7 = [[0, 0.5], [6, 8], [13, 15], [20, 22], [27, 29]]


        #### Train 0 ####
        X0 = X[df.freq == 0]
        df0 = df.loc[df.freq == 0].reset_index(drop=True)
        model0 = KDEBayesianModel(fs=fs_downsample, fbands=fbands, bands_to_erase=[], Selector2=False)
        X0 = model0.extract_features_bulk(X0)
        Y0 = df0.annotation.to_numpy()
        model0.fit(X0, Y0)
        self.MODEL[0] = model0

        #### Train 2 ####
        X2 = X[(df.freq == 2) | (df.freq == 0)]
        df2 = df.loc[(df.freq == 2) | (df.freq == 0)].reset_index(drop=True)
        model2 = KDEBayesianModel(fs=fs_downsample, fbands=fbands, bands_to_erase=bands_to_erase_2, Selector2=False)
        X2 = model2.extract_features_bulk(X2)
        Y2 = df2.annotation.to_numpy()
        model2.fit(X2, Y2)
        self.MODEL[2] = model2

        #### Train 7 ####
        X7 = X[(df.freq == 7) | (df.freq == 0)]
        df7 = df.loc[(df.freq == 7) | (df.freq == 0)].reset_index(drop=True)
        model7 = KDEBayesianModel(fs=fs_downsample, fbands=fbands, bands_to_erase=bands_to_erase_7, Selector2=False)
        X7 = model7.extract_features_bulk(X7)
        Y7 = df7.annotation.to_numpy()
        model7.fit(X7, Y7)
        self.MODEL[7] = model7

        #### Train 72.5 ####
        X725 = X[(df.freq == 72.5)]
        df725 = df.loc[(df.freq == 72.5)].reset_index(drop=True)
        model725 = KDEBayesianModel(fs=fs_downsample, fbands=fbands, bands_to_erase=[], Selector2=False)
        X725 = model725.extract_features_bulk(X725)
        Y725 = df725.annotation.to_numpy()
        model725.fit(X725, Y725)
        self.MODEL[72.5] = model725

    def predict_signal(self, X, fs, stim_freq):
        assert (fs == 250 or fs == 500), 'Sampling frequency has to be 250 or 500 Hz!!!!'

        if fs == 500:
            b, a = signal.butter(6, 40, fs=fs, btype='low', analog=False)
            nans = np.isnan(X)
            X[nans] = np.nanmean(X)
            X = signal.filtfilt(b, a, X)
            X[nans] = np.nan
            X = X[::2]

        clf = self.MODEL[stim_freq]
        return clf.predict_signal(X, 250, 0.85)


class MultiChannelMVGaussBayesClassifier:
    __name__ = "KDEBayesianModel"

    def __init__(self, fbands=[[0.5, 5],  # delta
                               [4, 9],  # theta
                               [8, 14],  # alpha
                               [11, 16],  # spindle
                               [14, 20],
                               [20, 30]], segm_size=30, fs=200, bands_to_erase=[], filter_bands=True, nfft=12000,
                 window_smooth_n=3, window_std=1, cat_bias={'AWAKE': 1, 'N2': 1, 'N3': 1, 'REM': 1},
                 Selector2=True):

        self.fbands = fbands
        self.segm_size = segm_size
        self.fs = fs
        self.bands_to_erase = bands_to_erase
        self.filter_bands = filter_bands
        self.nfft = nfft
        self.filter_order = 100

        self.STATES = []
        self.KDE = []
        self.PipelineClustering = None
        self.FeatureSelector = None

        self.SELECTOR2 = Selector2

        self.FeatureExtractor_MeanBand = SleepSpectralFeatureExtractor(
            fs=self.fs,
            segm_size=self.segm_size,
            fbands=self.fbands,
            bands_to_erase=self.bands_to_erase,
            filter_bands=self.filter_bands,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )

        self.FeatureExtractor_MeanBand = SleepSpectralFeatureExtractor(
            fs=self.fs,
            segm_size=self.segm_size,
            fbands=self.fbands,
            bands_to_erase=self.bands_to_erase,
            filter_bands=self.filter_bands,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )

        self.FeatureExtractor_MeanBand._extraction_functions = \
            [
                mean_bands,
            ]

        self.FeatureExtractor = SleepSpectralFeatureExtractor(
            fs=self.fs,
            fbands=self.fbands,
            bands_to_erase=self.bands_to_erase,
            segm_size=self.segm_size,
            sperwelchseg=10,
            soverlapwelchseg=5,
            nfft=self.nfft,
            datarate=False
        )

        self.FeatureExtractor._extraction_functions = \
            [
                mean_frequency,
                relative_bands,
            ]

        self.WINDOW = gaussian(window_smooth_n, window_std)
        self.WINDOW = self.WINDOW / self.WINDOW.sum()
        self.CAT_BIAS = cat_bias
        self.feature_names = None

    def extract_features(self, signal, return_names=False):
        if signal.ndim > 1:
            raise AssertionError('[INPUT ERROR]: Input data has to be of a dimension size 1 - raw signal')
        if signal.shape[0] != self.fs * self.segm_size:
            print('[INPUT WARNING]: input data is not a defined size fs*segm_size ' + str(
                self.fs * self.segm_size) + '; Signal of a size ' + str(
                signal.shape[0]) + ' found instead. Extracted features might be inaccurate.')

        ## Mean band-derived features - delta/beta ratio etc
        mean_bands, feature_names = self.FeatureExtractor_MeanBand(signal)
        mean_bands = np.concatenate(mean_bands)

        functions = [np.divide]
        symbols = ['/']
        mean_band_derived_features, mean_band_derived_names = mean_bands, feature_names
        for idx in range(functions.__len__()):
            mean_band_derived_features, mean_band_derived_names = augment_features(

                mean_band_derived_features.reshape(1, -1),
                feature_indexes=np.arange(mean_band_derived_features.shape[0]), operation=functions[idx],
                mutual=True, operation_str=symbols[idx], feature_names=mean_band_derived_names

            )

        mean_band_derived_names = mean_band_derived_names[feature_names.__len__():]
        mean_band_derived_features = mean_band_derived_features[0, feature_names.__len__():]
        # mean_band_derived_names = mean_band_derived_names.squeeze()

        # features = np.log10(np.append(other_features, mean_band_derived_features))
        # feature_names = feature_names + mean_band_derived_names
        features = np.log10(mean_band_derived_features)
        # feature_names = mean_band_derived_names

        ## other features
        other_features, feature_names_other = self.FeatureExtractor(signal)
        other_features = np.concatenate(other_features)
        features = np.append(other_features, features)
        feature_names = list(feature_names_other) + list(mean_band_derived_names)

        self.feature_names = feature_names
        if return_names:
            return features, feature_names
        return features

    def extract_features_bulk(self, list_of_signals, fsamp_list, return_names=False):
        data = list_of_signals
        # data, fs = unify_sampling_frequency(data, sampling_frequency=fsamp_list, fs_new=self.fs)
        x = []
        for k in tqdm(range(data.__len__())):
            x += [self.extract_features(data[k])]
        if return_names:
            _, feature_names = self.extract_features(data[k], return_names=True)
            return np.array(x), feature_names
        return np.array(x), fsamp_list[0]

    def fit(self, X, y):
        self.classifier = GaussianNB()
        self.classifier.fit(X, y)

    def scores(self, X):
        scr = self.classifier.predict_proba(X)
        return pd.DataFrame([{c: scr[:, idx]} for idx, c in enumerate(self.classifier.classes_)])

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        return np.array(self.scores(X).idxmax(axis=1))

    def preprocess_signal(self, signal, fs, datarate_threshold=0.85):
        data = buffer(signal, fs, self.segm_size)
        start_time = np.array([k * self.segm_size for k in range(data.__len__())])
        end_time = start_time + self.segm_size
        datarate = np.array(get_datarate(data))

        data = data[datarate >= datarate_threshold]
        start_time = start_time[datarate >= datarate_threshold]
        end_time = end_time[datarate >= datarate_threshold]
        return list(data), start_time, end_time

    def predict_signal(self, signal, fs, datarate_threshold=0.85):
        data, start_time, end_time = self.preprocess_signal(signal, fs, datarate_threshold)
        x, fs = self.extract_features_bulk(data, [fs] * data.__len__())
        scores = self.scores(x)
        df = pd.DataFrame(
            {'annotation': scores.idxmax(axis=1), 'start': start_time, 'end': start_time + 30, 'duration': 30})
        df = time_to_utc(df)
        df = merge_annotations(df)
        df = time_to_timestamp(df)
        df = df[['annotation', 'start', 'end', 'duration']]
        return df

    def predict_signal_scores(self, signal, fs, datarate_threshold=0.85):
        data, start_time, end_time = self.preprocess_signal(signal, fs, datarate_threshold)
        x, fs = self.extract_features_bulk(data, [fs] * data.__len__())
        scores = self.scores(x)
        return scores





