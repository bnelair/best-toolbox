# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Sleep Feature Extractor
^^^^^^^^^^^^^^^^^^^^^^^

The Feature Extractor package contains SleepSpectralFeatureExtractor for spectral feature extraction from designed for sleep classification from a raw EEG signal.


The extractor can return **data rate** which gives a relative ratio of valid values in the input signal based on a number of NaN values. The extractor requires information about frequency bands at which parameters will be calculated. Please see an example bellow.



Example
^^^^^^^^^

.. code-block:: python

    import sys
    import numpy as np
    from sleep_classification.FeatureExtractor.FeatureExtractor import SleepSpectralFeatureExtractor

    # Example synthetic signal generator
    fs = 500 # sampling frequency
    f = 10 # sin frequyency
    a = 1 # amplitude
    b = 0 # bias
    t = np.arange(0, 1000, 1/fs)
    x = a * np.sin(2*np.pi*f*t) + b


    # Spectral Feature  Extraction
    fs = 500 # sampling frequency of an analysed signal
    segm_size = 30 # time length of a segment which is used for extraction of each feature
    fbands = [[1, 4],
     [4, 8],
     [8, 12],
     [12, 14],
     [14, 20],
     [20, 30]] # frequency bands at which you want to extract features

    from sleep_classification.FeatureExtractor.SpectralFeatures import mean_bands, relative_bands

    Extractor = SleepSpectralFeatureExtractor(
        fs=fs,
        segm_size=segm_size,
        fbands=fbands,
        datarate=True
    )


    Extractor_MeanBand._extraction_functions = \
        [
            mean_bands,
            mean_frequency,
            relative_bands,
        ]

    feature_values, feature_names = Extractor(x)



Sources
^^^^^^^^^^^

This Feature Extractor implementation is based on the following papers (when use whole, parts, or are inspired by, we appreciate you acknowledge and refer these journal papers)


| Kremen, V., Duque, J. J., Brinkmann, B. H., Berry, B. M., Kucewicz, M. T., Khadjevand, F., G.A. Worrell, G. A. (2017). Behavioral state classification in epileptic brain using intracranial electrophysiology. Journal of Neural Engineering, 14(2), 026001. https://doi.org/10.1088/1741-2552/aa5688


| Kremen, V., Brinkmann, B. H., Van Gompel, J. J., Stead, S. (Matt) M., St Louis, E. K., & Worrell, G. A. (2018). Automated Unsupervised Behavioral State Classification using Intracranial Electrophysiology. Journal of Neural Engineering. https://doi.org/10.1088/1741-2552/aae5ab


| Gerla, V., Kremen, V., Macas, M., Dudysova, D., Mladek, A., Sos, P., & Lhotska, L. (2019). Iterative expert-in-the-loop classification of sleep PSG recordings using a hierarchical clustering. Journal of Neuroscience Methods, 317(February), 61?70. https://doi.org/10.1016/j.jneumeth.2019.01.013


and on repository `Semi Automated Sleep Classifier <https://github.com/vkremen/Semi_Automated_Sleep_Classifier_iEEG>`_, see details in the original repository.


"""

import multiprocessing

import numpy as np
import scipy.signal as signal

from functools import partial

from best.types import ObjDict
from best.feature_extraction.SpectralFeatures import *
from best.signal import buffer, PSD


class SleepSpectralFeatureExtractor_trial:
    __version__ = "2.0.0"
    """
    Spectral feature extractor designed for sleep classification

    ...

    Attributes
    ----------
    extraction_functions : list
        List of functions for parameter extraction from np.ndarray where shape is (n_spectrums, spectrum_samples).
        Each function has to return tuple (features : list, name_of_features : list). By default set to extract all features within this class.

    ...

    Methods
    --------

    ...

    Notes
    ------
    v0.1.0 Updates
    - communication between functions (Pxx, fs, ...) changed to ObjDict - see AISC.types.ObjDict
    - float value frequency bands enabled
    v0.1.1 Updates
    - bands_to_erase as an input into __call__ - erases defined bands of psd
    v0.1.2 Updates
    - self._extraction_functions init moved to __init__    
    v0.2.0
        - methods buffer, PSD transferred into AISC.utils.signal
        - PSD replaced by welch's method implementation by scipy.signal.welch
        - implementation of spectral feature extraction from shorter segments and average them
        - removed filtering due switch to welch method of periodogram - constant detrending
            - can be replaced with LowFrequencyFilter later on
        - automatic windown implemented in welch method - hann window

    v1.0.0
        - implemented combining of fbands and bands to erase to get bands for subsequent filtration
        - implemented filtering of frequency bands to eliminate leakage of surrounding frequencies
        - parameter init for feature extraction was moved from __call__ and process_signal to __init__
        - feature extraction functions were moved into AISC.FeatureExtractor.SpectralFeatures

    v2.0.0
        - notch filter implemented for stim frequency attenuation
        - chebyshev (cheb2) filter implemented for filtering individual bands
        - bands_to_erase modified from list of bands to list of single values which are to be filtered using notch filter

    """

    def __init__(self,
                 fs: float = None,
                 segm_size: float = None,
                 fbands: list = None,
                 sperwelchseg: float = None,
                 soverlapwelchseg: float = 0.0,
                 nfft: int = None,
                 filters_attenuate: list = [],
                 repeat_attenuate: int = 1,
                 ignore_bands: list = [],
                 datarate: bool = False,
                 n_processes: int = 1):

        self._fs = self._verify_input_fs(fs)
        self._segm_size = self._verify_input_segm_size(segm_size)
        self._fbands = self._verify_input_fbands(fbands)

        ###### SIGNAL PREPROCESSING SETUP #########

        self.FILTERS = None
        self.FILTER_LOW = signal.butter(6, np.min(fbands), fs=fs, btype='high', analog=False)
        self.FILTER_HIGH = signal.butter(6, np.max(fbands) + 10, fs=fs, btype='low', analog=False)

        self.FILTERS_ATTENUATE = []
        if filters_attenuate == 2:
            self.FILTERS_ATTENUATE = [signal.iirnotch(2 * k, Q=(20 + k * 2) * 2, fs=fs) for k in
                                      range(1, int(np.floor(np.max(fbands) / 2)) + 1)]
        elif filters_attenuate == 7:
            self.FILTERS_ATTENUATE = [signal.iirnotch(7 * k, Q=(2 + k * 2) * 7, fs=fs) for k in
                                      range(1, int(np.floor(np.max(fbands) / 7)) + 1)]

        if isinstance(filters_attenuate, list):
            if filters_attenuate.__len__() > 0:
                self.FILTERS_ATTENUATE = filters_attenuate
        self.ATTENUATE_N = repeat_attenuate

        ##### FEATURE EXTRACTION SETUP ###########

        self._ignore_bands = ignore_bands
        self.datarate = datarate
        self.n_processes = self._verify_input_n_processes(n_processes)

        self._sperwelchseg = sperwelchseg
        self._soverlapwelchseg = soverlapwelchseg
        self._nfft = nfft

        # self.extraction_functions = [
        #    normalized_entropy, mean_frequency, median_frequency, mean_bands, relative_bands, normalized_entropy_bands
        # ]  # property setter

        self.extraction_functions = [
            mean_bands, relative_bands
        ]  # property setter

    def design_filters(self):
        if self._filter_bands:
            self.FILTERS = []
            self.FILTER_RESPONSE = None
            trans = None
            for band in self._filter_bands:
                cutoff_low = 2 * band[1] / self._fs
                cutoff_high = 2 * band[0] / self._fs

                b_low = signal.firwin(self._nfiltorder, cutoff_low, pass_zero='lowpass')
                if band[0] > 0:
                    b_high = signal.firwin(self._nfiltorder, cutoff_high, pass_zero='highpass')
                    b = signal.convolve(b_low, b_high, mode='same')
                else:
                    b = cutoff_low

                w, h = signal.freqz(b, [1])
                b /= h.__abs__().max()

                self.FILTERS += [b]

                w, h = signal.freqz(b, [1])
                w = w / w.max()
                w = w * 0.5 * self._fs

                if isinstance(trans, type(None)):
                    trans = h
                else:
                    trans = trans + h
            self.FILTER_RESPONSE = {'x': w, 'y': trans}

    def __call__(
            self,
            x: (np.ndarray, list) = None
    ):

        """

        Parameters
        ----------
        x : np.ndarray
        fs
        segm_size
        sub_segment_size
        sub_segment_overlap
        fbands
        bands_to_erase
        datarate
        n_processes

        Returns
        -------

        """
        # Standard parameters
        if isinstance(x, np.ndarray):
            return self.process_signal(x=x)

        if isinstance(x, list) and x.__len__() == 1:
            return self.process_signal(x=x[0])

        else:
            if self.n_processes == 1:
                output = []
                for signal in x:
                    out_tuple = self.process_signal(x=signal)
                    output.append(out_tuple)
                return output
            else:
                with multiprocessing.Pool(self.n_processes) as p:
                    print('It is happening')
                    pfunc = partial(self.process_signal)
                    output = p.map(pfunc, x)
                return output

    def prefilter(self, x):
        x = signal.filtfilt(*self.FILTER_LOW, x, axis=1)
        x = signal.filtfilt(*self.FILTER_HIGH, x, axis=1)
        return x

    def attenuate(self, x):
        if isinstance(self.FILTERS_ATTENUATE, list):
            if self.FILTERS_ATTENUATE.__len__() > 0 and self.ATTENUATE_N > 0:
                for k in range(self.ATTENUATE_N):
                    for f in self.FILTERS_ATTENUATE:
                        x = signal.filtfilt(*f, x, axis=1)
        return x

    def process_signal(self, x):
        """

        Parameters
        ----------
        x


        Returns
        -------

        """
        x = x.copy().squeeze()
        features = []
        msg = []

        segm_size = self._segm_size
        fs = self._fs
        datarate = self.datarate
        sperwelchseg = self._sperwelchseg
        soverlapwelchseg = self._soverlapwelchseg
        fbands = self._fbands
        nfft = self._nfft
        ignore_bands = self._ignore_bands

        if x.shape[0] < fs * segm_size:  # if the signal is shorter than segm_size, appends zeros
            x = np.append(x, np.zeros(int(round(fs * segm_size)) - x.shape[0]))

        xbuffered = buffer(x, fs, segm_size)
        if datarate is True:
            features = features + [1 - (np.isnan(xbuffered).sum(axis=1) / (segm_size * fs))]
            msg = msg + ['DATA_RATE']
        xbuffered = xbuffered - np.nanmean(xbuffered, axis=1).reshape((-1, 1))
        xbuffered[np.isnan(xbuffered)] = 0

        xbuffered = self.prefilter(xbuffered)
        xbuffered = self.attenuate(xbuffered)

        # TODO: REDO FILTERING

        if isinstance(sperwelchseg, type(None)):
            soverlapwelchseg = 0
        else:
            sperwelchseg = int(np.round(sperwelchseg * fs))
            soverlapwelchseg = int(np.round(soverlapwelchseg * fs))
        freq, psd = PSD(xbuffered, fs, nperseg=sperwelchseg, noverlap=soverlapwelchseg, nfft=nfft)
        freq = freq[1:]  # remove 0Hz sample
        psd = psd[:, 1:]

        if ignore_bands.__len__() > 0:
            for eband in ignore_bands:
                psd[:, (freq > eband[0]) & (freq < eband[1])] = np.nan

        inp_params = ObjDict({
            'psd': psd,
            'fs': fs,
            'fbands': fbands,
            'segm_size': segm_size,
            'freq': freq
        })

        for func in self._extraction_functions:
            feature, ftr_name = func(inp_params)
            features = features + feature
            msg = msg + ftr_name
        return features, msg

    @property
    def extraction_functions(self):
        return self._extraction_functions

    @extraction_functions.setter
    def extraction_functions(self, item: list):
        self._extraction_functions = item
        self._verify_extractor_functions()

    @staticmethod
    def _verify_input_fs(item):
        if not isinstance(item, (int, float)):
            raise TypeError('[INPUT TYPE ERROR] Sampling frequency \"fs\" has to be an integer or float!')
        if not item > 0:
            raise ValueError(
                '[INPUT VALUE ERROR] Sampling frequency is required to be higher than 0! Pasted value: ' + str(item))
        return item

    @staticmethod
    def _verify_input_segm_size(item):
        if not isinstance(item, (int, float)):
            raise TypeError(
                '[INPUT TYPE ERROR] A segment size \"segm_size\" is required to be an integer or float. Parsed data type is ' + str(
                    type(item)))
        if not item > 0:
            raise ValueError('[INPUT VALUE ERROR] A segment size \"segm_size\" is required to be  higher than 0!')
        if item == np.inf:
            raise ValueError('[INPUT VALUE ERROR] A segment size \"segm_size\" cannot be Inf')
        return item

    @staticmethod
    def _verify_input_fbands(item):
        if not isinstance(item, (list, np.ndarray)):
            raise TypeError(
                '[INPUT TYPE ERROR] fbands variable has to be of a list or numpy.array type. Pasted value: ' + str(
                    type(item)))
        if not item.__len__() > 0:
            raise ValueError(
                '[INPUT SIZE ERROR] Length of fbands has to be > 0. Current length: ' + str(item.__len__()))
        for idx, subitem in enumerate(item):
            if not subitem.__len__() == 2:
                raise TypeError(
                    '[INPUT SIZE ERROR] Length of each frequency band in fband variable has to contain exactly 2 elements min and max frequency for a given bandwidth. Current size: ' + str(
                        subitem.__len__()))
            if not subitem[0] < subitem[1]:
                raise ValueError('[INPUT VALUE ERROR] For a bandwidth in variable fbands with index ' + str(
                    idx) + ' an error has been found. The first value has to be lower than the second one! Current input: ' + str(
                    subitem))
        return np.array(item)

    @staticmethod
    def _verify_input_x(item):
        if not isinstance(item, (np.ndarray, list)):
            raise TypeError(
                '[INPUT TYPE ERROR] An input signal has to be a type of list or numpy.ndarray. Pasted ' + str(
                    type(item)) + ' instead.')

        if isinstance(item, np.ndarray):
            if not (item.shape.__len__() == 1 or item.shape.__len__() == 2):
                raise TypeError(
                    '[INPUT SIZE ERROR] An input signal has to consist of an input of a single dimension for a single signal, 2D numpy.ndarray field for multiple signals (n_signal, signal_length), or list containing multiple fields with a single signal in each of these cells.')

        if isinstance(item, list):
            for subitem in item:
                if not isinstance(subitem, np.ndarray):
                    raise TypeError(
                        '[INPUT SIZE ERROR] An input signal has to consist of an input of a single dimension for a single signal, 2D numpy.ndarray field for multiple signals (n_signal, signal_length), or list containing multiple fields with a single signal in each of these cells.')

        return item

    @staticmethod
    def _verify_input_n_processes(item):
        if not isinstance(item, int):
            raise TypeError('[INPUT TYPE ERROR] Input n_processes has to be of a type int. Type ' + str(
                type(input)) + ' has found instead.')
        if item < 1:
            raise ValueError(
                '[INPUT VALUE ERROR] Number of processes dedicated to feature extraction should be > than 0.')
        if item > multiprocessing.cpu_count() / 2:
            raise PendingDeprecationWarning(
                '[INPUT VALUE ERROR] Number of processes dedicated to feature extraction shouldn\'t be higher than half of the number of processors. This can significantly slow down the processing time and decrease performance. Value is decreased to a number ' + str(
                    multiprocessing.cpu_count() / 2))
            return int(multiprocessing.cpu_count() / 2)
        return item

    def _verify_extractor_functions(self):
        if self._extraction_functions.__len__() < 1:
            raise TypeError('')

        for idx, func in enumerate(self._extraction_functions):
            if not callable(func):
                raise TypeError('[FUNCTION ERROR] A feature extraction function ' + str(func) + ' with an index ' + str(
                    idx) + ' is not callable')


class SleepSpectralFeatureExtractor:
    __version__ = "1.0.0"
    """
    Spectral feature extractor designed for sleep classification

    ...

    Attributes
    ----------
    extraction_functions : list
        List of functions for parameter extraction from np.ndarray where shape is (n_spectrums, spectrum_samples).
        Each function has to return tuple (features : list, name_of_features : list). By default set to extract all features within this class.

    ...

    Methods
    --------

    ...

    Notes
    ------
    v0.1.0 Updates
    - communication between functions (Pxx, fs, ...) changed to ObjDict - see AISC.types.ObjDict
    - float value frequency bands enabled
    v0.1.1 Updates
    - bands_to_erase as an input into __call__ - erases defined bands of psd
    v0.1.2 Updates
    - self._extraction_functions init moved to __init__    
    v0.2.0
        - methods buffer, PSD transferred into AISC.utils.signal
        - PSD replaced by welch's method implementation by scipy.signal.welch
        - implementation of spectral feature extraction from shorter segments and average them
        - removed filtering due switch to welch method of periodogram - constant detrending
            - can be replaced with LowFrequencyFilter later on
        - automatic windown implemented in welch method - hann window

    v1.0.0
        - implemented combining of fbands and bands to erase to get bands for subsequent filtration
        - implemented filtering of frequency bands to eliminate leakage of surrounding frequencies
        - parameter init for feature extraction was moved from __call__ and process_signal to __init__
        - feature extraction functions were moved into AISC.FeatureExtractor.SpectralFeatures

    """

    def __init__(self,
                 fs: float = None,
                 segm_size: float = None,
                 fbands: list = None,
                 sperwelchseg: float = None,
                 soverlapwelchseg: float = 0.0,
                 nfft: int = None,
                 bands_to_erase: list = None,
                 filter_bands: bool = False,
                 nfiltorder: int = 1001,
                 datarate: bool = False,
                 n_processes: int = 1):

        self._fs = self._verify_input_fs(fs)
        self._segm_size = self._verify_input_segm_size(segm_size)
        self._fbands = self._verify_input_fbands(fbands)
        if isinstance(bands_to_erase, type(None)): bands_to_erase = []

        self.FILTERS = None
        if bands_to_erase.__len__() == 0:
            self._bands_to_erase = []
        else:
            self._bands_to_erase = self._verify_input_fbands(bands_to_erase)

        self._nfiltorder = nfiltorder
        self._filter_bands = filter_bands
        # self._filter_bands = None
        if self._filter_bands:
            self._filter_bands = self._get_filter_bands(self._fbands, self._bands_to_erase)
            self.design_filters()

        self.datarate = datarate
        self.n_processes = self._verify_input_n_processes(n_processes)

        self._sperwelchseg = sperwelchseg
        self._soverlapwelchseg = soverlapwelchseg
        self._nfft = nfft

        self.extraction_functions = [
            normalized_entropy, mean_frequency, median_frequency, mean_bands, relative_bands, normalized_entropy_bands
        ]  # property setter

    @staticmethod
    def _get_filter_bands(fbands, bands_to_erase=[]):
        """
        Creates bands as cross-section of frequency bands to keep and bands_to_erase for bandpass filtering

        Parameters
        ----------
        fbands
        bands_to_erase

        Returns
        -------

        """
        min_band = np.min(fbands)
        max_band = np.max(fbands)
        if min_band > 0:
            min_step = np.log10(min_band)
            min_step = 10 ** np.floor(min_step)
        else:
            min_step = np.log10(np.unique(fbands)[1])
            min_step = 10 ** np.floor(min_step)

        min_step = min_step / 10

        frange = np.arange(min_band, max_band, min_step)
        area_erase = np.zeros_like(frange, dtype=bool)
        area_keep = np.zeros_like(frange, dtype=bool)

        for band in fbands:
            area_keep[(frange >= band[0]) & (frange <= band[1])] = True

        for band in bands_to_erase:
            area_erase[((frange >= band[0]) & (frange <= band[1]))] = True

        area = ((area_keep) & (~area_erase)).astype(np.float)
        diff = np.diff(np.concatenate((np.array([0]), area, np.array([0]))))
        t0 = frange[np.where(diff == 1)[0]]
        t1 = frange[np.where(diff == -1)[0] - 1] + min_step
        outp_bands = [list(z) for z in zip(t0, t1)]
        return outp_bands

    def design_filters(self):
        if self._filter_bands:
            self.FILTERS = []
            self.FILTER_RESPONSE = None
            trans = None
            for band in self._filter_bands:
                cutoff_low = 2 * band[1] / self._fs
                cutoff_high = 2 * band[0] / self._fs

                b_low = signal.firwin(self._nfiltorder, cutoff_low, pass_zero='lowpass')
                if band[0] > 0:
                    b_high = signal.firwin(self._nfiltorder, cutoff_high, pass_zero='highpass')
                    b = signal.convolve(b_low, b_high, mode='same')
                else:
                    b = cutoff_low

                w, h = signal.freqz(b, [1])
                b /= h.__abs__().max()

                self.FILTERS += [b]

                w, h = signal.freqz(b, [1])
                w = w / w.max()
                w = w * 0.5 * self._fs

                if isinstance(trans, type(None)):
                    trans = h
                else:
                    trans = trans + h
            self.FILTER_RESPONSE = {'x': w, 'y': trans}

    def __call__(
            self,
            x: (np.ndarray, list) = None
    ):

        """

        Parameters
        ----------
        x : np.ndarray
        fs
        segm_size
        sub_segment_size
        sub_segment_overlap
        fbands
        bands_to_erase
        datarate
        n_processes

        Returns
        -------

        """
        # Standard parameters

        if isinstance(x, np.ndarray):
            return self.process_signal(x=x)

        if isinstance(x, list) and x.__len__() == 1:
            return self.process_signal(x=x[0])

        else:
            if self.n_processes == 1:
                output = []
                for signal in x:
                    out_tuple = self.process_signal(x=signal)
                    output.append(out_tuple)
                return output
            else:
                with multiprocessing.Pool(self.n_processes) as p:
                    pfunc = partial(self.process_signal)
                    output = p.map(pfunc, x)
                return output

    def process_signal(self, x):
        """

        Parameters
        ----------
        x


        Returns
        -------

        """
        x = x.copy().squeeze()
        features = []
        msg = []

        segm_size = self._segm_size
        fs = self._fs
        datarate = self.datarate
        sperwelchseg = self._sperwelchseg
        soverlapwelchseg = self._soverlapwelchseg
        bands_to_erase = self._bands_to_erase
        fbands = self._fbands
        nfft = self._nfft

        if x.shape[0] < fs * segm_size:  # if the signal is shorter than segm_size, appends zeros
            x = np.append(x, np.zeros(int(round(fs * segm_size)) - x.shape[0]))

        xbuffered = buffer(x, fs, segm_size)
        if datarate is True:
            features = features + [1 - (np.isnan(xbuffered).sum(axis=1) / (segm_size * fs))]
            msg = msg + ['DATA_RATE']
        xbuffered = xbuffered - np.nanmean(xbuffered, axis=1).reshape((-1, 1))
        xbuffered[np.isnan(xbuffered)] = 0

        # filter bands
        if self._filter_bands:
            xbuffered_ = None
            for b in self.FILTERS:
                xbuffered__ = np.apply_along_axis(lambda m: np.convolve(m, b, mode='full'), axis=1, arr=xbuffered)
                if isinstance(xbuffered_, type(None)):
                    xbuffered_ = xbuffered__
                else:
                    xbuffered_ += xbuffered__
            xbuffered = xbuffered_

        if isinstance(sperwelchseg, type(None)):
            soverlapwelchseg = 0
        else:
            sperwelchseg = int(np.round(sperwelchseg * fs))
            soverlapwelchseg = int(np.round(soverlapwelchseg * fs))
        freq, psd = PSD(xbuffered, fs, nperseg=sperwelchseg, noverlap=soverlapwelchseg, nfft=nfft)
        freq = freq[1:]  # remove 0Hz sample
        psd = psd[:, 1:]

        if bands_to_erase.__len__() > 0:
            for eband in bands_to_erase:
                psd[:, (freq > eband[0]) & (freq < eband[1])] = 0

        inp_params = ObjDict({
            'psd': psd,
            'fs': fs,
            'fbands': fbands,
            'segm_size': segm_size,
            'freq': freq
        })

        for func in self._extraction_functions:
            feature, ftr_name = func(inp_params)
            features = features + feature
            msg = msg + ftr_name
        return features, msg

    @property
    def extraction_functions(self):
        return self._extraction_functions

    @extraction_functions.setter
    def extraction_functions(self, item: list):
        self._extraction_functions = item
        self._verify_extractor_functions()

    @staticmethod
    def _verify_input_fs(item):
        if not isinstance(item, (int, float)):
            raise TypeError('[INPUT TYPE ERROR] Sampling frequency \"fs\" has to be an integer or float!')
        if not item > 0:
            raise ValueError(
                '[INPUT VALUE ERROR] Sampling frequency is required to be higher than 0! Pasted value: ' + str(item))
        return item

    @staticmethod
    def _verify_input_segm_size(item):
        if not isinstance(item, (int, float)):
            raise TypeError(
                '[INPUT TYPE ERROR] A segment size \"segm_size\" is required to be an integer or float. Parsed data type is ' + str(
                    type(item)))
        if not item > 0:
            raise ValueError('[INPUT VALUE ERROR] A segment size \"segm_size\" is required to be  higher than 0!')
        if item == np.inf:
            raise ValueError('[INPUT VALUE ERROR] A segment size \"segm_size\" cannot be Inf')
        return item

    @staticmethod
    def _verify_input_fbands(item):
        if not isinstance(item, (list, np.ndarray)):
            raise TypeError(
                '[INPUT TYPE ERROR] fbands variable has to be of a list or numpy.array type. Pasted value: ' + str(
                    type(item)))
        if not item.__len__() > 0:
            raise ValueError(
                '[INPUT SIZE ERROR] Length of fbands has to be > 0. Current length: ' + str(item.__len__()))
        for idx, subitem in enumerate(item):
            if not subitem.__len__() == 2:
                raise TypeError(
                    '[INPUT SIZE ERROR] Length of each frequency band in fband variable has to contain exactly 2 elements min and max frequency for a given bandwidth. Current size: ' + str(
                        subitem.__len__()))
            if not subitem[0] < subitem[1]:
                raise ValueError('[INPUT VALUE ERROR] For a bandwidth in variable fbands with index ' + str(
                    idx) + ' an error has been found. The first value has to be lower than the second one! Current input: ' + str(
                    subitem))
        return np.array(item)

    @staticmethod
    def _verify_input_x(item):
        if not isinstance(item, (np.ndarray, list)):
            raise TypeError(
                '[INPUT TYPE ERROR] An input signal has to be a type of list or numpy.ndarray. Pasted ' + str(
                    type(item)) + ' instead.')

        if isinstance(item, np.ndarray):
            if not (item.shape.__len__() == 1 or item.shape.__len__() == 2):
                raise TypeError(
                    '[INPUT SIZE ERROR] An input signal has to consist of an input of a single dimension for a single signal, 2D numpy.ndarray field for multiple signals (n_signal, signal_length), or list containing multiple fields with a single signal in each of these cells.')

        if isinstance(item, list):
            for subitem in item:
                if not isinstance(subitem, np.ndarray):
                    raise TypeError(
                        '[INPUT SIZE ERROR] An input signal has to consist of an input of a single dimension for a single signal, 2D numpy.ndarray field for multiple signals (n_signal, signal_length), or list containing multiple fields with a single signal in each of these cells.')

        return item

    @staticmethod
    def _verify_input_n_processes(item):
        if not isinstance(item, int):
            raise TypeError('[INPUT TYPE ERROR] Input n_processes has to be of a type int. Type ' + str(
                type(input)) + ' has found instead.')
        if item < 1:
            raise ValueError(
                '[INPUT VALUE ERROR] Number of processes dedicated to feature extraction should be > than 0.')
        if item > multiprocessing.cpu_count() / 2:
            raise PendingDeprecationWarning(
                '[INPUT VALUE ERROR] Number of processes dedicated to feature extraction shouldn\'t be higher than half of the number of processors. This can significantly slow down the processing time and decrease performance. Value is decreased to a number ' + str(
                    multiprocessing.cpu_count() / 2))
            return int(multiprocessing.cpu_count() / 2)
        return item

    def _verify_extractor_functions(self):
        if self._extraction_functions.__len__() < 1:
            raise TypeError('')

        for idx, func in enumerate(self._extraction_functions):
            if not callable(func):
                raise TypeError('[FUNCTION ERROR] A feature extraction function ' + str(func) + ' with an index ' + str(
                    idx) + ' is not callable')




















