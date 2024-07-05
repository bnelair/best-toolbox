# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import numpy as np
from copy import deepcopy
from best.signal import LowFrequencyFilter, fft_filter
import os
import matplotlib.pyplot as plt

class WaveDetector:
    """
    A generic wave detector - WaveDetector object detects minimums and maximums of waves in a given frequency bandpass. Designed mainly to detect slow (delta - 0.5 - 4 Hz) waves. WaveDetector can also return statistical report of detected waves such as Δt, slope, peak2peak and min and max values.

    Designed for Delta wave detection. Can be deployed on any frequency range.

    Example
    ^^^^^^^^

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from sleep_classification.WaveDetector import WaveDetector

        fs = 200
        signal_length = 5 # s
        wave_freq1 = 2
        wave_freq2 = 1
        noise_level = 0.5

        t = np.arange(0, signal_length * fs) / fs
        noise = np.random.randn(t.shape[0]) * noise_level
        x1 = np.sin(2*np.pi * wave_freq1 * t)
        x2 = np.sin(2*np.pi * wave_freq2 * t) * 0.5
        x_noise = x1 + x2 + noise

        WDet = WaveDetector(fs=fs, cutoff_low=0.5, cutoff_high=4)

        stats, det = WDet(x_noise)
        min_pos = det['min_pos']
        min_val = det['min_val']
        max_pos = det['max_pos']
        max_val = det['max_val']

        #plt.plot(t, x_noise)
        #plt.xlabel('t [s]')
        #plt.title('Original')
        #plt.show()

        plt.plot(t, x_noise)
        plt.stem(t[min_pos], min_val, 'r')
        plt.stem(t[max_pos], max_val, 'k')
        plt.title('Detections')
        plt.show()

        print('Slope Stats')
        print(stats['slope_stats'])

        print('Peak2Peak Stats')
        print(stats['pk2pk_stats'])

        print('delta_t Stats')
        print(stats['delta_t_stats'])


    .. image:: ../../_images/1_det.png
       :width: 300



    """

    def __init__(self, fs, cutoff_low=0.5, cutoff_high=4):
        """

        Parameters
        ----------
        fs: float
            Sampling frequency
        cutoff_low: float
            The lowest frequency of the wave detects
        cutoff_high : float
            The highest frequency of the wave detects
        """
        self.fs = fs
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.LFFilter = LowFrequencyFilter(fs=fs, cutoff=cutoff_low, n_decimate=2, n_order=101, dec_cutoff=0.3, filter_type='lp')

        self.statistic_functions = [self.min_stats, self.max_stats, self.pk2pk_stats, self.slope_stats, self.delta_t_stats]


    def __call__(self, X, stats=True):
        if not isinstance(X, (list, np.ndarray)):
            raise AssertionError('')

        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = [X]
            else:
                X = list(X)


        min_pos = []
        min_val = []
        max_pos = []
        max_val = []

        for X_ in X:
            min_pos_temp, min_val_temp, max_pos_temp, max_val_temp = self.detect_waves(X_)
            min_pos += list(min_pos_temp)
            min_val += list(min_val_temp)
            max_pos += list(max_pos_temp)
            max_val += list(max_val_temp)


        min_pos = np.array(min_pos)
        min_val = np.array(min_val)
        max_pos = np.array(max_pos)
        max_val = np.array(max_val)

        outp_data = {
            'min_pos': min_pos,
            'min_val': min_val,
            'max_pos': max_pos,
            'max_val': max_val
        }


        if stats is True:
            outp_stats = {}
            for func in self.statistic_functions:
                name = func.__name__
                dict_ = func(min_pos, min_val, max_pos, max_val, self.fs)
                outp_stats[name] = dict_
            return outp_stats, outp_data

        return outp_data


    def detect_waves(self, X):
        X = X - X.mean()
        X = self.LFFilter(X)
        min_pos, max_pos = _find_wave_extremes(X, fs=self.fs, cutoff_low=self.cutoff_low, cutoff_high=self.cutoff_high)
        min_val = X[min_pos]
        max_val = X[max_pos]
        return min_pos, min_val, max_pos, max_val

    @classmethod
    def min_stats(cls, min_pos=None, min_vals=None, max_pos=None, max_vals=None, fs=None):
        return cls.stat(min_vals)

    @classmethod
    def max_stats(cls, min_pos=None, min_vals=None, max_pos=None, max_vals=None, fs=None):
        return cls.stat(max_vals)

    @classmethod
    def pk2pk_stats(cls, min_pos=None, min_vals=None, max_pos=None, max_vals=None, fs=None):
        pk2pk = max_vals - min_vals
        return cls.stat(pk2pk)

    @classmethod
    def slope_stats(cls, min_pos=None, min_vals=None, max_pos=None, max_vals=None, fs=None):
        slope = (max_vals - min_vals) / ((max_pos - min_pos) / fs)
        return cls.stat(slope)

    @classmethod
    def delta_t_stats(cls, min_pos=None, min_vals=None, max_pos=None, max_vals=None, fs=None):
        delta_t = (max_pos - min_pos) / fs
        return cls.stat(delta_t)

    @staticmethod
    def stat(X):
        return {
            'min': X.min(),
            'max': X.max(),
            'mean': X.mean(),
            'std': X.std(),
            'median': np.median(X),
        }


def _find_wave_extremes(X, fs, cutoff_low=0.5, cutoff_high=4):
    X_ = X.copy()
    X = fft_filter(X, fs, cutoff_low, 'hp')
    X = fft_filter(X, fs, cutoff_high, 'lp')

    X_sign = np.sign(np.diff(X))
    X_sign = X_sign[:-1] - X_sign[1:]


    mins = []
    maxes = []
    pos_max = 0
    pos_min = 0
    while pos_max != -1:
        # find 1st min lower than zero
        pos_min, val_min = _get_next_extrem_by_zero(X, X_sign, pos_max, tag='min')
        if pos_min == None:
            pos_max = -1
            break
        pos_min, val_min = _check_next_extreme(X, X_sign, pos_min, tag='min')
        if pos_min == None:
            pos_max = -1
            break
        pos_max, val_max = _get_next_extrem_by_zero(X, X_sign, pos_min, tag='max')
        if pos_max == None:
            pos_max = -1
            break
        pos_max, val_max = _check_next_extreme(X, X_sign, pos_max, tag='max')
        if pos_max == None:
            pos_max = -1
            break

        if pos_max == None:
            pos_max = -1
            break
        else:
            mins += [pos_min]
            maxes += [pos_max]
            #pos_min = pos_max + n_plato

    mins = np.array(mins)
    maxes = np.array(maxes)

    mins = _correct_extreme_positions(X_, extreme_positions=mins, n_samples_interval=round(int((1 / cutoff_high) * fs / 2)), tag='min')
    maxes = _correct_extreme_positions(X_, extreme_positions=maxes, n_samples_interval=round(int((1 / cutoff_high) * fs / 2)), tag='max')

    dif = maxes - mins
    maxes = maxes[dif > ((1/cutoff_high) * fs / 2)]
    mins = mins[dif > ((1/cutoff_high) * fs / 2)]

    dif = maxes - mins
    maxes = maxes[dif < ((1/cutoff_low) * fs / 2)]
    mins = mins[dif < ((1/cutoff_low) * fs / 2)]
    return mins, maxes


def _get_first_extreme(X, X_signum, idx=0, tag=''):
    if tag == 'min':
        func_value = -2
    elif tag == 'max':
        func_value = 2

    pos = np.where(X_signum[idx:] == func_value)[0]
    if pos.__len__() == 0:
        return None, None
    pos = pos[0] + idx
    val = X[pos]
    return pos, val


def _get_next_extrem_by_zero(X, X_signum, curr_pos=0, tag=''):
    if tag == 'min':
        func_value = -2
        zero_comp_func = np.less
    elif tag == 'max':
        func_value = 2
        zero_comp_func = np.greater

    pos = np.where(X_signum[curr_pos:] == func_value)[0]
    if pos.__len__() == 0:
        return None, None
    pos = pos + 1 + curr_pos
    vals = X[pos]

    pos = pos[zero_comp_func(vals, 0)]

    pos = pos[0]
    val = X[pos]
    return pos, val


def _check_next_extreme(X, X_signum, curr_pos=0, tag=''):
    if tag == 'min':
        argfunc = np.argmin
        func_value = -2
        comp_func = np.less
        rev_comp_func = np.greater
        rev_tag = 'max'
    elif tag == 'max':
        argfunc = np.argmax
        func_value = 2
        comp_func = np.greater
        rev_comp_func = np.less
        rev_tag = 'min'

    tag_pos = curr_pos
    tag_val = X[curr_pos]

    curr_pos = curr_pos
    curr_val = X[curr_pos]

    next_rtag_pos = curr_pos + 1
    next_rtag_val = 0

    next_tag_pos = curr_pos + 1
    next_tag_val = 0
    # MAX
    while not rev_comp_func(next_rtag_val, 0):
        next_tag_pos, next_tag_val = _get_first_extreme(X, X_signum, next_tag_pos + 1, tag)
        next_rtag_pos, next_rtag_val = _get_first_extreme(X, X_signum, next_rtag_pos + 1, rev_tag)

        if isinstance(next_rtag_pos, type(None)) or isinstance(next_tag_pos, type(None)):
            tag_pos, tag_val = None, None
            break

        if not rev_comp_func(next_rtag_val, 0):
            if comp_func(next_tag_val, tag_val):
                tag_pos = next_tag_pos
                tag_val = next_tag_val

    return tag_pos, tag_val


def _correct_extreme_positions(X, extreme_positions, n_samples_interval=10, tag=''):
    if tag == 'min':
        argfunc = np.argmin
    elif tag == 'max':
        argfunc = np.argmax

    extreme_positions = deepcopy(extreme_positions)
    for idx, pos in enumerate(extreme_positions):
        idx_from = pos - n_samples_interval
        idx_to = pos + n_samples_interval

        if idx_from < 0: idx_from = 0
        if idx_to >= X.shape[0]: idx_to = X.shape[0] - 1

        new_pos = argfunc(X[idx_from:idx_to]) + idx_from
        extreme_positions[idx] = new_pos
    return extreme_positions


##### Vaclav Detector

def _bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the data.

    Parameters:
    data (array-like): Input data to be filtered.
    lowcut (float): Lower frequency limit of the bandpass filter.
    highcut (float): Upper frequency limit of the bandpass filter.
    fs (float): Sampling frequency of the data.
    order (int, optional): Order of the filter. Default is 4.

    Returns:
    array-like: Filtered data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def _moving_average(data, window_size):
    """
    Apply a moving average filter to the data.

    Parameters:
    data (array-like): Input data to be filtered.
    window_size (int): Window size for the moving average filter.

    Returns:
    array-like: Filtered data.
    """
    # Filter the signal by moving average filter
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def _zero_crossings(data):
    """
    Find the indices where the data crosses zero.

    Parameters:
    data (array-like): Input data.

    Returns:
    array-like: Indices where the data crosses zero.
    """
    return np.where(np.diff(np.signbit(data)))[0]

def _detect_slow_waves(data, zeros, fs, max_distance, min_distance, amplitude_threshold):
    """
    Detect slow waves in the data.

    Parameters:
    data (array-like): Input data.
    zeros (array-like): Indices of zero crossings in the data.
    fs (float): Sampling frequency of the data.
    max_distance (float): Maximum distance between zero crossings for a valid slow wave.
    min_distance (float): Minimum distance between zero crossings for a valid slow wave.
    amplitude_threshold (float): Minimum amplitude for a valid slow wave.

    Returns:
    list: List of tuples, each containing the start index, end index, amplitude, slope, and trough index of each detected slow wave.
    """
    slow_waves = []
    for i in range(len(zeros) - 1):
        start = zeros[i]
        end = zeros[i + 1]
        if min_distance <= (end - start) / fs <= max_distance:  # Check if within min to max distance
            segment = data[start:end]
            trough_idx = np.argmin(segment)
            trough_value = segment[trough_idx]
            if abs(trough_value) >= amplitude_threshold:
                amplitude = abs(trough_value)
                # Ensure trough_idx is not zero to avoid division by zero
                if trough_idx == 0:
                    slope = np.nan  # The slope is undefined
                    amplitude = np.nan  # The amplitude is undefined
                else:
                    slope = amplitude / (trough_idx / fs)
                slow_waves.append((start, end, amplitude, slope, trough_idx + start))
    return slow_waves

def slow_wave_detect(data, fs, max_distance, min_distance, amplitude_threshold, pdf_path, sleep_state,
                     epoch_number, slow_waves_to_remove=None, time_threshold=None, verbose=True):
    """

        The code is based on the following publication and extends and improves the features and methods of original
    wave detections from the paper:

    - Riedner, B. a., Vyazovskiy, V. V., Huber, R., Massimini, M., Esser, S., Murphy, M., & Tononi, G. (2007).
    Sleep homeostasis and cortical synchronization: III. A high-density EEG study of sleep slow waves in humans. Sleep, 30(12), 1643–1657.

    Parameters:
    data (array-like): Input EEG data.
    fs (float): Sampling frequency of the data.
    max_distance (float): Maximum distance between zero crossings for a valid slow wave.
    min_distance (float): Minimum distance between zero crossings for a valid slow wave.
    amplitude_threshold (float): Minimum amplitude for a valid slow wave.
    pdf_path (str): Path to save the plot as a PDF.
    sleep_state (str): Current sleep state.
    epoch_number (int): Current epoch number.
    slow_waves_to_remove (list, optional): List of slow waves to remove.
    time_threshold (float, optional): Time threshold for removing slow waves.
    verbose (bool, optional): If True, plot the results. Default is True.

    Returns:
    tuple: Tuple containing the detected slow waves, their amplitudes and slopes, the mean and standard deviation of the amplitudes and slopes, and the number of detected waves.
    """
    # Filter the signal by 50 msec moving average filter
    filtered_eeg = _moving_average(data, int(0.05 * fs))  # 50 msec moving average filter

    # Detect zero crossings
    zeros = _zero_crossings(filtered_eeg)

    # Identify slow waves
    slow_waves = _detect_slow_waves(filtered_eeg, zeros, fs, max_distance, min_distance, amplitude_threshold)

    # Remove slow waves close to those in 'slow_waves_to_remove' if provided
    if slow_waves_to_remove is not None and time_threshold is not None:
        # Convert time_threshold from seconds to samples
        time_threshold_samples = time_threshold * fs

        def is_wave_close(wave, waves_to_remove, threshold_samples):
            for remove_wave in waves_to_remove:
                if abs(wave[0] - remove_wave[0]) <= threshold_samples:
                    return True
            return False

        slow_waves = [wave for wave in slow_waves if
                      not is_wave_close(wave, slow_waves_to_remove, time_threshold_samples)]

    # Check if no waves are found
    if not slow_waves:
        return None

    # Count the number of detected waves
    num_waves = len(slow_waves)

    # Save the slow wave details to variables
    slow_wave_amplitudes = [wave[2] for wave in slow_waves]
    slow_wave_slopes = [wave[3] for wave in slow_waves]

    # Calculate mean and standard deviation
    mean_amplitude = np.mean(slow_wave_amplitudes) if slow_wave_amplitudes else 0
    std_amplitude = np.std(slow_wave_amplitudes) if slow_wave_amplitudes else 0
    mean_slope = np.mean(slow_wave_slopes) if slow_wave_slopes else 0
    std_slope = np.std(slow_wave_slopes) if slow_wave_slopes else 0

    # Plot the results
    if verbose:
        dpi = 300
        width = 3840 / dpi
        height = 2160 / dpi
        fig, axs = plt.subplots(4, 1, figsize=(width, height), dpi=dpi)
        time_axis = np.arange(len(filtered_eeg)) / fs

        axs[0].plot(time_axis, filtered_eeg, label='Fz-Cz EEG signal', color='lightgray')
        for wave in slow_waves:
            start, end, amplitude, slope, trough = wave
            axs[0].plot(time_axis[start:end], filtered_eeg[start:end], color='black', linewidth=2, label='Found waves')
            axs[0].scatter(time_axis[start], 0, color='red')  # Zero crossing
            axs[0].scatter(time_axis[trough], filtered_eeg[trough], color='blue')  # Trough
            axs[0].plot([time_axis[start], time_axis[trough]], [0, filtered_eeg[trough]], 'g--', label='Slope')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude (µV)')
        axs[0].set_title(
            f'Detected Slow Waves in EEG Signal in {sleep_state} 30-second epoch #{epoch_number}'
            f'\nMean Amplitude: {mean_amplitude:.2f} µV (±{std_amplitude:.2f}), '
            f'Mean Slope: {mean_slope:.2f} µV/s (±{std_slope:.2f}), Number of Waves: {num_waves}')

        for i in range(1, 4):
            start_time = (i - 1) * 10
            end_time = i * 10
            axs[i].plot(time_axis, filtered_eeg, label='Fz-Cz EEG signal', color='lightgray')
            for wave in slow_waves:
                start, end, amplitude, slope, trough = wave
                if start_time <= time_axis[start] <= end_time:
                    axs[i].plot(time_axis[start:end], filtered_eeg[start:end], color='black', linewidth=2)
                    axs[i].scatter(time_axis[start], 0, color='red')  # Zero crossing
                    axs[i].scatter(time_axis[trough], filtered_eeg[trough], color='blue')  # Trough
                    axs[i].plot([time_axis[start], time_axis[trough]], [0, filtered_eeg[trough]], 'g--')
            axs[i].set_xlim(start_time, end_time)
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Amplitude (µV)')
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        plt.savefig(pdf_path, dpi=300, format='pdf')
        plt.close(fig)

    return slow_waves, slow_wave_amplitudes, slow_wave_slopes, mean_amplitude, std_amplitude, mean_slope, std_slope, num_waves



__all__ = [
    'WaveDetector',
    'slow_wave_detect'
]