
EEG Slow Wave Detection and Analysis
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Here we conveniently provide a standalone fully functional code example for analysis related to this project - `One file example <./example_one_file.py>`_.

This enables trialing this code without installing the whole Best Toolbox library.
The codes were also embedded in the BEST Toolbox so they can be freely available upon installing the whole `BEST Toolbox <https://github.com/bnelair/best-toolbox>`_ library.

The documentation to the toolbox is available at `BEST Toolbox <https://github.com/bnelair/best-toolbox>`_.

Tools specific to the project can be accessed on: `BEST Toolbox <https://github.com/bnelair/best-toolbox>`_


Acknowledgement
"""""""""""""""""""""""""""
 |
 | F. Mivalt et V. Kremen et al., “Electrical brain stimulation and continuous behavioral state tracking in ambulatory humans,” J. Neural Eng., vol. 19, no. 1, p. 016019, Feb. 2022, doi: 10.1088/1741-2552/ac4bfd.
 |
 | F. Mivalt et V. Sladky et al., “Automated sleep classification with chronic neural implants in freely behaving canines,” J. Neural Eng., vol. 20, no. 4, p. 046025, Aug. 2023, doi: 10.1088/1741-2552/aced21.
 |
 | Gerla, V., Kremen, V., Macas, M., Dudysova, D., Mladek, A., Sos, P., & Lhotska, L. (2019). Iterative expert-in-the-loop classification of sleep PSG recordings using a hierarchical clustering. Journal of Neuroscience Methods, 317(February), 61?70. https://doi.org/10.1016/j.jneumeth.2019.01.013
 |
 | Kremen, V., Brinkmann, B. H., Van Gompel, J. J., Stead, S. (Matt) M., St Louis, E. K., & Worrell, G. A. (2018). Automated Unsupervised Behavioral State Classification using Intracranial Electrophysiology. Journal of Neural Engineering. https://doi.org/10.1088/1741-2552/aae5ab
 |


Version
""""""""""""""""""
Version 1.0 (2024-06-01) by V. Kremen (Kremen.Vaclav@mayo.edu)


Code
""""""""""""""""""
The code is also attached for convenience in here:

.. code-block:: python

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt

    def bandpass_filter(data, lowcut, highcut, fs, order=4):
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

    def moving_average(data, window_size):
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

    def zero_crossings(data):
        """
        Find the indices where the data crosses zero.

        Parameters:
        data (array-like): Input data.

        Returns:
        array-like: Indices where the data crosses zero.
        """
        return np.where(np.diff(np.signbit(data)))[0]

    def detect_slow_waves(data, zeros, fs, max_distance, min_distance, amplitude_threshold):
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

    def SlowWaveDetect(data, fs, max_distance, min_distance, amplitude_threshold, pdf_path, sleep_state,
                       epoch_number, slow_waves_to_remove=None, time_threshold=None, verbose=True):
        """
        Detect slow waves in EEG data and plot the results.

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
        filtered_eeg = moving_average(data, int(0.05 * fs))  # 50 msec moving average filter

        # Detect zero crossings
        zeros = zero_crossings(filtered_eeg)

        # Identify slow waves
        slow_waves = detect_slow_waves(filtered_eeg, zeros, fs, max_distance, min_distance, amplitude_threshold)

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


