import warnings
import numpy as np
from scipy.stats import ttest_1samp


def crp_method(v, t_win, prune_the_data):
    """
    :param v: ndarray
        Voltage matrix of shape (n_samples, n_channels).
    :param t_win: ndarray
        Time array of shape (n_samples,) representing the time window.
    :param prune_the_data: bool
        Flag indicating whether to prune the data or not.
    :return: tuple
        Tuple containing the output parameters and projections.

    The function `crp_method` implements the Canonical Response Parametrization (CRP) method for ERP analysis of brain electrophysiology data. It takes in a voltage matrix `v`, a time array `t_win`, and a flag `prune_the_data
    *`. It returns a tuple containing the output parameters and projections. Note that the methodology behind this is described in the manuscript:
    "Canonical Response Parametrization: Quantifying the structure of responses to single-pulse intracranial electrical brain stimulation"
    by Kai J. Miller, et al., 2022. Python implementation by Vaclav Kremen, 1/2024 version 1.0.

    The voltage matrix `v` is a 2D ndarray of shape `(n_samples, n_channels)`, where `n_samples` is the number of samples and `n_channels` is the number of channels.

    The time array `t_win` is a 1D ndarray of shape `(n_samples,)` representing the time window.

    The flag `prune_the_data` is a boolean indicating whether to prune the data or not.

    The function performs the following steps:

    1. Calculates the sampling rate based on the time window.
    2. Calculates sets of normalized single stimulation cross-projection magnitudes using a specified time step.
    3. Parameterizes the trials by reducing the voltage matrix to the response duration, performing kernel trick PCA to capture structure, and calculating the first principal component and
    * residual epsilon.
    4. Returns the projections data, including projection timepoints, mean and variance of projection profiles, response duration index, average and standard deviation of input traces.
    5. Calculates significance statistics, such as the t-value and p-value at the response duration and full time sent in.
    6. Returns the parameterization data, including the reduced voltage matrix, alpha coefficient weights, canonical shape, residual epsilon, response time, parameter timepoints, average
    * and standard deviation of response traces, alpha coefficient weights normalized by the square root of the length of C, and the square root of the diagonal elements of ep.T @ ep.
    7. Calculates extracted single-trial quantities, such as signal-to-noise ratio (Vsnr) and explained variance (expl_var) for each trial.
    8. Optionally prunes the data if requested, by removing trials that are too far from the given template and outliers.
    9. Returns the final output parameters and projections.

    Example usage:
    ```python
    v = np.zeros((10, 1000))
    t_win = np.arange(0, 1000/fs, 1/fs)
    prune_the_data = True

    crp_parameters, crp_projections = crp_method(v, t_win, prune_the_data)
    ```
    """

    # region For testing purposes
    # Define the time range and sampling rate 1 kHz
    # fs = 1000
    # t_win = np.arange(0, 1000/fs, 1/fs)
    # fs = 1 / (t_win[1] - t_win[0])
    # # Create the ndarray 'v' with 1 sinusoid for testing
    # v = np.zeros((10, 1000))
    # for i in range(10):
    #     v[i] = np.sin(2 * np.pi * t_win)
    # v = np.transpose(v)
    # plt.plot(t_win, v[:, 9])
    # plt.show()
    # endregion For testing purposes

    # Initial housekeeping
    sampling_rate = 1 / np.mean(np.diff(t_win))  # Get sampling rate

    # Calculate sets of normalized single stimulation cross-projection magnitudes
    t_step = 5  # Timestep between timepoints (in samples)
    proj_tpts = np.arange(10, v.shape[0], t_step)  # Timepoints for calculation of profile (in samples)
    m = []  # Mean projection magnitudes
    v2 = []  # Variance of projection magnitudes
    for k in proj_tpts:  # Parse through time and perform projections for different data lengths
        # Get projection magnitudes for this duration
        s = ccep_proj(v[:k, :])
        # Change units from uV*sqrt(samples) to sqrt(seconds)
        s = s / np.sqrt(sampling_rate)
        # Calculate mean and variance of projections for this duration
        m.append(np.mean(s))
        v2.append(np.var(s))
        try:
            s_all
        except NameError:
            s_all = np.zeros((len(s), 1))
            s_all = np.append(s_all, s.reshape((-1, 1)), axis=1)  # Store projection weights
        else:
            s_all = np.append(s_all, s.reshape((-1, 1)), axis=1)  # Store projection weights
    s_all = s_all[:, 1:]  # Remove the first column of zeros
    tt = np.argmax(m)  # tt is the sample corresponding to response duration

    # Parameterize trials
    v_t_r = v[:proj_tpts[tt], :]  # Reduced length voltage matrix (to response duration)
    e_t_r, _ = kt_pca(v_t_r)  # Linear kernel trick PCA method to capture structure
    # 1st PC, canonical shape, C(t) from paper
    c = e_t_r[:, 0]
    # Mean shape
    # c = np.mean(v_tR, axis=1)
    # c = c / np.linalg.norm(c)
    al = np.dot(c, v_t_r)  # Alpha coefficient weights for C into V
    ep = v_t_r - np.outer(c, al)  # Residual epsilon after removal of a form of CCEP

    # Output variables, package data out
    # Projections data
    crp_projections = {'proj_tpts': t_win[proj_tpts], 's_all': s_all, 'mean_proj_profile': m, 'var_proj_profile': v2,
                       'tR_index': tt, 'avg_trace_input': np.mean(v, axis=1), 'std_trace_input': np.std(v, axis=1)}

    # Significance statistics - note that have to send in only non-overlapping trials.
    # Each trial is represented half of the time as the normalized projected, and half as non-normalized projected-into
    stat_indices = get_stat_indices(v.shape[1])
    crp_projections['stat_indices'] = stat_indices

    # t-statistic at response duration \tau_R
    crp_projections['t_value_tR'] = np.mean(s_all[stat_indices, tt]) / (
            np.std(s_all[stat_indices, tt]) / np.sqrt(len(s_all[stat_indices, tt])))  # Calculate t-statistic

    # p-value at response duration \tau_R (extraction significance)
    _, crp_projections['p_value_tR'] = ttest_1samp(s_all[stat_indices, tt], 0, alternative='greater')

    # t-statistic at full time sent in
    crp_projections['t_value_full'] = np.mean(s_all[stat_indices, -1]) / (
            np.std(s_all[stat_indices, -1]) / np.sqrt(len(s_all[stat_indices, -1])))  # Calculate t-statistic

    # p-value at full time sent in (extraction significance)
    _, crp_projections['p_value_full'] = ttest_1samp(s_all[stat_indices, -1], 0, alternative='greater')

    # Parameterization
    crp_parameters = {'V_tR': v_t_r, 'al': al, 'C': c, 'ep': ep, 'tR': t_win[proj_tpts[tt]],
                      'parms_times': t_win[:proj_tpts[tt]], 'avg_trace_tR': np.mean(v_t_r, axis=1),
                      'std_trace_tR': np.std(v_t_r, axis=1), 'al_p': al / (len(c) ** 0.5),
                      'epep_root': np.sqrt(np.diag(ep.T @ ep))}

    # Extracted single-trial quantities (e.g. Table 1 in manuscript)
    denominator = np.sqrt(np.diag(ep.T @ ep))
    denominator[denominator == 0] = 1  # Set the denominator to 1 if it is zero
    crp_parameters['Vsnr'] = al / denominator  # "signal-to-noise" for each trial
    denominator = np.diag(np.dot(v_t_r.T, v_t_r))
    denominator = denominator.copy()
    denominator[denominator == 0] = 1  # Set the denominator to 1 if it is zero
    crp_parameters['expl_var'] = 1 - np.diag(np.dot(ep.T, ep)) / denominator

    # If the data pruning was requested, then prune the data that are too far from given template
    # (just one more cycle and get rid of outliers)
    # Aggregated epsilon per trials
    eps = np.sum(np.abs(ep), axis=0)
    # Select the indexes of eps that are higher than the 2*std
    high_eps_indexes = np.where(eps < 2 * np.std(eps))[0]
    if len(high_eps_indexes) > 12:
        if prune_the_data and len(high_eps_indexes) > 6:
            # Prune the data - rerun the CRP with only selected trials
            [crp_parameters, crp_projections] = crp_method(v[:, high_eps_indexes], t_win, False)

    return crp_parameters, crp_projections


def ccep_proj(V):
    """
    Perform projections of each trial onto all other trials, and return the internal projections.

    :param V: The input matrix of shape (M, N), collector of trials (each trial is a column).
    :type V: numpy.ndarray
    :return: The calculated internal projections vector after removing self-projections.
    :rtype: numpy.ndarray
    """

    # Normalize (L2 norm) each trial
    denominator = np.sqrt(np.sum(V ** 2, axis=0))[np.newaxis, :]
    denominator[denominator == 0] = 1  # Set the denominator to 1 if it is zero
    V0 = V / denominator
    V0[np.isnan(V0)] = 0  # Taking care of a divide-by-zero situation in normalization

    # Calculate internal projections (semi-normalized - optimal)
    P = np.dot(V0.T, V)  # Calculate internal projections (semi-normalized - optimal)

    # Get only off-diagonal elements of P (i.e., ignore self-projections)
    p0 = P.copy()
    np.fill_diagonal(p0, np.nan)
    S0 = np.reshape(p0, (1, -1))  # Reshaping to 1D array
    S0 = S0[~np.isnan(S0)]  # Removing diagonal elements (self-projections)
    return S0


def kt_pca(X):
    """
    This is an implementation of the linear kernel PCA method ("kernel trick")
    described in "Kernel PCA Pattern Reconstruction via Approximate Pre-Images"
    by Scholkopf et al., ICANN, 1998, pp 147-15.

    param: X - Matrix of data in. Only need this trick if T>>N

    :return: E, S - Eigenvectors and Eigenvalues of X in descending order
    """

    # Use the "kernel trick" to estimate eigenvectors of this cluster of pair groups
    S2, F = np.linalg.eig(X.T @ X)  # Eigenvector decomposition of (covariance of transpose)

    idx = np.argsort(S2)[::-1]  # Indices to sort eigenvectors in descending order
    S2 = S2[idx]  # Sort eigenvalues in descending order
    F = F[:, idx]  # Sort eigenvectors in descending order
    # Ignore warnings of the DeprecationWarning category
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # A statement that may raise a warning
    # S = np.sqrt(S2)  # Estimated eigenvalues of both X.T @ X and X @ X.
    # Catch any remaining warnings
    with warnings.catch_warnings(record=True) as w:
        # Execute another statement that may raise a warning
        # TODO: check with Kai or somewhere if all these exceptions handling
        S = np.sqrt(np.abs(S2))  # Estimated eigenvalues of both X.T @ X and X @ X.
        # Print any warnings that were caught
        for warning in w:
            print(warning)

    ES = X @ F  # Kernel trick
    denominator = (np.ones((X.shape[0], 1)) @ S.reshape(1, -1))  # Denominator for normalization
    denominator[denominator == 0] = 1  # Set the denominator to 1 if it is zero
    E = ES / denominator  # Divide through to obtain unit-normalized eigenvectors

    return E, S


def get_stat_indices(N):
    """
    This function picks out the indices of S that can be used for statistical comparison.
    For each trial, half of normalized projections to other trials are used,
    and the other half of trials are the projected into ones. No overlapping comparison pairs are used.

    :param N: Scalar - number of trials
    :return: stat_indices (N^2-N,1) - Vector of indices to be used for statistical comparison
    """
    stat_indices = np.arange(1, N ** 2 - N + 1, 2)  # Indices used for statistics

    if N % 2 == 1:  # Odd number of trials - need to offset every other column in the original P matrix
        b = np.zeros_like(stat_indices)  # Initializes what is indexed
        for k in range(1, N + 1):
            if k % 2 == 0:  # Offset what would have been every even column in the original matrix
                b[((k - 1) * ((N - 1) // 2) + 1):(k * ((N - 1) // 2))] = 1

        stat_indices = stat_indices + b

    return stat_indices
