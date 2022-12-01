# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import tz
from copy import deepcopy
from tqdm import tqdm

from best.hypnogram.visualisation import plot_hypnogram
from best.hypnogram.utils import merge_annotations, filter_by_key


"""
Tools for analyzing hypnograms such as number of cycles, sleep time etc.
"""


def get_hypnogram_datarate(df):
    #return (df['start'][1:].reset_index(drop=True) - df['end'][:-1].reset_index(drop=True)).sum().seconds
    return ((df['duration'].sum()) / (df.iloc[-1]['end'] - df.iloc[0]['start']).seconds)


def get_fell_asleep_time(df, t_sleep_check=60, t_awake_threshold=10, awake_tag='AWAKE', sleep_cycle_tags=['REM', 'N1', 'N2', 'N3']):
    df = filter_by_key(df, 'annotation', 'Arrousal')
    # parameters
    t_sleep = datetime.timedelta(minutes=t_sleep_check) # interval since 1st asleep checked
    t_awake = datetime.timedelta(minutes=t_awake_threshold) # length of all awake cycles during the interval defined by t_sleep since the beginning of the sleep

    # get first asleep
    awake_to_sleep_changes = np.where(
        np.array([df.annotation[0] in sleep_cycle_tags] +
                 [(df.annotation[k-1] in awake_tag) and (df.annotation[k] in sleep_cycle_tags) for k in range(1, df.__len__())
                  ]))[0]

    fell_asleep_time = df.iloc[0].start
    for idx in awake_to_sleep_changes:
        start = df.start[idx]
        all_starts = df.start
        t_sleep_window_df = df.loc[(start <= all_starts) & (all_starts < (start + t_sleep))]
        time_awake = t_sleep_window_df.duration[t_sleep_window_df.annotation == awake_tag].sum()
        if time_awake < t_awake.seconds:
            fell_asleep_time = start
            break
    return fell_asleep_time


def get_awakening_time(df, t_awake_threshold=90, t_sleep_threshold=10, awake_tag='AWAKE', sleep_cycle_tags=['REM', 'N2', 'N3']):
    df = filter_by_key(df, 'annotation', 'Arrousal')
    t_awake = datetime.timedelta(minutes=t_awake_threshold)
    t_sleep = datetime.timedelta(minutes=t_sleep_threshold)

    sleep_to_awake_changes = np.where(
        np.array([df.annotation[0] in sleep_cycle_tags] +
                 [(df.annotation[k-1] in sleep_cycle_tags) and (df.annotation[k] in awake_tag) for k in range(1, df.__len__())]
                 ))[0]

    if df.iloc[-1].annotation == awake_tag:
        last_awake_time = df.iloc[sleep_to_awake_changes[-1]].start
    else:
        last_awake_time = df.iloc[-1].end

    for idx in sleep_to_awake_changes:
        start = df.start[idx]
        all_starts = df.start
        t_awake_window_df = df.loc[(start <= all_starts) & (all_starts < (start + t_awake))]
        time_awake = t_awake_window_df.duration[t_awake_window_df.annotation == awake_tag].sum()
        time_asleep = get_time_by_key(t_awake_window_df, ['N1', 'N2', 'N3', 'REM'])
        if time_awake >= t_awake.seconds and time_asleep <= t_sleep_threshold:
            last_awake_time = start
            break
    return last_awake_time


def is_sleep_complete(df, awake_tag='AWAKE'):
    return df.iloc[0].annotation == awake_tag == df.iloc[-1].annotation


def get_rem_latency(df, rem_tag='REM', awake_tag='AWAKE'):
    first_rem_start = df.loc[df.annotation == rem_tag].reset_index(drop=True).iloc[0].start
    fall_asleep_start = get_fell_asleep_time(df)
    df_rem = df.loc[(df.end <= first_rem_start) & (df.annotation == awake_tag)].reset_index(drop=True)
    if df_rem.__len__() == 0:
        last_awake_end = fall_asleep_start
    else:
        last_awake_end = df_rem.iloc[-1].end

    return {'last_awake': first_rem_start - last_awake_end, 'fall_asleep': first_rem_start - fall_asleep_start}


def get_number_of_sleep_stages(df, tags ='REM', delay=30):
    if isinstance(tags, str):
        tags = [tags]

    delay = datetime.timedelta(minutes=delay)
    bool_idxes = np.ones(df.__len__(), dtype=bool)
    for tag in tags:
        bool_idxes = (bool_idxes) & (df.annotation == tag)

    df = df.loc[bool_idxes].reset_index(drop=True)

    stage_df = pd.DataFrame()
    for idx, row in enumerate(df.iterrows()):
        if idx == 0:
            stage_df = stage_df.append(row[1], ignore_index=True)
        else:
            if (row[1].start - stage_df.iloc[-1].end).seconds >= delay.seconds:
                stage_df = stage_df.append(row[1], ignore_index=True)

    return (stage_df.annotation == tag).sum()




def get_number_of_awakenings(df, awake_tag='AWAKE', n1_tag='N1', sleep_tags=['N2', 'N3', 'REM']):
    awake_bool = df.annotation == awake_tag
    n1_bool = df.annotation == n1_tag
    sleep_bool = np.zeros_like(n1_bool, dtype=bool)
    for tag in sleep_tags:
        sleep_bool = (sleep_bool) | (df.annotation == tag)

    awake_n1_bool = (awake_bool) | (n1_bool)

    n_awakenings = 0
    sleep_happened = False
    awake_happened = False
    for k in range(sleep_bool.shape[0]):
        if sleep_bool[k] == True:
            sleep_happened = True

        if awake_bool[k] == True:
            awake_happened = True
        else:
            if not awake_n1_bool[k] == True:
                awake_happened = False

        if sleep_happened == True and awake_happened == True:
            n_awakenings += 1
            sleep_happened = False
            awake_happened = False
    return n_awakenings


def get_time_by_key(df, key):
    if isinstance(key, list):
        value = 0
        for single_key in key:
            value += (df.duration[(df.annotation == single_key)]).sum()
        return value
    else:
        return (df.duration[(df.annotation == key)]).sum()


def get_stage_times(df, keys):
    return [get_time_by_key(df, key) for key in keys]


def get_stage_times_dataset(hypnograms:list, keys, verbose=True):
    if verbose:
        return pd.DataFrame([dict([(state, get_time_by_key(hyp, state)) for state in keys]) for hyp in tqdm(hypnograms)])
    return pd.DataFrame([dict([(state, get_time_by_key(hyp, state)) for state in keys]) for hyp in hypnograms])


def score_night(df, plot=False):
    df = filter_by_key(df, 'annotation', 'Arrousal')
    df = merge_annotations(df)

    fell_asleep_time = get_fell_asleep_time(df)
    awakening_time = get_awakening_time(df)
    sleep_complete = is_sleep_complete(df)

    sleep_df = df.loc[(df.start >= fell_asleep_time) & (df.start < awakening_time)].reset_index(drop=True)

    n_complete_sleep_cycles = get_number_of_sleep_stages(sleep_df, tags='REM', delay=30)
    n_awakenings = get_number_of_awakenings(sleep_df)
    rem_latency = get_rem_latency(df)


    n1_sleep_time = get_time_by_key(sleep_df, 'N1')
    n2_sleep_time = get_time_by_key(sleep_df, 'N2')
    n3_sleep_time = get_time_by_key(sleep_df, 'N3')
    rem_sleep_time = get_time_by_key(sleep_df, 'REM')
    awake_sleep_time = get_time_by_key(sleep_df, 'AWAKE')


    if plot == True:
        plot_hypnogram(df)
        plt.stem([fell_asleep_time, awakening_time], [7, 7], linefmt='r', markerfmt='or', basefmt='r')

    return {
        'sleep_complete': sleep_complete,
        'fell_asleep_time': fell_asleep_time,
        'rem_latency_fell_asleep': rem_latency['fall_asleep'].seconds,
        'rem_latency_last_awake': rem_latency['last_awake'].seconds,
        'awakening_time': awakening_time,
        'n_complete_sleep_cycles': n_complete_sleep_cycles,
        'n_awakenings': n_awakenings,
        'n1_sleep_time': n1_sleep_time,
        'n2_sleep_time': n2_sleep_time,
        'n3_sleep_time': n3_sleep_time,
        'rem_sleep_time': rem_sleep_time,
        'awake_sleep_time': awake_sleep_time
    }


def print_sleep_score(score):
    total_sleep_time =(score['awakening_time'] - score['fell_asleep_time']).seconds
    hours, remainder = divmod(total_sleep_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    rem_lat1 = score['rem_latency_last_awake'].seconds
    hrs_rem1, remainder = divmod(rem_lat1, 3600)
    mins_rem1, secs_rem1 = divmod(remainder, 60)

    rem_lat2 = score['rem_latency_fell_asleep'].seconds
    hrs_rem2, remainder = divmod(rem_lat2, 3600)
    mins_rem2, secs_rem2 = divmod(remainder, 60)

    non_REM = score['n3_sleep_time'] + score['n2_sleep_time'] + score['n1_sleep_time']

    print('Sleep Complete: ', score['sleep_complete'])
    print('Falling asleep: ', score['fell_asleep_time'].strftime('%H:%M:%S'))
    print('Awakening: ', score['awakening_time'].strftime('%H:%M:%S'))
    print('Total Sleep Time: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))
    print('Rem latency - last_awake: {:02}:{:02}:{:02}'.format(int(hrs_rem1), int(mins_rem1), int(secs_rem1)))
    print('Rem latency - fall_asleep: {:02}:{:02}:{:02}'.format(int(hrs_rem2), int(mins_rem2), int(secs_rem2)))

    print('Number of sleep cycles: ', score['n_complete_sleep_cycles'])
    print('Number of awakenings', score['n_awakenings'])
    print()
    print('Sleep-time non-REM')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(non_REM),  non_REM/total_sleep_time))
    print('Sleep-time REM')
    print()
    print('Sleep-time REM')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['rem_sleep_time']),  score['rem_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time awake')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['awake_sleep_time']),  score['awake_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time N1')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['n1_sleep_time']),  score['n1_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time N2')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['n2_sleep_time']),  score['n2_sleep_time']/total_sleep_time))
    print()
    print('Sleep-time N3')
    print('Absolute: {0}  Relative: {1:0.3f}'.format(int(score['n3_sleep_time']),  score['n3_sleep_time']/total_sleep_time))


def get_transition_counts(hyp, states=['AWAKE', 'N1', 'N2', 'N3', 'REM']):
    states = np.array(states)
    matrix = np.zeros((states.__len__(), states.__len__()))
    for k in range(hyp.__len__()-1):
        s1 =  hyp['annotation'][k]
        s2 =  hyp['annotation'][k+1]
        if s1 in states and s2 in states:
            idx1 = np.where(states == hyp['annotation'][k])[0][0]
            idx2 = np.where(states == hyp['annotation'][k+1])[0][0]
            matrix[idx1, idx2] += 1
    return matrix


def get_transition_matrix(hyp, states=['AWAKE', 'N1', 'N2', 'N3', 'REM']):
    m = get_transition_counts(hyp, states)
    m = m / m.sum(axis=1).reshape(-1, 1)
    m[np.isnan(m)] = 0
    return m


def get_transition_matrix_dataset(hypnograms, states=['AWAKE', 'N1', 'N2', 'N3', 'REM']):
    ms = []
    for hyp in hypnograms:
        m = get_transition_counts(hyp, states)
        m = m / m.sum(axis=1).reshape(-1, 1)
        ms += [m]
    ms = np.array(ms)
    return np.nanmean(ms, axis=0), np.nanstd(ms, axis=0)


def transition_matrix_to_change_matrix(m):
    m = deepcopy(m)
    np.fill_diagonal(m, 0)
    m = m / m.sum(axis=1).reshape(-1, 1)
    m[np.isnan(m)] = 0
    return m


def valid_dataset_index_by_duration(hypnograms:list, filt_dict:dict):
    valid_hypnograms = []
    for idx, hyp in enumerate(hypnograms):
        if sum([hyp['duration'].sum() >= v for k, v in filt_dict.items() if k in hyp['annotation'].unique()]) == filt_dict.keys().__len__():
            valid_hypnograms += [idx]
    return valid_hypnograms






