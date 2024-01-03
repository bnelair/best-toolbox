# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pandas as pd
from copy import deepcopy

"""
Tools for correcting telemetry-based hypnograms
"""

def do_median_filtration(df):
    for k in range(1, df.__len__() - 1):
        if df.iloc[k - 1].annotation == df.iloc[k + 1].annotation and df.iloc[k].duration == 30:
            df.iloc[k]['annotation'] = df.iloc[k - 1].annotation
    return df


def fill_same_voids(df, time_threshold=5*60, initial_state='AWAKE'):
    new_df = []
    current_state = initial_state
    current_state_duration = time_threshold
    last_annotation_end = df.iloc[0]['start']

    for idx in range(df.__len__()):
        crow = df.iloc[idx]
        s_ = crow['start']
        e_ = crow['end']
        void = crow['start'] - last_annotation_end
        if void > 0 and void <= time_threshold and current_state == crow['annotation']:
            crow['start'] = last_annotation_end
            crow['duration'] = crow['end'] - crow['start']

        if crow['annotation'] == current_state:
            current_state_duration += crow['duration']
        else:
            current_state_duration = crow['duration']
            current_state = crow['annotation']

        last_annotation_end = e_
        new_df += [crow]
    return pd.DataFrame(new_df).reset_index(drop=True)


def fill_wakerem_voids(df, time_threshold=5*60, initial_state='AWAKE'):
    new_df = []
    current_state = initial_state
    current_state_duration = time_threshold
    last_annotation_end = df.iloc[0]['start']

    for idx in range(df.__len__()):
        crow = df.iloc[idx]
        s_ = crow['start']
        e_ = crow['end']
        void = crow['start'] - last_annotation_end
        if void > 0 and void <= time_threshold and current_state == 'AWAKE' and crow['annotation'] == 'REM':
            vrow = deepcopy(crow)
            vrow['annotation'] = 'AWAKE'
            vrow['start'] = last_annotation_end
            vrow['end'] = crow['start']
            vrow['duration'] = crow['end'] - crow['start']
            new_df += [vrow]

        if crow['annotation'] == current_state:
            current_state_duration += crow['duration']
        else:
            current_state_duration = crow['duration']
            current_state = crow['annotation']

        last_annotation_end = e_
        new_df += [crow]
    return pd.DataFrame(new_df).reset_index(drop=True)


def fill_nonrem_voids(df, time_threshold=5*60, initial_state='AWAKE'):
    new_df = []
    current_state = initial_state
    current_state_duration = time_threshold
    last_annotation_end = df.iloc[0]['start']

    for idx in range(df.__len__()):
        crow = df.iloc[idx]
        s_ = crow['start']
        e_ = crow['end']
        void = crow['start'] - last_annotation_end
        if void > 0 and void <= time_threshold and current_state in ('N2', 'N3', 'N') and crow['annotation'] in (
        'N2', 'N3', 'N'):
            vrow = deepcopy(crow)
            vrow['annotation'] = 'N'
            vrow['start'] = last_annotation_end
            vrow['end'] = crow['start']
            vrow['duration'] = crow['end'] - crow['start']
            new_df += [vrow]

        if crow['annotation'] == current_state:
            current_state_duration += crow['duration']
        else:
            current_state_duration = crow['duration']
            current_state = crow['annotation']

        last_annotation_end = e_
        new_df += [crow]
    return pd.DataFrame(new_df).reset_index(drop=True)


def fill_sleep_voids(df, time_threshold=5*60, initial_state='AWAKE'):
    new_df = []
    current_state = initial_state
    current_state_duration = time_threshold
    last_annotation_end = df.iloc[0]['start']

    for idx in range(df.__len__()):
        crow = df.iloc[idx]
        s_ = crow['start']
        e_ = crow['end']
        void = crow['start'] - last_annotation_end
        if void > 0 and void <= time_threshold and current_state != 'AWAKE' and crow['annotation'] != 'AWAKE':
            vrow = deepcopy(crow)
            vrow['annotation'] = 'SLP'
            vrow['start'] = last_annotation_end
            vrow['end'] = crow['start']
            vrow['duration'] = crow['end'] - crow['start']
            new_df += [vrow]

        if crow['annotation'] == current_state:
            current_state_duration += crow['duration']
        else:
            current_state_duration = crow['duration']
            current_state = crow['annotation']

        last_annotation_end = e_
        new_df += [crow]
    return pd.DataFrame(new_df).reset_index(drop=True)


def correct_rem(df, time_threshold=5*60, initial_state='AWAKE'):
    new_df = []
    current_state = initial_state
    current_state_duration = time_threshold
    last_annotation_end = df.iloc[0]['start']

    for idx in range(df.__len__()):
        crow = df.iloc[idx]
        s_ = crow['start']
        e_ = crow['end']
        void = crow['start'] - last_annotation_end
        if void <= 60 and current_state == 'AWAKE' and crow[
            'annotation'] == 'REM' and current_state_duration >= time_threshold:
            crow['annotation'] = 'AWAKE'

        if crow['annotation'] == 'AWAKE':
            current_state_duration += crow['duration']
        else:
            current_state_duration = 0
        current_state = crow['annotation']

        last_annotation_end = e_
        new_df += [crow]
    return pd.DataFrame(new_df).reset_index(drop=True)


def correct_hypnogram(df, time_threshold=60):
    new_df = deepcopy(df)
    new_df = fill_same_voids(new_df, time_threshold=time_threshold)
    new_df = fill_wakerem_voids(new_df, time_threshold=time_threshold)
    new_df = fill_nonrem_voids(new_df, time_threshold=time_threshold)
    new_df = fill_sleep_voids(new_df, time_threshold=time_threshold)
    new_df = correct_rem(new_df, time_threshold=time_threshold)
    # new_df = do_median_filtration(new_df)
    return new_df













