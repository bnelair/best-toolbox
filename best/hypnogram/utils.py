# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from dateutil import tz
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm


def _convert_to_timestamp(x):
    if isinstance(x, (datetime, Timestamp)):
        assert x.tzinfo, '[TIMEZONE ERROR] We allow operating with timezone-aware datatypes. This helps preventing inconsistency and errors.'
        return x.timestamp()
    if isinstance(x, (float, int)): return x
    raise TypeError('[TYPE ERROR]: input variable has to be of a type pandas Timestamp, datetime, float, or int. However ' + type(x) + ' recieved.')


def _convert_to_datetime_utc(x):
    if isinstance(x, (datetime, Timestamp)):
        assert x.tzinfo, '[TIMEZONE ERROR] We allow operating with timezone-aware datatypes. This helps preventing inconsistency and errors.'
        x = x.timestamp()
    if isinstance(x, (float, int)):
        utc = datetime.utcfromtimestamp(x)
        utc = utc.replace(tzinfo=tz.tzutc())

        return utc
    raise TypeError('[TYPE ERROR]: input variable has to be of a type pandas Timestamp, datetime, float, or int. However ' + type(x) + ' recieved.')


def _convert_to_pandas_timestamp_utc(x):
    if isinstance(x, (datetime, Timestamp)):
        assert x.tzinfo, '[TIMEZONE ERROR] We allow operating with timezone-aware datatypes. This helps preventing inconsistency and errors.'
        x = x.timestamp()
    if isinstance(x, (float, int)):
        utc = datetime.utcfromtimestamp(x)
        utc = utc.replace(tzinfo=tz.tzutc())
        utc = Timestamp(utc)
        return utc
    raise TypeError('[TYPE ERROR]: input variable has to be of a type pandas Timestamp, datetime, float, or int. However ' + type(x) + ' recieved.')


def _convert_to_utc(x):
    x = _convert_to_datetime_utc(x)
    return x


def _convert_to_local(x):
    x = _convert_to_datetime_utc(x)
    x = x.astimezone(tz.tzlocal())
    return x


def _convert_to_timezone(x, tzinfo):
    x = _convert_to_datetime_utc(x)
    x = x.astimezone(tzinfo)
    return x


def time_to_local(dfHyp):
    """
    Converts the time into the local timezone. Default by python and PC. Does not enter the timezone explicitely. Cannot be used for creating a hypnogram figure.
    """
    def convert(x, col_key):
        return _convert_to_local(x[col_key])

    dfHyp['start'] = dfHyp.apply(lambda x: convert(x, 'start'), axis=1)
    dfHyp['end'] = dfHyp.apply(lambda x: convert(x, 'end'), axis=1)
    return dfHyp


def time_to_utc(dfHyp):
    """
    Converts time to the UTC format.

    """

    def convert(x, col_key):
        return _convert_to_utc(x[col_key])

    dfHyp['start'] = dfHyp.apply(lambda x: convert(x, 'start'), axis=1)
    dfHyp['end'] = dfHyp.apply(lambda x: convert(x, 'end'), axis=1)
    return dfHyp


def time_to_timezone(dfHyp, tzinfo):
    """
    Converts the hypnogram into a timezone. The timezone has to be from a python library dateutil
    """
    def convert(x, col_key):
        return _convert_to_timezone(x[col_key], tzinfo)

    dfHyp['start'] = dfHyp.apply(lambda x: convert(x, 'start'), axis=1)
    dfHyp['end'] = dfHyp.apply(lambda x: convert(x, 'end'), axis=1)
    return dfHyp


def time_to_timestamp(dfHyp):
    """
    Converts the hypnogram time to timestamp.
    """
    def convert(x, col_key):
        return _convert_to_timestamp(x[col_key])

    dfHyp['start'] = dfHyp.apply(lambda x: convert(x, 'start'), axis=1)
    dfHyp['end'] = dfHyp.apply(lambda x: convert(x, 'end'), axis=1)
    return dfHyp


def create_duration(dfHyp):
    """
    Creates duration for each epoch within the hypnogram. (Faster on timestamp)
    """
    def duration(x):
        if type(x['start']) in (datetime, Timestamp):
            return _convert_to_timestamp(x['end']) - _convert_to_timestamp(x['start'])
        else:
            return x['end'] - x['start']
    dfHyp['duration'] = dfHyp.apply(lambda x: duration(x), axis=1)
    return dfHyp


def create_day_indexes(dfHyp, hour=12, tzinfo=tz.tzlocal):
    """
    Creates a day index for each epoch within the hypnogram, given the day-time hour supplied as input parameter.
    The format of start and end has to be an integer or a float in a form representing a timestamp, or a timezone aware datetime  or Timestamp object.
    If the start and end format do not include timezone, the local timezone will be used.
    """
    
    if not isinstance(dfHyp, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: Variable dfHyp must be of a type pandas.DataFrame.')

    if hour < 0 or hour > 23:
        raise ValueError(
            '[VALUE ERROR] - An input variable hour_cut indicating at which hour days are separated from each other must be on the range between 0 - 23. Pasted value: ',
            hour)

    timezone_counter = 0 # timezone_counter == dfHyp.__len__() if timeaware; == 0 if not timeaware; >0 & <dfHyp.__len__() if mismatch
    datetime_format = False
    tzinfo = None
    # check if the data is in
    for ridx, row in dfHyp.iterrows():
        if isinstance(row['start'], (Timestamp, datetime)) and isinstance(row['end'], (Timestamp, datetime)):
            if row['start'].tzinfo and (row['start'].tzinfo == row['start'].tzinfo == row['end'].tzinfo == row['end'].tzinfo):
                if ridx == 0:
                    tzinfo = row['start'].tzinfo
                    timezone_counter += 1
                elif row['start'].tzinfo == tzinfo:
                    timezone_counter += 1

    if timezone_counter > 0 and timezone_counter != dfHyp.__len__():
        raise ValueError('[VALUE ERROR] - Time zones in the start and end fields are inconsistent')


    dfHyp = dfHyp.sort_values('start').reset_index(drop=True)
    dfHyp['day'] = 0

    max_day = int(np.ceil((dfHyp.iloc[-1]['end'] - dfHyp.iloc[0]['start']).total_seconds() / (24*3600)))
    ref = dfHyp['start'][0].replace(hour=hour, minute=0, second=0, microsecond=0)

    for idx in range(max_day):
        dfHyp['day'][dfHyp['start'] >= ref] = idx + 1
        ref += timedelta(days=1)
    dfHyp['day'] -= dfHyp['day'].min()
    return dfHyp


def merge_annotations(df):
    """
    Merges epochs with the same annotation and end[i-1] == start[i]
    """
    new_df = pd.DataFrame()
    for idx, row in enumerate(df.iterrows()):
        appbl = True
        if idx > 0:
            if new_df.iloc[-1].annotation == row[1].annotation and new_df.iloc[-1].end == row[1].start:
                appbl = False

        if appbl == True:
            new_df = new_df.append(row[1], ignore_index=True)
        else:
            new_df.loc[new_df.__len__() - 1, 'end'] = row[1].end
        if type(new_df.loc[new_df.__len__() - 1, 'end']) in (datetime, Timestamp):
            new_df.loc[new_df.__len__() - 1, 'duration'] = (new_df.loc[new_df.__len__() - 1, 'end'] - new_df.loc[new_df.__len__() - 1, 'start']).seconds
        else:
            new_df.loc[new_df.__len__() - 1, 'duration'] = new_df.loc[new_df.__len__() - 1, 'end'] - new_df.loc[new_df.__len__() - 1, 'start']
    return new_df


def _tile_row(row, dur_threshold):
    outp = []
    start_time = row['start']
    end_time = row['end']
    curr_time = row['start']

    if type(row['end']) in (datetime, Timestamp):
        delta = timedelta(seconds=dur_threshold)
    else:
        delta = dur_threshold

    for idx in range(int(np.ceil(row['duration']/dur_threshold))):
        row_ = row.copy(deep=True)
        row_['start'] = curr_time
        curr_time += delta
        row_['end'] = curr_time
        if type(row_['end']) in (datetime, Timestamp):
            row_['duration'] = (row_['end'] - row_['start']).seconds
        else:
            row_['duration'] = row_['end'] - row_['start']

        outp += [row_]

    outp[-1]['end'] = end_time

    if type(outp[-1]['end']) in (datetime, Timestamp):
        outp[-1]['duration'] = (outp[-1]['end'] - outp[-1]['start']).seconds
    else:
        outp[-1]['duration'] = outp[-1]['end'] - outp[-1]['start']

    return outp


def tile_annotations(df, dur_threshold, verbose=False):
    """
    Tiles epochs to the max duration given by dur_threshold in seconds. Reverse to the 'merge annotations'.
    """
    if not isinstance(df, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: Variable dfAnnotations must be of type pandas.DataFrame.')

    if not isinstance(dur_threshold, (int, float)):
        raise AssertionError(
            '[INPUT ERROR]: dur_threshold must be float or int format giving the maximum duration of a single annotation. All anotations above this duration threshold will be tiled.')

    if np.isnan(dur_threshold) or np.isinf(dur_threshold) or dur_threshold <= 0:
        raise AssertionError('[INPUT ERROR]: dur_threshold must be a valid number bigger than 0, not nan and not inf')

    if (df['duration'] > dur_threshold).sum() > 1:
        outp = []
        if verbose:
            it = tqdm(list(df.iterrows()))
        else:
            it = list(df.iterrows())

        for row in it:
            row = row[1]
            outp += _tile_row(row, dur_threshold)
        return pd.DataFrame(outp).reset_index(drop=True)
    else:
        return df


def filter_by_duration(dfAnnotations, duration):
    """
    Keeps only epochs of the duration given by the input.
    """
    if not isinstance(dfAnnotations, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: Variable dfAnnotations must be of a type pandas.DataFrame.')

    if not isinstance(duration, (int, float)):
        raise AssertionError(
            '[INPUT ERROR]: duration must be float or int format giving the maximum duration of a single annotation. All anotations above this duration threshold will be tiled.')

    if np.isnan(duration) or np.isinf(duration) or duration <= 0:
        raise AssertionError('[INPUT ERROR]: duration must be a valid number bigger than 0, not nan and not inf')

    dfAnnotations = dfAnnotations.loc[dfAnnotations['duration'] == duration].reset_index(drop=True)
    return dfAnnotations


def filter_by_key(dfAnnotations, key, value):
    """
    Keeps only annotations given by the key and value within the pandas DataFrame
    """
    return dfAnnotations.loc[dfAnnotations[key] != value].reset_index(drop=True)




