# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta, datetime


def plot_hypnogram(orig_df, hypnogram_values=None, hypnogram_colors=None, fontsize=12, fig=None, night_start=22):
    """
    Creates a Matplotlib figure of spectrogram from the hypnogram. Time must be in a time-zone aware format.

    Parameters
    ----------
    orig_df : hypnogram
    hypnogram_values : dict
        dict of a y-axis values for each sleep state
    hypnogram_colors : dict
        dict of color hex codes for each sleep state
    fontsize : int
        Fontsize
    fig : figure
        Already existing figure object.
    night_start : int
        A hour when does the night begin

    Returns
    -------

    """
    _hypnogram_values = {
        'AWAKE': 6,
        'Arousal': 5,
        'SLP': 4.5,
        'REM': 4,
        'N1': 3,
        'N2': 2,
        'N3': 1,
    }

    _hypnogram_colors = {
        'AWAKE': '#e7b233',
        'Arousal': '#d44b05',
        'SLP': '#3500d3',
        'REM': '#3500d3',
        'N1': '#2bc7c4',  # 2b7cc7
        'N2': '#2b5dc7',
        'N3': '#000000',
    }

    if isinstance(hypnogram_colors, type(None)):
        hypnogram_colors = _hypnogram_colors

    if isinstance(hypnogram_values, type(None)):
        hypnogram_values = _hypnogram_values

    def set_hypnogram_properties(x, ref_dict):
        return ref_dict[x.annotation]

    orig_df['state_id'] = orig_df.apply(lambda x: set_hypnogram_properties(x, hypnogram_values), axis=1)
    orig_df['state_color'] = orig_df.apply(lambda x: set_hypnogram_properties(x, hypnogram_colors), axis=1)
    df_arrousals = orig_df.loc[orig_df.annotation == 'Arrousal'].reset_index(drop=True)
    df = orig_df.loc[orig_df.annotation != 'Arrousal'].reset_index(drop=True)
    new_df = pd.DataFrame()
    for idx, row in enumerate(df.iterrows()):  # if 2 cons. states are same, merges them
        appbl = True
        if idx > 0:
            if new_df.iloc[-1].state_id == row[1].state_id and new_df.iloc[-1].end == row[1].start:
                appbl = False

        if appbl == True:
            new_df = new_df.append(row[1], ignore_index=True)
        else:
            new_df.loc[new_df.__len__() - 1, 'end'] = row[1].end
    df = new_df

    x_start = np.array(df['start'])
    x_end = np.array(df['end'])
    try:
        for k, time_sample in enumerate(x_start): x_start[k] = time_sample.to_pydatetime()
        for k, time_sample in enumerate(x_end): x_end[k] = time_sample.to_pydatetime()
    except:
        for k, time_sample in enumerate(x_start): x_start[k] = time_sample
        for k, time_sample in enumerate(x_end): x_end[k] = time_sample

    if not fig:
        plt.figure(dpi=200)
    plt.xlim(x_start[0], x_end[-1])
    """
    # set background color for days
    for idx, day_id in enumerate(np.unique(df.day)):
        if idx % 2 == 0:
            background_color = 'gray'
            background_alpha = 0.1
        else:
            background_color = 'gray'
            background_alpha = 0.3

        day_start = x_start[df.day == day_id][0]
        day_end = x_end[df.day == day_id][-1]
        plt.axvspan(day_start, day_end, facecolor=background_color, alpha=background_alpha)
    """

    # set background color for nights
    for idx, day_id in enumerate(np.unique(df.day)):
        background_color = 'gray'
        background_alpha = 0.3
        day_start = x_start[df.day == day_id][0]
        night_start_ = datetime(
            year=day_start.year,
            month=day_start.month,
            day=day_start.day,
            hour=night_start,
            tzinfo=day_start.tzinfo
        )
        night_end_ = night_start_ + timedelta(hours=12)
        plt.axvspan(night_start_, night_end_, facecolor=background_color, alpha=background_alpha)

    # plot columns
    for idx, row in enumerate(df.iterrows()):
        val = row[1]['state_id']
        clr = row[1]['state_color']

        plt.fill_between(
            [x_start[idx], x_end[idx]],
            [val, val],
            color=clr,
            alpha=0.5,
            linewidth=0
        )

    for idx in range(df.__len__() - 1):
        val0 = df.state_id[idx]
        val1 = df.state_id[idx + 1]
        start0 = df.start[idx]
        start1 = df.start[idx + 1]
        end0 = df.end[idx]
        end1 = df.end[idx + 1]

        if val0 == val1:
            if end0 == start1:
                x = [start0, start1]
                y = [val0, val1]
            else:
                x = [start0, end0]
                y = [val0, val0]
        else:
            if end0 == start1:
                x = [start0, end0, start1]
                y = [val0, val0, val1]
            else:
                x = [start0, end0]
                y = [val0, val0]

        plt.plot(x, y, color='black', alpha=1, linewidth=1)

    x = [start1, end1]
    y = [val1, val1]
    plt.plot(x, y, color='black', alpha=1, linewidth=1)

    # plot arrousals
    for row in df_arrousals.iterrows():
        val = row[1].state_id
        clr = row[1].state_color
        plt.fill_between(
            # plt.plot(
            [row[1].start.to_pydatetime(), row[1].start.to_pydatetime(row[1].start), row[1].end.to_pydatetime(),
             row[1].end.to_pydatetime(row[1].end)],
            [0, val, val, 0],
            color=clr,
            alpha=1,
            linewidth=1
        )

    # format y ticks
    plt.yticks(list(hypnogram_values.values()), hypnogram_values.keys())
    for ticklabel in plt.gca().get_yticklabels():
        clr = hypnogram_colors[ticklabel._text]
        ticklabel.set_color(clr)
        # ticklabel.set_fontsize(fontsize)

    # plot y grid
    for idx, key in enumerate(hypnogram_values.keys()):
        clr = hypnogram_colors[key]
        val = hypnogram_values[key]
        plt.plot([x_start[0], x_end[-1]], [val, val], color=clr, linewidth=0.7, alpha=0.7, linestyle=':')

    # format x_ticks
    plt.gcf().autofmt_xdate()
    # formatter = mdates.DateFormatter("%H:%M") #mdates.DateFormatter("%H:%M", tz=tz.tzlocal())
    formatter = mdates.DateFormatter("%H:%M", tz=df.start[0].tzinfo)
    plt.gcf().get_axes()[0].xaxis.set_major_formatter(formatter)

    # plot hour x grid
    plt.grid(True, axis='x', alpha=1, linewidth=0.5, linestyle=':')

    # axes labels
    # plt.title('Days  ' + df.start[0].strftime('%d.%m') +'-' + df.iloc[-1].end.strftime('%d.%m'))
    plt.xlabel('\n Time [' + df.start[0].strftime('%d.%m.%Y') + ' - ' + df.iloc[-1].end.strftime('%d.%m.%Y') + ']',
               fontsize=fontsize)
    plt.ylabel('Sleep state', fontsize=fontsize)
    plt.gca().tick_params(axis='both', which='major', labelsize=fontsize)

