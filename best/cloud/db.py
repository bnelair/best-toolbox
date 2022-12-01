# Copyright 2020-present, Mayo Clinic Department of Neurology - Bioelectronics Neurophysiology and Engineering Laboratory
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import os
import zmq
import pickle
import json
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from sqlalchemy.pool import NullPool
from sshtunnel import SSHTunnelForwarder

from best.hypnogram.utils import create_day_indexes, time_to_timezone, time_to_timestamp, tile_annotations, create_duration
from best.cloud._db_connection_variables import *

import pandas as pd
from datetime import datetime



class DatabaseHandler:
    __version__ = '0.0.1'
    def __init__(self, sql_db_name, sql_host=None, sql_user=None, sql_pwd=None, sql_port=None, ssh_host=None, ssh_user=None, ssh_pwd=None, ssh_port=None):
        if isinstance(sql_host, type(None)):
            sql_host = IP_SQL
        if isinstance(sql_user, type(None)):
            sql_user = USER_SQL
        if isinstance(sql_pwd, type(None)):
            sql_pwd = PW_SQL
        if isinstance(sql_port, type(None)):
            sql_port = PORT_SQL

        if isinstance(ssh_host, type(None)):
            ssh_host = IP_SSH
        if isinstance(ssh_user, type(None)):
            ssh_user = USER_SSH
        if isinstance(ssh_pwd, type(None)):
            ssh_pwd = PW_SSH
        if isinstance(ssh_port, type(None)):
            ssh_port = PORT_SSH

        self._ssh_host = ssh_host
        self._ssh_user = ssh_user
        self._ssh_pwd = ssh_pwd
        self._ssh_port = ssh_port

        self._sql_host = sql_host
        self._sql_port = sql_port
        self._sql_user = sql_user
        self._sql_pwd = sql_pwd
        self._sql_db_name = sql_db_name

        self._sql_connection = None
        self._ssh_tunnel = None
        self._engine = None


        self.open()
        self._init_sql_engine()

    def _init_sql_engine(self):
        if self.check_ssh_connection():
            self._engine = sqla.create_engine(
                'mysql+pymysql://{}:{}@{}:{}/{}'.format(self._sql_user, self._sql_pwd, 'localhost', self._ssh_tunnel.local_bind_port, self._sql_db_name), poolclass=NullPool)
        else:
            self._engine = sqla.create_engine(
                'mysql+pymysql://{}:{}@{}:{}/{}'.format(self._sql_user, self._sql_pwd, self._sql_host, self._sql_port, self._sql_db_name), poolclass=NullPool)

    def _open_sql(self):
        self._sql_connection = self._engine.connect()

    def _close_sql(self):
        self._sql_connection.close()

    def check_sql_connection(self):
        self._open_sql()
        self._close_sql()
        return True

    def _open_ssh(self):
        self._ssh_tunnel = SSHTunnelForwarder(
            (self._ssh_host, int(self._ssh_port)),
            ssh_username=self._ssh_user,
            ssh_password=self._ssh_pwd,
            remote_bind_address=(self._sql_host, int(self._sql_port)))
        self._ssh_tunnel.start()

    def check_ssh_connection(self):
        if self._ssh_tunnel:
            return self._ssh_tunnel.is_active
        return False

    def _close_ssh(self):
        if self.check_ssh_connection():
            self._ssh_tunnel.close()

    def open(self):
        if self._ssh_host:
            self._open_ssh()
            self.check_ssh_connection()

        self._init_sql_engine()
        self.check_sql_connection()

    def close(self):
        self._close_sql()
        self._close_ssh()

    def check_connection(self):
        self.check_ssh_connection()
        return self.check_sql_connection()

    def __del__(self):
        self.close()

    @property
    def db_name(self):
        return self._sql_db_name

    @db_name.setter
    def db_name(self, name):
        self._sql_db_name = name
        self.check_connection()

class SessionFinder(DatabaseHandler):
    #TODO: Enable searching for signals between multiple session
    __version__ = '0.0.1'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_mef_session(self, patient_id, uutc_start, uutc_stop):
        if not isinstance(uutc_start, (int, float, np.int, np.float)):
            raise TypeError('uutc_start has to be of a number type - int or float. Data type ' + type(uutc_start) + ' found instead.')
        if not isinstance(uutc_stop, (int, float, np.int, np.float)):
            raise TypeError('uutc_stop has to be of a number type - int or float. Data type ' + type(uutc_stop) + ' found instead.')

        uutc_start = int(round(uutc_start*1e6))
        uutc_stop = int(round(uutc_stop*1e6))


        self._sql_connection = self._engine.connect()
        query = f"SELECT uutc_start, uutc_stop, session, fsamp, channels FROM {self._sql_db_name}.Sessions where id='{patient_id}' and uutc_start<='{uutc_start}' and uutc_stop>='{uutc_stop}' order by uutc_start desc"
        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()

        if df_data.__len__() > 0:
            return df_data.loc[0, 'session']

    def find_mef_session_bulk(self, patient_id, df):
        sessions = []
        self._sql_connection = self._engine.connect()

        for row in tqdm(list(df.iterrows())):
            row = row[1]
            if not isinstance(row['start'], (int, float, np.int, np.float)):
                raise TypeError('start column has to be of a number type - int or float. Data type ' + type(row['start']) + ' found instead.')
            if not isinstance(row['end'], (int, float, np.int, np.float)):
                raise TypeError('end column has to be of a number type - int or float. Data type ' + type(row['end']) + ' found instead.')

            uutc_start = int(round(row['start']*1e6))
            uutc_stop = int(round(row['end']*1e6))
            query = f"SELECT uutc_start, uutc_stop, session, fsamp, channels FROM {self._sql_db_name}.Sessions where id='{patient_id}' and uutc_start<='{uutc_start}' and uutc_stop>='{uutc_stop}' order by uutc_start desc"
            df_data = pd.read_sql(query, self._sql_connection)
            if df_data.__len__() > 0:
                sessions += [df_data.loc[0, 'session']]
            else:
                sessions += ['']
        self._sql_connection.close()
        return sessions

    @property
    def patient_ids(self):
        self._open_sql()
        query = f"SELECT DISTINCT id FROM {self._sql_db_name}.Sessions"
        unique_ids = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        return unique_ids['id'].to_list()

    def data_range(self, patient_id):
        self._open_sql()
        query = f"SELECT MIN(uutc_start), MAX(uutc_stop)  FROM {self._sql_db_name}.Sessions where id='{patient_id}'"
        unique_ids = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        return unique_ids.values[0]



class SleepClassificationModelDBHandler(DatabaseHandler):
    keys_pickle = ['classifier']
    keys_unalter = ['pid', 'kappa_score', 'name', 'class_name', 'package', 'package_version']

    """Can read and save models from and into DB"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, save_classifier):
        self._sql_connection = self._engine.connect()
        vals = {}
        for k, it in save_classifier.items():
            if not k in self.keys_unalter:
                if k in self.keys_pickle:
                    it = pickle.dumps(it)
                else:
                    it = json.dumps(it)

            vals[k] = it

        df = pd.DataFrame([vals])
        df.to_sql('sleep_classifier', con=self._sql_connection, if_exists='append', index=False)
        self._sql_connection.close()

    def load_classifier(self, classifier_id):
        self._sql_connection = self._engine.connect()
        query = f"SELECT * FROM UH3.sleep_classifier where id={classifier_id}"
        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()

        if df_data.__len__() > 0:
            data = {}
            for k, it in df_data.iloc[0].to_dict().items():
                if k != 'id':
                    if not k in self.keys_unalter:
                        if k in self.keys_pickle:
                            it = pickle.loads(it)
                        else:
                            it = json.loads(it)

                    data[k] = it
            return data


    def find_classifiers(self, pid, channels, name):
        self._sql_connection = self._engine.connect()
        query = f"SELECT id, channels, results, name, ampl_values, freq_values, remove_artifacts FROM UH3.sleep_classifier where pid='{pid}' and name='{name}'"
        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()

        data = []
        for idx, row in df_data.iterrows():
            drow = {}

            drow['id'] = row['id']
            drow['channels'] = json.loads(row['channels'])
            drow['name'] = row['name']
            drow['ampl_values'] = json.loads(row['ampl_values'])
            drow['freq_values'] = json.loads(row['freq_values'])
            drow['remove_artifacts'] = json.loads(row['remove_artifacts'])
            results = json.loads(row['results'])
            for metric, it in results.items():
                for state, value in it.items():
                    drow[f"{metric}_{state}"] = value

            if sum([ch in channels for ch in drow['channels']]) == drow['channels'].__len__():
                data += [drow]
        data = pd.DataFrame(data)
        return data

    @property
    def classifier_names(self):
        self._sql_connection = self._engine.connect()
        query = f"SELECT DISTINCT(name) FROM UH3.sleep_classifier"
        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()
        return [ch['name'] for idx, ch in df_data.iterrows()]

    # def load_model(self, patient_id, channel):
    #     self._sql_connection = self._engine.connect()
    #
    #     query = f"SELECT * FROM {self._sql_db_name}.sleep_classifier where patient_id='{patient_id}' and channel='{channel}'"
    #
    #     df_data = pd.read_sql(query, self._sql_connection)
    #     self._sql_connection.close()
    #     cls_string = df_data['classifier'][0]
    #     print(df_data.keys())
    #     return pickle.loads(cls_string), df_data.drop(columns='classifier')


class SleepDataDBHandler(DatabaseHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write_data(self, df, patient_id, classifier_id):
        df = deepcopy(df)

        df['patient_id'] = patient_id
        df['classifier'] = classifier_id

        if 'annotation' in df.keys():
            df['sleep_stage'] = df['annotation']
            df = df.drop(['annotation'], axis=1)

        if 'start' in df.keys():
            df['start_uutc'] = df['start'] * 1e6
            df = df.drop(['start'], axis=1)

        if 'end' in df.keys():
            df['stop_uutc'] = df['end'] * 1e6
            df = df.drop(['end'], axis=1)


        self._sql_connection = self._engine.connect()
        df.to_sql('sleep_table', con=self._sql_connection, if_exists='append', index=False)
        self._sql_connection.close()

    def get_data(self, patient_id, start, end):
        start = int(round(start * 1e6))
        end = int(round(end * 1e6))

        self._sql_connection = self._engine.connect()
        query = f"SELECT sleep_stage, start_uutc, stop_uutc FROM {self._sql_db_name}.sleep_table where patient_id='{patient_id}' and start_uutc >= '{start}' and start_uutc <= '{end}'"
        df = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()

        df['end'] = df['stop_uutc'] / 1e6
        df = df.drop(['stop_uutc'], axis=1)

        df['start'] = df['start_uutc'] / 1e6
        df = df.drop(['start_uutc'], axis=1)
        df = create_duration(df)

        df['annotation'] = df['sleep_stage']
        df = df.drop(['sleep_stage'], axis=1)


        df = df[['annotation', 'start', 'end', 'duration']]
        return df


class SystemStateLoader(DatabaseHandler):

    def _get_sensing_channels(df, row_idx):
        det_channels = [df['sense_channel_name_' + str(idx)][row_idx] for idx in range(4)]
        det_channels = [chan_name.lower() for chan_name in det_channels if isinstance(chan_name, str)]
        return det_channels

    def _get_stim_channels(df, row_idx):
        stim_channels = []
        for idx in range(4):
            txt = df['current_group_prog' + str(idx) + 'electrodes'][row_idx]
            if isinstance(txt, str):
                stim_channels += re.findall(r'e\d{1,2}', txt.lower())
        return list(np.unique(stim_channels))

    def load_stimulation_info(self, patient_id, start_uutc):
        self._sql_connection = self._engine.connect()
        if np.log10(start_uutc) < 10: start_uutc = int(np.round(start_uutc*1e6))
        query = f"SELECT start_uutc, " \
                f"curr_Prog0AmpInMilliamps, curr_Prog1AmpInMilliamps, curr_Prog2AmpInMilliamps, curr_Prog3AmpInMilliamps, " \
                f"curr_program_pulsewidth0, curr_program_pulsewidth1, curr_program_pulsewidth2, curr_program_pulsewidth3, " \
                f"curr_rate_in_hz" \
                f" FROM {self.db_name}.System_Status_Processed WHERE patient_id='{patient_id}' and start_uutc<{start_uutc} ORDER BY start_uutc DESC LIMIT 1"

        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()

        if df_data.__len__() > 0:
            row = df_data.iloc[0]
            ampl = [row['curr_Prog'+str(k)+'AmpInMilliamps'] for k in range(4) if not isinstance(row['curr_Prog'+str(k)+'AmpInMilliamps'], type(None))]

            freq = row['curr_rate_in_hz']
            if isinstance(freq, type(None)): freq = 0

            if ampl.__len__() == 0: ampl = 0
            else: ampl = np.nanmax(ampl)

            pw = [row['curr_program_pulsewidth'+str(k)] for k in range(4) if not isinstance(row['curr_program_pulsewidth'+str(k)], type(None))]
            if pw.__len__() == 0: pw = 0
            else: pw = np.nanmax(pw)

            stim_info = {
                'freq': freq,
                'ampl': ampl,
                'pulsewidth': pw
            }
            return stim_info

    def load_stimulation_info_bulk(self, patient_id, df):
        df = time_to_timestamp(deepcopy(df))
        freq_ = []
        ampl_ = []
        pulsewidth_ = []

        self._sql_connection = self._engine.connect()


        for row in tqdm(range(df.__len__())):
            row = df.iloc[row]
            start_uutc = int(round(row['start'] * 1e6))
            query = f"SELECT start_uutc, " \
                    f"curr_Prog0AmpInMilliamps, curr_Prog1AmpInMilliamps, curr_Prog2AmpInMilliamps, curr_Prog3AmpInMilliamps, " \
                    f"curr_program_pulsewidth0, curr_program_pulsewidth1, curr_program_pulsewidth2, curr_program_pulsewidth3, " \
                    f"curr_rate_in_hz" \
                    f" FROM {self.db_name}.System_Status_Processed WHERE patient_id='{patient_id}' and start_uutc<{start_uutc} ORDER BY start_uutc DESC LIMIT 1"

            df_outp = pd.read_sql(query, self._sql_connection)

            if df_outp.__len__() > 0:
                row = df_outp.iloc[0]
                ampl = [row['curr_Prog'+str(k)+'AmpInMilliamps'] for k in range(4) if not isinstance(row['curr_Prog'+str(k)+'AmpInMilliamps'], type(None))]

                if ampl.__len__() == 0: ampl = 0
                else: ampl = np.nanmax(ampl)

                freq = row['curr_rate_in_hz']
                if isinstance(freq, type(None)): freq = 0


                pw = [row['curr_program_pulsewidth'+str(k)] for k in range(4) if not isinstance(row['curr_program_pulsewidth'+str(k)], type(None))]
                if pw.__len__() == 0: pw = 0
                else: pw = np.nanmax(pw)

            else:
                freq = ampl = pw = None


            freq_.append(freq)
            ampl_.append(ampl)
            pulsewidth_.append(pw)


        self._sql_connection.close()
        df['freq'] = freq_
        df['ampl'] = ampl_
        df['pulsewidth'] = pulsewidth_
        return df

    def load_stimulation_day_info(self, patient_id, start_uutc, stop_uutc):
        self._sql_connection = self._engine.connect()
        if np.log10(start_uutc) < 10: start_uutc = int(np.round(start_uutc*1e6))
        if np.log10(stop_uutc) < 10: stop_uutc = int(np.round(stop_uutc*1e6))

        query = f"SELECT start_uutc, " \
                f"curr_Prog0AmpInMilliamps, curr_Prog1AmpInMilliamps, curr_Prog2AmpInMilliamps, curr_Prog3AmpInMilliamps, " \
                f"curr_program_pulsewidth0, curr_program_pulsewidth1, curr_program_pulsewidth2, curr_program_pulsewidth3, " \
                f"curr_rate_in_hz" \
                f" FROM {self.db_name}.System_Status_Processed WHERE patient_id='{patient_id}' and start_uutc>{start_uutc} and start_uutc<{stop_uutc}"

        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()

        freqs = []
        ampls = []
        pws = []
        for row in df_data.iterrows():
            row = row[1]
            ampl = [row['curr_Prog'+str(k)+'AmpInMilliamps'] for k in range(4) if not isinstance(row['curr_Prog'+str(k)+'AmpInMilliamps'], type(None))]

            freq = row['curr_rate_in_hz']
            if isinstance(freq, type(None)): freq = 0

            if ampl.__len__() == 0: ampl = 0
            else: ampl = np.nanmax(ampl)

            pw = [row['curr_program_pulsewidth'+str(k)] for k in range(4) if not isinstance(row['curr_program_pulsewidth'+str(k)], type(None))]
            if pw.__len__() == 0: pw = 0
            else: pw = np.nanmax(pw)
            freqs += [freq]
            ampls += [ampl]
            pws += [pw]

        if freqs.__len__() == 0: freqs = None
        if ampls.__len__() == 0: ampls = None
        if pws.__len__() == 0: pws = None
        stim_info = {
            'freq': np.max(freqs),
            'ampl': np.max(ampls),
            'pulsewidth': np.max(pws)
        }
        return stim_info

        self._sql_connection = self._engine.connect()
        if np.log10(start_uutc) < 10: start_uutc = int(np.round(start_uutc * 1e6))
        if np.log10(stop_uutc) < 10: stop_uutc = int(np.round(stop_uutc * 1e6))

        query = f"SELECT start_uutc, " \
                f"curr_Prog0AmpInMilliamps, curr_Prog1AmpInMilliamps, curr_Prog2AmpInMilliamps, curr_Prog3AmpInMilliamps, " \
                f"curr_program_pulsewidth0, curr_program_pulsewidth1, curr_program_pulsewidth2, curr_program_pulsewidth3, " \
                f"curr_rate_in_hz" \
                f" FROM {self.db_name}.System_Status_Processed WHERE patient_id='{patient_id}' and start_uutc>{start_uutc} and start_uutc<{stop_uutc}"

        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()

        df_data['amplitude'] = df_data[['curr_Prog0AmpInMilliamps', 'curr_Prog1AmpInMilliamps', 'curr_Prog2AmpInMilliamps',
                                        'curr_Prog3AmpInMilliamps']].max(axis=1)
        df_data['frequency'] = df_data['curr_rate_in_hz']
        df_data['start'] = df_data['start_uutc'] / 1e6
        df_data['end'] = np.nan
        df_data.loc[:df_data.__len__() - 1, 'end'] = df_data.iloc[1:].reset_index(drop=True)['start']
        df_data = df_data.iloc[:-1]

        df_data = df_data[['frequency', 'amplitude', 'start', 'end']]
        df_data['duration'] = df_data['end'] - df_data['start']
        return df_data

    def load_stimulation_annotations(self, patient_id, start_uutc, stop_uutc):
        if np.log10(start_uutc) < 10: start_uutc = int(np.round(start_uutc * 1e6))
        if np.log10(stop_uutc) < 10: stop_uutc = int(np.round(stop_uutc * 1e6))

        query = f"SELECT start_uutc, " \
                f"curr_Prog0AmpInMilliamps, curr_Prog1AmpInMilliamps, curr_Prog2AmpInMilliamps, curr_Prog3AmpInMilliamps, " \
                f"curr_program_pulsewidth0, curr_program_pulsewidth1, curr_program_pulsewidth2, curr_program_pulsewidth3, " \
                f"curr_rate_in_hz, curr_stim_state, current_group, stim_mode" \
                f" FROM {self.db_name}.System_Status_Processed WHERE patient_id='{patient_id}' and start_uutc>{start_uutc} and start_uutc<{stop_uutc}"

        self._sql_connection = self._engine.connect()
        df_data = pd.read_sql(query, self._sql_connection)
        self._sql_connection.close()

        if df_data.__len__() > 0:
            df_data['amplitude'] = df_data[
                ['curr_Prog0AmpInMilliamps', 'curr_Prog1AmpInMilliamps', 'curr_Prog2AmpInMilliamps',
                 'curr_Prog3AmpInMilliamps']].max(axis=1)
            df_data['frequency'] = df_data['curr_rate_in_hz']
            df_data['start'] = df_data['start_uutc'] / 1e6
            df_data['end'] = np.nan
            df_data['pulse_width'] = df_data[['curr_program_pulsewidth0', 'curr_program_pulsewidth1', 'curr_program_pulsewidth2', 'curr_program_pulsewidth3']].max(axis=1)
            df_data.loc[:df_data.__len__() - 1, 'end'] = df_data.iloc[1:].reset_index(drop=True)['start']
            # df_data = df_data.iloc[:-1]
    
            df_data = df_data[['frequency', 'amplitude', 'start', 'end', 'pulse_width', 'curr_stim_state', 'current_group', 'stim_mode']]
            df_data['duration'] = df_data['end'] - df_data['start']
            
            return df_data
        
        else:
            return pd.DataFrame(columns=['frequency', 'amplitude', 'start', 'end', 'pulse_width', 'curr_stim_state', 'current_group', 'stim_mode'])
            

class ScientificDataLoader(DatabaseHandler):
    def get_impedance_min_max(self, patient_id=None, start_timestamp=None, stop_timestamp=None):
        if isinstance(start_timestamp, type(None)):
            start_timestamp = 0
        if isinstance(stop_timestamp, type(None)):
            stop_timestamp = datetime.now().timestamp()

        start_timestamp *= 1e6
        stop_timestamp *= 1e6

        query = f"SELECT MIN(start_uutc), MAX(stop_uutc)" \
                f" FROM {self._sql_db_name}.Impedance where patient_id='{patient_id}' " \
                f"and start_uutc>={start_timestamp} " \
                f"and stop_uutc<={stop_timestamp}"
        self._open_sql()
        min_max = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        if not isinstance(min_max.iloc[0]['MIN(start_uutc)'], type(None)):
            min_max = [min_max.iloc[0]['MIN(start_uutc)'] / 1e6, min_max.iloc[0]['MAX(stop_uutc)'] / 1e6]
            return min_max
        else:
            return None

    def get_impedance_data(self, patient_id=None, start_timestamp=None, stop_timestamp=None):
        if isinstance(start_timestamp, type(None)):
            start_timestamp = 0
        if isinstance(stop_timestamp, type(None)):
            stop_timestamp = datetime.now().timestamp()

        start_timestamp *= 1e6
        stop_timestamp *= 1e6

        query = f"SELECT " \
                f"start_uutc, " \
                f"el_0, el_1, el_2, el_3, el_4, el_5, el_6, el_7, el_8, el_9, el_10, el_11, el_12, el_13, el_14, el_15 " \
                f"FROM {self._sql_db_name}.Impedance WHERE patient_id='{str(patient_id)}' " \
                f"and start_uutc>={start_timestamp} " \
                f"and start_uutc<={stop_timestamp}"
        self._open_sql()
        df = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        df['start_uutc'] /= 1e6
        return df


    def get_impedance_data_bipolar(self, patient_id=None, start_timestamp=None, stop_timestamp=None):
        if isinstance(start_timestamp, type(None)):
            start_timestamp = 0
        if isinstance(stop_timestamp, type(None)):
            stop_timestamp = datetime.now().timestamp()

        start_timestamp *= 1e6
        stop_timestamp *= 1e6

        query = f"SELECT " \
                f" * FROM {self._sql_db_name}.Impedance_Bipol WHERE id='{str(patient_id)}' " \
                f"and start_uutc>={start_timestamp} " \
                f"and start_uutc<={stop_timestamp}"
        self._open_sql()
        df = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        df['start_uutc'] /= 1e6
        return df


    ############ Spike ##################
    def get_spike_rate_channels(self, patient_id=None, start_timestamp=None, stop_timestamp=None):
        if isinstance(start_timestamp, type(None)):
            start_timestamp = 0
        if isinstance(stop_timestamp, type(None)):
            stop_timestamp = datetime.now().timestamp()

        start_timestamp *= 1e6
        stop_timestamp *= 1e6

        self._open_sql()
        query = f"SELECT DISTINCT channel " \
                f"FROM {self._sql_db_name}.Automated_Events WHERE id='{str(patient_id)}' " \
                f"and ev_type='spike_rate' " \
                f"and start_uutc>={start_timestamp} " \
                f"and start_uutc<={stop_timestamp}"
        df = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        list_of_channels = np.array(df['channel'].unique())
        channels_by_location = {}
        for ch in list_of_channels:
            loc = self._get_channel_location(ch)
            if not loc in channels_by_location.keys():
                channels_by_location[loc] = []
            channels_by_location[loc].append(ch)
        return channels_by_location


    def get_spike_rate_data(self, patient_id=None, channel=None, start_timestamp=None, stop_timestamp=None):
        if isinstance(start_timestamp, type(None)):
            start_timestamp = 0
        if isinstance(stop_timestamp, type(None)):
            stop_timestamp = datetime.now().timestamp()

        start_timestamp *= 1e6
        stop_timestamp *= 1e6

        if channel:
            query = f"SELECT " \
                    f"start_uutc, value  " \
                    f"FROM {self._sql_db_name}.Spike_Rate_Events WHERE id='{str(patient_id)}' " \
                    f"and channel='{channel}' " \
                    f"and start_uutc>={start_timestamp} " \
                    f"and start_uutc<={stop_timestamp}"
            self._open_sql()
            df = pd.read_sql(query, self._sql_connection)
            self._close_sql()
        else:
            query = f"SELECT " \
                    f"start_uutc, value, channel  " \
                    f"FROM {self._sql_db_name}.Spike_Rate_Events WHERE id='{str(patient_id)}' " \
                    f"and start_uutc>={start_timestamp} " \
                    f"and start_uutc<={stop_timestamp}"
            self._open_sql()
            df = pd.read_sql(query, self._sql_connection)
            self._close_sql()


        df['start_uutc'] /= 1e6
        df['start_uutc'] += 10*60
        return df

    def get_spike_n_detections(self,  patient_id=None, channel=None, start_timestamp=None, stop_timestamp=None):
        if isinstance(start_timestamp, type(None)):
            start_timestamp = 0
        if isinstance(stop_timestamp, type(None)):
            stop_timestamp = datetime.now().timestamp()

        start_timestamp *= 1e6
        stop_timestamp *= 1e6

        query = f"SELECT " \
                f"COUNT(start_uutc)  " \
                f"FROM {self._sql_db_name}.Features WHERE id='{str(patient_id)}' " \
                f"and ev_type=\"spike\" " \
                f"and channel='{channel}' " \
                f"and start_uutc>={start_timestamp} " \
                f"and start_uutc<={stop_timestamp}"
        self._open_sql()
        df = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        return df['COUNT(start_uutc)'][0]


    def get_spike_detections(self,  patient_id=None, channel=None, start_timestamp=None, stop_timestamp=None):
        if isinstance(start_timestamp, type(None)):
            start_timestamp = 0
        if isinstance(stop_timestamp, type(None)):
            stop_timestamp = datetime.now().timestamp()

        start_timestamp *= 1e6
        stop_timestamp *= 1e6

        query = f"SELECT " \
                f"start_uutc " \
                f"FROM {self._sql_db_name}.SpikesEvents WHERE id='{str(patient_id)}' " \
                f"and channel='{channel}' " \
                f"and start_uutc>={start_timestamp} " \
                f"and start_uutc<={stop_timestamp}"
        self._open_sql()
        df = pd.read_sql(query, self._sql_connection)
        self._close_sql()

        df['start_uutc'] /= 1e6
        df['start_uutc'] += 10*60
        return df


    def get_seizures(self, patient_id=None, start_timestamp=None, stop_timestamp=None):
        if isinstance(start_timestamp, type(None)):
            start_timestamp = 0
        if isinstance(stop_timestamp, type(None)):
            stop_timestamp = datetime.now().timestamp()

        start_timestamp *= 1e6
        stop_timestamp *= 1e6

        query = f"SELECT " \
                f"start_uutc, stop_uutc " \
                f"FROM {self._sql_db_name}.Manual_Events WHERE id='{str(patient_id)}' " \
                f"and start_uutc>={start_timestamp} " \
                f"and start_uutc<={stop_timestamp} " \
                f"and ev_type='seizure'  and source='eeg_review'"\
                #f"and source='eeg_review'"
        self._open_sql()
        df = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        df['start_uutc'] /= 1e6
        # df['start_uutc'] += 10*60
        df['stop_uutc'] /= 1e6
        # df['stop_uutc'] += 10*60
        return df


    def get_seizure_probability(self, patient_id, seizures):
        seizures_ = []
        self._open_sql()
        for idx, szr in tqdm(list(seizures.iterrows())):
            query = f"SELECT " \
                    f"* " \
                    f"FROM {self._sql_db_name}.{patient_id}_prob WHERE " \
                    f"uutc_start>={szr.start_uutc * 1e6} " \
                    f"and uutc_stop<={szr.stop_uutc * 1e6} "
            # f"and source='eeg_review'"

            df = pd.read_sql(query, self._sql_connection)
            
            l = df.el1.to_numpy()
            r = df.el2.to_numpy()
            s_ = df.uutc_start.to_numpy()
            e_ = df.uutc_stop.to_numpy()

            fs = 1
            s_prob = szr.start_uutc
            e_prob = szr.stop_uutc
            if s_.__len__() > 1:
                s_prob = s_[0] / 1e6
                e_prob = e_[-1] / 1e6

            seizures_ += [
                {
                    'start_annotation': szr.start_uutc,
                    'end_annotation': szr.stop_uutc,
                    'prob_right': r,
                    'prob_left': l,
                    'self_reported': l.__len__() <= 1,
                    's_prob': s_prob,
                    'e_prob': e_prob,
                    'fs' : fs
                }
            ]
            
        self._close_sql()
        return seizures_


    def get_seizures_patient(self, patient_id=None, start_timestamp=None, stop_timestamp=None):
        if isinstance(start_timestamp, type(None)):
            start_timestamp = 0
        if isinstance(stop_timestamp, type(None)):
            stop_timestamp = datetime.now().timestamp()

        start_timestamp *= 1e6
        stop_timestamp *= 1e6

        query = f"SELECT " \
                f"start_uutc, stop_uutc " \
                f"FROM {self._sql_db_name}.Annotations_Events WHERE id='{str(patient_id)}' " \
                f"and start_uutc>={start_timestamp} " \
                f"and start_uutc<={stop_timestamp} " \
                f"and (ev_type='aura' or ev_type='seizure') " \
                f"and source='patient'"
        self._open_sql()
        df = pd.read_sql(query, self._sql_connection)
        self._close_sql()
        df['start_uutc'] /= 1e6
        df['start_uutc'] += 10*60
        df['stop_uutc'] /= 1e6
        df['stop_uutc'] += 10*60
        return df


    def get_datarate(self,  patient_id=None, start_timestamp=None, stop_timestamp=None):
        if isinstance(start_timestamp, type(None)):
            start_timestamp = 0
        if isinstance(stop_timestamp, type(None)):
            stop_timestamp = datetime.now().timestamp()

        start_timestamp *= 1e6
        stop_timestamp *= 1e6

        query = f"SELECT " \
                f"start_uutc, value  " \
                f"FROM {self._sql_db_name}.Automated_Events WHERE id='{str(patient_id)}' " \
                f"and ev_type='data_rate' " \
                f"and start_uutc>={start_timestamp} " \
                f"and start_uutc<={stop_timestamp}"
        self._open_sql()
        df = pd.read_sql(query, self._sql_connection)
        self._close_sql()

        df['start_uutc'] /= 1e6
        df['start_uutc'] += 10*60
        return df

























