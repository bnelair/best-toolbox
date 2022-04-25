# Copyright 2020-present, Mayo Clinic Department of Neurology - Bioelectronics Neurophysiology and Engineering Laboratory
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import zmq
import numpy as np
from best.cloud import *
from best.cloud._mefclient_connection_variables import *

class MefClient:
    __version__ = '3.0.0'
    def __init__(self,ports=None, server_ip=None, response_wait=30):
        if isinstance(ports, type(None)):
            ports = PORT_MEF
        if isinstance(server_ip, type(None)):
            server_ip = IP_MEF

        context = zmq.Context(1)
        self.client = context.socket(zmq.REQ)
        self.poll = zmq.Poller()
        self.ports = ports
        self.server_ip = server_ip
        self.response_wait = response_wait

    def request_data(self, path, channel, passwd='', start=None, stop=None, sample=0):
        a, b, c = self._request_data(path, channel, passwd, start, stop, sample)
        if a: return a, b, c
        a, b, c = self._request_data(path, channel, passwd, start, stop, sample)
        if a: return a, b, c
        a, b, c = self._request_data(path, channel, passwd, start, stop, sample)
        return a, b, c


    def _request_data(self, path, channel, passwd='', start=None, stop=None, sample=0):
        #if start and stop:
            #if np.log10(start) < 15:
                #start = int(round(start * 1e6))
                #stop = int(round(stop * 1e6))

        # sample arg must be 1 if data are read by sample
        client = self.client
        for p in self.ports:
            client.connect(f"tcp://{self.server_ip}:{p}")

        self.poll.register(client, zmq.POLLIN)
        try:
            client.send_json({'path': path, 'channel': channel, 'passwd': passwd, 'start': start, 'stop': stop, 'sample': sample,
                              'flag':'data'})
            socks = dict(self.poll.poll(self.response_wait * 1000))
            if socks.get(client) == zmq.POLLIN:
                md = client.recv_json(flags=0, )
                msg = client.recv(flags=0, copy=True, track=False)
                buf = memoryview(msg)
                if md['error'] == 1:
                    return False, md['error_message'], False
                else:
                    res_numpy = np.frombuffer(buf, dtype=md['dtype'])
                    return True, res_numpy.reshape(md['shape']), md['fsamp']
            else:
                client.setsockopt(zmq.LINGER, 0)
                client.close()
                self.poll.unregister(client)
                # rebind
                context = zmq.Context(1)
                self.client = context.socket(zmq.REQ)
                return False, f'Server response time elapsed {self.response_wait} s', False

        except Exception as exc:
            exce = f'{exc}'
            return False, exce, False

    def request_processed_data(self, path, channel, passwd='', start=None, stop=None, sample=0, b_low=None, a_low=None, b_high=None, a_high=None, zero_phase=True, fs_target=200, correct_timestamp=True):
        #if start and stop:
            #if np.log10(start) < 15 and correct_timestamp:
                #start = int(round(start * 1e6))
                #stop = int(round(stop * 1e6))

        # sample arg must be 1 if data are read by sample
        client = self.client
        for p in self.ports:
            client.connect(f"tcp://{self.server_ip}:{p}")

        self.poll.register(client, zmq.POLLIN)
        try:
            client.send_json({
                'path': path,
                'channel': channel,
                'passwd': passwd,
                'start': int(start),
                'stop': int(stop),
                'sample': int(sample),
                'flag':'data',
                'filter_b_high': [float(v) for v in b_high],
                'filter_a_high': [float(v) for v in a_high],
                'filter_b_low': [float(v) for v in b_low],
                'filter_a_low': [float(v) for v in a_low],
                'zero_phase': bool(zero_phase),
                'fs_target': float(fs_target)
            })


            socks = dict(self.poll.poll(self.response_wait * 1000))
            if socks.get(client) == zmq.POLLIN:
                md = client.recv_json(flags=0, )
                msg = client.recv(flags=0, copy=True, track=False)
                buf = memoryview(msg)
                if md['error'] == 1:
                    return False, md['error_message'], False
                else:
                    res_numpy = np.frombuffer(buf, dtype=md['dtype'])
                    return True, res_numpy.reshape(md['shape']), md['fsamp']
            else:
                client.setsockopt(zmq.LINGER, 0)
                client.close()
                self.poll.unregister(client)
                # rebind
                context = zmq.Context(1)
                self.client = context.socket(zmq.REQ)
                return False, f'Server response time elapsed {self.response_wait} s', False

        except Exception as exc:
            exce = f'{exc}'
            return False, exce, False




    def request_data_bulk(self, df):
        data = []
        fs = []
        for row in df.iterrows():
            row = row[1]
            outp = self.request_data(row['path'], row['channel'], start=row['start'], stop=row['end'])
            if not outp[0]:
                raise AssertionError('[ERROR] Data haven\'t been read \n' + str(row) + ' \n Error Message: ' + str(outp))
            data += [outp[1]]
            fs += [outp[2]]
        return data, fs



    def request_metadata(self, path, passwd=''):
        client = self.client
        for p in self.ports:
            client.connect(f"tcp://{self.server_ip}:{p}")
        self.poll.register(client, zmq.POLLIN)
        try:
            client.send_json({'path': path,  'passwd': passwd, 'flag': 'basic_info'})
            socks = dict(self.poll.poll(self.response_wait * 1000))
            if socks.get(client) == zmq.POLLIN:
                md = client.recv_json(flags=0, )
                if md['error'] == 1:
                    return False, md['error_message']
                else:
                    return True, md['bi']
            else:
                client.setsockopt(zmq.LINGER, 0)
                client.close()
                self.poll.unregister(client)
                # rebind
                context = zmq.Context(1)
                self.client = context.socket(zmq.REQ)
                return False, f'Server response time elapsed {self.response_wait} s'

        except Exception as exc:
            exce = f'{exc}'
            return False, exce

