import os
import pandas as pd

from tqdm import tqdm
from mef_tools.io import MefReader

from best.files import get_files

def list_mefd_files(path_bidsdata):
    path_participants = os.path.join(path_bidsdata, 'participants.tsv')
    if os.path.exists(path_participants):
        participants = pd.read_csv(path_participants, sep='\t')
    else:
        participants = pd.DataFrame()

    data = []
    for fid in [f for f in get_files(path_bidsdata, 'mefd') if not '._' in f]:
        try:
            pid = fid.split(os.sep)[-1].split('sub-')[1].split('_')[0]
        except:
            pid = ''
        try:
            session = fid.split(os.sep)[-1].split('ses-')[1].split('_')[0]
        except:
            session = ''
        try:
            task = fid.split(os.sep)[-1].split('task-')[1].split('_')[0]
        except:
            task = ''
        try:
            acquisition = fid.split(os.sep)[-1].split('acq-')[1].split('_')[0]
        except:
            acquisition = ''
        try:
            run = fid.split(os.sep)[-1].split('run-')[1].split('_')[0]
        except:
            run = ''
        try:
            modality = fid.split(os.sep)[-1].split('.')[-2].split('_')[-1]
        except:
            modality = ''

        pwd = ''
        if 'mef_pwd' in participants.keys() and pid in participants['subject_id'].values:
            pwd = participants[participants['subject_id'] == pid]['mef_pwd'].values[0]



        dat = {
            'subject': pid,
            'session': session,
            'task': task,
            'acquisition': acquisition,
            'run': run,
            'modality': modality,
            'path': fid,
            'mef_pwd': pwd
        }
        data += [dat]

    data = pd.DataFrame(data)
    return data

def scan_mefd_properties(df_files):
    for idx, row in tqdm(list(df_files.iterrows())):
        path = row['path']
        pwd = row['mef_pwd']

        Rdr = MefReader(path, pwd)
        start = min(*Rdr.get_property('start_time')) / 1e6
        end = max(*Rdr.get_property('end_time')) / 1e6
        fsamp = max(Rdr.get_property('fsamp'))

        df_files.loc[idx, 'start'] = start
        df_files.loc[idx, 'end'] = end
        df_files.loc[idx, 'duration'] = end - start
        df_files.loc[idx, 'fsamp'] = fsamp

    return df_files


