# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from shutil import rmtree
from time import sleep


def create_folder(PATH_TO_CREATE):
    """
    Creates a folder on PATH_TO_CREATE position.
    """
    from os import mkdir
    mkdir(PATH_TO_CREATE)

def remove_folder(PATH_TO_REMOVE):
    """
    Creates a folder on PATH_TO_CREATE position.
    """
    try: rmtree(PATH_TO_REMOVE)
    except: pass
    sleep(0.05)

def get_files(path, endings=None, creation_time=False):
    """File list generator.

        For each file in path directory and subdirectories with endings from endings_tuple returns path in a list variable

        path is a string, the path to the directory where you wanna search specific files.

        endings_tuple is a tuple of any length

        The function returns paths to all files in folder and files in all subfolders as well.

        .. code-block:: python

            from best.files import get_files
            import pandas as pd

            path = "root/data"
            files_list = get_files(path, ('.jpg', '.jpeg'))
            files_pd = pd.DataFrame(files_list)

            def get_FID(x):
                temp = x['path'].split('\\')
                return temp[len(temp) - 1][0:-4]
            files_pd = files_pd.rename(columns={0: "path"})
            files_pd['fid'] = files_pd.apply(lambda x: get_FID(x), axis=1)
        """

    if isinstance(endings, str):
        endings = (endings, )

    if isinstance(endings, type(None)):
        data = [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files]

        data += [os.path.join(root, dir)
                for root, dirs, files in os.walk(path)
                for dir in dirs]
    else:
        data = [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files
                if name.endswith(endings)]

        data += [os.path.join(root, dir)
                for root, dirs, files in os.walk(path)
                for dir in dirs
                if dir.endswith(endings)]


    data = [file for file in data if not '._' in file]
    data.sort()

    if creation_time == False:
        return data

    creation_times = [os.stat(path).st_ctime for path in data]
    data = np.array(data)
    creation_times = np.array(creation_times)
    idx = creation_times.argsort()
    sorted_data = data[idx]
    sorted_creation_times = creation_times[idx]
    return sorted_data, sorted_creation_times

def get_folders(path):
    """
    Returns sub-folders present in the folder specified by the path.
    """
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and ((not '.mefd' in name) and (not '.segd' in name) and (not '.timd' in name))]

def split_files(file_paths, ratio=0.5):
    """
    Randomly shuffles list of files and splits given by the ratio.
    """
    mix_idxes = np.random.permutation(file_paths.__len__())
    paths = list(np.array(file_paths)[mix_idxes])
    split_idx = int(np.round(ratio*paths.__len__()))
    return paths[:split_idx], paths[split_idx:]