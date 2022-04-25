# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import yaml
import os
import numpy as np


class ObjDict(dict):
    """
    Dictionary which you can access a) as a dict; b) as a struct with attributes. Can use both foo adding and deleting
    attributes resp items. Inherits from dict
    """

    def __init__(self, VT_={}):
        super().__init__(VT_)
        for key in VT_.keys():
            self.__setattr__(key, VT_[key])

    def __setitem__(self, key, value):
        if key in self:
            del self[key]

        super().__setitem__(key, value)
        if not key in self.__dir__():
            self.__setattr__(key, value)

    def __setattr__(self, key, value):
        if key in self.__dir__():
            self.__delattr__(key)

        super().__setattr__(key, value)
        if not key in self:
            self.__setitem__(key, value)

    def __delitem__(self, key):
        value = super().pop(key)
        try:
            super().pop(value, None)
        except: pass
        if key in dir(self):
            self.__delattr__(key)

    def __delattr__(self, key):
        super().__delattr__(key)
        if key in self:
            self.__delitem__(key)

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    def __missing__(self, key):
        self[key] = ObjDict()
        return self[key]

    def __getattr__(self, item):
        if not item in self.__dir__():
            self.__missing__(item)
        return super().__getattribute__(item)


def DictToObjDict(d):
    """
    Converts dictionary into object where, keys - converts keys into attributes
    """
    if isinstance(d, dict):
        #print(d.keys())
        top = ObjDict(d)

        #print(top.keys())
        for key in d.keys():
            if isinstance(d[key], dict):
                tmp = DictToObjDict(d[key])
                del top[key]
                top[key] = tmp

        return top
    else: return d


def config(path_config):
    with open(path_config, 'r') as stream:
        Cfg = yaml.safe_load(stream)
    return DictToObjDict(Cfg)


def get_files(path, endings=None, creation_time=False):
    """File list generator.

        For each file in path directory and subdirectories with endings from endings_tuple returns path in a list variable

        path is a string, the path to the directory where you wanna search specific files.

        endings_tuple is a tuple of any length

        The function returns paths to all files in folder and files in all subfolders as well.

        Caution:  ending must be tuple, even for length of 1 ending. ('png')

        Example:

        from myFiles import get_files
        import pandas as pd

        path = "root/data"
        files_list = get_files(path)
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


def ObjDictToDict(d):
    """
    Converts dictionary into object where, keys - converts keys into attributes
    """
    if isinstance(d, ObjDict):
        #print(d.keys())
        top = dict(d)

        #print(top.keys())
        for key in d.keys():
            if isinstance(d[key], ObjDict):
                tmp = ObjDictToDict(d[key])
                del top[key]
                top[key] = tmp

        return top
    else: return d


