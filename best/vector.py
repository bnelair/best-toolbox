# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from copy import deepcopy

def _check_dimensions(x, n_low=1, n_high=3):
    if x.shape[1] > n_high or x.shape[1] < n_low:
        raise AssertionError('[ERROR]: Data has to be 1-3 dimensions')
    return True


def _check_scale(x, m):
    if x.shape[1] != m.__len__():
        raise AssertionError('[ERROR]: N-dim of data has to be the same as scale coefficients')
    return True


def translate(x, m):
    _check_scale(x, m)
    for idx, s in enumerate(m):
        x[:, idx] = x[:, idx] + s
    return x


def scale(x, m):
    _check_scale(x, m)
    sc = np.array(m).reshape(1, -1)
    mn = x.mean(axis=0).reshape(1, -1)
    x = (x-mn)*sc + mn
    return x


def rotate(x, angl):
    x = deepcopy(x)
    if isinstance(angl, (tuple, np.ndarray, list)):
        if angl.__len__() == 3 and x.shape[1] == 3:
            return _rotate_3d(x, angl)
        if angl.__len__() == 1:
            angl = angl[0]
    return _rotate_2d(x, angl)


def _rotate_2d(x, angl):
    m = get_rot_2d(angl)
    mn = x.mean(axis=0).reshape(1, -1)
    return np.dot(x-mn, m) + mn


def get_rot_2d(an):
    an = 2*np.pi*an/360
    m = np.array([
        [np.cos(an), -np.sin(an)],
        [np.sin(an), np.cos(an)]
    ])
    return m


def _rotate_3d(x, angl):
    ms = get_rot_3d(angl)
    mn = x.mean(axis=0).reshape(1, -1)
    x = x - mn
    for m in ms:
        x = np.dot(x, m)
    x = x + mn
    return x


def get_rot_3d(an):
    return [
        get_rot_3d_x(an[0]),
        get_rot_3d_y(an[1]),
        get_rot_3d_z(an[2])
    ]


def get_rot_3d_x(an):
    an = 2*np.pi*an/360
    m = np.array([
        [1, 0, 0],
        [0, np.cos(an), -np.sin(an)],
        [0, np.sin(an), np.cos(an)]
    ])
    return m


def get_rot_3d_y(an):
    an = 2*np.pi*an/360
    m = np.array([
        [np.cos(an), 0, np.sin(an)],
        [0, 1, 0],
        [-np.sin(an), 0,  np.cos(an)]
    ])
    return m


def get_rot_3d_z(an):
    an = 2*np.pi*an/360
    m = np.array([
        [np.cos(an), -np.sin(an), 0],
        [np.sin(an), np.cos(an), 0],
        [0, 0, 1]
    ])
    return m


def get_mutual_vectors(x, y=None):
    leg = []
    v = []
    for idx, x_ in enumerate(x):
        temp_x = x_ - x
        if not isinstance(y, type(None)):
            temp_y = np.array([y[idx] + '-' + y_ for y_ in y])
        v += [temp_x]
        leg += [temp_y]
    v = np.concatenate(v, axis=0)

    if not isinstance(y, type(None)):
        leg = np.concatenate(leg)
        return v, leg
    return v





