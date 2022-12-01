# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy.stats as stats

def tkeo(x):
    """
    Teager-Kaiser Energy Operator

    :param x: numpy ndarray
    :return: numpy ndarray
    """
    return (x[1:-1]**2) - (x[2:]*x[:-2])



