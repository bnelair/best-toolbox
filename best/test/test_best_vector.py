# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np

from copy import deepcopy
from shutil import rmtree

import unittest
from unittest import TestCase

from best.vector import get_mutual_vectors, get_rot_2d, get_rot_3d, get_rot_3d_x, get_rot_3d_y, get_rot_3d_z, rotate, \
    _rotate_3d, _rotate_2d, _check_scale, _check_dimensions, translate, scale

basedir = os.path.abspath(os.path.dirname(__file__))

class TestVector(TestCase):
    def test_import(self):
        print("Testing import 'from best.vector'")


if __name__ == '__main__':
    unittest.main()
