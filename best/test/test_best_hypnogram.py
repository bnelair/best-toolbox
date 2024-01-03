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

from best.annotations.analysis import *
from best.annotations.visualisation import *
from best.annotations.utils import *
from best.annotations.io import *
from best.annotations.correct import *
from best.annotations.CyberPSG import *
from best.annotations.NSRR import *

class TestHypnogram(TestCase):
    def test_import(self):
        print("Testing import 'from best.annotations'" )



if __name__ == '__main__':
    unittest.main()
