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

from best.sleep_classification.MultiChannelClassifier import *
from best.sleep_classification.models import *

class Test_sleep_classification(TestCase):
    def test_import(self):
        print("Testing import 'from best.sleep_classification'")

if __name__ == '__main__':
    unittest.main()
