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


# define env variables for the db testing
import os
os.environ['IP_SQL'] = '0.0.0.0'
os.environ['USER_SQL'] = 'db_usr'
os.environ['PW_SQL'] = '\'db_pwd\''
os.environ['PORT_SQL'] = '3360'
os.environ['IP_SSH'] = '\'0.0.0.0\''
os.environ['USER_SSH'] = '\'ssh_user\''
os.environ['PW_SSH'] = '\'ssh_pwd\''
os.environ['PORT_SSH'] = '\'22\''
os.environ['IP_MEF'] = '\'0.0.0.0\''
os.environ['PORT_MEF'] = '\'[5500,5501,5502]\''

# from best.cloud.db import *



class TestFeature(TestCase):
    def test_import_mefclient_connection_variables(self):
        print("Testing import from best.cloud._mefclient_connection_variables import *'")
        from best.cloud._mefclient_connection_variables import IP_MEF, PORT_MEF
        self.assertTrue(IP_MEF == '\'0.0.0.0\'')
        self.assertTrue(PORT_MEF == '\'[5500,5501,5502]\'')

    def test_import_mefclient_connection_variables(self):
        print("Testing import 'from best.cloud._db_connection_variables import *'")
        from best.cloud._db_connection_variables import IP_SQL, PORT_SQL, PW_SQL, USER_SQL, IP_SSH, PORT_SSH, PW_SSH, USER_SSH
        self.assertTrue(IP_SQL == '0.0.0.0')
        self.assertTrue(PORT_SQL == '3360')
        self.assertTrue(PW_SQL == 'db_pwd')
        self.assertTrue(USER_SQL == 'db_usr')
        self.assertTrue(USER_SSH == 'ssh_user')
        self.assertTrue(IP_SSH == '0.0.0.0')
        self.assertTrue(PW_SSH == 'ssh_pwd')
        self.assertTrue(PORT_SSH == '22')

if __name__ == '__main__':
    unittest.main()
