# Copyright 2020-present, Mayo Clinic Department of Neurology - Bioelectronics Neurophysiology and Engineering Laboratory
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

def validate_environment_variables():
    for key in mef_vars:
        if not key in environ:
            raise KeyError('[ENVIRONMENT VARIABLE MISSING]: Environment variable ' + key + ' was not found. '
                                                                                           'Following env variables required for a succesfull database connection setup: ' + str(mef_vars))



# SQL variables
PORT_MEF = IP_MEF = None
mef_vars = ['PORT_MEF', 'IP_MEF']



environ = list(dict(os.environ).keys())
validate_environment_variables()


for mefv in mef_vars:
    var = os.environ[mefv]

    if not '\'' in var: var = '\'' + var + '\''

    if mefv == 'PORT_MEF': # erase ' to create list and not string
        var = var.replace('\'', '')
    exec(mefv + '=' + var.replace(' ', ''))
