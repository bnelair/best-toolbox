# Copyright 2020-present, Mayo Clinic Department of Neurology - Bioelectronics Neurophysiology and Engineering Laboratory
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

def validate_environment_db_variables():
    for key in sql_vars:
        if not key in environ:
            raise KeyError('[ENVIRONMENT VARIABLE MISSING]: Environment variable ' + key + ' was not found. '
                                                                                           'Following env variables required for a succesfull database connection setup: ' + str(sql_vars))


    w_raised = False
    for key in sql_vars:
        if not key in environ:
            raise Warning('[ENVIRONMENT VARIABLE MISSING]: Environment variable ' + key + ' was not found.')
            w_raised = True
    if w_raised:
        raise Warning('[SSH CONNECTION IGNORED]: Connection details for SSH connection were not found. DatabaseHandler will'
                      'connect directly to database from your current IP address.')

# SQL variables
IP_SQL = USER_SQL = PW_SQL = PORT_SQL = None
sql_vars = ['IP_SQL', 'USER_SQL', 'PW_SQL', 'PORT_SQL']
# SSH_variables
IP_SSH = USER_SSH = PW_SSH = PORT_SSH = None
ssh_vars = ['IP_SSH', 'USER_SSH', 'PW_SSH', 'PORT_SSH']


environ = list(dict(os.environ).keys())
validate_environment_db_variables()


for sqlv in sql_vars:
    var = os.environ[sqlv]
    if not '\'' in var: var = '\'' + var + '\''
    exec(sqlv + '=' + var.replace(' ', ''))

for sshv in ssh_vars:
    if sshv in environ:
        var = os.environ[sshv]
        if not '\'' in var: var = '\'' + var + '\''
        exec(sshv + '=' + var.replace(' ', ''))
    else:
        exec(sshv + '=' + 'None')
