# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from best import DELIMITER
from best._config import get_files, config

configs_DCGAN = dict([(f.split(DELIMITER)[-1].split('.')[0], config(f)) for f in get_files(DELIMITER.join(__file__.split(DELIMITER)[:-1]), 'yaml') if not '.-' in f.split(DELIMITER)[-1]])




