# Copyright 2020-present, Mayo Clinic Department of Neurology - Bioelectronics Neurophysiology and Engineering Laboratory
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from scipy.io import loadmat
from best import DELIMITER
from best.files import get_files


class _stimulation_artifacts(dict):
    _keys = dict([
        ( '.'.join(f.split(DELIMITER)[-1].split('.')[:-1]), f )
        for f in get_files(DELIMITER.join(__file__.split(DELIMITER)[:-1]), 'mat')
        if not '.-' in f.split(DELIMITER)[-1]
    ])

    def keys(self):
        return self._keys.keys()

    def get_artifact(self, item):
        f = self._keys[item]
        tmp = loadmat(f)
        return {'X': tmp['X'][0], 'fs': tmp['fs'][0, 0]}

    def __getitem__(self, item):
        return self.get_artifact(item)

    def __call__(self, item):
        return self[item]

    def __str__(self):
        return 'Stimulation Artifacts: ' + str(list(self._keys.keys()))

    def __repr__(self):
        return self.__str__()

stimulation_artifacts = _stimulation_artifacts()

"""

"""





