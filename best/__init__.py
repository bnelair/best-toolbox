# Copyright 2020-present, Mayo Clinic Department of Neurology - Bioelectronics Neurophysiology and Engineering Laboratory
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

__version__ = '0.0.4'



import os
# Check windows or linux and sets separator
DELIMITER = os.path.join(' ', ' ')[1]
PATH_PACKAGE = DELIMITER.join(__file__.split(DELIMITER)[::-1])


"""

A Python package providing tools and utilities which are nice to have while working not only on this project.

"""