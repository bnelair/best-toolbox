# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import setuptools
from glob import glob

from setuptools import Command, Extension
import shlex
import subprocess
import os
import re

## get version from file
VERSIONFILE="./best/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))



setuptools.setup(
    name="best_toolbox",
    version=verstr,
    license='',
    url="https://github.com/mselair/best_toolbox",

    author="Filip Mivalt",
    author_email="mivalt.filip@mayo.edu",


    description="Python package designed for iEEG analysis and sleep classification.",
    long_description="Python package for EEG sleep classification and analysis. Developed by the laboratory of Bioelectronics Neurophysiology and Engineering - Mayo Clinic",
    long_description_content_type="",

    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        'best_toolbox': ["best/dbs/artifact_removal/_configs/*.yaml", "best/dbs/artifact_removal/_models/*.pt", "dbs/artifact_bank/*.mat"]
    },
    data_files=[
        ('dbs_artifact_removal_cfg', glob('best/dbs/artifact_removal/_configs/*.yaml')),
        ('dbs_artifact_removal_mdl', glob('best/dbs/artifact_removal/_models/*.pt')),
        ('dbs_artifact_bank_artifact', glob('best/dbs/artifact_bank/*.mat')),
    ],


    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: POSIX :: Linux",
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    python_requires='>=3.6',
    install_requires =[
        'numpy',
        'pandas',
        'scipy',
        'tqdm',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'umap-learn',
        'pymef',
        'mef_tools',
        'pyedflib',
        'h5py',
        'sqlalchemy',
        'PyMySQL',
        'pytz',
        'python-dateutil',
        'pyzmq',
        'sshtunnel',
        'torch',
        'pyyaml',
        'mne'
    ]
)






