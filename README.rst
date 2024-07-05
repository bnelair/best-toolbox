
.. image:: https://github.com/mselair/best_toolbox/actions/workflows/test_publish.yml/badge.svg
    :target: https://pypi.org/project/best-toolbox/

.. image:: https://readthedocs.org/projects/best-toolbox/badge/?version=latest
     :target: https://best-toolbox.readthedocs.io/en/latest
     :alt: Documentation Status



BEhavioral STate Analysis Toolbox (BEST)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

A Python package for EEG and behavioral state analysis EEG using multiple tools.
Including tools for automated sleep classification, analysis of long-term and short-term electrical brain signals
recorded both invasively and non-invasively for an acute period of time but also for long-term data spanning years.

The tools were developed in the `Bioelectronics Neurophysiology and Engineering Laboratory <https://www.mayo.edu/research/labs/bioelectronics-neurophysiology-engineering/overview>`_ at Mayo Clinic, Rochester, MN, USA.


Installation
"""""""""""""""""""""""""""

.. code-block:: bash

    pip install best-toolbox


Acknowledgement
"""""""""""""""""""""""""""
BEST was developed and originally published for the first time with by (Mivalt 2022, and Sladky 2022).
We apreciate you citing these papers when utilizing this toolbox in further research work.

 | F. Mivalt et V. Kremen et al., “Electrical brain stimulation and continuous behavioral state tracking in ambulatory humans,” J. Neural Eng., vol. 19, no. 1, p. 016019, Feb. 2022, doi: 10.1088/1741-2552/ac4bfd.
 |
 | V. Sladky et al., “Distributed brain co-processor for tracking spikes, seizures and behaviour during electrical brain stimulation,” Brain Commun., vol. 4, no. 3, May 2022, doi: 10.1093/braincomms/fcac115.

Please, see the sections below for references to individual submodules.

Sleep classification and feature extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Automated feature extraction and sleep classification was developed during the following projects.

 | F. Mivalt et V. Kremen et al., “Electrical brain stimulation and continuous behavioral state tracking in ambulatory humans,” J. Neural Eng., vol. 19, no. 1, p. 016019, Feb. 2022, doi: 10.1088/1741-2552/ac4bfd.

 | F. Mivalt et V. Sladky et al., “Automated sleep classification with chronic neural implants in freely behaving canines,” J. Neural Eng., vol. 20, no. 4, p. 046025, Aug. 2023, doi: 10.1088/1741-2552/aced21.

Our work was based on the following references:

 | Gerla, V., Kremen, V., Macas, M., Dudysova, D., Mladek, A., Sos, P., & Lhotska, L. (2019). Iterative expert-in-the-loop classification of sleep PSG recordings using a hierarchical clustering. Journal of Neuroscience Methods, 317(February), 61?70. https://doi.org/10.1016/j.jneumeth.2019.01.013

 | Kremen, V., Brinkmann, B. H., Van Gompel, J. J., Stead, S. (Matt) M., St Louis, E. K., & Worrell, G. A. (2018). Automated Unsupervised Behavioral State Classification using Intracranial Electrophysiology. Journal of Neural Engineering. https://doi.org/10.1088/1741-2552/aae5ab

 | Kremen, V., Duque, J. J., Brinkmann, B. H., Berry, B. M., Kucewicz, M. T., Khadjevand, F., G.A. Worrell, G. A. (2017). Behavioral state classification in epileptic brain using intracranial electrophysiology. Journal of Neural Engineering, 14(2), 026001. https://doi.org/10.1088/1741-2552/aa5688


Seizure detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 | V. Sladky et al., “Distributed brain co-processor for tracking spikes, seizures and behaviour during electrical brain stimulation,” Brain Commun., vol. 4, no. 3, May 2022, doi: 10.1093/braincomms/fcac115.

Artificial Signal Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 | F. Mivalt et al., “Deep Generative Networks for Algorithm Development in Implantable Neural Technology,” in 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Oct. 2022, pp. 1736–1741, doi: 10.1109/SMC53654.2022.9945379.

Evoked Response Potential Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 | K. J. Miller et al., “Canonical Response Parameterization: Quantifying the structure of responses to single-pulse intracranial electrical brain stimulation,” PLOS Comput. Biol., vol. 19, no. 5, p. e1011105, May 2023, doi: 10.1371/journal.pcbi.1011105.

EEG Slow Wave Detection and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Readme to the EEG Slow Detection project available in this repository in this repository: `projects/slow_wave_detection.rst <./projects/slow_wave_detection.rst>`_.

 | XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX



Funding
""""""""""""""""""

BEST was developed under projects supported by

- NIH Brain Initiative UH2&3 NS095495 - *Neurophysiologically-Based Brain State Tracking & Modulation in Focal Epilepsy*,
- NIH U01-NS128612 - *An Ecosystem of Techmology and Protocols for Adaptive Neuromodulation Research in Humans*,
- DARPA - HR0011-20-2-0028 *Manipulating and Optimizing Brain Rhythms for Enhancement of Sleep (Morpheus)*.

Filip Mivalt was also partially supported by the grant FEKT-K-22-7649 realized within the project Quality Internal Grants of the Brno University of Technology (KInG BUT), Reg. No. CZ.02.2.69/0.0/0.0/19_073/0016948, which is financed from the OP RDE.


License
""""""""""""""""""

This software is licensed under BSD-3 license. For details see the `LICENSE <https://github.com/bnelair/best-toolbox/blob/master/LICENSE>`_ file in the root directory of this project.


Documentation
"""""""""""""""""""""""""""
Documentation is available on `Read the Docs <https://best-toolbox.readthedocs.io/en/latest/>`_.


