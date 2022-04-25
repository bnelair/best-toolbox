# SignalPlant
SignalPlant is free software tool for signal examination, scoring and post-processing. It has been developed by Medical Signals group of Institute of Scientific Instruments of CAS for projects concerning ECG and EEG signals. Although it is aimed to biological signals, it contains tools useful for any other area of signal processing.

- [Oficial webpage](https://www.medisig.com/signalplant/)
- DOI: [10.1088/0967-3334/37/7/N38](http://doi.org/10.1088/0967-3334/37/7/N38)
- [ResearchGate](https://www.researchgate.net/publication/303687877_SignalPlant_An_open_signal_processing_software_platform)
- No time-aware data
- Strengths
    - Fast signal viewer
    - Simple annotations
    - Good for simple tasks
    - Compatible with many formats (edf, h5, mat)
    - Simple and intuitive annotation file formats
    - Advanced analysis tools
- Weaknesses
    - Loads all signals into memory
    - Not compatible with MEF
    - Not good for user-sliding window feature. (Features/spectra needs to be always calculated again for a whole view window.)

### Features

SignalPlant architecture allows extendibility via plugins. This allows 3rd parties to develop plugins for their own file-formats or processing and displaying method. The most significant features are:

Real-time response while examining large files
Mark (trigger) operations
Non-destructive signal processing
Default plugins set offers over 30 tools for:
filtering (FFT/IIR/FIR ...) with real-time preview
analysis tool (FFT/TFA ....)
detection (QRS complex, local extrems ...)
file I/O operations for HDF5, BIN, MAT, EDF, EGI, CSV, M&I d-files and others
export images as SVG, EPS or PNG
Signal acquisition from a COM port (experimental)
and many more...


![SignalPlant Example](https://www.medisig.com/signalplant/images/signalplant.jpg?crc=225103499 )