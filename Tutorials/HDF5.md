# Convert H5 to be compatible with SignalPlant

```python
def mef2hdf(hfile, multmef, ch_names, fs):
    """
    Create SignalPlant HDF file from mef data files previously 
    loaded by pymef.
    !!! overwrites any existing file with the same name !!!
    Multiscale Eletrophysiology Format:
    http://msel.mayo.edu/research.html,
    https://pymef.readthedocs.io/en/latest/getting_started.html
    Parameters
    ----------
    hfile: string
        full path and name of the new hdf file
    multmef: list of lists with numpy arrays
        data loaded by pymef
    ch_names: list of strings
        list of channel names
    fs: float
        sampling frequency   

    Returns
    -------
    """ 
    # convert mefmat list to float32 array
    multmef = np.asarray(multmef)
    multmef = np.float32(multmef)

    # check rows and columns of data matrix
    rows,columns = multmef.shape
    if rows>columns:
        multmef = multmef.T    

    # create hdf file
    hf = h5py.File(hfile,'w')
    # write data to hdf file
    hf.create_dataset('Data', data=multmef)   
    # write channel name
    channel_list = []
    datacache = 'RAW'
    datacache = datacache.encode()
    units = 'mV'
    units = units.encode()

    for ch in ch_names:
        ch_name = (ch.encode(),datacache,units)
        channel_list.append(ch_name)

    info = np.array(channel_list, dtype=[('ChannelName', 'S256'), ('DatacacheName', 'S256'), ('Units', 'S256')])
    hf.create_dataset('Info', data=info)
    # write attributes
    hf.attrs['Fs'] = np.float32(fs)  
    current_date = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    hf.attrs['GeneratedBy'] = 'python multmef2hdf, ' + current_date   
    # close hdf file

    hf.close()
```








