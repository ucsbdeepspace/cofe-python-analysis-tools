"""This file contains tools for parsing the data contained in the .dat files
produced by the telescopes' data acquisition code.
"""
import numpy as np
import logging as l

import utils
from dtypes import *

def open_raw(filename):
    """Reads a .dat file into a memmap

    Parameters
    ----------
    filename : str
        input .dat filename

    Returns
    -------
    out : memmap
        numpy memory map array
    """
    l.info('Loading raw file %s' % filename)
    raw_data = np.memmap( filename, dtype=dat_dtype, mode='r')
    return raw_data

def remove_noise_triggers(d):
    """Find noise triggered data

    returns cut down memmap whie still needs to be0
    tested for incomplete revolutions etc.
    """
    encoder=d['enc']/16
    enc_diff=np.diff(encoder)
    good=[i for i in range(len(enc_diff)) if (enc_diff[i] != 0)]
    
    return d[good]

def create_revdata(raw_data, volts=True,supply_index=False, printstatements=True):

    """Deletes invalid revolutions and shapes the array on revolutions
    
    Parameters
    ----------
    raw_data : ndarray
        input array with dtype dat_dtype

    Returns
    -------
    revdata : ndarray
        reshaped output dataset
    """

    # remove partial revolutions at the beginning and end of dataset
    d=raw_data.copy()
    if supply_index==True:
        d['enc']=np.mod(d['enc']+6,4096)
    start_of_revs, = np.where(d['enc'] < config['ENC_START_TRIGGER'])
    if len(start_of_revs)>0:
        d = np.array(d[start_of_revs[0]:start_of_revs[-1]])
    d = remove_noise_triggers(d)

    if printstatements:
        print len(start_of_revs)
        print config['ENC_START_TRIGGER']

    # remove revolutions with bad number of samples 
    start_of_revs, = np.where(d['enc'] < config['ENC_START_TRIGGER'])
    # add the end of array to compute the length of the last revolution
    start_of_revs = np.append(start_of_revs, len(d['enc']))
    samples_per_rev = np.diff(start_of_revs)
    invalid_revs, = np.where(samples_per_rev != config['SEC_PER_REV'])

    if printstatements:
        if len(invalid_revs) > 0:
            l.warning('Removing invalid revolutions (index from beginning of file): %s' % invalid_revs)
        else:
            l.info('No invalid revolutions')

    # remove the samples of the bad revolutions from the array
    #print('dshape',np.shape(d))
    for i in invalid_revs[::-1]:
	d = np.delete(d, np.s_[start_of_revs[i]:start_of_revs[i+1]])
    #print('dshape after cleaning',np.shape(d))

	
    #for i in invalid_revs:
      #  d = np.delete(d, np.s_[start_of_revs[i]:start_of_revs[i+1]])

    out_dtype = rev_dtype if volts else rev_dtype_adu

    # create the rev output array
    if len(d) == 0:
        l.error('NO VALID DATA IN FILE')
        data = np.zeros(0, dtype=out_dtype)
    else:
        data = np.zeros(len(d)/config['SEC_PER_REV'], dtype=out_dtype)
        d_rev = d[::config['SEC_PER_REV']]
	if len(d_rev)>len(data):
		data=np.append(data,data[0])
        data['rev'] = d_rev['rev0'].astype(np.long) + \
                      d_rev['rev1'].astype(np.long) * 256 + \
                      d_rev['rev2'].astype(np.long) * 256**2
        data['azi'] = d_rev['rev0'].astype(np.long)+d_rev['rev1'].astype(np.long)*256
        for ch in channels_labels:
            chdata = d[ch].reshape((-1, config['SEC_PER_REV']))
            if volts:
                data[ch] = utils.adu2volts(chdata)
            else:
                data[ch] = chdata 

    return data

def read_raw(filenames, volts=True,supply_index=False, printstatements = True):
    """Reads a list of filenames, creates revdata dataset and concatenates them
    
    Parameters
    ----------
    filenames : list
        list of .dat filenames to read
    volts : bool, optional
        whether to convert to volts or keep ADU

    Returns
    -------
    revdata : array
        reshaped concatenated array
    """
    return np.concatenate(
                [create_revdata(open_raw(f), volts,supply_index=supply_index, printstatements=printstatements) for f in filenames]
                         )

if __name__=="__main__":
    print 'what the fuck!!!!!!!!!'