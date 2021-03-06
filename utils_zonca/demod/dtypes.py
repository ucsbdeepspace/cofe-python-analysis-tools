import numpy as np
import os
from ConfigParser import ConfigParser

config = dict(  NCHAN=16, 
                SEC_PER_REV=256, 
                ENC_START_TRIGGER=16
             )

# Names of the data channels:
channels_labels = ['ch%d' % i for i in range(config['NCHAN'])]

# Read phases
phases = ConfigParser(dict(((ch,'0') for ch in channels_labels)))
phases.read(os.path.join(os.path.dirname(__file__), 'phases.cfg'))

# Structure of the data we read from the .dat files:
dat_dtype = np.dtype( [(ch,np.uint16) for ch in channels_labels] + 
                      [('enc',np.uint16)]+[('dummy',np.uint16)]  + 
                      [('rev%d' % i,np.uint16) for i in range(3)])

# Structure of revdata before demodulation
# in volts
rev_dtype = np.dtype( [('azi',np.uint16)] +
                      [('rev',np.long)] + 
                      [(ch,np.float,config['SEC_PER_REV']) for ch in channels_labels] )
# in ADU
rev_dtype_adu = np.dtype([('azi',np.uint16)] +
                        [('rev',np.long)] + 
                      [(ch,np.uint16,config['SEC_PER_REV']) for ch in channels_labels] )

# Structure of a single channel
ch_dtype = np.dtype( [('T',np.float),('Q',np.float),('U',np.float)] )

# Structure of the output demodulated data
demod_dtype = np.dtype( [('azi',np.uint16)] + [('rev',np.float)] + [(ch,ch_dtype) for ch in channels_labels] )
