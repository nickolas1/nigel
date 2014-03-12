from __future__ import division

"""This module provides an interface for the HDF5 files created by n6tohdf5.

Classes:
Nbody6Snapshot -- container for the data in the HDF5 file

Nbody6Subset -- a subclass of Nboy6Snapshot, used to analyse a subset of stars.

Nbody6Header -- a class to grab just the header information.
"""

import numpy as np
from numpy.core.umath_tests import inner1d
import h5py
import sys
from scipy import spatial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ..datastructures.nbstate import NBodyState
try:
    import h5py
except ImportError:
    h5py = None


class NigelHDF5State(NBodyState):
    def __init__(self, infile):
        """Read a nigel HDF5 snapshot and populate the NBodyState
    
        this is not usually called directly from the user, rather it's called
        as part of the 'load' command.
        """
        super(NigelHDF5State, self).__init__()
        
        f = h5py.File(infile, 'r')
        # get some information from the header. 
        header = f['Header']
        self.n = int(header.attrs['Nstars'])
        self.time = header.attrs['Time']
       # self.TimeMyr = header.attrs['TimeMyr']
        self.tscale = header.attrs['Tscale']
        self.rscale = header.attrs['Rbar']
        self.vscale = header.attrs['Vstar']
        self.mscale = header.attrs['Mbar']
    
        self.pos = np.array(f['Stars/Positions'])
        self.vel = np.array(f['Stars/Velocities'])
        self.mass = np.array(f['Stars/Masses']).reshape(self.n)
        self.id = np.array(f['Stars/Names'], dtype=np.int64).reshape(self.n)
        try:
            self.luminosity = 10**np.array(f['Stars/Luminosity']).reshape(self.n)
        except KeyError: 
            print 'no luminosities for this snapshot'
        try:
            self.temperature = 10**np.array(f['Stars/Teff']).reshape(self.n)
        except KeyError: 
            print 'no temperatures in this snapshot'       
    
        # close the HDF5 file
        f.close()

    @staticmethod
    def _is_readable(infile):
        print infile
        try:
            if h5py.is_hdf5(infile):
                return True
            else:
                return False
        except AttributeError:
            if 'hdf5' in infile:
                print 'problem with HDF5 support!'
            return False