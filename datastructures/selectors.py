"""This module provides an interface for the HDF5 files created by n6tohdf5.

Classes:
Nbody6Snapshot -- container for the data in the HDF5 file

Nbody6Subset -- a subclass of Nboy6Snapshot, used to analyse a subset of stars.

Nbody6Header -- a class to grab just the header information.
"""
import numpy as np
from .nbstate import NBodySubset

def MassSelection(nbstate, mlow = None, mhigh = None):
    """make a subset of stars based on a mass range
    
    Arguments:
    nbstate -- a valid NBodyState instance
    mlow -- lower mass range in code units. default: minimum mass
    mhigh -- upper mass range in code units. default: maximum mass
    
    Returns:
    NBodySubset containing the stars in the requested mass range
    """
    if mlow is None:
        mlow = self.masses.min()
    else:
        if mlow < self.masses.min():
            print 'WARNING: specified lower mass cut is below the minimum mass star'
    if mhigh is None:
        mhigh = self.masses.max()
    else:
        if mhigh > self.masses.max():
            print 'WARNING: specified upper mass cut is above the maximum mass star'

    selection = (self.masses >= mlow) & (self.masses <= mhigh)
    if selection.sum() == 0:
        print 'WARNING: No stars are in the mass range!'
        print '       : selection failed, returning None'
        return None
    else:
        return NBodySubset(nbstate, nbstate.id[selection])            


def SphereSelection(nbstate, origin = [0, 0, 0], radius = 1.0):
    """make a spherical subset of stars
    
    Arguments:
    origin -- origin of the spherical region. default: [0, 0, 0]
    radius -- radius range in code units of the spherical region
    
    Returns:
    Nbody6Subset containing the stars in the requested mass range
    """
    if isinstance(origin, basestring):
        if (origin == 'origin') | (origin == 'o') | (origin == '0'):
            origin = [0, 0, 0]
        elif (origin == 'c') | (origin == 'dc'):
            origin = nbstate.dc_pos
    
    if np.allclose(origin, [0, 0, 0]):
        rads = nbstate.radii_origin
    elif np.allclose(origin, nbstate.dc_pos):
        rads = nbstate.radii_dc
    else:
        rads = np.linalg.norm(self.pos - origin, axis=0)
    selection = (rads <= radius)
    if selection.sum() == 0:
        print 'WARNING: No stars are in the requested sphere!'
        print '       : selection failed, returning None'
        return None
    else:
        return NBodySubset(nbstate, nbstate.id[selection])            
        
def BoundSelection(self, lagrangecut = 0.9):
    """make a subset of stars that are probably bound.
    
    Arguments:
    lagrangecut -- lagrangian radius outside which stars may be escapers. default: 0.9
    
    Returns:
    Nbody6Subset containing the stars that are probably bound.    
    """
    """
    self.calc_lagrangian_radii([lagrangecut])
    # get the desired lagrangian radius
    #rcut = self.Lagr_rads[np.where(self.Lagr_fractions == lagrangecut)]
    rcut = self.Lagr_rads[0]
    # get stars inside the cutoff
    selection = (self.Radii_dcm_new <= rcut)
    # and outside
    outside = (self.Radii_dcm_new > rcut)

    # get stars with positive v.r
    vdotr = inner1d(self.Pos - self.dc_pos_new, 
                    self.Vel - self.dc_vel_new)
    streamers = (vdotr > 0)
   
    
    # safe ones: inside or outside and not streaming
    selection = -outside | -(outside & streamers)
    possibles = outside & streamers
    possiblenames = self.Names[possibles]    
            
            
    # get velocities of stars outside the cut
    velrels = self.Vel[possibles] - self.dc_vel_new
    vel2 = np.array(inner1d(velrels, velrels))
    # get squared escape velocity from the masscut at each radius in nbody units
    vesc2 = 2 * self.Masses.sum() * lagrangecut / self.Radii_dcm_new[possibles]
    keepers = (vel2 < vesc2)
    keepnames = possiblenames[keepers] 
    selectionnames = np.vstack((self.Names[selection], keepnames))    
        
    # update radii from density center
    poscm = self.Pos - self.dc_pos
    self.Radii_dcm_new = np.array(np.sqrt(inner1d(poscm, poscm)))  
    if selection.sum() == 0:
        print 'No stars are in the sphere!'
        return None
    else:
        return Nbody6Subset(self, selectionnames) 
    """
    pass
