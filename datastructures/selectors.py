"""This module provides an interface for the HDF5 files created by n6tohdf5.

Classes:
Nbody6Snapshot -- container for the data in the HDF5 file

Nbody6Subset -- a subclass of Nboy6Snapshot, used to analyse a subset of stars.

Nbody6Header -- a class to grab just the header information.
"""
import numpy as np
from .nbstate import NBodySubset

def MassSelection(nbstate, mlow = None, mhigh = None):
    """Select a subset of stars based on mass ranges.
        
        Args:
            nbstate (NBodyState): a valid NBodyState object
            mlow (float): the lower mass cutoff in code units.
                            Default is the minimum stellar mass in the NbodyState.
            mhigh (float): the upper mass cutoff in code units.
                            Default is the maximum stellar mass in the NbodyState.
        Returns:
            NBodySubset: a subset containing all lying between the mass limits, inclusive.
    """
    if mlow is None:
        mlow = nbstate.mass.min()
    else:
        if mlow < nbstate.mass.min():
            print 'WARNING: specified lower mass cut is below the minimum mass star'
    if mhigh is None:
        mhigh = nbstate.mass.max()
    else:
        if mhigh > nbstate.mass.max():
            print 'WARNING: specified upper mass cut is above the maximum mass star'

    selection = (nbstate.mass >= mlow) & (nbstate.mass <= mhigh)
    if selection.sum() == 0:
        print 'WARNING: No stars are in the mass range!'
        print '       : selection failed, returning None'
        return None
    else:
        return NBodySubset(nbstate, nbstate.id[selection])            


def SphereSelection(nbstate, origin = [0, 0, 0], radius = 1.0):
    """Select a spherical subset of stars.
        
        Args:
            nbstate (NBodyState): a valid NBodyState object
            origin (list): the origin of the sphere in code units. defaults to the origin
                           can also accept strings 'origin', 'o', '0' for origin, or 'c',
                           'dc' for the nbstate density center.
            radius (float): the radius of the spherical selection in code units.
        Returns:
            NBodySubset: a subset containing all stars in the specified sphere.
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
        rads = np.linalg.norm(nbstate.pos - origin, axis=1)

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
