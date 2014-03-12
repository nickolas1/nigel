from __future__ import division
from __future__ import absolute_import

"""This module provides an interface for the HDF5 files created by n6tohdf5.

Classes:
Nbody6Snapshot -- container for the data in the HDF5 file

Nbody6Subset -- a subclass of Nboy6Snapshot, used to analyse a subset of stars.

Nbody6Header -- a class to grab just the header information.
"""
import numpy as np
from scipy.spatial import cKDTree
from .. import units as units



class NBodyState(object):
    """Read in an hdf5 snapshot from an nbody6 simulation.
    
    All the information in the hdf5 file is read in and stored as numpy arrays. 
    Additional information is calculated, such as the radius of each star from the 
    center of the domain as well as the density center. 
    
    Public functions:
    calc_new_density_center -- self-contained density center calculation.
    
    calc_lagrangian_radii -- determine the Lagrangian radii of a snapshot.
    
    mass_selection -- get a Nbody6Subset based on a mass range.
    
    sphere_selection -- get a Nbody6Subset based on a spherical region.
    
    make_pretty_image_data -- get a 'realistic' rendering of the stars
    """
    
    def __init__(self):
        # number of stars
        self.n = None
           
        # time in nbody units
        self.time = None
            
        # radius in cgs
        self.rscale = None
    
        # velocity scaling to cgs
        self.vscale = None
    
        # mass scaling to cgs
        self.mscale = None
        
        # time scaling to cgs
        self.tscale = None
        
        # positions of stars in nbody units
        self.pos = None
        
        # velocities of stars in nbody units
        self.vel = None
        
        # masses of stars in nbody units
        self.mass = None
        
        # id numbers of stars
        self.id = None
        
        # luminosities in Lsun
        self.luminosity = None
        
        # temperatures in K
        self.temperature = None   
        
        # these next ones are computationally involved or derived from other quantities
        # so they are made lazy. properties for these are below.
        # density center
        self._dc_pos = None
        self._dc_vel = None
        
        # radius of the stars from the origin
        self._radii_origin = None
              
        # radius of the stars from the density center
        self._radii_dc = None
        
        # core radius and mass density and number
        self._core_radius = None
        self._core_density = None
        self._core_n = None
        
        # half mass radius
        self._half_mass_radius = None
        
        # velocity dispersion
        self._sigmas = None
        
        # shortcuts to x, y, z
        self._x = None
        self._y = None
        self._z = None

    
    @property
    def time_yr(self):
        # time in yr
        return self.time * self.tscale / units.yr_cgs
    
    @property
    def tcross(self):
        # crossing time in nbody units
        pass
        
    @property
    def trh(self):
        # relaxation time in nbody units
        pass
        
    @property
    def sigmas(self):
        if self._sigmas is None:
            self._sigmas = np.std(self.vel, axis=0)
        return self._sigmas
        
    @property
    def tcross_yr(self):
        return self.tcross * self.tscale / units.yr_cgs
        
    @property
    def trh_yr(self):
        return self.trh * self.tscale / units.yr_cgs
        
    @property
    def dc_pos(self):
        if self._dc_pos is None:
            self.find_density_center()
        return self._dc_pos    
        
    @property
    def dc_vel(self):
        if self._dc_vel is None:
            self.find_density_center()
        return self._dc_vel    
        
    @property
    def radii_origin(self):
        if self._radii_origin is None:
             self._radii_origin = np.linalg.norm(self.pos, axis=1)
        return self._radii_origin 
    
    @property
    def radii_dc(self):
        if self._radii_dc is None:
            self._find_radii_from_density_center()
        return self._radii_dc
    
    @property
    def core_radius(self):
        if self._core_radius is None:
            self._find_core_properties()
        return self._core_radius
    
    @property
    def core_density(self):
        if self._core_density is None:
            self._find_core_properties()
        return self._core_density
    
    @property
    def core_n(self):
        if self._core_n is None:
            self._find_core_properties()
        return self._core_n
    
    @property
    def half_mass_radius(self):
        if self._half_mass_radius is None:
            self._find_half_mass_radius()
        return self._half_mass_radius

    @property
    def x(self):
        if self._x is None:
            self._x = self.pos[:,0]
        return self._x
    @property
    def y(self):
        if self._y is None:
            self._y = self.pos[:,1]
        return self._y

    @property
    def z(self):
        if self._z is None:
            self._z = self.pos[:,2]
        return self._z                   
     
    def rescale_length(self, scale):
        """ Rescale the simulation to a new characteristic length.
        
        Arguments:
        scale -- the new scale length of the simulation in nbody units 
        """
        self.tscale *= (scale / self.rscale)**1.5 
        self.rscale = scale
        
        
    def find_density_center(self):
        """Calculate the density center of a set of stars
        
        Uses the method of Casertano & Hut ApJ 1985, 298, 80. 
        """
        #  get a nearest neighbor tree  
        kdtree = cKDTree(self.pos)    
        # the first result is the point itself, so sixth neighbor is the seventh result
        (dists, indices) = kdtree.query(self.pos, 7)
        near6 = kdtree.query(self.pos, 7)[0][:,6] # distance to 6th nearest neighbor
        vols = near6**3   # no need for 4/3 pi I guess
        masses = self.mass[indices[:,1:6]].sum(axis=1) # total mass of 5 nearest neighbors
        densities = masses / vols 
        # density center is density weighted radius of the stars
        self._dc_pos = (densities[:,np.newaxis] * self.pos).sum(axis=0) / densities.sum()    
        self._dc_vel = (densities[:,np.newaxis] *   \
            self.vel).sum(axis=0) / densities.sum()
        (densities[:,np.newaxis] * self.pos).sum(axis=0) / densities.sum()
        # update radii
        self._find_radii_from_density_center()
    
    
    def _find_radii_from_density_center(self):
        """Calculate radii from the density center"""
        self._radii_dc = np.linalg.norm(self.pos - self.dc_pos, axis=1)
    
    
    def _find_core_properties(self):
        """Calculate the core radius of the cluster. This also sets core density and n"""
        pass    
    
    def _find_half_mass_radius(self):
        """Calculate the half mass radius of the cluster from the density center."""
        pass
    
    
    
class NBodySubset(NBodyState):
    """Get a subset of a snapshot, selected by the Names of the stars.
    
    This is a subclass of NBodyState. It can be used to run analyses (Lagrangian
    Radii, density center, velocity dispersions etc.) on a subset of the stars in 
    a simulation. Stars are added to the subset based on a list of names.
    """
    def __init__(self, nbstate, nameselection):
        """Get a subset of a NBodyState
        
        Arguments:
        nbstate -- a valid NBodyState instance.
        nameselection -- an array of names to select.
        """
        super(NBodySubset, self).__init__()
        
        selection = np.in1d(nbstate.id, nameselection)


        self.pos = nbstate.pos[selection]
        self.vel = nbstate.vel[selection]
        self.mass = nbstate.mass[selection]
        self.id = nbstate.id[selection]
        self.luminosity = nbstate.luminosity[selection]
        self.temperature = nbstate.temperature[selection]        
            
        self.nstars = len(self.id)    
            
        # time and in Nbody
        self.time = nbstate.time
        # scaling factors
        self.rscale = nbstate.rscale
        self.tscale = nbstate.tscale
        self.mscale = nbstate.mscale
        self.vscale = nbstate.vscale

        # keep the density center of the whole ensemble
        self._dc_pos = nbstate.dc_pos
        self._dc_vel = nbstate.dc_vel
        # radius of the stars from the origin and density center of the whole ensemble
        self._radii_origin = nbstate.radii_origin[selection]   
        self._radii_dc = nbstate.radii_dc[selection]
        
        # density center of the subset only, as well as radius from that density center
        self._dc_pos_subset = None
        self._dc_vel_subset = None
        self._radii_dc_subset = None

              
    @property
    def dc_pos_subset(self):
        if self._dc_pos_subset is None:
            self.find_density_center_subset()
        return self._dc_pos_subset    
        
    @property
    def dc_vel_subset(self):
        if self._dc_vel_subset is None:
            self.find_density_center_subset()
        return self._dc_vel_subset    
    
    @property
    def radii_dc_subset(self):
        if self._radii_dc_subset is None:
            self._find_radii_from_density_center_subset()
        return self._radii_dc_subset
    
    def find_density_center_subset(self):
        """Calculate the density center of a set of stars
        
        Uses the method of Casertano & Hut ApJ 1985, 298, 80. 
        """
        if self.nstars < 7:
            print "Warning! There are only %d stars in this subset!" %(self.nstars)
            print "Calculating density center using %d neighbors."
            self._dc_pos_subset = (self.mass[:,np.newaxis] *   \
                self.pos).sum(axis=0) / self.mass.sum()
            self._dc_vel_subset = (self.mass[:,np.newaxis] *   \
                self.vel).sum(axis=0) / self.mass.sum()
            return
                
        #  get a nearest neighbor tree  
        kdtree = cKDTree(self.pos)    
        # the first result is the point itself, so sixth neighbor is the seventh result
        (dists, indices) = kdtree.query(self.pos, 7)
        near6 = kdtree.query(self.pos, 7)[0][:,6] # distance to 6th nearest neighbor
        vols = near6**3   # no need for 4/3 pi I guess
        masses = self.mass[indices[:,1:6]].sum(axis=1) # total mass of 5 nearest neighbors
        densities = masses / vols 
        # density center is density weighted radius of the stars
        self._dc_pos_subset = (densities[:,np.newaxis] *   \
            self.pos).sum(axis=0) / densities.sum()    
        self._dc_vel_subset = (densities[:,np.newaxis] *   \
            self.vel).sum(axis=0) / densities.sum()
        (densities[:,np.newaxis] * self.pos).sum(axis=0) / densities.sum()
        # update radii
        self._find_radii_from_density_center_subset()           
    
    def _find_radii_from_density_center_subset(self):
        """Calculate radii from the density center"""
        self._radii_dc_subset = np.linalg.norm(self.pos - self.dc_pos_subset, axis=1)
    
        