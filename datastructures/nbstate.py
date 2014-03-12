from __future__ import division
from __future__ import absolute_import

"""This module provides storage for the state of an N-body simulation at some time

Classes:
NBodyState -- the state of an N-body simulation. stellar positions, masses, etc.
              plus derived quantities like density center, lagrangian radii etc.

NBodySubset -- a subclass of Nboy6Snapshot, used to analyse a subset of stars selected
               based on some physical criterion like mass, position, boundedness etc.
"""

import numpy as np
from scipy.spatial import cKDTree
from .. import units as units


class RangeError(Exception): 
    pass


class NBodyState(object):
    """Store the state of an N-body simulation
    
    All the information in the snapshot file is read in and stored as numpy arrays. 
    Additional information is calculated, such as the radius of each star from the 
    center of the domain as well as the density center. 
    
    The attributes are populated by the datahandlers, e.g. nigelhdf5reader
    
    Attributes:
        n (int): number of stars in the snapshot
        time (float): time of the snapshot in code units
        rscale (float): position scaling from code units to cgs. R_cgs = rscale * R_code
        vscale (float): velocity scaling from code units to cgs. V_cgs = vscale * V_code
        mscale (float): mass scaling from code units to cgs. M_cgs = mscale * M_code
        tscale (float): time scaling from code units to cgs. T_cgs = tscale * T_code
        pos (numpy float array, shape(n, 3)): positions of the stars in code units.
        vel (numpy float array, shape(n, 3)): velocities of the stars in code units.
        mass (numpy float array, shape(n, )): masses of the stars in code units.
        id (numpy int array, shape(n, )): unique id number of the stars.
        luminosity(numpy float array, shape(n, )): luminosity of the stars in L_sun. 
        temperature(numpy float array, shape(n, )): temperature of the stars in K. 
        dc_pos (numpy float array, shape(3, )): density center of the stars. 
        dc_vel (numpy float array, shape(3, )): velocity of the density center of the stars. 
        radii_origin (numpy float array, shape(n, )): radius of each star from the origin in code units. 
        radii_dc (numpy float array, shape(n, )): radius of each star from the density center in code units.
        core_radius (float): core radius in code units.
        core_density (float): core density in code units. 
        core_n (int): number of stars in the core.
        half_mass_radius (float): half-mass radius of the cluster in code units.
        sigmas (numpy float array, shape(3, )): standard deviation of the velocities in code units.
        x (numpy float array, shape(n, )): the x-positions of the stars in code units. 
        y (numpy float array, shape(n, )): the y-positions of the stars in code units.
        z (numpy float array, shape(n, )): the z-positions of the stars in code units. 
    """
    
    def __init__(self):
        """Define empty storage for the N-body simulation state.
        
        Note that computationally intensive quantities like radii, core properties, etc.
        are lazily created using properties.
        """
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
        """Rescale the simulation to a new characteristic length.
        
        Args:
            scale (float): the new length scale 
        """
        try:
            scale = float(scale)
            if scale <= 0:
                raise RangeError
            self.tscale *= (scale / self.rscale)**1.5 
            self.rscale = scale
        except TypeError:
            print "bad scale value passed to rescale_length; leaving the scale alone."
        except RangeError:
            print "scale needs to be greater than zero; leaving the scale alone."
            
    def _get_local_densities(self):
        """Calculate the local stellar density.
        
        Uses the method of Casertano & Hut ApJ 1985, 298, 80. 
        This is used to calculate the density centers.
        
        Returns:
            numpy float array, shape(n, ) containing density for each star in code units
        """
        n_neighbors = 7
        if self.n < 7:
            print "Warning! There are only %d stars!" %(self.n)
            print "Calculating density center using %d neighbors."
            n_neighbors = self.n
            
        #  get a nearest neighbor tree  
        kdtree = cKDTree(self.pos)    
        # the first result is the point itself, so sixth neighbor is the seventh result
        (dists, indices) = kdtree.query(self.pos, n_neighbors)
        near6 = kdtree.query(self.pos, n_neighbors)[0][:,n_neighbors - 1] # distance to 6th nearest neighbor
        vols = near6**3   # no need for 4/3 pi I guess
        masses = self.mass[indices[:,1:n_neighbors - 1]].sum(axis=1) # total mass of 5 nearest neighbors
        densities = masses / vols
        return densities
        
    def find_density_center(self):
        """Calculate and set the density center and velocity the stars"""
        densities = self._get_local_densities() 
        # density center is density weighted radius of the stars
        self._dc_pos = (densities[:,np.newaxis] * self.pos).sum(axis=0) / densities.sum()    
        self._dc_vel = (densities[:,np.newaxis] *   \
            self.vel).sum(axis=0) / densities.sum()
        (densities[:,np.newaxis] * self.pos).sum(axis=0) / densities.sum()
        # update radii
        self._find_radii_from_density_center()
    
    
    def _find_radii_from_density_center(self):
        """Calculate and set radii from the density center"""
        self._radii_dc = np.linalg.norm(self.pos - self.dc_pos, axis=1)
    
    
    def _find_core_properties(self):
        """Calculate the core radius of the cluster. This also sets core density and n"""
        raise NotImplementedError   
    
    def _find_half_mass_radius(self):
        """Calculate the half mass radius of the cluster from the density center."""
        raise NotImplementedError   
    
    
    
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
            
        self.n = len(self.id)    
            
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
        """Calculate the density center and velocity of a subset of stars."""
        densities = self._get_local_densities() 
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
    
        