#!/usr/local/bin/python
#n6hdf5reader.py

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


class Nbody6Snapshot:
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
    
    def __init__(self, h5file, scale = 1.0):
        """Read the HDF5 file into numpy arrays.
        
        Arguments:
        h5file -- an HDF5 snapshot created by n6tohdf5
        scale -- optionally rescale a simulation by this distance factor. default: 1.0
        """
        try:
            f = h5py.File(h5file, 'r')
        except IOError:
            print "can't open the file", h5file, '!'
            sys.exit(0)
        
            
        # get some information from the header. 
        header = f['Header']
        
        # number of stars
        self.Nstars = int(header.attrs['Nstars'])
           
        # time in Nbody, Myr, Tcross
        self.Time = header.attrs['Time']
        self.TimeMyr = header.attrs['TimeMyr'] * scale**1.5
        self.Tstar = header.attrs['Tscale'] * scale**1.5
            
        # radius scaling
        self.Rbar = np.array(header.attrs['Rbar']) * scale
    
        # velocity scaling
        self.Vstar = np.array(header.attrs['Vstar']) / scale**0.5
    
        # mass scaling
        self.Mbar = np.array(header.attrs['Mbar'])
        
        
        self.Pos = np.array(f['Stars/Positions'])
        self.Vel = np.array(f['Stars/Velocities'])
        self.Masses = np.array(f['Stars/Masses'])
        self.Names = np.array(f['Stars/Names'], dtype=np.int64)
        try:
            self.Lum = np.array(f['Stars/Luminosity'])
        except KeyError: 
            self.Lum = None
        try:
            self.Teff = np.array(f['Stars/Teff'])
        except KeyError: 
            self.Teff = None       
                
        
        # initialise density center with nbody6 version
        self.dc_pos = np.array([header.attrs['DensityCenterX'],
            header.attrs['DensityCenterY'],header.attrs['DensityCenterZ']])
        self.dc_vel = np.zeros(3)
        # space for recalculation of the density center
        self.dc_pos_new = np.zeros(3)
        self.dc_vel_new = np.zeros(3)
        
        # radius of the stars from the origin
        self.Radii = np.array(np.sqrt(inner1d(self.Pos, self.Pos)))        
        # radius of the stars from the density center
        poscm = self.Pos - self.dc_pos
        self.Radii_dcm = np.array(np.sqrt(inner1d(poscm, poscm)))  
        
        # space for values using recalculated density center
        self.Radii_dcm_new = np.zeros(len(self.Pos))     
        
        # lagrangian radii
        self.Lagr_rads = np.zeros(5)
        self.Lagr_fractions = None
        
        # velocity dispersion in nbody units
        self.sigmas = np.array([np.std(self.Vel[:, 0]), np.std(self.Vel[:, 1]), 
                        np.std(self.Vel[:, 2])])
        
        # close the HDF5 file
        f.close()
        
        
        
    def calc_lagrangian_radii(self, fractions, axes = [0, 1, 2], newcm = True):
        """Calculate the Lagrangian radii of a set of stars.
        
        Arguments:
        fractions -- an array of mass fractions to get radii for. e.g. [0.1, 0.5, 0.9]
        axes -- use the full 3d information (default mode) or pass in two axes to  
                use the projection along the missing axis.
        newcm -- use the density center as calculated in this module (default) or use the
                 density center calculated from the code.
        """
        if newcm:
            if self.dc_pos_new.sum() == 0 or len(axes) < 3: 
                self.calc_new_density_center(axes)
            poscm = self.Pos[:, axes] - self.dc_pos_new[axes]  
        else:
            # this isn't fully consistent, so let the user know
            print 'WARNING: using projected positions in the lagrangian radius', \
                  'calculation and not calculating a new centroid is not self-consistent.'
            poscm = self.Pos[:, axes] - self.dc_pos[axes]
            
        if(len(axes) == 3):
            if newcm:
                rads = self.Radii_dcm_new.copy()
            else:
                rads = self.Radii_dcm.copy()
        else:
            # poscm has the right dimmensionality
            rads = np.array(np.sqrt(inner1d(poscm, poscm))) 

        # sort by radius
        sortindices = rads.argsort()
        masses = self.Masses[sortindices].T[0]
       # cummasses = [reduce(lambda c, d: c+d, masses[:i], 0) for i in range(1, len(masses)+1)]
        cummasses = np.add.accumulate(masses)
        rads.sort()
        radmassfunc = interp1d(np.hstack((0.0, cummasses)), 
            np.hstack((0.0, rads)), kind='linear')
        lagrsels = np.array(fractions) * masses.sum()
        self.Lagr_rads = radmassfunc(lagrsels)
        self.Lagr_fractions = fractions
    
    
    def calc_new_density_center(self, axes = [0, 1, 2]): 
        """Calculate the density center of a set of stars
        
        Uses the method of Casertano & Hut ApJ 1985, 298, 80. 
        
        Arguments:
        axes -- use the full 3d information (default mode) or pass in two axes to 
                use the projection along the missing axis.
        """
        positions = self.Pos[:, axes]
        #  get a nearest neighbor tree  
        kdtree = spatial.cKDTree(positions)    
        # the first result is the point itself, so sixth neighbor is the seventh result
        near6 = kdtree.query(positions, 7)[0][:,6]
        vols = np.pi * near6**2
        densities = 5.0 / vols
    
        # density center is density weighted radius of the stars
        self.dc_pos_new = (densities[:,np.newaxis] * positions).sum(0) / densities.sum()    
        self.dc_vel_new = (densities[:,np.newaxis] * 
            self.Vel[:, axes]).sum(0) / densities.sum()
            
        # update radii from density center
        poscm = self.Pos - self.dc_pos_new
        self.Radii_dcm_new = np.array(np.sqrt(inner1d(poscm, poscm)))  
        
        
        
    def find_binaries(self):
        """find binaries that are present at this time
        """ 
        # make copies of position, velocity, mass
        pos = self.Pos.copy()
        vel = self.Vel.copy()
        masses = self.Masses.copy()
        # get a neighbor tree set up
        kdtree = spatial.cKDTree(pos)
        near10 = kdtree.query(pos, 6)       

        allpairs = []
        allpairsegy = []
        for i in range(self.Nstars):
            dists = near10[0][i,1:]
            ngbs = near10[1][i,1:]
            vrel = vel[ngbs] - vel[i]
            mus = (masses[i] * masses[ngbs] / 
                (masses[i] + masses[ngbs])).T[0]
            Ms = (masses[i] + masses[ngbs]).T[0]
            # get energies divided by reduced mass
            egys = 0.5 * inner1d(vrel, vrel) - Ms / dists 
            # select negative energies
            negs = (egys < 0)
            pairs = []
            pairsegy = []
            if (negs).sum() > 0:
                # accumulate all pairs of negative energies
                for j in range(negs.sum()):
                    pair = [i, ngbs[negs][j]]
                    pair.sort()
                    pairs.append(pair)
                    pairsegy.append(egys[negs][j])
            allpairs.append(pairs)
            allpairsegy.append(pairsegy)
            
        # now go through and accept mutually most-bound pairs
        mostboundoptions = []
        for i in range(len(allpairs)):
            # find the most bound option
            if len(allpairs[i]) > 0:
                j = allpairsegy[i].index(min(allpairsegy[i]))
                mostboundoptions.append(allpairs[i][j])

        # each mutually bound pair is now in mostboundoptions twice. grab those.
        mostboundconfirmed = []
        for i in range(len(mostboundoptions)):
            if mostboundoptions.count(mostboundoptions[i]) > 1:
                mostboundconfirmed.append(mostboundoptions[i])
                

        # now remove duplicates
        mostboundconfirmed = dict((x[0], x) for x in mostboundconfirmed).values()
        mostboundconfirmed.sort()

        
        # get properties of the binaries 
        semis = np.zeros(len(mostboundconfirmed))
        eccs = np.zeros(len(mostboundconfirmed))
        mus = np.zeros(len(mostboundconfirmed))
        egys = np.zeros(len(mostboundconfirmed))
        for b in range(len(mostboundconfirmed)):
            i = mostboundconfirmed[b][0]
            j = mostboundconfirmed[b][1]
            prel = pos[i] - pos[j]
            sep = np.sqrt(np.dot(prel, prel))
            vrel = vel[i] - vel[j]
            M = masses[i] + masses[j]
            mus[b] = masses[i] * masses[j] / M
            sep
            egys[b] = 0.5 * np.dot(vrel, vrel) - M / sep
            semis[b] = -0.5 * M / egys[b]
            rdotv = np.dot(prel, vrel)
            eccs[b] = np.sqrt((1.0 - sep / semis[b])**2 + rdotv**2 / (semis[b] * M))
        return(mostboundconfirmed, semis, eccs, mus, egys)
            
        
        

    def mass_selection(self, mlow = -1.0, mhigh = -1.0):
        """get a subset of stars based on a mass range
        
        Arguments:
        mlow -- lower mass range in code units. default: minimum mass
        mhigh -- upper mass range in code units default: maximum mass
        
        Returns:
        Nbody6Subset containing the stars in the requested mass range
        """
        if mlow == -1.0:
            mlow = self.Masses.min()
        if mhigh == -1.0:
            mhigh = self.Masses.max()
        selection = (self.Masses > mlow) & (self.Masses < mhigh)
        if selection.sum() == 0:
            print 'No stars are in the mass range!'
            return None
        else:
            return Nbody6Subset(self, self.Names[selection])            


    def sphere_selection(self, origin = None, radius = 1.0):
        """get a spherical subset of stars
        
        Arguments:
        origin -- origin of the spherical region. default: [0, 0, 0]
        radius -- radius in code units of the spherical region
        
        Returns:
        Nbody6Subset containing the stars in the requested mass range
        """
    
        if origin is not None and np.allclose(origin, self.dc_pos_new):
            rads = self.Radii_dcm_new
        if origin is not None and np.allclose(origin, self.dc_pos):
            rads = self.Radii_dcm
        if origin is None:
            rads = self.Radii
        else:
            posorigin = self.Pos - origin
            rads = np.array(np.sqrt(inner1d(posorigin, posorigin)))  
        selection = (rads < radius)
        if selection.sum() == 0:
            print 'No stars are in the sphere!'
            return None
        else:
            return Nbody6Subset(self, self.Names[selection])            
        
    def boundedness_selection(self, lagrangecut = 0.9):
        """get a subset of stars that are probably bound.
        
        Arguments:
        lagrangecut -- lagrangian radius outside which stars may be escapers. default: 0.9
        
        Returns:
        Nbody6Subset containing the stars that are probably bound.    
        """
        """# everything here is using the new density center as the origin.
        # if lagrangian radii aren't calculated, get the desired one.
        if self.Lagr_fractions is None:
            self.calc_lagrangian_radii([lagrangecut])
            
        # if the desired cut isn't in the already calculated ones, get it
        if lagrangecut not in self.Lagr_fractions:
            newfracs = np.hstack((self.Lagr_fractions, lagrangecut))
            newfracs.sort()
            self.calc_lagrangian_radii(newfracs)
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
    
        
            
        
        
    def make_pretty_image_data(self, plotsizew = 5, plotsizeh = 5, dpival = 80, 
        rangew = 5, rangeh = 5, origin = [0, 0, 0], fov = np.pi / 4.0,
        gsmin = 1.05, gamin = 0.9, gspower = 0.15, gapower = 0.15,
        esmin = 1.2, eamin= 3, espower = 0.5, eapower = 0.3,
        minlum = -1.0, transitionlum = -1.0, tempmin = 1000.0, tempmax = 10000.0,
        phi = 0.0, theta = 0.0, psi = 0.0, project3d = True, paramtest = False,
        backgroundstars = False, nbackground = 2000, sbackground = 1,
        distantcut = False, backgroundopacity = False, luminositycut = -1,
        plottrails = False, trails=[], trailcolors=[]
        ):
        """render an image of the stars on a black background with 'realistic' colors
        
        By default the x-y plane is plotted. Other projections can be realized using
        the Euler angles. 
        order is phi around z axis, theta around new y axis, psi around new z axis.
        For example: 
        x-z plane: phi = pi/2, theta = pi/2, psi = -pi/2
        z-y plane: phi = 0, theta = Pi/2, psi = 0
        the third dimension is by default included via projection of 3d points onto a 2d 
        screen. With project3d = False the coordinates are simply plotted instead.
        
        Arguments:
        plotsizew -- width of final plot in inches. default: 5
        pltsizeh -- height of final plot in inches. default: 5
        dpival -- dpi of final image. default: 80
        rangew -- width range in code units. default: 5
        rangeh -- height range in code units. default: 5
        origin -- origin of the plot in code units. default: [0, 0, 0]
        fov -- field of view in the height direction in radians. default: pi/4
        gsmin -- minimum sigma of gaussian stars. default: 1.05
        gamin -- minimum amplitude of gaussian stars. default: 0.9
        gspower -- power law of luminosity scaling for gaussian sigma. default: 0.15
        gapower -- power law of luminosity scaling for gaussian amplitude. default: 0.15
        esmin -- minimum sigma of exponential stars. default: 1.2
        eamin -- minimum amplitude of exponential stars. default: 3
        espower -- power law of luminosity scaling for exponential sigma. default: 0.5
        eapower -- power law of luminosity scaling for exponential amplitude. default: 0.3     
        minlum -- minimum luminosity to plot
        transitionlum -- luminosity at which stars change from gaussian to exponential
        tempmin -- temperature floor for the color scaling default: 1000
        tempmax -- temperature ceiling for the color scaling. default: 10000
        phi -- Euler rotation angle. See rotation_matrix for convention.
        theta -- Euler rotation angle. See rotation_matrix for convention.     
        psi -- Euler rotation angle. See rotation_matrix for convention. 
        project3d -- use 3d projection. default: True
        paramtest -- just plot stars in a line to test the aesthetics. default: False
        backgroundstars -- add background stars to the plot. default: False
        nbackground -- number of background stars. default: 2000
        sbackground -- random number seed of background stars. default: 1
        distantcut -- drop the luminosity of stars way in the background. default: false
        luminositycut -- maximum allowed luminosity. default: -1 (off)
        plottrails -- plot the path of a particular star. default: False
        trails -- input the paths of the stars you want plotted
        trailcolors -- colors for the paths you want plotted
        Returns:
        a numpy array containing RGB values of the rendered image     
        """
        # get teff and lum in linear units
        teff = 10**self.Teff
        lum = 10**self.Lum
        pos = self.Pos[:]
        
        # set up background star population
        if backgroundstars:
            np.random.seed(sbackground)
            randompos = np.random.uniform(-20000, 20000, (nbackground,3))
        
        # if we're testing plotting parameters, put some subset of stars on a line and
        # plot them to see what it looks like.
        if paramtest:
            masses = self.Masses*self.Mbar
            sortindices = masses.T[0].argsort()
            teff = teff[sortindices]
            lum = lum[sortindices]
            pos = pos[sortindices]
            masses = masses[sortindices]
            selection = np.array(np.abs(np.unique(np.floor(
                np.logspace(.1,np.log10(self.Nstars), 50))) - self.Nstars), 
                dtype = 'int64')
            teff = teff[selection]
            lum = lum[selection]
            masses = masses[selection]
            pos = np.array([range(len(lum)), range(len(lum)), np.zeros(len(lum))]).T
            pos -= pos[len(lum)//2]
            pos *= .4*rangeh / pos[:,0].max()
            masses.sort()
        teff = teff.reshape(len(teff))
        lum = lum.reshape(len(lum))

        # scale the inputs to the size of the image.
        scalefac = plotsizeh * dpival / 700.0
        
        # gaussian star parameters
        gsmin *= scalefac  # minimum gaussian sigma for lowest luminosity star
        esmin *= scalefac  # minimum exponential sigma 
        
        # automatically set some things if they aren't input
        if minlum < 0:
            minlum = lum.min()
        if transitionlum < 0:
            transitionlum = max((0.01, 2.0 * lum.mean()))

        
        # set up the rotation matrix
        if phi != 0 or theta != 0 or psi != 0:
            rotmatrix = rotation_matrix(phi, theta, psi)
            rotate = True
        else:
            rotate = False
            
        # find the screen distance
        screendistance = rangeh / (2.0 * np.tan(fov / 2.0))
        print 'screen distance = ',screendistance, screendistance*self.Rbar
        
        # get the colors
        colors = plt.get_cmap('RdYlBu')
        
        # restrict temps to some aesthetic range   
        teff[teff > tempmax] = tempmax
        teff[teff < tempmin] = tempmin        
            
        if paramtest:
            print 'masses, luminosity, temperatures plotted: '
            for i in range(len(pos)):
                print masses[len(pos) - i - 1], lum[i], teff[i]
            print 'tempmin ', tempmin
            print 'tempmax ', tempmax
            print 'minlum ', minlum
            print 'transitionlum ', transitionlum
            print 'gsmin ', gsmin / scalefac
            print 'gamin ', gamin
            print 'gspower ', gspower
            print 'gapower ', gapower
            print 'esmin ', esmin / scalefac
            print 'eamin ', eamin
            print 'espower ', espower
            print 'eapower ', eapower               
        # set up arrays to hold RGB image data        
        resx = plotsizew * dpival
        resy = plotsizeh * dpival      
        
        if backgroundopacity:
            ndims = 4
        else:
            ndims = 3
        imdat = np.zeros((resx, resy, ndims))
        imdatf = np.zeros((resy, resx, ndims)) # to hold rotated image  
            
        xpixorigin = resx // 2
        ypixorigin = resy // 2               
        
        # start adding stars        
        for i in range(len(pos)):
            if lum[i] > minlum:
                if luminositycut > -1:
                    if lum[i] > luminositycut:
                        lum[i] = luminositycut
                newpoint = pos[i] - origin # tranlate the point to our origin
        
                if rotate:
                    newpoint = rotmatrix.dot(newpoint)
                
                # don't plot stars behind the viewer
                if(newpoint[2] >= screendistance):
                    continue
                # this is a bit ad hoc- fade out things that are way in the back in case
                # they were put there 'out of the way' by nbody6
                if(distantcut and (newpoint[2] < -20*screendistance)):
                    lum[i] *= -1/newpoint[2]  
                if project3d:
                    newpoint2d = project_to_screen(newpoint, screendistance)
                else:
                    newpoint2d = newpoint[[0,1]]
                
                # get location in the image array            
                # use resy and rangeh only here
                xpix = int(round(resy * newpoint2d[0] / rangeh) + xpixorigin)
                ypix = int(round(resy * newpoint2d[1] / rangeh) + ypixorigin)
                
                # map the temp range from 0-1 and get the star's color
                # take 0:3 from the color- 4th value is the opacity 
                basecolor = colors(min((teff[i] - tempmin) / tempmax, 0.8))#[0,0:3]
                       
                # crossover between gaussian and exponential
                if lum[i] < transitionlum:
                    gaussfrac = 1.0
                    expfrac = 0.0
                if lum[i] > 1.5 * transitionlum:
                    gaussfrac = 0.1
                    expfrac = 1.0
                if transitionlum <= lum[i] and lum[i] <= 1.5 * transitionlum:
                    gaussfrac = 1.0 - (lum[i] - transitionlum) / (0.5 * transitionlum)
                    expfrac = 1.0 - gaussfrac**2
                
                if gaussfrac > 0:
                    # gaussian contribution
                    # sigma of the gaussian for this star. 
                    # the minimum mass star has sigma = gsmin
                    sig = gsmin * (lum[i] / minlum)**gspower
                    amp = gamin * (lum[i] / minlum)**gapower
                    gauss = amp * gaussian_point(sig)
                    patchsize = int(4 * sig)
                    xmin = xpix - patchsize
                    xmax = xpix + patchsize + 1
                    ymin = ypix - patchsize
                    ymax = ypix + patchsize + 1
                    # check if we're in the plot
                    if xmin >= 0 and xmax <= resx and ymin >= 0 and ymax <= resy:
                        for j in range(3):        
                            imdat[xmin:xmax, ymin:ymax, j] += basecolor[j] * gauss
                                            
                if expfrac > 0:
                    # exponential contribution
                    sig = esmin * (lum[i] / transitionlum)**espower
                    amp = eamin * (lum[i] / transitionlum)**eapower
                    exponential = amp * exponential_point(sig)
                    patchsize = int(4 * sig)
                    xmin = xpix - patchsize
                    xmax = xpix + patchsize + 1
                    ymin = ypix - patchsize
                    ymax = ypix + patchsize + 1
                    # check if we're in the plot
                    if xmin >= 0 and xmax <= resx and ymin >= 0 and ymax <= resy:
                        for j in range(3):        
                            imdat[xmin:xmax, ymin:ymax, j] += basecolor[j] * exponential                
        
        # add the background stars
        if backgroundstars:
            for i in range(nbackground):
                newpoint = randompos[i] - origin # tranlate the point to our origin
        
                if rotate:
                    newpoint = rotmatrix.dot(newpoint)
            
                if project3d:
                    newpoint2d = project_to_screen(newpoint, screendistance)
                else:
                    newpoint2d = newpoint[[0,2]]

                # get location in the image array            
                # use resy and rangeh only here
                xpix = int(round(resy * newpoint2d[0] / rangeh) + xpixorigin)
                ypix = int(round(resy * newpoint2d[1] / rangeh) + ypixorigin)
                
                # use fixed neutral color for these stars
                basecolor = np.floor(np.random.uniform(low=80,high=230))*np.array([1,1,1])
            
                randomfactor = np.random.uniform(low = 0.5, high = 1)
                sig = .4*gsmin*randomfactor
                amp = .4*gamin*randomfactor
                gauss = amp * gaussian_point(sig)
                patchsize = int(4 * sig)
                xmin = xpix - patchsize
                xmax = xpix + patchsize + 1
                ymin = ypix - patchsize
                ymax = ypix + patchsize + 1
                # check if we're in the plot
                if xmin >= 0 and xmax <= resx and ymin >= 0 and ymax <= resy:
                    for j in range(3):        
                        imdat[xmin:xmax, ymin:ymax, j] += basecolor[j] * gauss 
                    if backgroundopacity:
                        imdat[xmin:xmax, ymin:ymax, 3] += 100*gauss        

        # add the trails stars
        if plottrails:
            for k in range(len(trails)):
                trail = trails[k]
                for i in range(len(trail)):
                    newpoint = trail[i] - origin # tranlate the point to our origin
        
                    if rotate:
                        newpoint = rotmatrix.dot(newpoint)
            
                    if project3d:
                        newpoint2d = project_to_screen(newpoint, screendistance)
                    else:
                        newpoint2d = newpoint[[0,2]]

                    # get location in the image array            
                    # use resy and rangeh only here
                    xpix = int(round(resy * newpoint2d[0] / rangeh) + xpixorigin)
                    ypix = int(round(resy * newpoint2d[1] / rangeh) + ypixorigin)
                
                    # use input color
                    basecolor = trailcolors[k]
            
                    sig = 1.*gsmin
                    amp = 1.*gamin
                    gauss = amp * gaussian_point(sig)
                    patchsize = int(4 * sig)
                    xmin = xpix - patchsize
                    xmax = xpix + patchsize + 1
                    ymin = ypix - patchsize
                    ymax = ypix + patchsize + 1
                    # check if we're in the plot
                    if xmin >= 0 and xmax <= resx and ymin >= 0 and ymax <= resy:
                        for j in range(3):        
                            imdat[xmin:xmax, ymin:ymax, j] += basecolor[j] * gauss/10
                        if backgroundopacity:
                            imdat[xmin:xmax, ymin:ymax, 3] += 100*gauss 

        # the image array is rotated strangely. unrotate it.
        if not backgroundopacity:
            imdat[imdat > 1.0] = 1.0   
        for k in range(ndims):
            imdatf[:,:,k] = imdat[:,:,k].T            
        del(imdat)
               
        return imdatf    
    
    
def rotation_matrix(phi, theta, psi):
    """given three Euler angles return a rotation matrix.
    
    Arguments:
    phi -- Euler angle for first rotation about the z axis
    theta -- Euler angle for rotation about the rotated y axis
    psi -- Euler angle for rotation about the rotated z axis

    Returns:
    a 3x3 numpy array containing the rotation matrix
    """
    """a11 = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi)
    a12 = np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi)
    a13 = np.sin(psi) * np.sin(theta)
    a21 = -np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi)
    a22 = -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi)        
    a23 = np.cos(psi) * np.sin(theta)
    a31 = np.sin(theta) * np.sin(phi)
    a32 = -np.sin(theta) * np.cos(phi)
    a33 = np.cos(theta)    
    """
    
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)
    cps = np.cos(psi)
    sps = np.sin(psi)
    
    a11 = cp * ct * cps - sp * sps
    a12 = sp * ct * cps + cp * sps
    a13 = -st * cps
    a21 = -cp * ct * sps - sp * cps
    a22 = -sp * ct * sps + cp * cps      
    a23 = st * sps
    a31 = cp * st
    a32 = sp * st
    a33 = ct
   # print np.array([a11, a12, a13, a21, a22, a23, a31, a32, a33]).reshape((3, 3)).T.dot([1,0,0]) 
   # print np.array([a11, a12, a13, a21, a22, a23, a31, a32, a33]).reshape((3, 3)).T.dot([0,1,0])
   # print np.array([a11, a12, a13, a21, a22, a23, a31, a32, a33]).reshape((3, 3)).T.dot([0,0,1])
    return np.array([a11, a12, a13, a21, a22, a23, a31, a32, a33]).reshape((3, 3)).T    
    

def project_to_screen(point, screendistance):
    """project a 3d point to a 2d screen
    
    Arguments: 
    point -- a 3d numpy array containing the coordinates of a point
    screendistance -- distance to the projection plane
    
    Returns:
    a numpy array containing the 2d projection of the 3d point
    """
    # that small value is in the denominator in case the point is at the screen distance
    return point[[0,1]] * screendistance / (screendistance - point[2] + 1.e-10)
    
    
def gaussian_point(sigma):
    """make a gaussian splotch.
    
    Arguments:
    sigma -- dispersion of the gaussian
    
    Returns:
    a numpy array covering a square 4 sigma on a side with guassian values.
    """
    size = int(4*sigma)
    sigma2 = sigma**2
    if sigma2 == 0:
        sigma2 = 1.e-3
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp( -(x**2 + y**2)/(2 * sigma2) )
    return g # / g.sum()      
    
def exponential_point(sigma):
    """make an exponential splotch.
    
    Arguments:
    sigma -- scale length of the exponential
    
    Returns:
    a numpy array covering a square 4 sigma on a side with guassian values.
    """
    size = int(4*sigma)
    if sigma == 0:
        sigma = 1.e-3
    x, y = np.mgrid[-size:size+1, -size:size+1]
    e = np.exp( -np.sqrt(x**2 + y**2)/sigma )
    return e # / g.sum()



class Nbody6Subset(Nbody6Snapshot):
    """Get a subset of a snapshot, found by the Names of the stars.
    
    This is a subclass of Nbody6Snaphot. It can be used to run analyses (Lagrangian
    Radii, density center, velocity dispersions etc.) on a subset of the stars in 
    a simulation. Stars are added to the subset based on a list of names.
    """
    def __init__(self, n6snap, nameselection):
        """Get a subset of a Nbody6Snapshot
        
        Arguments:
        n6snap -- a valid Nbody6Snapshot instance.
        nameselection -- an array of names to select.
        """
        #selection = [i for i, item in enumerate(n6snap.Names) if item in nameselection]
        selection = np.in1d(n6snap.Names.reshape(len(n6snap.Names)), nameselection)

        self.Pos = n6snap.Pos[selection]
        self.Vel = n6snap.Vel[selection]
        self.Masses = n6snap.Masses[selection]
        self.Names = n6snap.Names[selection]
        if n6snap.Lum is None:
            self.Lum = None
        else:
            self.Lum = n6snap.Lum[selection]
        if n6snap.Teff is None:
            self.Teff = None
        else: 
            self.Teff = n6snap.Teff[selection]        
            
        self.Nstars = len(self.Pos)    
            
        # time in Nbody, Myr, Tcross. these are scaled the same as the full snapshot!
        self.Time = n6snap.Time
        self.TimeMyr = n6snap.TimeMyr
        self.Tstar = n6snap.Tstar
        
        # radius scaling
        self.Rbar = n6snap.Rbar
    
        # velocity scaling
        self.Vstar = n6snap.Vstar
    
        # mass scaling
        self.Mbar = n6snap.Mbar      
        
        # initialise density center with nbody6 version
        self.dc_pos = n6snap.dc_pos
        self.dc_vel = n6snap.dc_vel
        # space for recalculation of the density center
        self.dc_pos_new = np.zeros(3)
        self.dc_vel_new = np.zeros(3)

        # radius of the stars from the origin
        self.Radii = np.array(np.sqrt(inner1d(self.Pos, self.Pos)))   
        
        # radius of the stars from the density center
        poscm = self.Pos - self.dc_pos
        self.Radii_dcm = np.array(np.sqrt(inner1d(poscm, poscm)))  
        # space for values using recalculated density center
        self.Radii_dcm_new = np.zeros(len(self.Pos))
        
        # lagrangian radii
        self.Lagr_rads = np.zeros(5)
        
        # velocity dispersion in nbody units
        self.sigmas = np.array([np.std(self.Vel[:, 0]), np.std(self.Vel[:, 1]), 
                        np.std(self.Vel[:, 2])])        



            
            
class Nbody6Header:
    """Scan the header of an HDF5 file created by n6tohdf5
    
    This is useful mainly to scan through a list of snapshots, e.g. to find the snapshot
    closest to some physical time when radii are scaled by 0.5.
    """
    def __init__(self, h5file, scale = 1.0):
        """Read the HDF5 header.
        
        Arguments:
        h5file -- an HDF5 snapshot created by n6tohdf5
        scale -- optionally rescale a simulation by this distance factor. default: 1.0
        """
        try:
            f = h5py.File(h5file, 'r')
        except IOError:
            print "can't open the file", h5file, '!'
            sys.exit(0)
            
        header = f['Header']

        # get some information from the header. 
        # number of stars
        self.Nstars = int(header.attrs['Nstars'])
            
        # time in Nbody, Myr, Tcross
        self.Time = header.attrs['Time']
        self.TimeMyr = header.attrs['TimeMyr'] * scale**1.5
        self.Tstar = header.attrs['Tscale'] * scale **1.5
            
        # radius scaling
        self.Rbar = np.array(header.attrs['Rbar']) * scale
    
        # velocity scaling
        self.Vstar = np.array(header.attrs['Vstar']) / scale**0.5
    
        # mass scaling
        self.Mbar = np.array(header.attrs['Mbar'])            
        
        # close the HDF5 file
        f.close()