from __future__ import division

"""This module provides an interface for the HDF5 files created by n6tohdf5.

Classes:
Nbody6Snapshot -- container for the data in the HDF5 file

Nbody6Subset -- a subclass of Nboy6Snapshot, used to analyse a subset of stars.

Nbody6Header -- a class to grab just the header information.
"""

"""
TODO
add wrapper that takes psf, distance, band, converts to observation w/ light profile
"""


import numpy as np
import matplotlib.pyplot as plt
from ..units import *


class Camera(object):
    """Create a 'realistic' rendering of the star cluster
    """
    def __init__(self, nbstate, point_to = [0, 0, 0], camera_vector = [0, 0, 1],
                     up_vector = [0, 1, 0], fov = np.pi / 4, mapping = 'OpticalNatural'):
        self._point_to = point_to
        self._camera_vector = camera_vector
        self._up_vector = up_vector
        self.fov = fov
        self._mapping = mapping
        self.x = nbstate.x
        self.y = nbstate.y
        self.z = nbstate.z
        self.T = nbstate.temperature
        self.L = nbstate.luminosity
        self.sigma_gaussian = 0.001
        self.sigma_exponential =  0.0004     
        self._min_amplitude = 0.25
        self._max_amplitude = 1.0
        self._min_log_luminosity_solar = -6
        self._max_log_luminosity_solar = 6
        self._min_temp = 1000
        self._max_temp = 100000
        
        self.tempkeys = None
        self.rkeys = None
        self.gkeys = None
        self.bkeys = None
        
        # if we're plotting a dimensionless simulation, need to treat things
        # differently. e.g. make fake temperatures and luminosities. that is TODO
        self.nbody_units = False
        
        self._transx = None
        self._transy = None
        self._transz = None
        # track if the stars have been transformed into camera's reference frame
        self._transformed = False
        
        self._colors = None
        # track if stars have colors assigned 
        self._mapped = False
        
        self._amplitudes = None
        self._gaussian_fraction = None
        # track if stars have amplitudes assigned
        self._amped = False
        
    # these three are properties so that we can keep track of self._transformed 
    # whenever they change. everything else we can change without worrying about altering 
    # the transformed coordinates of the stars.
    @property
    def point_to(self):
        return self._point_to
    @point_to.setter
    def point_to(self, point_to):
        self._transformed = False
        self._point_to = point_to
        
    @property
    def camera_vector(self):
        return self._camera_vector
    @camera_vector.setter
    def camera_vector(self, camera_vector):
        self._transformed = False
        self._camera_vector = camera_vector

    @property
    def up_vector(self):
        return self._up_vector
    @up_vector.setter
    def up_vector(self, up_vector):
        self._transformed = False
        self._up_vector = up_vector
    
    # likewise, we want to track if the colormap changes
    @property
    def mapping(self):
        return self._mapping
    @mapping.setter
    def mapping(self, mapping):
        self._mapped = False
        self._mapping = mapping    

    # likewise, we want to track if the amplitude assignments change
    @property
    def min_amplitude(self):
        return self._min_amplitude
    @min_amplitude.setter
    def min_amplitude(self, min_amplitude):
        self._amped = False
        self._min_amplitude = min_amplitude  
     
    @property
    def max_amplitude(self):
        return self._max_amplitude
    @max_amplitude.setter
    def max_amplitude(self, max_amplitude):
        self._amped = False
        self._max_amplitude = max_amplitude   

    @property
    def min_log_luminosity_solar(self):
        return self._min_log_luminosity_solar
    @min_log_luminosity_solar.setter
    def min_log_luminosity_solar(self, min_log_luminosity_solar):
        self._amped = False
        self._min_log_luminosity_solar = min_log_luminosity_solar 
        
    @property
    def max_log_luminosity_solar(self):
        return self._max_log_luminosity_solar
    @max_log_luminosity_solar.setter
    def max_log_luminosity_solar(self, max_log_luminosity_solar):
        self._amped = False
        self._max_log_luminosity_solar = max_log_luminosity_solar                   

    @property
    def min_temp(self):
        return self._min_temp
    @min_temp.setter
    def min_temp(self, min_temp):
        self._amped = False
        self._min_temp = min_temp 
        
    @property
    def max_temp(self):
        return self._max_temp
    @max_temp.setter
    def max_temp(self, max_temp):
        self._amped = False
        self._max_temp = max_temp    
    
    
    def mapping_test(self):
        """output a fake HR diagram to test the color mapping   
        """         
        # set up a black canvas
        resx = 800
        resy = 800
        imarray = np.ones((resx, resy, 4)) * TINY # background black image
        imarrayrot = np.ones((resy, resx, 4)) * TINY # rotated image
        
        print self._mapped
        
        # if we haven't assigned colors etc., do that now.
        if not self._mapped:
            self._map_colors()
            
        # populate log Teff, and log Lum
        temps = np.logspace(np.log10(2000), np.log10(50000), 12)
        logtemps = np.log10(temps)
        lums = np.linspace(-4, 6, 12) 
        
        # assign colors to these
        rvals = np.interp(temps, self.tempkeys, self.rkeys)
        gvals = np.interp(temps, self.tempkeys, self.gkeys)
        bvals = np.interp(temps, self.tempkeys, self.bkeys)
        colors = tuple(map(tuple, np.vstack((rvals, gvals, bvals)).T/255))
        
        # limits of the hr diagram
        xmin = np.log10(1000)
        xmax = np.log10(100000)
        rangex = xmax - xmin
        ymin = -5
        ymax = 7
        rangey = ymax - ymin
        
        # assign amplitudes
        # luminosity part of amplitude 
        amplum = (lums - self.min_log_luminosity_solar) / (self.max_log_luminosity_solar - self.min_log_luminosity_solar)  
        # temperature part of amplitude
        amptemp = (logtemps - np.log10(self.min_temp)) / (np.log10(self.max_temp) - np.log10(self.min_temp))       
        amplitudes = np.outer(amplum , amptemp)        
        # scale to min and max amplitude
        amplitudes = amplitudes / np.max(amplitudes) * (self.max_amplitude - self.min_amplitude) + self.min_amplitude
        
        amplum = 10**lums
        amptemp = np.zeros(len(temps))
        for i in xrange(len(temps)):
            # this part finds the fraction of the integrated planck function lying
            # between 4000 and 7000 angstroms. total planck function in dimensionless
            # units integrates to pi^4 / 15 ~ 6.4939394 
            ktoverhc = kb_cgs * temps[i] / (h_cgs * c_cgs)
            u1 = 4000 * angstrom_cgs * ktoverhc
            u2 = 7000 * angstrom_cgs * ktoverhc
            u = np.logspace(np.log10(u1), np.log10(u2), 32)
            planckfunc = 1 / (u**5 * (np.exp(1/u) - 1))   
            amptemp[i] = np.trapz(planckfunc, u) / 6.4939394
        print 'amplum', amplum
        print 'amptemp', amptemp
        amplitudes = np.outer(amplum , amptemp)      
        # scale to min and max amplitude
        #amplitudes = np.log10(amplitudes)
         
        #amplitudes = amplitudes / np.max(amplitudes) * (self.max_amplitude - self.min_amplitude) + self.min_amplitude    
        
        
        gaussfracs = 1 - ((lums - self.min_log_luminosity_solar) / 
            (self.max_log_luminosity_solar - self.min_log_luminosity_solar))
        print gaussfracs
        for i in xrange(len(temps)):
            temp = logtemps[i]
            color = colors[i]
            for j in xrange(len(lums)):
                lum = lums[j]
                amp = amplitudes[j,i]
                gfrac = gaussfracs[j]
                # get location in the image array            
                imx = (temp - xmin) / rangex
                imy = (lum - ymin) / rangey
             #   print  temp, lum, amplum[j], amptemp[i], amp
                imarray = self._deposit_star(imarray, imx, imy, color, amp, gfrac)
        # imarray now has total rgb values and total intensity in linear units.
        # get a rotated version, with colors scaled to total intensity
        for j in xrange(3):
            imtemp = imarray[:,:,j]      
            imtemp[imtemp < 2*TINY] = 0
            imarrayrot[:,:,j] = (imtemp/imarray[:,:,3]).T  # still linear
        imarrayrot[:,:,3] = imarray[:,:,3].T
        del(imarray)
        # scale total intensity to the logarithmic range we're interested in
        imarrayrot[:,:,3] = np.log10(imarrayrot[:,:,3])
        imarrayrot[:,:,3] = ((imarrayrot[:,:,3] - self._min_log_luminosity_solar) / 
            (self._max_log_luminosity_solar - self._min_log_luminosity_solar))
        # apply this scaled intensity to the colors
        for j in xrange(3):
            imarrayrot[:,:,j] *= imarrayrot[:,:,3]
        # clip to 0-1
        imarrayrot[imarrayrot > 1] = 1
        imarrayrot[imarrayrot < 0] = 0
        imarrayrot = imarrayrot[:,:,0:3]#imarrayrot[:,:,3] = 1.0
        
             
        fig = plt.figure(figsize = (5, 5), dpi=200)
        ax = fig.add_axes([.15, .15, .8, .8])
        ax.imshow(imarrayrot,interpolation = 'nearest',origin = 'lower')
        
        plt.savefig('maptest.png', dpi = 200)
    
    
    def _deposit_star(self, imarray, x, y, color, amplitude, gaussfrac):
        gaussfrac = max(gaussfrac, 0)
       # gaussfrac = 0.0
        expfrac = 1.0 - gaussfrac
    
        print gaussfrac * amplitude, (1.0 - gaussfrac) * amplitude
        splotch = gaussfrac * amplitude * gaussian_point(self.sigma_gaussian * imarray.shape[0])

        patchsize = int(5 * self.sigma_gaussian * imarray.shape[0])
        # get the limits of the splotch and convert to pixels
        xmin = x * imarray.shape[0] - patchsize 
        xmax = x * imarray.shape[0] + patchsize + 1
        ymin = y * imarray.shape[1] - patchsize
        ymax = y * imarray.shape[1] + patchsize + 1
        # check if we're in the plot   
        if xmin >= 0 and xmax <= imarray.shape[0] and ymin >= 0 and ymax <= imarray.shape[1]:
            for j in range(3):     
                imarray[xmin:xmax, ymin:ymax, j] += color[j] * splotch  # track rgb fractions
            imarray[xmin:xmax, ymin:ymax, 3] += splotch # track total intensity
                
        splotch = expfrac * amplitude * exponential_point(self.sigma_exponential * imarray.shape[0])
        patchsize = int(32 * self.sigma_exponential * imarray.shape[0])
        # get the limits of the splotch and convert to pixels
        xmin = x * imarray.shape[0] - patchsize 
        xmax = x * imarray.shape[0] + patchsize + 1
        ymin = y * imarray.shape[1] - patchsize
        ymax = y * imarray.shape[1] + patchsize + 1
        # check if we're in the plot   
        if xmin >= 0 and xmax <= imarray.shape[0] and ymin >= 0 and ymax <= imarray.shape[1]:
            for j in range(3):     
                imarray[xmin:xmax, ymin:ymax, j] += color[j] * splotch # track rgb fractions               
            imarray[xmin:xmax, ymin:ymax, 3] += splotch # track total intensity 
        return imarray         
        
    def snapshot(self, width = 5, height = 5, dpi = 200):
        """create the actual image
        """
        # if there have been changes to the camera, transform the stars again
        if not self._transformed:
            self._transform()
            
        # if we haven't assigned colors etc., do that now.
        if not self._mapped:
            self._map_colors()
        
     
    def _map_amplitudes(self):
        """map amplitudes for the gaussian and exponential splotches based on luminosity
        
           this maps luminosities to the range [self.min_amplitude, self.max_amplitude]        
        """
        self._amplitudes = ( (self.luminosities - self.min_log_luminosity_solar)   
            / (self.max_log_luminosity_solar - self.min_log_luminosity_solar)    
            * (1 - self.min_amplitude) + self.min_amplitude )
        
        # figure out fraction of splotch that is gaussian versus exponential    
        # map the point in Luminosity - Temperature to a distance in which the range
        # of temperature and luminosity is 0 - 1
        coordT = self.T    
        self._amped = True
    
    
    
    def _map_colors(self):
        """transform luminosity and temperature into colors and intensities
        """
        if self.mapping == 'OpticalNatural':
            # this mapping is from http://www.vendian.org/mncharity/dir3/starcolor/
            # temperature ranges are from http://outreach.atnf.csiro.au/education/senior/astrophysics/photometry_colour.html
            # beginning and ending temperatures are the inner edges of the terminal bins
            # the other temperatures are the middle of the temperature ranges
            # colors are linearly interpolated between these, saturating at the end states
            self.tempkeys = [3500, 4200, 5450, 6750, 8750, 19000, 28000]
            self.rkeys = [255, 255, 255, 248, 202, 170, 155]
            self.gkeys = [204, 210, 244, 247, 215, 191, 176]
            self.bkeys = [111, 161, 234, 255, 255, 255, 255]
            
        # interpolate the colors for each star
        rvals = np.interp(self.T, self.tempkeys, self.rkeys)
        gvals = np.interp(self.T, self.tempkeys, self.gkeys)
        bvals = np.interp(self.T, self.tempkeys, self.bkeys)
            
        # convert these numpy arrays into tuples 
        self._colors = tuple(map(tuple, np.vstack((rvals, gvals, bvals)).T))
        self._mapped = True
    
    
    def _transform(self):
        """transform the raw x, y, z of the stars into the camera's reference frame
        """
        print "not implemented, defaulting to no transformation"
        print self._transformed
        self._transx = self.x
        self._transy = self.y
        self._transz = self.z
        self._transformed = True
    
    
    
        
    
    
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
    

def deposit_star(imarray, x, y, temp, lum, sigma):
    splotch = gaussian_point(sigma)
    ima
    
    
def gaussian_point(sigma):
    """make a gaussian splotch.
    
    Arguments:
    sigma -- dispersion of the gaussian
    
    Returns:
    a numpy array covering a square 5 sigma on a side with guassian values.
    """
    size = int(5*sigma)
    sigma2 = sigma**2
    if sigma2 == 0:
        sigma2 = 1.e-3
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp( -(x**2 + y**2)/(2 * sigma2) ) / twopi
    return g      
    
def exponential_point(sigma):
    """make an exponential splotch.
    
    Arguments:
    sigma -- scale length of the exponential
    
    Returns:
    a numpy array covering a square 5 sigma on a side with exponential values.
    """
    size = int(32*sigma)
    if sigma == 0:
        sigma = 1.e-3
    x, y = np.mgrid[-size:size+1, -size:size+1]
    e = np.exp( -np.sqrt(x**2 + y**2)/sigma )
    return e 



