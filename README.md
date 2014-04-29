nigel
=====

N-body analysis and visualization tools

This is a package that's supposed to make dealing with collisional N-body data a bit more user friendly. At the moment it only handles NBODY6 data that has been converted from the OUT3 file to an hdf5 format. This can be done using the script `n6tohdf5.py`. For example: 

```$ n6tohdf5.py OUT3```

will convert each time in OUT3 into a series of snapshot files called `n6snap.xxxx.hdf5`. These can then be used with the rest of the analysis tools.

### Examples
These can all be tried in ipython or whatever else you want.
If the `nigel` package is not in your python path, add it like so before trying these with your data:
```
$ import sys
$ sys.path.append('/PATH/CONTAINING/NIGEL/DIRECTORY')
```
####isolate the stars inside the half-mass radius of a cluster
```
$ import numpy as np
$ import nigel
$ nb = nigel.load('n6snap.0010.hdf5')
n6snap.0010.hdf5
no luminosities for this snapshot
no temperatures in this snapshot
```
Through `nb` you can now access the ids, masses, positions, etc. of the stars in your cluster. For instance, here's the median radius calculated from the origin and from the density center, the number of stars, and the total mass:
```
$ np.median(nb.radii_origin)
11.281018
$ np.median(nb.radii_dc)
9.0040725
$ nb.n
2542
$ np.sum(nb.mass)
0.9998993
```
Here's how to get the half-mass radius, and make a subset of stars within that radius. We need to get the half mass radius as well as the position of the density center to pass to the sphere selector:
```
$ rhm = nb.half_mass_radius
$ rhm
9.2056963
$ dc = nb.dc_pos
$ dc
array([ 3.7554911 ,  0.92901707,  3.21264892])
$ innerHalf = nigel.SphereSelection(nb, origin = dc, radius = rhm)
$ innerHalf.n
1298
$ np.sum(innerHalf.mass)
0.4992601
```
####velocity dispersion of high mass stars
Most things you can do with the full set of stars you can do with selections. Here's the velocity dispersions of stars more massive than 10 times the median mass:
```
$ medianMass = np.median(nb.mass)
$ highMass = nigel.MassSelection(nb, mlow = 10*medianMass)
$ 10 * medianMass * nb.mscale
1.99164629
$ highMass.n
159
$ highMass.sigmas
array([ 0.55474067,  0.59045607,  0.44086817], dtype=float32)
```

####rescale the length 
Rescaling the length will also adjust the timescale
```
$ nb.rscale
0.13946099
$ nb.tscale
0.01953192
$ nb.rescale_length(0.5)
$ nb.rscale
0.5
$ nb.tscale
0.13259307
```

##the future
The plan originally was to have this be able to read in native output formats from a variety of N-body codes, sort of a unified analysis framework ala yt for hydro codes. Inspecting the source code can show what it's capable of, especially the code in `datastructures`. The `visualization` directory currently has an old version of some code (`n6hdf5reader.py`) and the beginnings of some new code (`rendering.py`) to make attractive visualizations of simulations. Since I'm leaving the field, I'm not sure how much more development this will see. Hopefully this can be useful as a launching point for someone's analysis projects!
