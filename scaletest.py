import sys
sys.path.append('/Users/moeckel/Codes/')

import nigel

nb = nigel.load('n6snap.0100.hdf5')


nb.rescale_length(3)
print nb.rscale, nb.tscale

nb.rescale_length(.3)
print nb.rscale, nb.tscale

nb.rescale_length("3")
print nb.rscale, nb.tscale

nb.rescale_length(-3)
print nb.rscale, nb.tscale

nb.rescale_length("-3")
print nb.rscale, nb.tscale



print nb.dc_pos

sp = nigel.SphereSelection(nb, 10)