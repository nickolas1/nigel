import sys
sys.path.append('/Users/moeckel/Codes/n6tohdf5')

import nigel

nb = nigel.load('n6snap.0100.hdf5')

cam = nigel.Camera(nb)

cam.mapping_test()