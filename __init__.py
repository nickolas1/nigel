from __future__ import absolute_import

import sys
import os
import numpy as np


sys.path.append('/Users/moeckel/Codes/')



# get packages
from .datahandlers.nigelhdf5reader import \
    NigelHDF5State
  
from .datastructures.nbstate import \
    NBodyState, NBodySubset

from .datastructures.selectors import \
    MassSelection, SphereSelection, BoundSelection
    
from .visualization.rendering import \
    Camera

# supported file types
_outputtypes = [NigelHDF5State]
    
def load(infile):
    for outputtype in _outputtypes:
        if outputtype._is_readable(infile):
            return outputtype(infile)
    print 'oh no, unsupported output type'
    