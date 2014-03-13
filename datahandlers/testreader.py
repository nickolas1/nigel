from __future__ import division

import numpy as np
from ..datastructures.nbstate import NBodyState

class TestState(NBodyState):
    def __init__(self, infile):
        """Create a very small and simple test system.
        """
        super(TestState, self).__init__()
        print 'creating infile'
        self.n = 10
        self.time = 1
        self.tscale = 1.0
        self.rscale = 1.0
        self.vscale = 1.0
        self.mscale = 1.0
    
        self.pos = np.array(
            [[1.0, 0.0, 1.0], [1.0, 0.0, -1.0],
            [1.0, 1.0, 0.0], [1.0, -1.0, 0.0],
            [2.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], [2.0, 1.0, 1.0],
            [0.0, -1.0, -1.0], [11.0, 10.0, 10.0]], dtype=np.float32)
        self.vel = np.array(
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.1],
            [0.0, 0.1, 0.0], [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0], [0.0, 0.1, 0.1],
            [1.0, 1.0, 1.0], [0.0, -1.0, 2.0],
            [0.0, 1.0, 0.0], [0.0, 0.4, 0.0]], dtype=np.float32)
        self.mass = np.array(
            [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.11], dtype=np.float32)
        self.id = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        
    @staticmethod
    def _is_readable(infile):
        print infile
        if infile == 'simpletest':
            return True
        