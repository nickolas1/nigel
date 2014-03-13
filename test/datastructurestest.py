import sys
sys.path.append('/Users/moeckel/Codes/')

import nigel
import unittest

nb = nigel.load('n6snap.0075.hdf5')

class RescaleTest(unittest.TestCase):    
    def testIntegerInput(self):
        """rescale_length should set rscale with no problems"""
        nb.rescale_length(2)
        self.assertEqual(2.0, nb.rscale)
    
    def testFloatInput(self):
        """rescale_length should set rscale with no problems"""
        nb.rescale_length(2.0)
        self.assertEqual(2.0, nb.rscale)

    def testStringInput(self):
        """rescale_length should set rscale with no problems"""
        nb.rescale_length("2.0")
        self.assertEqual(2.0, nb.rscale)
    
    def testNegativeInput(self):
        """rescale_length should refuse to rescale with negative input"""
        nb.rescale_length(2.0)
        nb.rescale_length(-1.0)
        self.assertEqual(2.0, nb.rscale)
    
    def testZeroInput(self):
        """rescale_length should refuse to rescale with zero input"""
        nb.rescale_length(2.0)
        nb.rescale_length(0)
        self.assertEqual(2.0, nb.rscale)
        
    def testZeroInput(self):
        """rescale_length should fail with no input"""
        self.assertRaises(TypeError, nb.rscale,)
        

class RadiiTest(unittest.TestCase):
    nb = NBodyState()
    nb.n = 10
    print nb.n
    print nb.pos


    """nb.rescale_length(3)
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
    """
    
    
if __name__ == "__main__":
    unittest.main() 