import sys
sys.path.append('/Users/moeckel/Codes/')

import nigel
import unittest
import numpy as np

nb = nigel.load('simpletest')

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
        

class DensityTest(unittest.TestCase):
    def testDensityCalculation(self):
        """compare density calculation to hand-calculated values"""
        known_densities = np.array([1.76776695e-01, 1.76776695e-01, 1.76776695e-01,
            4.59619433e-01, 4.59619433e-01, 1.76776695e-01, 5.00000000e-01, 
            8.84538011e-02, 3.40206909e-02, 2.26040275e-04])
        densities = nb._get_local_densities() 
        np.testing.assert_allclose(densities, known_densities)       
    
    def testDensityCenter(self):
        """compare density center to hand-calculated values"""
        known_dc = np.array([1.1509689991515422, -0.10055339316744231, 0.025207802992049881])
        np.testing.assert_allclose(nb.dc_pos, known_dc)
    
    def testDensityCenterVelocity(self):
        """compare density center velocity to hand-calculated values"""
        known_dcv = np.array([0.24275266063732542, 0.25474645145914782, 0.32455563530545328])
        np.testing.assert_allclose(nb.dc_vel, known_dcv)
    

class RadiiTest(unittest.TestCase):
    def testRadiiFromOrigin(self):
        known_radii_origin = np.array([np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0), 
            2.0, 0.0, 1.0, np.sqrt(6.0), np.sqrt(2.0), np.sqrt(200 + 11**2)])
        np.testing.assert_allclose(nb.radii_origin, known_radii_origin)
    
    def testRadiiFromDensityCenter(self):
        known_radii_dc = np.array([0.99152531532477317, 1.0411309537700291, 1.1111457344780995,
            0.91237671527670305,  0.85533622547563493, 1.1556279917071388, 0.18313398623409738, 
            1.6977312033893031, 1.7845965580114431,  17.277762293693911])
        np.testing.assert_allclose(nb.radii_dc, known_radii_dc)
        
        
class SphereSelectionTest(unittest.TestCase):
    def testDefaultSphere(self):
        """this should return only two stars"""
        sp = nigel.SphereSelection(nb)
        self.assertEqual(sp.n, 2)     
    
    def testSphereRadius(self):
        """this should exclude the outlying star"""
        sp = nigel.SphereSelection(nb, radius=10)
        self.assertEqual(sp.n, 9)
        
    def testSphereOrigin(self):
        """this should return three stars"""
        sp = nigel.SphereSelection(nb, origin=[1,1,1])
        self.assertEqual(sp.n, 3)
    
    def testEmptySphere(self):
        """this should return None"""
        sp = nigel.SphereSelection(nb, origin=[100,100,100])
        self.assertIsNone(sp)
        
        
class MassSelectionTest(unittest.TestCase):
    def testDefaults(self):
        """defaults should return the full set of stars"""
        mc = nigel.MassSelection(nb)
        self.assertEqual(mc.n, 10)
        
    def testLowMass(self):
        """this should exclude all but two stars"""
        mc = nigel.MassSelection(nb, mlow = 0.101)
        self.assertEqual(mc.n, 2)
        
    def testHighMass(self):
        """this should exclude two stars"""
        mc = nigel.MassSelection(nb, mhigh = 0.101)
        self.assertEqual(mc.n, 8)

    def testNoMatchingMasses(self):
        """this should return None"""
        mc = nigel.MassSelection(nb, mlow = 10, mhigh = 10.101)
        self.assertIsNone(mc)
        
    
if __name__ == "__main__":
    unittest.main() 