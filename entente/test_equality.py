import unittest
from .equality import have_same_topology

class TestEquality(unittest.TestCase):
    def test_have_same_topology(self):
        import numpy as np
        from lace.shapes import create_cube

        cube_1 = create_cube(np.zeros(3), 1.)
        cube_2 = create_cube(np.zeros(3), 1.)
        self.assertTrue(cube_1.has_same_topology(cube_2))

        cube_1 = create_cube(np.zeros(3), 1.)
        cube_2 = create_cube(np.ones(3), 1.)
        self.assertTrue(cube_1.has_same_topology(cube_2))

        cube_1 = create_cube(np.zeros(3), 1.)
        cube_2 = create_cube(np.zeros(3), 1.)
        cube_2.f = np.roll(cube_2.f, 1, axis=1)
        self.assertFalse(cube_1.has_same_topology(cube_2))

        cube_1 = create_cube(np.zeros(3), 1.)
        cube_2 = create_cube(np.zeros(3), 1.)
        del cube_2.f
        self.assertFalse(cube_1.has_same_topology(cube_2))
