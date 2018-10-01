import unittest
import numpy as np
from .scramble import scramble_vertices, scramble_faces


def coord_set(a):
    return set(tuple(coords) for coords in a)


class TestScramble(unittest.TestCase):
    def setUp(self):
        from .testing import vitra_mesh

        self.test_mesh = vitra_mesh()

    def test_scramble_vertices(self):
        scrambled = self.test_mesh.copy_fv()

        scramble_vertices(scrambled)

        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, scrambled.v, self.test_mesh.v
        )
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, scrambled.f, self.test_mesh.f
        )
        self.assertItemsEqual(coord_set(self.test_mesh.v), coord_set(scrambled.v))

    def test_scramble_faces(self):
        scrambled = self.test_mesh.copy_fv()

        scramble_faces(scrambled)

        np.testing.assert_array_equal(scrambled.v, self.test_mesh.v)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, scrambled.f, self.test_mesh.f
        )
        self.assertItemsEqual(coord_set(self.test_mesh.f), coord_set(scrambled.f))
