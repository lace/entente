import unittest
import numpy as np
from .scramble import scramble_vertices, scramble_faces
from .testing import ExtraAssertions


class TestScramble(ExtraAssertions, unittest.TestCase):
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
        self.assertSameVertexSet(self.test_mesh, scrambled)

    def test_scramble_faces(self):
        scrambled = self.test_mesh.copy_fv()

        scramble_faces(scrambled)

        np.testing.assert_array_equal(scrambled.v, self.test_mesh.v)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, scrambled.f, self.test_mesh.f
        )
        self.assertSameFaceSet(self.test_mesh, scrambled)
