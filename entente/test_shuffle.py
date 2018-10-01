import unittest
import numpy as np
from .shuffle import shuffle_vertices, shuffle_faces
from .testing import ExtraAssertions


class TestShuffle(ExtraAssertions, unittest.TestCase):
    def setUp(self):
        from .testing import vitra_mesh

        self.test_mesh = vitra_mesh()

    def test_shuffle_vertices(self):
        shuffled = self.test_mesh.copy_fv()

        shuffle_vertices(shuffled)

        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, shuffled.v, self.test_mesh.v
        )
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, shuffled.f, self.test_mesh.f
        )
        self.assertSameVertexSet(self.test_mesh, shuffled)

    def test_shuffle_faces(self):
        shuffled = self.test_mesh.copy_fv()

        shuffle_faces(shuffled)

        np.testing.assert_array_equal(shuffled.v, self.test_mesh.v)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, shuffled.f, self.test_mesh.f
        )
        self.assertSameFaceSet(self.test_mesh, shuffled)
