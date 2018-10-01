import unittest
import numpy as np
from .restore_correspondence import find_correspondence, restore_correspondence


class TestScramble(unittest.TestCase):
    def setUp(self):
        from .testing import vitra_mesh

        self.test_mesh = vitra_mesh()
        # For performance.
        self.test_mesh.keep_vertices(np.arange(1000))

    def test_find_correspondence_produces_expected_correspondence(self):
        a = self.test_mesh.v
        permutation = np.random.permutation(len(a))
        b = a[permutation]

        correspondence = find_correspondence(a, b, progress=False)

        np.testing.assert_array_equal(correspondence, permutation)
        np.testing.assert_array_equal(a[correspondence], b)

    def test_restore_correspondence_restores_original_array(self):
        return
        from lace.mesh import Mesh

        scrambled_mesh = Mesh(f=self.test_mesh.f)

        restore_correspondence(scrambled),

        np.testing.assert_array_equal(indexes)
