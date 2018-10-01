import unittest
import numpy as np
from .restore_correspondence import find_correspondence, restore_correspondence


class TestScramble(unittest.TestCase):
    def setUp(self):
        from .testing import vitra_mesh

        self.test_mesh = vitra_mesh()

    def test_find_correspondence_produces_expected_correspondence(self):
        return
        unscrambled = self.test_mesh.v
        indexes = np.random.permutation(len(unscrambled))
        scrambled = unscrambled[indexes]

        np.testing.assert_array_equal(
            find_correspondence(unscrambled, scrambled), indexes
        )

    def test_restore_correspondence_restores_original_array(self):
        return
        from lace.mesh import Mesh

        scrambled_mesh = Mesh(f=self.test_mesh.f)

        restore_correspondence(scrambled),

        np.testing.assert_array_equal(indexes)
