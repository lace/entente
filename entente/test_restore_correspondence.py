import unittest
import numpy as np
from .restore_correspondence import find_permutation, restore_correspondence
from .testing import ExtraAssertions


class TestRestoreCorrespondence(ExtraAssertions, unittest.TestCase):
    def setUp(self):
        from .testing import vitra_mesh

        self.test_mesh = vitra_mesh()
        # For performance.
        self.test_mesh.keep_vertices(np.arange(1000))

    def test_find_permutation(self):
        a = self.test_mesh.v
        expected_permutation = np.random.permutation(len(a))
        b = a[expected_permutation]

        result_permutation = find_permutation(a, b, progress=False)

        np.testing.assert_array_equal(result_permutation, expected_permutation)
        np.testing.assert_array_equal(a[result_permutation], b)

    def test_restore_correspondence(self):
        from .shuffle import shuffle_vertices

        working = self.test_mesh.copy_fv()
        permutation = shuffle_vertices(working)

        result_permutation = restore_correspondence(
            working, self.test_mesh, progress=False
        )

        np.testing.assert_array_equal(working.v, self.test_mesh.v)
        np.testing.assert_array_equal(working.f, self.test_mesh.f)

        # Compute the inverse of the permutation.
        # https://stackoverflow.com/a/11649931/893113
        v_old_to_new = np.argsort(permutation)
        np.testing.assert_array_equal(result_permutation, v_old_to_new)
