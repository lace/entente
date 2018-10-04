import unittest
import numpy as np
from .restore_correspondence import find_correspondence, restore_correspondence
from .testing import ExtraAssertions


class TestRestoreCorrespondence(ExtraAssertions, unittest.TestCase):
    def setUp(self):
        from .testing import vitra_mesh

        self.test_mesh = vitra_mesh()
        # For performance.
        self.test_mesh.keep_vertices(np.arange(1000))

    def test_find_correspondence_matched(self):
        b = self.test_mesh.v
        expected_correspondence = np.random.permutation(len(b))
        a = b[expected_correspondence]

        correspondence = find_correspondence(a, b, progress=False)

        np.testing.assert_array_equal(correspondence, expected_correspondence)
        np.testing.assert_array_equal(b[correspondence], a)

    def test_find_correspondence_unmatched(self):
        b = self.test_mesh.v
        expected_correspondence = np.random.permutation(len(b))
        a = b[expected_correspondence]

        a = np.vstack([a, np.array([1.0, 2.0, 3.0])])
        expected_correspondence = np.append(1 + expected_correspondence, np.array([-1]))
        b = np.vstack([np.array([3.0, 2.0, 1.0]), b])
        expected_unmatched_b = np.array([0])

        with self.assertRaises(ValueError):
            find_correspondence(a, b, progress=False)

        correspondence, unmatched_b = find_correspondence(
            a, b, all_must_match=False, ret_unmatched_b=True, progress=False
        )

        np.testing.assert_array_equal(correspondence, expected_correspondence)
        np.testing.assert_array_equal(unmatched_b, expected_unmatched_b)
        reconstructed_a = np.vstack(
            [
                b[correspondence[np.where(correspondence != -1)]],
                np.array([1.0, 2.0, 3.0]),
            ]
        )
        np.testing.assert_array_equal(reconstructed_a, a)

    def test_restore_correspondence(self):
        from .shuffle import shuffle_vertices

        working = self.test_mesh.copy_fv()
        v_new_to_old = shuffle_vertices(working)
        # Compute the inverse of the permutation.
        # https://stackoverflow.com/a/11649931/893113
        expected_v_old_to_new = np.argsort(v_new_to_old)

        v_old_to_new = restore_correspondence(working, self.test_mesh, progress=False)

        np.testing.assert_array_equal(working.v, self.test_mesh.v)
        np.testing.assert_array_equal(working.f, self.test_mesh.f)
        np.testing.assert_array_equal(v_old_to_new, expected_v_old_to_new)
