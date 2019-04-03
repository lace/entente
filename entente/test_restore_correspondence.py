import pytest
import numpy as np
from .restore_correspondence import find_correspondence, restore_correspondence


def create_test_mesh():
    from .testing import vitra_mesh

    result = vitra_mesh()
    # For performance.
    result.keep_vertices(np.arange(1000))
    return result


def test_find_correspondence_matched():
    b = create_test_mesh().v
    expected_correspondence = np.random.permutation(len(b))
    a = b[expected_correspondence]

    correspondence = find_correspondence(a, b, progress=False)

    np.testing.assert_array_equal(correspondence, expected_correspondence)
    np.testing.assert_array_equal(b[correspondence], a)


def test_find_correspondence_unmatched():
    b = create_test_mesh().v
    expected_correspondence = np.random.permutation(len(b))
    a = b[expected_correspondence]

    a = np.vstack([a, np.array([1.0, 2.0, 3.0])])

    with pytest.raises(ValueError):
        find_correspondence(a, b, progress=False)

    expected_correspondence = np.append(1 + expected_correspondence, np.array([-1]))
    b = np.vstack([np.array([3.0, 2.0, 1.0]), b])
    expected_unmatched_b = np.array([0])

    with pytest.raises(ValueError):
        find_correspondence(a, b, progress=False)

    correspondence, unmatched_b = find_correspondence(
        a, b, all_must_match=False, ret_unmatched_b=True, progress=False
    )

    np.testing.assert_array_equal(correspondence, expected_correspondence)
    np.testing.assert_array_equal(unmatched_b, expected_unmatched_b)
    reconstructed_a = np.vstack(
        [b[correspondence[np.where(correspondence != -1)]], np.array([1.0, 2.0, 3.0])]
    )
    np.testing.assert_array_equal(reconstructed_a, a)


def test_restore_correspondence():
    from .shuffle import shuffle_vertices

    test_mesh = create_test_mesh()
    working = test_mesh.copy_fv()
    v_new_to_old = shuffle_vertices(working)
    # Compute the inverse of the permutation.
    # https://stackoverflow.com/a/11649931/893113
    expected_v_old_to_new = np.argsort(v_new_to_old)

    v_old_to_new = restore_correspondence(working, test_mesh, progress=False)

    np.testing.assert_array_equal(working.v, test_mesh.v)
    np.testing.assert_array_equal(working.f, test_mesh.f)
    np.testing.assert_array_equal(v_old_to_new, expected_v_old_to_new)
