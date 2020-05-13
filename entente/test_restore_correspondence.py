import pytest
import numpy as np
from .restore_correspondence import (
    _maybe_tqdm,
    find_correspondence,
    restore_correspondence,
)


def create_truncated_test_mesh():
    from .testing import vitra_mesh

    # For performance.
    return vitra_mesh().picking_vertices(np.arange(1000))


def test_helper():
    assert [x for x in _maybe_tqdm(iter([1, 2, 3]), progress=True)] == [1, 2, 3]
    assert [x for x in _maybe_tqdm(iter([1, 2, 3]), progress=False)] == [1, 2, 3]


def test_find_correspondence_matched():
    b = create_truncated_test_mesh().v
    expected_correspondence = np.random.permutation(len(b))
    a = b[expected_correspondence]

    correspondence = find_correspondence(a, b, progress=False)

    np.testing.assert_array_equal(correspondence, expected_correspondence)
    np.testing.assert_array_equal(b[correspondence], a)


def test_find_correspondence_unmatched():
    b = create_truncated_test_mesh().v
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

    test_mesh = create_truncated_test_mesh()
    shuffled, ordering = shuffle_vertices(test_mesh, ret_new_ordering=True)

    restored, v_old_to_new = restore_correspondence(shuffled, test_mesh, progress=False)

    np.testing.assert_array_equal(restored.v, test_mesh.v)
    np.testing.assert_array_equal(restored.f, test_mesh.f)
    np.testing.assert_array_equal(v_old_to_new, ordering)
