import numpy as np
from .shuffle import shuffle_vertices, shuffle_faces
from .testing import vitra_mesh, assert_same_face_set, assert_same_vertex_set


def test_shuffle_vertices():
    test_mesh = vitra_mesh()
    shuffled, ordering = shuffle_vertices(test_mesh, ret_new_ordering=True)

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, shuffled.v, test_mesh.v
    )
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, shuffled.f, test_mesh.f
    )
    assert_same_vertex_set(test_mesh, shuffled)
    np.testing.assert_array_equal(test_mesh.v[ordering], shuffled.v)


def test_shuffle_faces():
    test_mesh = vitra_mesh()
    shuffled, ordering = shuffle_faces(test_mesh, ret_new_ordering=True)

    np.testing.assert_array_equal(shuffled.v, test_mesh.v)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, shuffled.f, test_mesh.f
    )
    assert_same_face_set(test_mesh, shuffled)
    np.testing.assert_array_equal(test_mesh.f[ordering], shuffled.f)
