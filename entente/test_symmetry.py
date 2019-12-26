import pytest
import numpy as np
from polliwog import Plane
from .symmetry import find_opposite_vertices
from .testing import vitra_mesh

FAST = True


def create_seat_and_arm_mesh():
    from .testing import vitra_mesh

    result = vitra_mesh()
    if FAST:
        # For performance, work with part of the chair back instead of the
        # whole thing.
        result.cut_across_axis(1, minval=42, maxval=57)
        result.cut_across_axis(2, maxval=-17)
    return result


def test_find_opposite_vertices():
    mesh = create_seat_and_arm_mesh()
    # Select a tolerance at which the vast majority of vertices of the chair
    # will match.
    atol = 1e-1

    indices_of_opposite_vertices = find_opposite_vertices(
        vertices=mesh.v, plane_of_symmetry=Plane.yz, all_must_match=False, atol=atol
    )

    matched = indices_of_opposite_vertices != -1
    assert np.count_nonzero(matched) > 0.95 * len(mesh.v)

    exact_mirrored_vs = np.copy(mesh.v)
    exact_mirrored_vs[:, 0] = -exact_mirrored_vs[:, 0]

    np.testing.assert_allclose(
        mesh.v[indices_of_opposite_vertices[matched]],
        exact_mirrored_vs[matched],
        atol=1.1 * atol,
    )
