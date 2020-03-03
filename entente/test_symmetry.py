import pytest
import numpy as np
from polliwog import Plane
from .symmetry import find_opposite_vertices

FAST = True


def create_seat_and_arm_mesh():
    from .testing import vitra_mesh

    if FAST:
        # For performance, work with part of the chair back instead of the
        # whole thing.
        return (
            vitra_mesh()
            .select()
            .vertices_at_or_above(dim=1, point=np.array([0.0, 42.0, 0.0]))
            .vertices_at_or_below(dim=1, point=np.array([0.0, 57.0, 0.0]))
            .vertices_at_or_below(dim=2, point=np.array([0.0, 0.0, -17.0]))
            .end()
        )
    else:
        return vitra_mesh()


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


def test_find_opposite_vertices_validation():
    mesh = create_seat_and_arm_mesh()
    with pytest.raises(ValueError, match=r"Expected a Plane"):
        find_opposite_vertices(vertices=mesh.v, plane_of_symmetry="not a plane")
