import numpy as np
from polliwog import Plane
from vg.compat import v1 as vg
from ..symmetry import find_opposite_vertices


def symmetrize_landmarks_using_plane(plane_of_symmetry, landmark_coords):
    """
    TODO
    """
    vg.shape.check(locals(), "landmark_coords", (2, 3))
    if not isinstance(plane_of_symmetry, Plane):
        raise ValueError("plane_of_symmetry should be a Plane")

    mirrored_landmarks = plane_of_symmetry.mirror_point(landmark_coords)

    return np.average(
        np.vstack([landmark_coords.flatten(), np.flipud(mirrored_landmarks).flatten()]),
        axis=0,
    ).reshape(2, 3)


def symmetrize_landmarks_using_topology(
    mesh, plane_of_symmetry, landmark_coords, atol=1e-4
):
    """
    TODO
    """
    from polliwog.tri import barycentric_coordinates_of_points
    from proximity import faces_nearest_to_points

    vg.shape.check(locals(), "landmark_coords", (2, 3))
    if not isinstance(plane_of_symmetry, Plane):
        raise ValueError("plane_of_symmetry should be a Plane")

    # Compute the barycentric coordinates of each landmark.
    indices_of_nearest_faces = faces_nearest_to_points(mesh.v, mesh.f, landmark_coords)
    vertex_indices = mesh.f[indices_of_nearest_faces]
    vertex_coeffs = barycentric_coordinates_of_points(
        mesh.v[vertex_indices], landmark_coords
    )

    # Find the opposite of each of these vertices, and use that to compute the
    # opposite of each landmark.
    # TODO: Add a parameter to `find_opposite_vertices()` that only computes the
    # points we need (i.e. `mesh.v[vertex_indices.flatten()]``). Then use
    # `all_must_match=True`.
    indices_of_opposite_vs = find_opposite_vertices(
        mesh.v,
        plane_of_symmetry=plane_of_symmetry,
        all_must_match=False,
        atol=atol,
    )[vertex_indices]
    if np.any(indices_of_opposite_vs == -1):
        raise ValueError("Some landmarks are near triangles which are not mirrored")

    # TODO: Add a `barys_to_points()` function in polliwog.
    mirrored_landmarks = (
        (vertex_coeffs.reshape(-1, 1) * mesh.v[indices_of_opposite_vs].reshape(-1, 3))
        .reshape(-1, 3, 3)
        .sum(axis=1)
    )

    return np.average(
        np.vstack([landmark_coords.flatten(), np.flipud(mirrored_landmarks).flatten()]),
        axis=0,
    ).reshape(-1, 3)
