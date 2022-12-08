"""
This module requires entente to be installed with the `surface_regressor` extra:

    pip install entente[surface_regressor]
"""

import numpy as np
from vg.compat import v2 as vg


def surface_regressor_for(faces, source_mesh_vertices, query_points):
    """
    Low-level function to create a regressor for the given query points on the
    given mesh. The regressor computes the corresponding points on a target
    mesh (given by its vertices).

    The query points are projected to the nearest point on the mesh surface.

    It works like this:

    1. Find the face on which each query point sits, projecting it if
       necessary. Then describe its position as a linear combination of the
       three vertices of that face.
    2. Represent this as a sparse matrix with a column for each coordinate of
       the source vertices and a row for each coordinate of the landmarks.
    3. Push target mesh vertices through the matrix to transfer the query points
       to the target mesh.

    In most cases, prefer `Landmarker` or `PathTransfer`, which present
    friendlier and safer interfaces.

    See also:
        - `entente.landmarking.Landmarker`
        - `entente.path_transfer.PathTransfer`
    """
    from lacecore import check_indices
    from proximity import faces_nearest_to_points
    from polliwog.tri import barycentric_coordinates_of_points
    from scipy.sparse import csc_matrix

    vg.shape.check(locals(), "faces", (-1, 3))
    vg.shape.check(locals(), "source_mesh_vertices", (-1, 3))
    check_indices(faces, len(source_mesh_vertices), "faces")

    face_indices = faces_nearest_to_points(source_mesh_vertices, faces, query_points)
    vertex_indices = faces[face_indices]
    vertex_coeffs = barycentric_coordinates_of_points(
        source_mesh_vertices[vertex_indices], query_points
    )

    # Note the `.transpose()` at the end. The matrix is initially created
    # from data organized along columns of the result, not rows.
    values = np.repeat(vertex_coeffs, 3, axis=0).ravel()
    indices = (
        (3 * vertex_indices).reshape(-1, 1, 3)
        + np.arange(3, dtype=np.uint64).reshape(-1, 1)
    ).ravel()
    indptr = np.arange(len(values) + 1, step=3)
    return csc_matrix(
        (values, indices, indptr),
        shape=(3 * len(source_mesh_vertices), 3 * len(query_points)),
    ).transpose()


def apply_surface_regressor(regressor, target_mesh_vertices):
    from scipy.sparse import isspmatrix

    num_vertices = vg.shape.check(locals(), "target_mesh_vertices", (-1, 3))
    if not isspmatrix(regressor) or regressor.shape[1] != 3 * num_vertices:
        assert regressor.shape[1] % 3 == 0
        expected_num_vertices = int(regressor.shape[1] / 3)
        raise ValueError(f"This regressor expects {expected_num_vertices} vertices")

    return (regressor * target_mesh_vertices.ravel()).reshape(-1, 3)
