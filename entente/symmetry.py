import numpy as np
from lace.mesh import Mesh
import vg


def find_opposite_vertices(
    vertices, plane_of_symmetry, all_must_match=False, atol=1e-4, progress=True
):
    """
    Given a plane of symmetry and a point cloud, match each vertex to its
    mirror image.

    When `all_must_match=True`, raise an error if, for any vertex, a
    corresponding mirrored vertex can't be found. When
    `all_must_match=False`, return an index of `-1` for any vertex
    with no mirror image.

    Vertices on the plane of symmetry correspond to themselves.

    Args:
        mesh (lace.mesh.Mesh): A mesh with `k` vertices.
        atol (float): Match tolerance.
        all_must_match (bool): When `True`, `mesh` must contain a mirror-image
            vertex for each of its vertices.
        progress (bool): When `True`, show a progress bar.

    Return:
        np.ndarray: For each of `mesh.v`, the index in `mesh.v` of its mirror
            image, with shape `(k,)`.

    Note:
        This relies on a brute-force algorithm.

        For the interpretation of `atol`, see documentation for `np.isclose`.
    """
    from polliwog import Plane
    from polliwog.plane import mirror_point_across_plane
    from .restore_correspondence import find_correspondence

    vg.shape.check(locals(), "vertices", (-1, 3))
    if not isinstance(plane_of_symmetry, Plane):
        raise ValueError("Expected a Plane")

    mirrored_vertices = mirror_point_across_plane(vertices, plane_of_symmetry.equation)
    return find_correspondence(
        vertices,
        mirrored_vertices,
        atol=atol,
        all_must_match=all_must_match,
        progress=progress,
    )
