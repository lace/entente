"""
Functions relating to mesh geometry.
"""


def compute_barycentric_coordinates(vertices_of_tris, points):
    """
    Compute barycentric coordinates for the projection of a set of points to a
    given set of triangles specfied by their vertices.

    These barycentric coordinates can refer to points outside the triangle.
    This happens when one of the coordinates is negative. However they can't
    specify points outside the triangle's plane. (That requires tetrahedral
    coordinates.)

    The returned coordinates supply a linear combination which, applied to the
    vertices, returns the projection of the original point the plane of the
    triangle.

    Args:
        vertices_of_tris (np.arraylike): A set of triangle vertices as `kx3x3`.
        points (np.arraylike): Coordinates of points as `kx3`.

    Returns:
        np.ndarray: Barycentric coordinates as `kx3`

    See Also:
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system

    Note:
        A function with this signature probably belongs in `blmath`.
    """
    from blmath.geometry.barycentric import barycentric_coordinates_of_projection
    from .validation import validate_shape_from_ns

    k = validate_shape_from_ns(locals(), "vertices_of_tris", -1, 3, 3)
    validate_shape_from_ns(locals(), "points", k, 3)

    return barycentric_coordinates_of_projection(
        points,
        vertices_of_tris[:, 0],
        vertices_of_tris[:, 1] - vertices_of_tris[:, 0],
        vertices_of_tris[:, 2] - vertices_of_tris[:, 0],
    )
