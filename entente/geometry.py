def compute_barycentric_coordinates(vertices_of_tris, points):
    """
    Compute barycentric coordinates for the projection of a set of points
    (specified as kx3) to a given set of triangles (specified by their
    vertices as kx3x3).

    These barycentric coordinates, can refer to points outside the triangle.
    This happens when one of the coordinates is negative. However they can't
    specify points outside the triangle's plane.

    https://en.wikipedia.org/wiki/Barycentric_coordinate_system

    The returned coordinates supply a kx3 linear combination which, applied
    to the vertices, returns the original point projected to the plane of
    the triangle.

    A function with this signature probably belongs in blmath.
    """
    from blmath.geometry.barycentric import barycentric_coordinates_of_projection
    from .validation import validate_shape_from_ns

    k = validate_shape_from_ns(locals(), 'vertices_of_tris', '*', 3, 3)
    validate_shape_from_ns(locals(), 'points', k, 3)

    return barycentric_coordinates_of_projection(
        points,
        vertices_of_tris[:, 0],
        vertices_of_tris[:, 1] - vertices_of_tris[:, 0],
        vertices_of_tris[:, 2] - vertices_of_tris[:, 0])
