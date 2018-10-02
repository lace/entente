"""
`cgal_search` provides spatial search on vertices and faces. This is
implemented atop [CGAL's axis-aligned-bounding-box search tree][aabb].

[CGAL][] is a heavy dependency and therefore is optional. Before using this
module, you must install CGAL and its Python bindings (which take quite some
time to build).

On Mac OS:

    ```sh
    brew install cgal swig
    pip install cgal-bindings
    # wait a year
    ```

Note: The AABB tree is in the [GPLv3-licensed portion of CGAL][cgal license].

[aabb]: https://doc.cgal.org/latest/AABB_tree/index.html
[cgal]: https://www.cgal.org/
[cgal license]: https://www.cgal.org/license.html
"""
from __future__ import print_function


def require_cgal():
    """
    Check that CGAL is installed, and raise an error with a helpful error message
    if it is not.
    """
    try:
        from CGAL.CGAL_Kernel import Point_3, Triangle_3
        from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

        # For pyflakes.
        assert Point_3
        assert Triangle_3
        assert AABB_tree_Triangle_3_soup
    except ImportError:
        print(
            """
            CGAL is not installed. On Mac OS:

            $ brew install cgal swig
            $ pip install cgal-bindings
            $ wait a year

            Note: The AABB code in CGAL has a GPLv3 license.
            """
        )
        raise ImportError("CGAL is not installed")


def create_aabb_tree(mesh):
    """
    Create a CGAL AABB tree from the given mesh.

    These trees may rely on some shared internal storage in CGAL, so to be
    conservative, *finish using one before creating another*.

    For more information, see:

    - https://doc.cgal.org/latest/AABB_tree/index.html
    - https://github.com/CGAL/cgal-swig-bindings/blob/master/examples/python/AABB_triangle_3_example.py

    Returns:
        CGAL_AABB_tree: A CGAL AABB tree.
    """
    from CGAL.CGAL_Kernel import Point_3, Triangle_3
    from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

    cgal_verts = [Point_3(*xyz) for xyz in mesh.v]
    cgal_faces = [
        Triangle_3(cgal_verts[a], cgal_verts[b], cgal_verts[c]) for a, b, c in mesh.f
    ]
    tree = AABB_tree_Triangle_3_soup(cgal_faces)
    tree.accelerate_distance_queries()
    return tree


def faces_nearest_to_points(mesh, query_points, ret_points=False):
    """
    Find the triangular faces on a mesh which are nearest to the given query
    points.

    Args:
        query_points (arraylike): The points to query, with shape kx3
        ret_points (bool): When `True`, return both the indices of the
            nearest faces and the closest points to the query points, which
            are not necessarily vertices. When `False`, return only the
            face indices.

    Returns:
        object: np.ndarray with shape kx1 of face indices, or when `ret_points`
        is `True`, a tuple which also contains a np.ndarray with shape kx3 with
        the coordinates of the closest points.
    """
    import numpy as np
    from CGAL.CGAL_Kernel import Point_3

    tree = create_aabb_tree(mesh)
    face_indices = np.empty(shape=(len(query_points),), dtype=np.uint64)
    if ret_points:
        closest_points = np.empty(shape=(len(query_points), 3))
    for i, p in enumerate(query_points):
        point, face_index = tree.closest_point_and_primitive(Point_3(*p))
        face_indices[i] = face_index
        if ret_points:
            closest_points[i] = np.array([point.x(), point.y(), point.z()])
    return (face_indices, closest_points) if ret_points else face_indices
