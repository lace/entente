def require_cgal():
    """
    Check that CGAL is installed, and exit with a helpful error message
    if it is not. CGAL is optional because it is a very heavy dependency.
    """
    message = """
    CGAL is not installed.

    brew install cgal swig
    pip install cgal-bindings
    wait a year

    Note: CGAL has a GPLv3 license.
    """
    try:
        from CGAL.CGAL_Kernel import Point_3, Triangle_3
        from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
    except ImportError:
        raise ImportError(message)

def create_aabb_tree(mesh):
    from CGAL.CGAL_Kernel import Point_3, Triangle_3
    from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
    cgal_verts = [Point_3(*xyz) for xyz in mesh.v]
    cgal_faces = [
        Triangle_3(cgal_verts[a], cgal_verts[b], cgal_verts[c])
        for a, b, c in mesh.f
    ]
    tree = AABB_tree_Triangle_3_soup(cgal_faces)
    tree.accelerate_distance_queries()
    return tree

def faces_nearest_to_points(mesh, to_points, ret_points=False):
    import numpy as np
    from CGAL.CGAL_Kernel import Point_3
    tree = create_aabb_tree(mesh)
    face_indices = np.empty(shape=(len(to_points),), dtype=np.uint64)
    if ret_points:
        closest_points = np.empty(shape=(len(to_points), 3))
    for i, p in enumerate(to_points):
        point, face_index = tree.closest_point_and_primitive(Point_3(*p))
        face_indices[i] = face_index
        if ret_points:
            closest_points[i] = np.array([point.x(), point.y(), point.z()])
    return (face_indices, closest_points) if ret_points else face_indices
