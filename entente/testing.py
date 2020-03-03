def relative_to_project(*components):
    import os

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", *components))


def mesh_asset(*components):
    import lacecore

    return lacecore.load_obj(relative_to_project(*components), triangulate=True)


def vitra_mesh():
    result = mesh_asset("examples/vitra/vitra_without_materials_triangulated.obj")
    assert len(result.v) > 100
    return result


def coord_set(a):
    return set(tuple(coords) for coords in a)


def assert_same_vertex_set(a, b):
    assert coord_set(a.v) == coord_set(b.v)


def assert_same_face_set(a, b):
    assert coord_set(a.f) == coord_set(b.f)
