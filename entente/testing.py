def relative_to_project(*components):
    import os

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", *components))


def mesh_asset(*components):
    from lace.mesh import Mesh

    return Mesh(filename=relative_to_project(*components))


def vitra_mesh():
    result = mesh_asset("examples/vitra/vitra_without_materials.obj")
    assert len(result.v) > 100
    return result
