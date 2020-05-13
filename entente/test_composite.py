import lacecore
import numpy as np
import pytest
import vg
from .composite import composite_meshes
from .testing import vitra_mesh


def mesh():
    # For performance.
    return vitra_mesh().picking_vertices(np.arange(1000))


def test_composite_meshes(tmp_path):
    base_mesh = mesh()

    # Create several translations of an example mesh which, composited,
    # should return the original mesh.
    mesh_paths = []
    for i, offset in enumerate(np.linspace(-25.0, 25.0, num=9)):
        mesh_path = str(tmp_path / "mesh_{}.obj".format(i))
        lacecore.Mesh(v=base_mesh.v + offset * vg.basis.y, f=base_mesh.f).write_obj(
            mesh_path
        )
        mesh_paths.append(mesh_path)

    # Confidence check.
    composite = composite_meshes(mesh_paths[:-1])
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_almost_equal, composite.v, base_mesh.v
    )

    composite = composite_meshes(mesh_paths)

    np.testing.assert_array_almost_equal(composite.v, base_mesh.v)


def test_composite_meshes_error(tmp_path):
    test_mesh = mesh()
    test_mesh_path = str(tmp_path / "test_mesh.obj")
    test_mesh.write_obj(test_mesh_path)

    # Create a mesh with a different topology.
    flipped = test_mesh.faces_flipped()
    test_mesh_flipped_path = str(tmp_path / "test_mesh_flipped.obj")
    flipped.write_obj(test_mesh_flipped_path)

    with pytest.raises(
        ValueError, match=r"Expected .+ to have the same topology as .*"
    ):
        composite_meshes([test_mesh_path, test_mesh_flipped_path])

    with pytest.raises(ValueError, match="Expected at least one mesh path"):
        composite_meshes([])
