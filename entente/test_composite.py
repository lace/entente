import numpy as np
from .composite import composite_meshes
from .testing import vitra_mesh


def test_composite_meshes(tmp_path):
    base_mesh = vitra_mesh()
    # For performance.
    base_mesh.keep_vertices(np.arange(1000))

    # Create several translations of an example mesh which, composited,
    # should return the original mesh.
    working_mesh = base_mesh.copy_fv()
    mesh_paths = []
    for i, offset in enumerate(np.linspace(-25.0, 25.0, num=9)):
        mesh_path = str(tmp_path / "mesh_{}.obj".format(i))
        working_mesh.v[:1] = base_mesh.v[:1] + offset
        working_mesh.write(mesh_path)
        mesh_paths.append(mesh_path)

    # Confidence check.
    composite = composite_meshes(mesh_paths[:-1])
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_almost_equal, composite.v, base_mesh.v
    )

    composite = composite_meshes(mesh_paths)

    np.testing.assert_array_almost_equal(composite.v, base_mesh.v)
