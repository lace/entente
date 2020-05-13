import lacecore
from .equality import have_same_topology


def composite_meshes(mesh_paths):
    """
    Create a composite as a vertex-wise average of several meshes in
    correspondence. Faces, groups, and other attributes are loaded from the
    first mesh given.

    Args:
        mesh_paths (list): Paths of the meshes to average.

    Returns:
        lacecore.Mesh: The composite mesh.
    """
    if not len(mesh_paths):
        raise ValueError("Expected at least one mesh path")

    first_mesh_path, remaining_mesh_paths = mesh_paths[0], mesh_paths[1:]

    working_mesh = lacecore.load_obj(first_mesh_path, triangulate=True)
    working_v = working_mesh.v.copy()

    for this_mesh_path in remaining_mesh_paths:
        this_mesh = lacecore.load_obj(this_mesh_path, triangulate=True)
        if not have_same_topology(this_mesh, working_mesh):
            raise ValueError(
                "Expected {} to have the same topology as {}".format(
                    this_mesh_path, first_mesh_path
                )
            )
        working_v += this_mesh.v

    working_v /= len(mesh_paths)

    return lacecore.Mesh(
        v=working_v, f=working_mesh.f, face_groups=working_mesh.face_groups
    )
