def composite_meshes(mesh_paths):
    """
    Create a composite as a vertex-wise average of several meshes in
    correspondence. Faces and groups are loaded from the first mesh given.

    Args:
        mesh_paths (list): Paths of the meshes to average.

    Returns:
        lace.mesh.Mesh: The composite mesh.
    """
    from lace.mesh import Mesh

    if not len(mesh_paths):
        raise ValueError('Expected at least one mesh path')

    first_mesh_path, remaining_mesh_paths = mesh_paths[0], mesh_paths[1:]

    working_mesh = Mesh(filename=mesh_paths[0])

    for this_mesh_path in mesh_paths[1:]:
        this_mesh = Mesh(filename=this_mesh_path)
        if not this_mesh.has_same_topology(working_mesh):
            raise ValueError('Expected {} to have the same topology as {}'.format(
                this_mesh_path,
                first_mesh_path))
        working_mesh.v += this_mesh.v

    working_mesh.v /= len(mesh_paths)

    return working_mesh
