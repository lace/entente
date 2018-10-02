import numpy as np


def shuffle_vertices(mesh):
    """
    Shuffle the mesh's vertex ordering, preserving the integrity of the faces.
    The mesh is mutated.

    Args:
        mesh (lace.mesh.Mesh): A mesh.

    Returns:
        np.ndarray: `vx1` mapping of old vertex indices to new.
    """
    v_old_to_new = np.random.permutation(len(mesh.v))
    mesh.reorder_vertices(v_old_to_new)
    return v_old_to_new


def shuffle_faces(mesh):
    """
    Shuffle the mesh's face ordering. The mesh is mutated.

    Args:
        mesh (lace.mesh.Mesh): A mesh.

    Returns:
        np.ndarray: `fx1` mapping of old face indices to new.
    """
    f_old_to_new = np.random.permutation(len(mesh.f))
    mesh.f = mesh.f[f_old_to_new]
    return f_old_to_new
