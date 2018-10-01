import numpy as np


def shuffle_vertices(mesh):
    """
    Shuffle the mesh's vertex ordering.

    Mutate the mesh and return np array that maps from old vertex indices to new.
    """
    v_old_to_new = np.random.permutation(len(mesh.v))
    mesh.reorder_vertices(v_old_to_new)
    return v_old_to_new


def shuffle_faces(mesh):
    """
    Shuffle the mesh's face ordering.

    Mutate the mesh and return np array that maps from old face indices to new.
    """
    f_old_to_new = np.random.permutation(len(mesh.f))
    mesh.f = mesh.f[f_old_to_new]
    return f_old_to_new
