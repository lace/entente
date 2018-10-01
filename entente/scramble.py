import numpy as np


def scramble_vertices(mesh):
    """
    Scramble the vertex ordering.

    Return np array that maps from old vertex indices to new.
    """
    v_old_to_new = np.random.permutation(len(mesh.v))
    mesh.v = mesh.v[v_old_to_new]
    mesh.f = np.vstack(
        [
            v_old_to_new[mesh.f[:, 0]],
            v_old_to_new[mesh.f[:, 1]],
            v_old_to_new[mesh.f[:, 2]],
        ]
    ).T
    return v_old_to_new


def scramble_faces(mesh):
    """
    Scramble the face ordering.

    Return np array that maps from old face indices to new.
    """
    f_old_to_new = np.random.permutation(len(mesh.f))
    mesh.f = mesh.f[f_old_to_new]
    return f_old_to_new
