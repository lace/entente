import lacecore
import numpy as np


def shuffle_vertices(mesh, ret_new_ordering=False):
    """
    Shuffle the mesh's vertex ordering, preserving the integrity of the faces.
    The mesh is mutated.

    Args:
        mesh (lacecore.Mesh): A mesh.
        ret_new_ordering (bool): When `True`, return the new vertex ordering.

    Returns:
        object: When `ret_new_ordering` is `True`, return a tuple containing
            the new mesh and the new vertex ordering. When `False`, return only
            the new mesh.
    """
    ordering = np.random.permutation(len(mesh.v))
    new_mesh = lacecore.reindex_vertices(mesh, ordering)
    if ret_new_ordering:
        return new_mesh, ordering
    else:
        return new_mesh


def shuffle_faces(mesh, ret_new_ordering=False):
    """
    Shuffle the mesh's face ordering. The mesh is mutated.

    Args:
        mesh (lacecore.Mesh): A mesh.
        ret_new_ordering (bool): When `True`, return the new face ordering.

    Returns:
        object: When `ret_new_ordering` is `True`, return a tuple containing
            the new mesh and the new vertex ordering. When `False`, return only
            the new mesh.
    """
    ordering = np.random.permutation(len(mesh.f))
    new_mesh = lacecore.reindex_faces(mesh, ordering)
    if ret_new_ordering:
        return new_mesh, ordering
    else:
        return new_mesh
