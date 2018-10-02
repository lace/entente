def _maybe_tqdm(iterable, progress):
    if progress:
        from tqdm import tqdm

        return tqdm(iterable)
    else:
        return iterable


def find_permutation(a, b, progress=True):
    """
    Given a `kxn` array `a` and a permutation of it `b`, return the indices of
    `a` ordered such that `a[find_permutation(a, b)]` is equal to `b`.

    The permutation must be along the first axis, such `a[0], a[1], ...` and
    `b[0], b[1], ...` have the same elements.

    Args:
        a (np.arraylike): `kxn` array
        b (np.arraylike): Another `kxn` array
        progress (bool): When `True`, show a progress bar.

    Return:
        np.ndarray: Indices of `b` as `kx1`

    Note:
        This relies on a brute-force algorithm.
    """
    import numpy as np

    if not len(a) == len(b):
        raise

    a_remaining = np.ones(len(a), dtype=np.bool_)
    a_to_b = np.zeros(len(a), dtype=np.uint64)

    for i, item in _maybe_tqdm(enumerate(b), progress):
        indices, = np.nonzero(np.logical_not(a - item).all(axis=1))
        if len(indices) != 1:
            raise ValueError(
                "Couldn't find corresponding element in a for item {} in b".format(i)
            )
        index = indices[0]
        a_remaining[index] = 0
        a_to_b[i] = index

    return a_to_b


def restore_correspondence(mesh, reference_mesh, progress=True):
    """
    Given `mesh` which has the same vertex set as a given `reference_mesh`, but
    which has lost its correspondence due to the vertices being scrambled,
    reorder the vertices in `mesh` so they match the order in `reference_mesh`.

    This was designed to assist in extracting face ordering and groups from a
    shuffled `mesh` that work on `reference_mesh` and may have other uses as
    well.

    Args:
        mesh (lace.mesh.Mesh): A mesh, which will be mutated
        reference_mesh (lace.mesh.Mesh): Another mesh with the same set of
            vertices in the desired order
        progress (bool): When `True`, show a progress bar.

    Returns:
        np.array: `vx1` mapping of old face indices to new

    Note:
        This relies on a brute-force algorithm.
    """
    v_old_to_new = find_permutation(reference_mesh.v, mesh.v, progress=progress)
    mesh.reorder_vertices(v_old_to_new)
    return v_old_to_new
