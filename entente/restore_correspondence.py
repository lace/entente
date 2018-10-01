def _maybe_tqdm(iterable, progress):
    if progress:
        from tqdm import tqdm

        return tqdm(iterable)
    else:
        return iterable


def find_permutation(a, b, progress=True):
    """
    Given two permutations of identical elements `a` and `b`, return an array
    of the indices of `a` ordered such that `a[find_permutation(a, b)]` is
    equal to `b`.

    progress: When `True`, show a progress bar.

    This relies on a brute-force algorithm.
    """
    import numpy as np

    if not len(a) == len(b):
        raise

    a_remaining = np.ones(len(a), dtype=np.bool_)
    a_to_b = np.zeros(len(a), dtype=np.uint64)

    for i, item in _maybe_tqdm(enumerate(b), progress):
        indices, = np.nonzero(~(a - item).any(axis=1))
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
    Reorder the vertices in `mesh` to match the desired order in
    `reference_mesh`. The sets of vertices in the two meshes must be exactly
    identical. `reference_mesh` is not modified. Face ordering and groups in

    This relies on a brute-force algorithm.

    progress: When `True`, show a progress bar.

    Return a np array mapping from old vertex indices to new.
    """
    v_old_to_new = find_permutation(reference_mesh.v, mesh.v)
    mesh.reorder_vertices(v_old_to_new)
    return v_old_to_new
