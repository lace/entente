def _maybe_tqdm(iterable, progress):
    if progress:
        from tqdm import tqdm

        return tqdm(iterable)
    else:
        return iterable


def find_correspondence(
    a, b, atol=1e-4, allow_unmatched=False, ret_unmatched_b=False, progress=True
):
    """
    Given a `kxn` array `a[0], a[1], ...` and `jxn` array `b[0], b[1], ...`,
    for each element in `a`, find the index of `b` with the corresponding
    element.

    When `allow_unmatched` is `True`, return an index of `-1` for elements in
    `a` having no match in `b`. When `False`, the default, `a` and `b` must
    contain the same set of elements and `b[find_correspondence(a, b)]` will
    equal `a`.

    Args:
        a (np.arraylike): `kxn` array.
        b (np.arraylike): `jxn` array.
        atol (float): Match tolerance.
        allow_unmatched (bool): When `True`
        ret_unmatched_b (bool): When `True`, return a tuple which also contains
            the indices of `b` which were not matched.
        progress (bool): When `True`, show a progress bar.

    Return:
        np.ndarray: Indices of `b` as `kx1`

    Note:
        This relies on a brute-force algorithm.

        For the interpretation of `atol`, see documentation for `np.isclose`.
    """
    import numpy as np

    if not len(a) == len(b) and not allow_unmatched:
        raise ValueError("a and b do not contain the same number of elements")

    a_to_b = np.repeat(-1, len(a))
    b_matched = np.zeros(len(b), dtype=np.bool_)

    for a_index, item in _maybe_tqdm(enumerate(a), progress):
        indices, = np.nonzero(np.all(np.isclose(b, item, atol=atol), axis=1))
        if len(indices) >= 1:
            b_index = indices[0]
            b_matched[b_index] = True
            a_to_b[a_index] = b_index
        elif not allow_unmatched:
            raise ValueError(
                "Couldn't find corresponding element in b for item {} in a".format(
                    a_index
                )
            )

    if ret_unmatched_b:
        unmatched_b, = np.where(b_matched == 0)
        return a_to_b, unmatched_b
    else:
        return a_to_b


def restore_correspondence(shuffled_mesh, reference_mesh, atol=1e-4, progress=True):
    """
    Given a reference mesh, reorder the vertices of a shuffled copy to restore
    correspondence with the reference mesh. The vertex set of the shuffled
    mesh and reference mesh must be the same within `atol`. Mutates
    `reference_mesh`. Faces are preserved but ignored.

    Args:
        reference_mesh (lace.mesh.Mesh): A mesh with the vertices in the
            desired order.
        shuffled_mesh (lace.mesh.Mesh): A mesh with the same vertex set as
            `reference_mesh`.
        progress (bool): When `True`, show a progress bar.

    Returns:
        np.ndarray: `vx1` which maps old vertices in `shuffled_mesh` to new.

    Note:
        This was designed to assist in extracting face ordering and groups from a
        shuffled `mesh` that "work" with `reference_mesh`.

        It relies on a brute-force algorithm.
    """
    v_old_to_new = find_correspondence(
        shuffled_mesh.v, reference_mesh.v, atol=atol, progress=progress
    )
    shuffled_mesh.reorder_vertices(v_old_to_new)
    return v_old_to_new
