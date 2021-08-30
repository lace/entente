import lacecore
import numpy as np
from vg.compat import v1 as vg


def _maybe_tqdm(iterable, progress):
    if progress:
        from tqdm import tqdm

        return tqdm(iterable)
    else:
        return iterable


def find_correspondence(
    a, b, atol=1e-4, all_must_match=True, ret_unmatched_b=False, progress=True
):
    """
    Given `a[0], a[1], ..., a[k]` and `b[0], b[1], ..., b[j]`, match each element
    of `a` to the corresponding element of `b`.

    If `a` contains elements which do not exist in `b` and
    `all_must_match=False`, return an index of `-1` for the unmatched elements
    in `a`.

    If `a` contains elements which do not exist in `b` and
    `all_must_match=True`, raise an error. With `all_must_match=True`, unless
    either contains duplicates, `b` is a shuffled copy of `a`, and
    `b[find_correspondence(a, b)]` equals `a`.

    Args:
        a (np.arraylike): `kxn` array.
        b (np.arraylike): `jxn` array.
        atol (float): Match tolerance.
        all_must_match (bool): When `True`, `a` and `b` must contain the
            same elements.
        ret_unmatched_b (bool): When `True`, return a tuple which also contains
            the indices of `b` which were not matched.
        progress (bool): When `True`, show a progress bar.

    Return:
        np.ndarray: Indices of `b` with shape `(k,)`.

    Note:
        This relies on a brute-force algorithm.

        For the interpretation of `atol`, see documentation for `np.isclose`.
    """
    if all_must_match and len(a) != len(b):
        raise ValueError("a and b do not contain the same number of elements")

    a_to_b = np.repeat(-1, len(a))
    b_matched = np.zeros(len(b), dtype=np.bool_)

    for a_index, item in _maybe_tqdm(enumerate(a), progress):
        (indices,) = np.nonzero(np.all(np.isclose(b, item, atol=atol), axis=1))
        if len(indices) >= 1:
            if len(indices) > 1:
                closest_index = np.argmin(vg.euclidean_distance(b[indices], item))
                b_index = indices[closest_index]
            else:
                b_index = indices[0]
            b_matched[b_index] = True
            a_to_b[a_index] = b_index
        elif all_must_match:
            raise ValueError(
                f"Couldn't find corresponding element in b for item {a_index} in a"
            )

    if ret_unmatched_b:
        (unmatched_b,) = np.where(b_matched == 0)
        return a_to_b, unmatched_b
    else:
        return a_to_b


def restore_correspondence(shuffled_mesh, reference_mesh, atol=1e-4, progress=True):
    """
    Given a reference mesh, reorder the vertices of a shuffled copy to restore
    correspondence with the reference mesh. The vertex set of the shuffled
    mesh and reference mesh must be equal within `atol`. Mutate
    `reference_mesh`. Ignore faces but preserves their integrity.

    Args:
        reference_mesh (lacecore.Mesh): A mesh with the vertices in the
            desired order.
        shuffled_mesh (lacecore.Mesh): A mesh with the same vertex set as
            `reference_mesh`.
        progress (bool): When `True`, show a progress bar.

    Returns:
        tuple: The reordered mesh, and an array which maps old vertices in
        `shuffled_mesh` to new.

    Note:
        This was designed to assist in extracting face ordering and groups from a
        `shuffled_mesh` that "work" with `reference_mesh`, such that face
        ordering and groups can be used on shuffled sets of vertices.

        It relies on a brute-force algorithm.
    """
    v_old_to_new = find_correspondence(
        shuffled_mesh.v, reference_mesh.v, atol=atol, progress=progress
    )
    ordering = np.argsort(v_old_to_new)
    new_mesh = lacecore.reindex_vertices(shuffled_mesh, ordering)
    return new_mesh, v_old_to_new
