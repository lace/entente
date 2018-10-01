def find_correspondence(a, b):
    """
    Given two orderings of identical elements, return an array that maps
    indices of the first to the indices of the second.
    """
    import numpy as np

    if not len(a) == len(b):
        raise

    b_remaining = np.ones(len(b), dtype=np.bool_)
    for item in a:
        idx = np.logical_and(b_remaining, b == item)


def restore_correspondence():
    pass
