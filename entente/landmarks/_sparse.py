def sparse(row_indices, column_indices, data, num_rows, num_columns):
    # Adapted from blmath.
    import numpy as np
    from scipy.sparse import csc_matrix

    indices = np.vstack(
        (row_indices.flatten().reshape(1, -1), column_indices.flatten().reshape(1, -1))
    )

    return csc_matrix((data, indices), shape=(num_rows, num_columns))
