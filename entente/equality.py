def attr_has_same_shape(first_obj, second_obj, attr):
    first, second = getattr(first_obj, attr), getattr(second_obj, attr)
    if first is None or second is None:
        return first is second
    else:
        return first.shape == second.shape

def attr_is_equal(first_obj, second_obj, attr):
    import numpy as np
    # Avoid comparing None's.
    return attr_has_same_shape(first_obj, second_obj, attr) and \
        np.array_equal(getattr(first_obj, attr), getattr(second_obj, attr))

def have_same_topology(first_mesh, second_mesh):
    return attr_has_same_shape(first_mesh, second_mesh, 'v') and \
        attr_is_equal(first_mesh, second_mesh, 'f')
