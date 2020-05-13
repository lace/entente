"""
Utilities related to mesh equality.
"""


def attr_has_same_shape(first_obj, second_obj, attr):
    """
    Given two objects, check if the given arraylike attributes of those
    objects have the same shape. If one object has an attribute value of
    ``None``, the other must too.

    Args:
        first_obj (obj): A object with an arraylike ``attr`` attribute.
        second_obj (obj): Another object with an arraylike ``attr`` attribute.
        attr (str): The name of the attribute to test.

    Returns:
        bool: `True` if attributes are the same shape
    """
    # Support legacy `lace.mesh.Mesh`, where attrs are set to `None` instead
    # of empty arrays.
    first, second = getattr(first_obj, attr), getattr(second_obj, attr)
    if first is None or second is None:
        return first is second
    else:
        return first.shape == second.shape


def attr_is_equal(first_obj, second_obj, attr):
    """
    Given two objects, check if the given arraylike attributes of those
    objects are equal. If one object has an attribute value of ``None``, the
    other must too.

    Args:
        first_obj (obj): A object with an arraylike `attr` attribute.
        second_obj (obj): Another object with an arraylike `attr` attribute.
        attr (str): The name of the attribute to test.

    Returns:
        bool: `True` if attributes are equal
    """
    import numpy as np

    # Avoid comparing None's.
    return attr_has_same_shape(first_obj, second_obj, attr) and np.array_equal(
        getattr(first_obj, attr), getattr(second_obj, attr)
    )


def have_same_topology(first_mesh, second_mesh):
    """
    Given two meshes, check if they have the same vertex count and same faces.
    In other words, check if they have the same topology.

    Args:
        first_mesh (lacecore.Mesh): A mesh.
        second_mesh (lacecore.Mesh): Another mesh.

    Returns:
        bool: `True` if meshes have the same topology
    """
    return attr_has_same_shape(first_mesh, second_mesh, "v") and attr_is_equal(
        first_mesh, second_mesh, "f"
    )
