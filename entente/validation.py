def validate_shape(a, *shape, **kwargs):
    """
    a: An array-like input.
    shape: Shape to validate. To require 3 by 1, pass (3,). To require n by 3,
      pass (-1, 3). Pass '*' for a wildcard dimension.
    name: Variable name to embed in the error message.

    Return: The wildcard dimension or a tuple of wildcard dimensions.
    """
    is_wildcard = lambda dim: dim == "*"
    if all(not isinstance(dim, int) and not is_wildcard(dim) for dim in shape):
        raise ValueError("Expected shape dimensions to be int or '*'")

    if "name" in kwargs:
        preamble = "{} must be an array".format(kwargs["name"])
    else:
        preamble = "Expected an array"

    if a is None:
        raise ValueError("{} with shape {}; got None".format(preamble, shape))
    try:
        len(a.shape)
    except (AttributeError, TypeError):
        raise ValueError(
            "{} with shape {}; got {}".format(preamble, shape, a.__class__)
        )

    # Check non-wildcard dimensions.
    if len(a.shape) != len(shape) or any(
        actual != expected
        for actual, expected in zip(a.shape, shape)
        if not is_wildcard(expected)
    ):
        raise ValueError("{} with shape {}; got {}".format(preamble, shape, a.shape))

    wildcard_dims = [
        actual for actual, expected in zip(a.shape, shape) if is_wildcard(expected)
    ]
    if len(wildcard_dims) == 0:
        return None
    elif len(wildcard_dims) == 1:
        return wildcard_dims[0]
    else:
        return tuple(wildcard_dims)


def validate_shape_from_ns(namespace, name, *shape):
    """
    Convenience function.
    Usage: validate_shape_from_namespace(locals(), 'points', '*', 3)
    """
    return validate_shape(namespace[name], *shape, name=name)
