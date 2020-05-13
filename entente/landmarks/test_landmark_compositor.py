from lacecore import shapes
import numpy as np
import pytest
from .landmark_compositor import LandmarkCompositor


def composite_landmark_examples():
    """
    Create three meshes in correspondence.
    """
    base_mesh = shapes.cube(np.zeros(3), 1.0)
    example_mesh_1 = shapes.cube(np.repeat(-3.0, 3), 8.0)
    near_origin_1 = [-2.98, -2.98, -3.0]
    example_mesh_2 = shapes.cube(np.repeat(4.0, 3), 0.3)
    near_origin_2 = [4.006, 4.001, 4.0]
    return base_mesh, example_mesh_1, near_origin_1, example_mesh_2, near_origin_2


def test_landmark_compositor():
    (
        base_mesh,
        example_mesh_1,
        near_origin_1,
        example_mesh_2,
        near_origin_2,
    ) = composite_landmark_examples()
    compositor = LandmarkCompositor(base_mesh=base_mesh, landmark_names=["near_origin"])
    compositor.add_example(example_mesh_1, {"near_origin": near_origin_1})
    compositor.add_example(example_mesh_2, {"near_origin": near_origin_2})
    np.testing.assert_array_almost_equal(
        compositor.result["near_origin"], np.zeros(3), decimal=2
    )


def test_landmark_compositor_error():
    (
        base_mesh,
        example_mesh_1,
        near_origin_1,
        example_mesh_2,
        near_origin_2,
    ) = composite_landmark_examples()
    compositor = LandmarkCompositor(base_mesh=base_mesh, landmark_names=["near_origin"])
    with pytest.raises(
        ValueError, match="Expected examples to contain keys near_origin"
    ):
        compositor.add_example(example_mesh_1, {"oops": near_origin_1})
