import numpy as np
from .landmark_composite_recipe import LandmarkCompositeRecipe
from .test_landmark_compositor import composite_landmark_examples


def write_recipe_assets(relative_to):
    (
        base_mesh,
        example_mesh_1,
        near_origin_1,
        example_mesh_2,
        near_origin_2,
    ) = composite_landmark_examples()

    base_mesh_path = str(relative_to / "base.obj")
    example_mesh_path_1 = str(relative_to / "example1.obj")
    example_mesh_path_2 = str(relative_to / "example2.obj")

    recipe = {
        "base_mesh": base_mesh_path,
        "landmarks": ["near_origin"],
        "examples": [
            # Define a "near origin" point close to the origin of each cube.
            {
                "id": "example1",
                "mesh": example_mesh_path_1,
                "near_origin": near_origin_1,
            },
            {
                "id": "example2",
                "mesh": example_mesh_path_2,
                "near_origin": near_origin_2,
            },
        ],
    }

    base_mesh.write(base_mesh_path)
    example_mesh_1.write(example_mesh_path_1)
    example_mesh_2.write(example_mesh_path_2)

    return recipe


def test_landmark_compositor(tmp_path):
    recipe = write_recipe_assets(tmp_path)

    recipe = LandmarkCompositeRecipe(recipe)
    np.testing.assert_array_almost_equal(
        recipe.composite_landmarks["near_origin"], np.zeros(3), decimal=2
    )

    _, example_mesh_1, _, example_mesh_2, _ = composite_landmark_examples()
    np.testing.assert_array_almost_equal(
        recipe.reprojected_landmarks["example1"]["near_origin"],
        example_mesh_1.v[0],
        decimal=1,
    )
    np.testing.assert_array_almost_equal(
        recipe.reprojected_landmarks["example2"]["near_origin"],
        example_mesh_2.v[0],
        decimal=2,
    )
