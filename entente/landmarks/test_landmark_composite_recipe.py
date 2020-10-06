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
        "decimals": 2,
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

    base_mesh.write_obj(base_mesh_path)
    example_mesh_1.write_obj(example_mesh_path_1)
    example_mesh_2.write_obj(example_mesh_path_2)

    return recipe


def test_landmark_compositor(tmp_path):
    recipe = write_recipe_assets(tmp_path)

    recipe = LandmarkCompositeRecipe(recipe)
    np.testing.assert_array_almost_equal(
        recipe.composite_landmarks["near_origin"], np.zeros(3), decimal=2
    )

    (
        _,
        example_mesh_1,
        near_origin_1,
        example_mesh_2,
        near_origin_2,
    ) = composite_landmark_examples()
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

    original_and_reprojected_landmarks = recipe.original_and_reprojected_landmarks
    assert set(original_and_reprojected_landmarks.keys()) == set(
        ["example1", "example2"]
    )

    result1 = original_and_reprojected_landmarks["example1"]["near_origin"]
    np.testing.assert_array_equal(result1["original"], near_origin_1)
    np.testing.assert_array_almost_equal(
        result1["reprojected"],
        example_mesh_1.v[0],
        decimal=1,
    )

    result2 = original_and_reprojected_landmarks["example2"]["near_origin"]
    np.testing.assert_array_equal(result2["original"], near_origin_2)
    np.testing.assert_array_almost_equal(
        result2["reprojected"],
        example_mesh_2.v[0],
        decimal=1,
    )
