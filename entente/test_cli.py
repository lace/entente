import numpy as np
import meshlab_pickedpoints
import yaml
from click.testing import CliRunner
import vg
from .cli import cli
from .landmarks.test_landmarker import source_target_landmarks
from .landmarks.test_landmark_compositor import composite_landmark_examples


def test_transfer_landmarks_cli(tmp_path):
    source_mesh, target_mesh, landmarks, expected_landmarks = source_target_landmarks()

    source_mesh_path = str(tmp_path / "source.obj")
    landmark_path = str(tmp_path / "landmarks.pp")
    target_mesh_path = str(tmp_path / "target.obj")

    source_mesh.write_obj(source_mesh_path)
    with open(landmark_path, "w") as f:
        meshlab_pickedpoints.dump(landmarks, f)
    target_mesh.write_obj(target_mesh_path)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["transfer-landmarks", source_mesh_path, landmark_path, target_mesh_path],
        )
        assert result.exit_code == 0

        transferred = meshlab_pickedpoints.load("target.pp")
        np.testing.assert_array_equal(
            transferred["origin"], expected_landmarks["origin"]
        )
        np.testing.assert_array_equal(
            transferred["near_opposite_corner"],
            expected_landmarks["near_opposite_corner"],
        )


def test_composite_landmarks_cli(tmp_path):
    (
        base_mesh,
        example_mesh_1,
        near_origin_1,
        example_mesh_2,
        near_origin_2,
    ) = composite_landmark_examples()

    base_mesh_path = str(tmp_path / "base.obj")
    example_mesh_path_1 = str(tmp_path / "example1.obj")
    example_mesh_path_2 = str(tmp_path / "example2.obj")

    base_mesh.write_obj(base_mesh_path)
    example_mesh_1.write_obj(example_mesh_path_1)
    example_mesh_2.write_obj(example_mesh_path_2)

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

    recipe_path = str(tmp_path / "recipe.yml")
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["composite-landmarks", recipe_path])
        if result.exception:
            raise result.exception

        assert result.exit_code == 0

        with open("composite_result/landmarks.yml", "r") as f:
            result = yaml.safe_load(f)
        np.testing.assert_array_almost_equal(
            result["composited"]["near_origin"], np.zeros(3), decimal=2
        )


def test_composite_landmarks_cli_symmetrized(tmp_path):
    (
        base_mesh,
        example_mesh_1,
        near_origin_1,
        example_mesh_2,
        near_origin_2,
    ) = composite_landmark_examples()

    base_mesh_path = str(tmp_path / "base.obj")
    example_mesh_path_1 = str(tmp_path / "example1.obj")
    example_mesh_path_2 = str(tmp_path / "example2.obj")

    base_mesh.write_obj(base_mesh_path)
    example_mesh_1.write_obj(example_mesh_path_1)
    example_mesh_2.write_obj(example_mesh_path_2)

    bottom_left_1 = near_origin_1
    bottom_right_1 = np.array([5.22, -3.08, -3.1]).tolist()

    bottom_left_2 = near_origin_2
    bottom_right_2 = np.array([4.356, 3.9801, 4.0]).tolist()

    recipe = {
        "base_mesh": base_mesh_path,
        "decimals": 2,
        "landmarks": ["bottom_left", "bottom_right"],
        "symmetrize": {
            "reference_point": [0.5, 0.0, 0.0],
            "normal": vg.basis.x.tolist(),
        },
        "examples": [
            {
                "id": "example1",
                "mesh": example_mesh_path_1,
                "bottom_left": bottom_left_1,
                "bottom_right": bottom_right_1,
            },
            {
                "id": "example2",
                "mesh": example_mesh_path_2,
                "bottom_left": bottom_left_2,
                "bottom_right": bottom_right_2,
            },
        ],
    }

    recipe_path = str(tmp_path / "recipe.yml")
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["composite-landmarks", recipe_path])
        if result.exception:
            raise result.exception

        assert result.exit_code == 0

        with open("composite_result/landmarks.yml", "r") as f:
            result = yaml.safe_load(f)

        np.testing.assert_array_almost_equal(
            result["composited"]["bottom_left"], np.zeros(3), decimal=1
        )
        np.testing.assert_array_almost_equal(
            result["composited_and_symmetrized"]["bottom_left"], np.zeros(3), decimal=1
        )
