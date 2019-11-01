import numpy as np
from click.testing import CliRunner
from lace.serialization import meshlab_pickedpoints
from .cli import transfer_landmarks
from .landmarks.test_landmarker import source_target_landmarks


def test_cli(tmp_path):
    source_mesh, target_mesh, landmarks, expected_landmarks = source_target_landmarks()

    source_mesh_path = str(tmp_path / "source.obj")
    landmark_path = str(tmp_path / "landmarks.pp")
    target_mesh_path = str(tmp_path / "target.obj")

    source_mesh.write(source_mesh_path)
    meshlab_pickedpoints.dump(landmarks, landmark_path)
    target_mesh.write(target_mesh_path)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            transfer_landmarks, [source_mesh_path, landmark_path, target_mesh_path]
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
