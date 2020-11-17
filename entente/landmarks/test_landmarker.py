from lacecore import shapes, Mesh
import numpy as np
import meshlab_pickedpoints
import pytest
from .landmarker import Landmarker


def source_target_landmarks():
    source_mesh = shapes.cube(np.zeros(3), 1.0)
    target_mesh = (
        source_mesh.transform()
        .uniform_scale(5.0)
        .translate(np.array([0, 3.5, 1.0]))
        .end()
    )

    landmarks = {
        "origin": np.zeros(3),
        "near_opposite_corner": np.array([0.8, 0.9, 1.0]),
    }

    expected_landmarks = {
        "origin": np.array([0.0, 3.5, 1.0]),
        "near_opposite_corner": np.array([4.0, 8.0, 6.0]),
    }

    return source_mesh, target_mesh, landmarks, expected_landmarks


def test_landmarker(tmp_path):
    source_mesh, target_mesh, landmarks, expected_landmarks = source_target_landmarks()

    landmarker = Landmarker(source_mesh, landmarks)

    transferred = landmarker.transfer_landmarks_onto(target_mesh)
    np.testing.assert_array_equal(transferred["origin"], expected_landmarks["origin"])
    np.testing.assert_array_equal(
        transferred["near_opposite_corner"], expected_landmarks["near_opposite_corner"]
    )

    source_mesh_path = str(tmp_path / "source.obj")
    landmark_path = str(tmp_path / "landmarks.pp")

    source_mesh.write_obj(source_mesh_path)
    with open(landmark_path, "w") as f:
        meshlab_pickedpoints.dump(landmarks, f)

    landmarker = Landmarker.load(
        source_mesh_path=source_mesh_path, landmark_path=landmark_path
    )
    transferred = landmarker.transfer_landmarks_onto(target_mesh)
    np.testing.assert_array_equal(transferred["origin"], expected_landmarks["origin"])
    np.testing.assert_array_equal(
        transferred["near_opposite_corner"], expected_landmarks["near_opposite_corner"]
    )


def test_landmarker_wrong_topology():
    source_mesh = shapes.cube(np.zeros(3), 1.0)

    # Create a second mesh with a different topology.
    target_mesh = source_mesh.faces_flipped()

    # Landmarks are empty; we don't get that far.
    landmarker = Landmarker(source_mesh, {})

    with pytest.raises(ValueError, match="Target mesh must have the same topology"):
        landmarker.transfer_landmarks_onto(target_mesh)


def test_landmarker_quad():
    tri_mesh = shapes.cube(np.zeros(3), 1.0)
    quad_mesh = Mesh(v=np.zeros((0, 3)), f=np.zeros((0, 4)))

    with pytest.raises(ValueError, match="Source mesh should be triangulated"):
        # Landmarks are empty; we don't get that far.
        Landmarker(quad_mesh, {})

    landmarker = Landmarker(tri_mesh, {})
    with pytest.raises(ValueError, match="Target mesh must be triangulated"):
        landmarker.transfer_landmarks_onto(quad_mesh)
