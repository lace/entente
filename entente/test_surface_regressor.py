from entente.surface_regressor import regressor_for, apply_regressor
import numpy as np
from lacecore import shapes
import pytest


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


def test_regressor_for():
    source_mesh, target_mesh, landmarks, expected_landmarks = source_target_landmarks()
    query_points = np.array(list(landmarks.values()))

    regressor = regressor_for(
        faces=source_mesh.f,
        source_mesh_vertices=source_mesh.v,
        query_points=query_points,
    )
    target_landmark_coords = apply_regressor(regressor, target_mesh.v)

    np.testing.assert_array_almost_equal(
        target_landmark_coords, np.array(list(expected_landmarks.values()))
    )


def test_apply_regressor_errors():
    source_mesh, target_mesh, landmarks, expected_landmarks = source_target_landmarks()
    query_points = np.array(list(landmarks.values()))

    regressor = regressor_for(
        faces=source_mesh.f,
        source_mesh_vertices=source_mesh.v,
        query_points=query_points,
    )

    vertices_but_not_enough = target_mesh.v[:3]
    with pytest.raises(ValueError, match="This regressor expects 8 vertices"):
        apply_regressor(regressor, vertices_but_not_enough)


# def test_surface_regressor():
#     source_mesh, target_mesh, landmarks, expected_landmarks = source_target_landmarks()

#     regressor =

#     regressor = SurfaceRegressor()


#     landmarker = Landmarker(source_mesh, landmarks)

#     transferred = landmarker.transfer_landmarks_onto(target_mesh)
#     np.testing.assert_array_equal(transferred["origin"], expected_landmarks["origin"])
#     np.testing.assert_array_equal(
#         transferred["near_opposite_corner"], expected_landmarks["near_opposite_corner"]
#     )

#     source_mesh_path = str(tmp_path / "source.obj")
#     landmark_path = str(tmp_path / "landmarks.json")

#     source_mesh.write_obj(source_mesh_path)
#     dump_landmarks(landmarks, landmark_path)

#     landmarker = Landmarker.load(
#         source_mesh_path=source_mesh_path, landmark_path=landmark_path
#     )
#     transferred = landmarker.transfer_landmarks_onto(target_mesh)
#     np.testing.assert_array_equal(transferred["origin"], expected_landmarks["origin"])
#     np.testing.assert_array_equal(
#         transferred["near_opposite_corner"], expected_landmarks["near_opposite_corner"]
#     )

#     pass
