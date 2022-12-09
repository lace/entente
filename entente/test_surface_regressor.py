from entente.surface_regressor import apply_surface_regressor, surface_regressor_for
from lacecore import shapes
import numpy as np
import pytest


def source_target_landmarks():
    source_mesh = shapes.cube(np.zeros(3), 1.0)
    target_mesh = (
        source_mesh.transform()
        .uniform_scale(5.0)
        .translate(np.array([0, 3.5, 1.0]))
        .end()
    )

    landmarks = [
        {"name": "origin", "point": [0.0, 0.0, 0.0]},
        {"name": "near_opposite_corner", "point": [0.8, 0.9, 1.0]},
    ]

    expected_landmarks = [
        {"name": "origin", "point": [0.0, 3.5, 1.0]},
        {"name": "near_opposite_corner", "point": [4.0, 8.0, 6.0]},
    ]

    return source_mesh, target_mesh, landmarks, expected_landmarks


def test_surface_regressor_for():
    source_mesh, target_mesh, landmarks, expected_landmarks = source_target_landmarks()
    query_points = np.array([point["point"] for point in landmarks])

    regressor = surface_regressor_for(
        faces=source_mesh.f,
        source_mesh_vertices=source_mesh.v,
        query_points=query_points,
    )
    target_landmark_coords = apply_surface_regressor(regressor, target_mesh.v)

    np.testing.assert_array_almost_equal(
        target_landmark_coords,
        np.array([point["point"] for point in expected_landmarks]),
    )


def test_apply_surface_regressor_errors():
    source_mesh, target_mesh, landmarks, _ = source_target_landmarks()
    query_points = np.array(list(landmarks.values()))

    regressor = surface_regressor_for(
        faces=source_mesh.f,
        source_mesh_vertices=source_mesh.v,
        query_points=query_points,
    )

    vertices_but_not_enough = target_mesh.v[:3]
    with pytest.raises(ValueError, match="This regressor expects 8 vertices"):
        apply_surface_regressor(regressor, vertices_but_not_enough)
