import pytest
import numpy as np
from polliwog import Plane
import vg
from ..test_symmetry import create_seat_and_arm_mesh
from .symmetrize_landmarks import (
    symmetrize_landmarks_using_plane,
    symmetrize_landmarks_using_topology,
)


def test_symmetrize_landmarks_using_plane():
    original = np.array([[-18.5657, 54.7161, -19.5649], [20.0896, 54.919, -19.5738]])
    symmetrized = symmetrize_landmarks_using_plane(Plane.yz, original)

    np.testing.assert_allclose(symmetrized, original, atol=1)

    mirrored = np.copy(original)
    mirrored[:, 0] = -mirrored[:, 0]

    np.testing.assert_allclose(np.flipud(symmetrized), mirrored, atol=1)

    distances_to_original = vg.euclidean_distance(symmetrized, original)
    distances_to_mirrored = vg.euclidean_distance(np.flipud(symmetrized), mirrored)
    np.testing.assert_allclose(distances_to_original, distances_to_mirrored, atol=1e-1)


def test_symmetrize_landmarks_using_plane_non_plane():
    original = np.array([[-18.5657, 54.7161, -19.5649], [20.0896, 54.919, -19.5738]])
    with pytest.raises(ValueError, match=r"plane_of_symmetry should be a Plane"):
        symmetrize_landmarks_using_plane("not_a_plane", original)


def test_symmetrize_landmarks_using_topology():
    mesh = create_seat_and_arm_mesh()
    original = np.array([[-18.5657, 54.7161, -19.5649], [20.0896, 54.919, -19.5738]])
    symmetrized = symmetrize_landmarks_using_topology(
        mesh, Plane.yz, original, atol=1e-1
    )

    np.testing.assert_allclose(symmetrized, original, atol=1)

    mirrored = np.copy(original)
    mirrored[:, 0] = -mirrored[:, 0]

    np.testing.assert_allclose(np.flipud(symmetrized), mirrored, atol=1)

    distances_to_original = vg.euclidean_distance(symmetrized, original)
    distances_to_mirrored = vg.euclidean_distance(np.flipud(symmetrized), mirrored)
    np.testing.assert_allclose(distances_to_original, distances_to_mirrored, atol=1e-1)


def test_symmetrize_landmarks_using_topology_asymmetrical():
    mesh = create_seat_and_arm_mesh().translated(np.array([50.0, 0.0, 0.0]))
    original = np.array([[-18.5657, 54.7161, -19.5649], [20.0896, 54.919, -19.5738]])
    with pytest.raises(
        ValueError, match=r"Some landmarks are near triangles which are not mirrored"
    ):
        symmetrize_landmarks_using_topology(mesh, Plane.yz, original, atol=1e-1)


def test_symmetrize_landmarks_using_topology_non_plane():
    mesh = create_seat_and_arm_mesh().translated(np.array([50.0, 0.0, 0.0]))
    original = np.array([[-18.5657, 54.7161, -19.5649], [20.0896, 54.919, -19.5738]])
    with pytest.raises(ValueError, match=r"plane_of_symmetry should be a Plane"):
        symmetrize_landmarks_using_topology(mesh, "not_a_plane", original, atol=1e-1)
