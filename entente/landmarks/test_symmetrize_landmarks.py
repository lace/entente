import pytest
import numpy as np
import vg
from ..test_symmetry import create_seat_and_arm_mesh
from .symmetrize_landmarks import symmetrize_landmarks


def test_symmetrize_landamrks():
    mesh = create_seat_and_arm_mesh()
    original = np.array([[-18.5657, 54.7161, -19.5649], [20.0896, 54.919, -19.5738]])
    symmetrized = symmetrize_landmarks(mesh, original, atol=1e-1)

    np.testing.assert_allclose(symmetrized, original, atol=1)

    mirrored = np.copy(original)
    mirrored[:, 0] = -mirrored[:, 0]

    np.testing.assert_allclose(np.flipud(symmetrized), mirrored, atol=1)

    distances_to_original = vg.euclidean_distance(symmetrized, original)
    distances_to_mirrored = vg.euclidean_distance(np.flipud(symmetrized), mirrored)
    np.testing.assert_allclose(distances_to_original, distances_to_mirrored, atol=1e-1)

def test_symmetrize_landamrks_asymmetrical():
    mesh = create_seat_and_arm_mesh()
    original = np.array([[-18.5657, 54.7161, -19.5649], [20.0896, 54.919, -19.5738]])
    mesh.translate(np.array([50.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match=r"Some landmarks are near triangles which are not mirrored"):
        symmetrize_landmarks(mesh, original, atol=1e-1)