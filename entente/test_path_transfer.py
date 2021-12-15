from entente.path_transfer import PathTransfer
from lacecore import Mesh, shapes
import numpy as np
from polliwog import Polyline
import pytest
from .test_surface_regressor import source_target_landmarks

# A polyline, any polyline.
BOGUS_POLYLINE = Polyline(v=np.zeros(3).reshape(1, 3))


def test_path_transfer():
    (source_mesh, target_mesh, *_) = source_target_landmarks()

    is_closed = False
    source_path = Polyline(
        v=np.array([[0, 0, 0], [0.8, 0.9, 1.0], [1, 1, 1], [0, 0.5, 0.75]]),
        is_closed=is_closed,
    )

    path_transfer = PathTransfer(source_mesh=source_mesh, source_path=source_path)
    transferred = path_transfer.path_for(target_mesh)

    expected_target_path_v = np.array(
        [
            [0.0, 3.5, 1.0],
            [4.0, 8.0, 6.0],
            [5.0, 8.5, 6.0],
            [0, 6.0, 4.75],
        ]
    )
    np.testing.assert_array_equal(
        transferred.v,
        expected_target_path_v,
    )
    assert transferred.is_closed is is_closed


def test_path_transfer_wrong_topology():
    source_mesh = shapes.cube(np.zeros(3), 1.0)

    # Create a second mesh with a different topology.
    target_mesh = source_mesh.faces_flipped()

    path_transfer = PathTransfer(source_mesh, BOGUS_POLYLINE)

    with pytest.raises(ValueError, match="Target mesh must have the same topology"):
        path_transfer.path_for(target_mesh)


def test_path_transfer_quad():
    tri_mesh = shapes.cube(np.zeros(3), 1.0)
    quad_mesh = Mesh(v=np.zeros((0, 3)), f=np.zeros((0, 4), dtype=np.int64))

    path_transfer = PathTransfer(tri_mesh, BOGUS_POLYLINE)

    with pytest.raises(ValueError, match="Target mesh must be triangulated"):
        path_transfer.path_for(quad_mesh)
