from entente.path_transfer import PathTransfer
import numpy as np
from polliwog import Polyline
from .test_surface_regressor import source_target_landmarks


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
