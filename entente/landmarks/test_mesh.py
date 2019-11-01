import numpy as np
from lace.shapes import create_cube
from ._mesh import add_landmark_points


def test_add_landmark_points():
    mesh = create_cube(np.zeros(3), 1.0)
    add_landmark_points(
        mesh, np.array([[0.25, 0.25, 0.0], [0.75, 0.75, 0.0]]), radius=0.1
    )
    added_segments = mesh.v[mesh.e]
    np.testing.assert_array_equal(
        added_segments,
        np.array(
            [
                [[0.35, 0.25, 0.0], [0.15, 0.25, 0.0]],
                [[0.25, 0.35, 0.0], [0.25, 0.15, 0.0]],
                [[0.25, 0.25, 0.1], [0.25, 0.25, -0.1]],
                [[0.85, 0.75, 0.0], [0.65, 0.75, 0.0]],
                [[0.75, 0.85, 0.0], [0.75, 0.65, 0.0]],
                [[0.75, 0.75, 0.1], [0.75, 0.75, -0.1]],
            ]
        ),
    )
