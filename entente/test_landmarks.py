import numpy as np
from .landmarks import Landmarker


def test_landmarker():
    from lace.shapes import create_cube

    cube_1 = create_cube(np.zeros(3), 1.0)

    cube_2 = create_cube(np.zeros(3), 5.0)
    cube_2.translate(np.array([0, 3.5, 1.0]))

    landmarks = {
        "origin": np.zeros(3),
        "near_opposite_corner": np.array([0.8, 0.9, 1.0]),
    }

    landmarker = Landmarker(cube_1, landmarks)

    transferred = landmarker.transfer_landmarks_onto(cube_2)
    expected = {
        "origin": np.array([0.0, 3.5, 1.0]),
        "near_opposite_corner": np.array([4.0, 8.0, 6.0]),
    }
    np.testing.assert_array_equal(transferred["origin"], expected["origin"])
    np.testing.assert_array_equal(
        transferred["near_opposite_corner"], expected["near_opposite_corner"]
    )
