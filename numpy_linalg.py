import numpy as np


def numpy_regression():
    array = np.array([[-4.0, -2.0, -2.0], [-2.0, -4.0, -2.0], [-2.0, -2.0, -4.0]])
    expected_a = np.array(
        [
            [-5.77350269e-01, 8.16496581e-01, -5.04179082e-17],
            [-5.77350269e-01, -4.08248290e-01, -7.07106781e-01],
            [-5.77350269e-01, -4.08248290e-01, 7.07106781e-01],
        ]
    )
    expected_b = np.array(
        [
            [0.57735027, 0.57735027, 0.57735027],
            [-0.81649658, 0.40824829, 0.40824829],
            [-0.0, 0.70710678, -0.70710678],
        ]
    )
    (a, _, b) = np.linalg.svd(array, full_matrices=False)
    print(np.testing.assert_almost_equal(expected_a, a))
    print(np.testing.assert_almost_equal(expected_b, b))
