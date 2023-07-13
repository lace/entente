from entente.rigid_transform import find_rigid_rotation, find_rigid_transform
from env_flag import env_flag
import numpy as np
from polliwog import Box
from polliwog.transform import euler
import pytest


def test_rigid_transform_from_simple_translation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    b = Box(origin=np.array([1.0, 2.0, 3.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = np.eye(3)
    expected_t = np.array([[1.0, 2.0, 3.0]])
    R, t = find_rigid_transform(a, b)
    np.testing.assert_array_almost_equal(a.dot(expected_R) + expected_t, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_transform_from_simple_rotation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = euler([30, 15, 21])
    expected_t = np.array([[0, 0, 0]])
    b = a.dot(expected_R) + expected_t
    R, t = find_rigid_transform(a, b)
    np.testing.assert_array_almost_equal(a.dot(expected_R) + expected_t, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_transform_from_rotation_and_translation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = euler([30, 15, 21])
    expected_t = np.array([[1, 2, 3]])
    b = a.dot(expected_R) + expected_t
    R, t = find_rigid_transform(a, b)
    np.testing.assert_array_almost_equal(a.dot(expected_R) + expected_t, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_transform_from_rotation_translation_and_scale():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = euler([30, 15, 21])
    expected_t = np.array([[1, 2, 3]])
    expected_scale = 1.7
    b = expected_scale * a.dot(expected_R) + expected_t
    s, R, t = find_rigid_transform(a, b, compute_scale=True)
    np.testing.assert_array_almost_equal(s * a.dot(expected_R) + expected_t, b)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)
    np.testing.assert_almost_equal(s, expected_scale)


def test_rigid_transform_with_reflection():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    b = a * -1
    with pytest.raises(ValueError):
        find_rigid_transform(a, b)


def test_rigid_transform_with_reflection_but_solve_anyway():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    b = a * -1
    expected_R = np.eye(3)
    expected_R[0][0] = expected_R[1][1] = -1
    expected_t = np.array([[0.0, 0.0, -1.0]])
    R, t = find_rigid_transform(a, b, fail_in_degenerate_cases=False)
    np.testing.assert_array_almost_equal(R, expected_R)
    np.testing.assert_array_almost_equal(t, expected_t)


def test_rigid_rotation():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = euler([30, 15, 21])
    b = a.dot(expected_R)
    R = find_rigid_rotation(a, b)
    np.testing.assert_array_almost_equal(R, expected_R)


def test_rigid_rotation_single_point():
    a = np.array([[1, 2, 3]])
    expected_R = euler([30, 15, 21])
    b = a.dot(expected_R)
    R = find_rigid_rotation(a, b)
    # in this degenerate case, don't assert that we got the same
    # rotation, just that it is a valid solution
    np.testing.assert_array_almost_equal(a.dot(R), b)


@pytest.mark.skipif(env_flag("CI") is True, reason="failing in CI when numpy>=1.19.3")
# TODO: Clarify the intent of this test.
#
# This test produces different answers on numpy 1.19.3 and higher on certain architectures. This is
# because there are duplicate singular values.
#
# Since a reflection through the origin in 3-space is not a rotation at all, and there is no rotation
# matrix which can reflect through the origin, it's not clear what the correct result should be.
#
# https://github.com/numpy/numpy/issues/22945
# https://github.com/lace/entente/issues/195
def test_rigid_rotation_with_reflection():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    b = a * -1
    R = find_rigid_rotation(a, b)
    expected_R = np.zeros((3, 3))
    expected_R[0][0] = expected_R[1][2] = expected_R[2][1] = -1
    np.testing.assert_array_almost_equal(R, expected_R)


def test_numpy_regression():
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
    np.testing.assert_almost_equal(expected_a, a)
    np.testing.assert_almost_equal(expected_b, b)


def test_rigid_rotation_with_scale():
    a = Box(origin=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0])).v
    expected_R = euler([30, 15, 21])
    b = a.dot(expected_R)
    R = find_rigid_rotation(a, b, allow_scaling=True)
    np.testing.assert_array_almost_equal(R, expected_R)
