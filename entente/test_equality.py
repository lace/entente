from lacecore import shapes
import numpy as np
from .equality import have_same_topology


def test_have_same_topology():

    cube_1 = shapes.cube(np.zeros(3), 1.0)
    cube_2 = shapes.cube(np.zeros(3), 1.0)
    assert have_same_topology(cube_1, cube_2) is True

    cube_1 = shapes.cube(np.zeros(3), 1.0)
    cube_2 = shapes.cube(np.ones(3), 1.0)
    assert have_same_topology(cube_1, cube_2) is True

    cube_1 = shapes.cube(np.zeros(3), 1.0)
    cube_2 = shapes.cube(np.zeros(3), 1.0)
    cube_2.f = np.roll(cube_2.f, 1, axis=1)
    assert have_same_topology(cube_1, cube_2) is False

    cube_1 = shapes.cube(np.zeros(3), 1.0)
    cube_2 = shapes.cube(np.zeros(3), 1.0)
    cube_2.f = np.zeros((0, 3), dtype=np.uint64)
    assert have_same_topology(cube_1, cube_2) is False
