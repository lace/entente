from .equality import have_same_topology


def test_have_same_topology():
    import numpy as np
    from lace.shapes import create_cube

    cube_1 = create_cube(np.zeros(3), 1.0)
    cube_2 = create_cube(np.zeros(3), 1.0)
    assert have_same_topology(cube_1, cube_2) is True

    cube_1 = create_cube(np.zeros(3), 1.0)
    cube_2 = create_cube(np.ones(3), 1.0)
    assert have_same_topology(cube_1, cube_2) is True

    cube_1 = create_cube(np.zeros(3), 1.0)
    cube_2 = create_cube(np.zeros(3), 1.0)
    cube_2.f = np.roll(cube_2.f, 1, axis=1)
    assert have_same_topology(cube_1, cube_2) is False

    cube_1 = create_cube(np.zeros(3), 1.0)
    cube_2 = create_cube(np.zeros(3), 1.0)
    del cube_2.f
    assert have_same_topology(cube_1, cube_2) is False
