import numpy as np
from envs.common.sample_target import sample_target

def test_shape():
    point = sample_target(1, 1)
    assert point.shape == (2,)

    point = sample_target(1, 4)
    assert point.shape == (4, 2)


def test_radius():
    n_points = 100_000
    points = sample_target(5, n_points)

    norm = np.linalg.norm(points, axis=1)

    assert norm.all() <= 5

