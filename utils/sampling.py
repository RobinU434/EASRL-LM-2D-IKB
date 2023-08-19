import numpy as np


def sample_target(radius: float, num_samples: int = 1) -> np.ndarray:
    """
    sample goal position in a circular shape around the origin
    radius and angle is sampled uniformly

    Args:
        radius (float): maximum radius to sample fromm

    Returns:
        np.array: shape (num_samples, 2) if  n_point == 1 -> shape: (2)
    """
    # angle to sample from
    theta = np.random.uniform(0, 2 * np.pi, size=num_samples)
    radius_array = np.random.uniform(0, radius, size=num_samples)
    vector = radius_array * np.array([np.cos(theta), np.sin(theta)])
    return vector.squeeze().T