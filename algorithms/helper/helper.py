from typing import Iterable

import numpy as np


def get_space_size(shape: Iterable) -> int:
    if len(shape) == 1:
        return shape[0]
    else:
        return np.multiply(*shape)