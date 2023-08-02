from typing import Iterable

import numpy as np
import torch


def get_space_size(shape: Iterable) -> int:
    if len(shape) == 1:
        return shape[0]
    else:
        return np.multiply(*shape)

def get_dim(size: torch.Size):
    if len(size) == 2:
        batch_size, n = size
    elif len(size) == 1:
        batch_size = 1
        n = size[0]
    else:
        raise ValueError("size has to be shorter or equal two dimensions.")
    
    return batch_size, n