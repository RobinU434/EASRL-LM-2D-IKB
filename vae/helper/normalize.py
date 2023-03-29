import torch


def normalize(x: torch.tensor, mean: float, std: float):
    # normalize input values (mean and std extracted from ./data/distribution.ipynb)
    x = (x - mean) / std

    return x
