import torch


def scale_matrix(matrix: torch.tensor, batch_size):
    return matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    