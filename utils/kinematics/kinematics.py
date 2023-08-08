import logging
import torch
from torch import Tensor

def forward_kinematics(angles: Tensor) -> Tensor:
    """computes forward kinematics for a robot arm with segment length = 1

    Args:
        angles (Tensor): shape (num_arms, n_joints)

    Returns:
        Tensor: individual segment positions in 2D coordinates (num_arms, n_joints + 1, 2)
    """
    num_arms, n_joints = angles.size()
    positions = torch.zeros((num_arms, n_joints + 1, 2))

    for idx in range(n_joints):
        origin = positions[:, idx]

        # new position
        new_pos = torch.zeros((num_arms, 2))
        new_pos[:, 0] = torch.cos(angles[:, idx])
        new_pos[:, 1] = torch.sin(angles[:, idx])

        # translate position
        new_pos += origin

        positions[:, idx + 1] = new_pos

    return positions