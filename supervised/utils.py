from typing import Tuple

import torch
from torch import Tensor


def split_state_information(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """splits the state vector from the environment into its individual parts.

    Args:
        x (Tensor): state vector from the environment

    Returns:
        Tuple[Tensor, Tensor, Tensor]: target position, current position, state_angles
    """
    target_pos = x[:, 0:2]
    current_pos = x[:, 2:4]
    angles = x[:, 4:]
    return target_pos, current_pos, angles


def forward_kinematics(angles: torch.Tensor) -> torch.Tensor:
    """computes forward kinematics for a robot arm with segment length = 1

    Args:
        angles (np.array): shape (num_arms, num_joints)

    Returns:
        torch.Tensor: individual segment positions (num_arms, num_joints + 1, 2)
    """
    num_arms, num_joints = angles.size()
    positions = torch.zeros((num_arms, num_joints + 1, 2))

    for idx in range(num_joints):
        origin = positions[:, idx]

        # new position
        new_pos = torch.zeros((num_arms, 2))
        new_pos[:, 0] = torch.cos(angles[:, idx])
        new_pos[:, 1] = torch.sin(angles[:, idx])

        # translate position
        new_pos += origin

        positions[:, idx + 1] = new_pos

    return positions
