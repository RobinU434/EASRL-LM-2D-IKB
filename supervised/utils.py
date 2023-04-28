import torch
from typing import Tuple


def split_state_information(x: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """_summary_

    Args:
        x (torch.tensor): 

    Returns:
        _type_: _description_
    """
    target_pos = x[:, 0:2]
    current_pos = x[:, 2:4]
    angles = x[:, 4:]
    return target_pos, current_pos, angles


def forward_kinematics(angles: torch.tensor):
    """_summary_

    Args:
        angles (np.array): shape (num_arms, num_joints)

    Returns:
        _type_: _description_
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

