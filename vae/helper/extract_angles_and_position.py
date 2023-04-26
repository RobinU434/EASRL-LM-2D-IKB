import torch
from typing import Tuple


def split_conditional_info(array: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """splits the encoder input vector into target_angles and state, where the state is containing: target position, state_position, state_angles

    Args:
        array (torch.tensor): input vector for encoder with shape: (batch_size, output_dim * 2 + 4)

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: target_angles, target_position, state_position, state_angles
    """
    out_dim = array.size()[1]
    split_idx = (out_dim - 4) // 2  # assume conditional information contains (target, current_pos, current_angles)
    target_angles = array[:, :split_idx]
    state = array[:, split_idx:]
    target_position, state_position, state_angles = split_state_information(state)
    
    return target_angles, target_position, state_position, state_angles


def split_state_information(x: torch.tensor):
    target_pos = x[:, 0:2]
    current_pos = x[:, 2:4]
    angles = x[:, 4:]
    return target_pos, current_pos, angles