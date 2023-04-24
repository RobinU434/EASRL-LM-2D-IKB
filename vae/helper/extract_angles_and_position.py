from typing import Tuple
import torch


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
    target_position = state[:, :2]
    state_position = state[:, 2:4]
    state_angles = state[:, 4:]

    return target_angles, target_position, state_position, state_angles