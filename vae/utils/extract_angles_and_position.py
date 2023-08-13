import numpy as np
import torch
from typing import Tuple, Union


def split_conditional_info(
    array: torch.tensor,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """splits the encoder input vector into target_angles and state, where the state is containing: target position, state_position, state_angles

    Args:
        array (torch.tensor): input vector for encoder with shape: (batch_size, output_dim * 2 + 4)

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: target_angles, target_position, state_position, state_angles
    """
    out_dim = array.size()[1]
    split_idx = (
        out_dim - 4
    ) // 2  # assume conditional information contains (target, current_pos, current_angles)
    target_angles = array[:, :split_idx]
    state = array[:, split_idx:]
    target_position, state_position, state_angles = split_state_information(state)

    return target_angles, target_position, state_position, state_angles


def split_state_information(
    x: Union[torch.Tensor, np.ndarray]
) -> Tuple[
    Union[torch.Tensor, np.ndarray],
    Union[torch.Tensor, np.ndarray],
    Union[torch.Tensor, np.ndarray],
]:
    """the incoming tensor or ndarray must have 2 dimension [batch_size, state]

    Args:
        x (Union[torch.Tensor, np.ndarray]): object to be split with 2 dimensions [batch_size, state]

    Returns:
        Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]: target_position, current_position, state_angles
    """
    target_pos = x[:, 0:2]
    current_pos = x[:, 2:4]
    angles = x[:, 4:]
    return target_pos, current_pos, angles
