from typing import Tuple
import torch


def split_conditional_info(array: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    out_dim = array.size()[1]
    split_idx = (out_dim - 4) // 2  # assume conditional information contains (target, current_pos, current_angles)
    angles = array[:, :split_idx]
    position = array[:, split_idx:]

    return angles, position