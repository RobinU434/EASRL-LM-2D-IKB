from typing import Tuple
import torch


def extract_angles_and_position(array: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    angles = array[:, :-2]
    position = array[:, -2:]

    return angles, position