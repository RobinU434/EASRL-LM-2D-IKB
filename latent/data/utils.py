from enum import Enum
from typing import Tuple, Union

from numpy import ndarray
from torch import Tensor


class TargetMode(Enum):
    """class to define different target modes. Is the target information an action which can enable Imitation and Distance loss or only Position which enables only a distance loss"""

    UNDEFINED = 0
    ACTION = 1
    POSITION = 2
    INTERMEDIATE_POSITION = 3
    FINAL_POSITION = 4


def split_state_information(
    x: Union[Tensor, ndarray]
) -> Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray], Union[Tensor, ndarray],]:
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
