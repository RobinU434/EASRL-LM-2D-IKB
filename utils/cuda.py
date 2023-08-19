

from typing import Any, Dict

from torch import Tensor


def dict_to_device(tensor_dict: Dict[Any, Tensor], device: str) -> Dict[Any, Tensor]:
    """sends given dict to device

    Args:
        tensor_dict (Dict[Any, Tensor]): dict with tensors as values
        device (str): device to send to

    Returns:
        Dict[Any, Tensor]: dict with tensors on requested device
    """
    device_dict = {}
    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            device_dict[k] = dict_to_device(v, device)
        elif isinstance(v, Tensor):
            device_dict[k] = v.to(device)
    return device_dict