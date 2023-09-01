from abc import ABC, abstractmethod
import logging
from typing import Any

import torch
from torch import Tensor


class Criterion(ABC):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        logging.debug(f"Init {type(self).__name__}")
        self.loss = torch.inf
        self._device = device

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
       pass

    def __str__(self) -> str:
        return type(self).__name__