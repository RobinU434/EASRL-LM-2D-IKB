from abc import ABC, abstractmethod, abstractclassmethod
import logging
from typing import Any, Callable, Literal
from torch import Tensor
import torch
from torch.utils.data import Dataset

from latent.datasets.utils import TargetMode


class LatentDataset(Dataset, ABC):
    def __init__(
        self, target_mode: Literal[TargetMode.ACTION, TargetMode.POSITION], **kwargs
    ) -> None:
        super().__init__()

        self._target_mode = target_mode
        self._actions: Tensor
        """Tensor: with all actions which are leading from state to target inside, Shape: (num_actions, n_joints)"""
        self._states: Tensor
        """Tensor: with all states inside. Shape: (num_actions, 4 + n_joints)"""

    @abstractmethod
    def __len__(self):
        pass

    @abstractclassmethod
    def from_files(cls, file_path):
        pass

    def mean(self):
        raise NotImplementedError

    def std(self):
        raise NotImplementedError
    
    def _get_acquire_target_func(self) -> Callable[[int], Tensor]: 
        if self._target_mode == TargetMode.ACTION:
            return self.get_action
        elif self._target_mode == TargetMode.POSITION:
            return self.get_target
        else:
            logging.error(f"{self._target_mode=} not set correctly. Allowed are ACTION, POSITION")
            return lambda x: torch.zeros(0)
        
    def get_action(self, idx: int):
        return self._actions[idx]
    
    def get_target(self, idx: int):
        return self._states[idx][:2]

    @property
    def target_mode(self) -> TargetMode:
        return self._target_mode

    @property
    def input_dim(self) -> int:
        """returns the input dimension for the CVAE with out taking the conditional information into account

        Returns:
            int: input dimension
        """
        xy = self[0]
        return len(xy[0])