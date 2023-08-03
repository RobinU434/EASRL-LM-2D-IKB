from typing import Any, Literal, Tuple

from torch import Tensor
from abc import abstractmethod

from latent.datasets.datasets import StateActionDataset, TargetGaussianDataset
from latent.datasets.latent_dataset import LatentDataset
from latent.datasets.utils import TargetMode


class SupervisedDataset(LatentDataset):
    """Interface how to design the get item function"""

    def __init__(
        self, target_mode: Literal[TargetMode.ACTION, TargetMode.POSITION], **kwargs
    ) -> None:
        super().__init__(target_mode, **kwargs)

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """function provides input for supervised learning model with

        Args:
            index (int): index of the element you want to access

        Returns:
            Tuple[Tensor, Tensor]:
            (network input (x), ground truth for loss func (y))
        """
        pass


class SupervisedStateActionDataset(StateActionDataset, SupervisedDataset):
    def __init__(
        self,
        actions: Tensor,
        states: Tensor,
        target_mode: Literal[TargetMode.ACTION, TargetMode.POSITION],
        action_constrain_radius: float = 0,
        **kwargs
    ):
        super().__init__(
            actions, states, target_mode, action_constrain_radius, **kwargs
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self._states[idx]

        y = self._acquire_target_func(idx)

        return x, y


class SupervisedTargetGaussianDataset(TargetGaussianDataset, SupervisedDataset):
    def __init__(
        self,
        states: Tensor,
        std: float,
        truncation: float,
        target_mode: Literal[TargetMode.ACTION, TargetMode.POSITION],
        **kwargs
    ) -> None:
        super().__init__(states, std, truncation, target_mode, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self._states[idx]

        y = self._acquire_target_func(idx)

        return x, y
