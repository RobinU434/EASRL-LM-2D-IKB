from abc import ABC, abstractmethod
from typing import Literal, Tuple

import torch
from torch import Tensor

from latent.datasets.datasets import (
    ActionDataset,
    StateActionDataset,
    TargetGaussianDataset,
)
from latent.datasets.latent_dataset import LatentDataset
from latent.datasets.utils import TargetMode, split_state_information


class VAEDataset(LatentDataset, ABC):
    """Interface how to design the get item function and present dimension of conditional dimensions"""

    def __init__(
        self, target_mode: Literal[TargetMode.ACTION, TargetMode.POSITION], **kwargs
    ) -> None:
        super().__init__(target_mode, **kwargs)

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """function provides input for conditional variational autoencoder

        Args:
            index (int): index of the element you want to access

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
            (encoder input (x), conditional encoder input (c_enc), conditional decoder input (c_dec), ground truth for loss func (y))
        """
        pass

    @property
    def conditional_dim(self) -> Tuple[int, int]:
        """returns the dimension for the conditional input you want to encode

        Returns:
            Tuple[int, int]: first: conditional dim for encoder, second: conditional dim for decoder
        """
        _, c_enc, c_dec, _ = self[0]
        return len(c_enc), len(c_dec)


class VAEActionDataset(ActionDataset, VAEDataset):
    def __init__(
        self,
        target_mode: Literal[TargetMode.ACTION, TargetMode.POSITION],
        actions: Tensor,
        **kwargs,
    ) -> None:
        super().__init__(target_mode, actions, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self._actions[index]

        c_enc = torch.tensor([])
        c_dec = torch.tensor([])

        y = self._actions[index]

        return x, c_enc, c_dec, y


class VAEStateActionDataset(StateActionDataset, VAEDataset):
    def __init__(
        self,
        actions: Tensor,
        states: Tensor,
        target_mode: Literal[TargetMode.ACTION, TargetMode.POSITION],
        action_constrain_radius: float = 0,
    ):
        super().__init__(actions, states, target_mode, action_constrain_radius)

    def __getitem__(self, idx: int):
        target, current_position, state_angles = split_state_information(self._states)

        x = target[idx]

        c_enc = self._states[idx, 2:]
        c_dec = self._states[idx, 2:]
        print(self._states[idx])
        y = self._acquire_target_func(idx)

        return x, c_enc, c_dec, y


class VAETargetGaussianDataset(TargetGaussianDataset, VAEDataset):
    def __init__(
        self,
        states: Tensor,
        std: float,
        truncation: float,
        target_mode: Literal[TargetMode.ACTION, TargetMode.POSITION],
        **kwargs,
    ) -> None:
        super().__init__(states, std, truncation, target_mode, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self._states[idx, :2]

        c_enc = self._states[idx, 2:]
        c_dec = self._states[idx, 2:]
        y = self._acquire_target_func(idx)

        return x, c_enc, c_dec, y
