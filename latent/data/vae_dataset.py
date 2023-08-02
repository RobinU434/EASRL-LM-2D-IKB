from abc import ABC, abstractmethod
import torch
import logging
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from progress.bar import Bar
from typing import Any, Tuple
from enum import Enum
from envs.robots.ccd import IK

from latent.data.utils import TargetMode, split_state_information


class VAEDataset(Dataset, ABC):
    """Base class for VAE datasets

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()

        self._target_mode= TargetMode.UNDEFINED

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """function provides input for conditional variational autoencoder

        Args:
            index (int): index of the element you want to access

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            (encoder input (x), conditional encoder input (c_enc), conditional decoder input (c_dec), ground truth for loss func (y))
        """
        return super().__getitem__(index)

    @property
    def conditional_dim(self) -> Tuple[int, int]:
        """returns the dimension for the conditional input you want to encode

        Returns:
            Tuple[int, int]: first: conditional dim for encoder, second: conditional dim for decoder
        """
        _, c_enc, c_dec, _ = self[0]
        return len(c_enc), len(c_dec)

    @property
    def input_dim(self) -> int:
        """returns the input dimension for the CVAE with out taking the conditional information into account

        Returns:
            int: input dimension
        """
        x, _, _, _ = self[0]
        return len(x)
    
    @property
    def target_mode(self) -> TargetMode:
        return self._target_mode

    @target_mode.setter
    def target_mode(self, value: TargetMode):
        if not isinstance(value, TargetMode):
            logging.warning(
                f"no change in target_mode because of value error. Demanded: {type(TargetMode)}, given: {type(value)}"
            )
            logging.info(f"target_mode remains at {self._target_mode}")
            return
        if value == TargetMode.UNDEFINED:
            logging.warning("value == TargetMode.UNDEFINED is not allowed")
            logging.info(f"target_mode remains at {self._target_mode}")
            return
        self._target_mode = value


class ActionDataset(VAEDataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    Therefore we need no label which is in this case an empty tensor
    """

    def __init__(self, annotations_file: str):
        super().__init__()
        self.csv = pd.read_csv(annotations_file)

        self._target_mode = TargetMode.ACTION

    def __len__(self):
        return len(self.csv)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.csv.iloc[idx, 1:]).float()
        x = Variable(x, requires_grad=True)

        c_enc = torch.tensor([])
        c_enc = Variable(c_enc, requires_grad=True)

        c_dec = torch.tensor([])
        c_dec = Variable(c_dec, requires_grad=True)

        y = torch.tensor(self.csv.iloc[idx, 1:]).float()
        y = Variable(y, requires_grad=True)
        return x, c_enc, c_dec, y


class ActionTargetDatasetV1(VAEDataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    In this version we enhance the latent space with the label
    """

    def __init__(self, action_file, target_file, action_constrain_radius: float = 0):
        super().__init__()
        self.action_csv = pd.read_csv(action_file)
        self.target_csv = pd.read_csv(target_file)

        self._target_mode = TargetMode.ACTION

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        x = Variable(x, requires_grad=True)

        c_enc = torch.tensor([])
        c_enc = Variable(c_enc, requires_grad=True)

        c_dec = torch.tensor(
            self.target_csv.iloc[idx, 1:3]
        ).float()  # we work in a two dimensional space
        c_dec = Variable(c_dec, requires_grad=True)

        y = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        y = Variable(y, requires_grad=True)
        return x, c_enc, c_dec, y


class ActionTargetDatasetV2(VAEDataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    In this version we concatenate the action with the target position and feed it in the VAE
    """

    def __init__(self, action_file, target_file, action_constrain_radius: float = None):
        super().__init__()
        self.action_csv = pd.read_csv(action_file)
        self.target_csv = pd.read_csv(target_file)

        self._target_mode = TargetMode.ACTION

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        action = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        target = torch.tensor(
            self.target_csv.iloc[idx, 1:3]
        ).float()  # we work in a two dimensional space

        x = action

        c_enc = Variable(target, requires_grad=True)

        c_dec = torch.tensor([])
        c_dec = Variable(c_dec, requires_grad=True)

        y = action
        return x, c_enc, c_dec, y


class ConditionalActionTargetDataset(VAEDataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    In this version we concatenate the action with the target position and feed it in the VAE
    """

    def __init__(
        self, action_file, target_file, action_constrain_radius: float = None
    ):  # TODO: remove None and replace with 0
        super().__init__()
        self.action_csv = pd.read_csv(action_file)
        self.state_csv = pd.read_csv(target_file)

        self._target_mode = TargetMode.ACTION

        self.action_constrain_radius = action_constrain_radius
        if action_constrain_radius is not None:
            logging.info("start action constraining")
            self.action_csv = self.constrain_actions(self.action_constrain_radius)
            logging.info("done action constraining")
        logging.info("finished setting up conditional action target dataset")

    def constrain_actions(self, constrain_radius: float) -> pd.DataFrame:
        target_positions, current_positions, state_angles = split_state_information(
            self.state_csv.to_numpy().copy()[:, 1:]
        )
        target_positions = target_positions.squeeze()
        current_positions = current_positions.squeeze()
        state_angles = state_angles.squeeze()

        # get distance from current_pos to target_pos
        target_vectors = target_positions - current_positions
        target_dists = np.sqrt(np.sum(np.square(target_vectors), axis=1))

        action_array = np.zeros_like(self.action_csv.to_numpy())
        # add index to action array
        action_array[:, 0] = np.array(range(len(self)))

        bar = Bar("constraining actions", max=len(self))
        for state_idx in range(len(self)):
            if target_dists[state_idx] <= constrain_radius:
                action_array[state_idx] = self.action_csv.to_numpy().copy()[state_idx]

            # shrink target_vector to action_radius
            target_vector = np.where(
                target_dists[state_idx] == 0,
                np.zeros(2),
                target_vectors[state_idx] / target_dists[state_idx] * constrain_radius,
            )
            new_target = np.zeros(3)
            new_target[0:2] = current_positions[state_idx] + target_vector
            # solve IK for this new position
            label_angles, _, _, _ = IK(
                new_target,
                np.rad2deg(state_angles[state_idx]).copy(),
                np.ones_like(state_angles[state_idx]),
                err_min=0.001,
            )
            label_angles = np.deg2rad(label_angles)
            label_angles = np.cumsum(label_angles) - state_angles[state_idx]
            action_array[state_idx, 1:] = label_angles

            bar.next()
        bar.finish()

        action_df = pd.DataFrame(action_array)
        return action_df

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        action = torch.tensor(self.action_csv.iloc[idx, 1:].to_numpy()).float()
        state = torch.tensor(
            self.state_csv.iloc[idx, 1:].to_numpy()
        ).float()  # we work in a two dimensional space

        x = Variable(action, requires_grad=True)

        c_enc = Variable(state, requires_grad=True)

        c_dec = Variable(state, requires_grad=True)

        y = Variable(action, requires_grad=True)

        return x, c_enc, c_dec, y


class TargetGaussianDataset(VAEDataset):
    def __init__(self, state_file, std) -> None:
        super().__init__()
        self.state_csv = pd.read_csv(state_file)
        self.action_csv = None

        self._target_mode = TargetMode.POSITION  # can be flipped between ACTION and POSITION

        self.std = std
        if self.std > 0:
            logging.info("start action preprocessing")
            self.state_csv = self.preprocess_targets()
            logging.info("done action preprocessing")
        logging.info("finished setting up conditional target dataset")

        if self._target_mode == TargetMode.ACTION:
            logging.info("create action file")
            self.action_csv = self.generate_actions()
            logging.info("done creating action file")
        elif self._target_mode == TargetMode.POSITION:
            logging.info("no action calculation needed")
        else:
            logging.warning("no assignment to x from dataset -> default empty tensor")

    def preprocess_targets(self, truncation: float = 0):
        index = np.array(range(len(self.state_csv)))
        index = np.expand_dims(index, axis=1)
        target_positions, current_positions, state_angles = split_state_information(
            self.state_csv.to_numpy().copy()[:, 1:]
        )

        radius_noise = np.random.normal(0, self.std, (len(current_positions)))

        # truncate radius noise
        if truncation > 0:
            logging.info(f"truncate radius noise to {truncation}")
            radius_noise = np.fmod(radius_noise, truncation)
        elif truncation == 0:
            logging.info("no truncation")
        else:
            logging.warning(
                f"no truncation but truncation={truncation} is an invalid value"
            )

        radius_noise = np.abs(radius_noise)
        radius_noise = np.expand_dims(radius_noise, 1)

        target_vector = target_positions - current_positions
        target_dists = (
            np.linalg.norm(target_vector, axis=1) + 1e-15
        )  # to handle cases where target dists == 0 -> no div by 0
        target_dists = np.expand_dims(target_dists, 1)
        target_noise = (target_vector * np.repeat(radius_noise, 2, axis=1)) / np.repeat(
            target_dists, 2, axis=1
        )

        target_positions = current_positions + np.where(
            target_dists < radius_noise, target_vector, target_noise
        )
        state = np.concatenate(
            [index, target_positions, current_positions, state_angles], axis=1
        )
        state_df = pd.DataFrame(state)
        return state_df

    def generate_actions(self):
        index = np.array(range(len(self.state_csv)))
        index = np.expand_dims(index, axis=1)
        target_position, _, state_angles = split_state_information(
            self.state_csv.to_numpy().copy()[:, 1:]
        )

        action_array = np.zeros_like(state_angles)
        bar = Bar("get actions for targets", max=len(self))
        for state_idx in range(len(self)):
            new_target = np.zeros(3)
            new_target[0:2] = target_position[state_idx]
            # solve IK for this new position
            label_angles, _, _, _ = IK(
                new_target,
                np.rad2deg(state_angles[state_idx]).copy(),
                np.ones_like(state_angles[state_idx]),
                err_min=0.001,
            )
            label_angles = np.deg2rad(label_angles)
            label_angles = np.cumsum(label_angles) - state_angles[state_idx]
            action_array[state_idx] = label_angles
            bar.next()
        bar.finish()

        action_df = pd.DataFrame(np.concatenate([index, action_array], axis=1))
        return action_df

    def __len__(self):
        return len(self.state_csv)

    def __getitem__(self, idx):
        state = torch.tensor(self.state_csv.iloc[idx, 1:].to_numpy()).float()
        state = state.unsqueeze(dim=0)
        target_position, current_position, current_angles = split_state_information(
            state
        )

        x = torch.tensor([])
        if self._target_mode == TargetMode.ACTION and self.action_csv is not None:
            action = torch.tensor(self.action_csv.iloc[idx, 1:].to_numpy()).float()
            x = action.squeeze()
        elif self._target_mode == TargetMode.POSITION:
            x = (target_position - current_position).squeeze()

        c_enc = torch.cat([current_position, current_angles], dim=1).squeeze()
        c_enc = Variable(c_enc, requires_grad=True)

        c_dec = torch.cat([current_position, current_angles], dim=1).squeeze()
        c_dec = Variable(c_dec, requires_grad=True)

        y = target_position.squeeze()

        return x, c_enc, c_dec, y


