import logging
from typing import Callable, Literal, Union

import pandas as pd
import torch
import numpy as np
from numpy import ndarray
from progress.bar import Bar
from torch import Tensor
from torch.utils.data import Dataset
from envs.robots.ccd import IK

from latent.datasets.latent_dataset import LatentDataset
from latent.datasets.utils import TargetMode, split_state_information
from utils.file_system import load_csv


class ActionDataset(LatentDataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    Therefore we need no label which is in this case an empty tensor
    """

    def __init__(
        self,
        target_mode: Literal[TargetMode.ACTION, TargetMode.POSITION],
        actions: Tensor,
        **kwargs,
    ) -> None:
        super().__init__(target_mode, **kwargs)
        self._actions: Tensor = actions.float()
        """Tensor: with all actions inside"""
        self._target_mode = TargetMode.ACTION

    def __len__(self):
        return len(self._actions)

    @classmethod
    def from_files(cls, action_file: str, **dataset_args):
        actions = load_csv(action_file)
        return cls(actions=actions, **dataset_args)


class StateActionDataset(LatentDataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    In this version we concatenate the action with the target position and feed it in the VAE
    """

    def __init__(
        self,
        actions: Tensor,
        states: Tensor,
        target_mode: Union[Literal[TargetMode.ACTION], Literal[TargetMode.POSITION]],
        action_constrain_radius: float = 0,
    ):
        super().__init__(target_mode)
        self._actions = actions.float()
        self._states = states.float()

        self._acquire_target_func = self._get_acquire_target_func()

        self._max_step_width = action_constrain_radius
        if action_constrain_radius > 0:
            logging.info("start action constraining")
            self._actions = self._constrain_actions()
            logging.info("done action constraining")
        logging.info("finished setting up conditional action target dataset")

    @classmethod
    def from_files(cls, action_file: str, state_file: str, **dataset_args):
        actions = load_csv(action_file)
        states = load_csv(state_file)
        return cls(actions=actions, states=states, **dataset_args)

    def _constrain_actions(self) -> Tensor:
        """constrains maximum step width with one action

        actions are in default setting generated to go from one random point inside the arms reach to another random point inside the arms reach.
        This function constrains the step width but keeps the direction. But keeps the original target

        Returns:
            Tensor: constrained actions
        """
        target_positions, current_positions, state_angles = split_state_information(
            self._states
        )

        # get distance from current_pos to target_pos
        target_vectors = target_positions - current_positions
        target_dists = torch.linalg.norm(target_vectors, dim=1)

        action_array = torch.zeros_like(self._actions)

        bar = Bar("constraining actions", max=len(self))
        for state_idx in range(len(self)):
            if target_dists[state_idx] <= self._max_step_width:
                action_array[state_idx] = self._actions[state_idx]
                continue
            target_vector = target_vectors[state_idx]
            target_dist = target_dists[state_idx]
            # scale target vector
            target_vector = target_vector / target_dists * target_dist

            new_target = torch.zeros(3)
            new_target[:2] = current_positions[state_idx] + target_vector
            # solve IK for this new position
            label_angles, _, _, _ = IK(
                new_target,
                torch.rad2deg(state_angles[state_idx]).clone(),
                torch.ones_like(state_angles[state_idx]),
                err_min=0.001,
            )
            label_angles = torch.deg2rad(label_angles)
            label_angles = torch.cumsum(label_angles, dim=0) - state_angles[state_idx]
            action_array[state_idx, 1:] = label_angles

            bar.next()
        bar.finish()

        return action_array

    def __len__(self):
        return len(self._actions)

    def __getitem__(self, idx: int):
        x = self._actions[idx]

        c_enc = self._states[idx]
        c_dec = self._states[idx]

        y = self._acquire_target_func(idx)

        return x, c_enc, c_dec, y


class TargetGaussianDataset(LatentDataset):
    def __init__(
        self,
        states: Tensor,
        std: float,
        truncation: float,
        target_mode: Literal[TargetMode.ACTION, TargetMode.POSITION],
        **kwargs,
    ) -> None:
        super().__init__(target_mode, **kwargs)

        self._states = states.float()
        self._std = std
        """float: std for noise around current position"""
        self._truncation = truncation
        """float: hard border. If > 0 apply truncated normal distribution"""
        self._set_gaussian_targets()

        if self._target_mode == TargetMode.ACTION:
            logging.info("create action file")
            self._actions = self._generate_actions()
            logging.info("done creating action file")
        elif self._target_mode == TargetMode.POSITION:
            pass
        else:
            logging.error(
                f"{self._target_mode=} not set correctly. Allowed are ACTION, POSITION"
            )
        self._acquire_target_func = self._get_acquire_target_func()

    def __len__(self):
        return len(self._states)

    @classmethod
    def from_files(cls, state_file: str, **dataset_args):
        states = load_csv(state_file)
        return cls(states=states, **dataset_args)

    def _set_gaussian_targets(self):
        _, current_positions, state_angles = split_state_information(self._states)

        # assume segment length = 1
        arms_reach = state_angles.shape[1]

        radius_noise = torch.normal(
            torch.zeros(len(self)),
            torch.ones(len(self)) * self._std,
        )
        # truncate radius noise
        if self._truncation > 0:
            logging.info(f"truncate radius noise to {self._truncation}")
            radius_noise = torch.fmod(radius_noise, self._truncation)
        elif self._truncation == 0:
            logging.info("no truncation")
        else:
            logging.warning(
                f"no truncation. But {self._truncation=} is < 0 and therefor invalid."
            )

        radius_noise = torch.abs(radius_noise)[:, None]
        theta_noise = torch.rand(len(self)) * 2 * torch.pi
        noisy_intermediate_targets = (
            torch.stack([torch.cos(theta_noise), torch.sin(theta_noise)]).T
            * radius_noise
        )
        new_targets = current_positions + noisy_intermediate_targets

        # cap length where noise + current position > arms_reach
        capped_targets = (
            new_targets.T / torch.linalg.norm(new_targets, dim=1) * arms_reach
        ).T
        condition = torch.linalg.norm(new_targets, dim=1) > arms_reach
        condition = (torch.ones_like(new_targets, dtype=torch.bool).T * condition).T
        new_targets = torch.where(
            condition,
            capped_targets,
            new_targets,
        )

        self._states = torch.cat(
            [new_targets, current_positions, state_angles], dim=1
        ).float()

    def _generate_actions(self) -> Tensor:
        """generates actions for target positions from self._states

        Returns:
            Tensor: actions. Shape (num_samples, n_joints)
        """
        targets, _, state_angles = split_state_information(self._states)

        state_angles_deg = torch.rad2deg(state_angles.clone())
        joint_segments = torch.ones_like(state_angles_deg)

        n_joints = state_angles.size()[1]
        action_array = np.zeros((len(self), n_joints))
        bar = Bar("get actions for targets", max=len(self))
        for state_idx in range(len(self)):
            target = torch.zeros(3)
            target[:2] = targets[state_idx]
            # solve IK for this new position
            label_angles, _, _, _ = IK(
                target.numpy(),
                state_angles_deg[state_idx].numpy(),
                joint_segments[state_idx].numpy(),
                err_min=0.001,
            )
            label_angles = np.deg2rad(label_angles)
            label_angles = np.cumsum(label_angles) - state_angles[state_idx].numpy()
            action_array[state_idx] = label_angles
            bar.next()
        bar.finish()

        actions = torch.from_numpy(action_array)
        return actions.float()
