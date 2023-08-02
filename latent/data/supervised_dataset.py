import logging
from typing import Any, Literal, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from torch import Tensor
from matplotlib import pyplot as plt
from progress.bar import Bar
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from envs.robots.ccd import IK
from latent.data.utils import TargetMode, split_state_information


class ActionTargetDataset(Dataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    Therefore are features and labels the same!
    """

    def __init__(self, action_file, target_file):
        self.action_csv = pd.read_csv(action_file)
        self.target_csv = pd.read_csv(target_file)

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        features = torch.tensor(
            self.target_csv.iloc[idx, 1:3]
        ).float()  # 2D target position
        label = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        return features, label


class ActionStateDataset(Dataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    Therefore are features and labels the same!
    """

    def __init__(self, action_file, state_file, action_constrain_radius: float = 0.5):
        self.action_csv = pd.read_csv(action_file)
        self.state_csv = pd.read_csv(state_file)

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
            label_angles = np.deg2rad(label_angles)  # convert to rad
            label_angles = np.cumsum(label_angles) - state_angles[state_idx]
            action_array[state_idx, 1:] = label_angles

            bar.next()
        bar.finish()

        action_df = pd.DataFrame(action_array)  # cut indices
        return action_df

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        features = torch.tensor(
            self.state_csv.iloc[idx, 1:].to_numpy()
        ).float()  # state information
        label_angles = torch.tensor(self.action_csv.iloc[idx, 1:].to_numpy()).float()

        return features, label_angles


class TargetGaussianDataset(Dataset):
    def __init__(
        self,
        state_file: Union[str, ndarray, Tensor],
        std: float,
        target_mode: Union[
            Literal[TargetMode.ACTION],
            Literal[TargetMode.POSITION],
            Literal[TargetMode.UNDEFINED],
            Literal[TargetMode.INTERMEDIATE_POSITION],
            Literal[TargetMode.FINAL_POSITION],
        ] = TargetMode.UNDEFINED,
    ) -> None:
        super().__init__()

        self.target_mode = TargetMode.POSITION
        self.target_mode = target_mode

        self._final_targets = torch.empty(1)
        self._intermediate_targets = torch.empty(1)
        self._start_positions = torch.empty(1)
        self._state_angles = torch.empty(1)
        self.actions = torch.empty(1)

        self.std = std if std is not None else 0
        if isinstance(state_file, (ndarray, Tensor)):
            self.set_data_attributes(state_file)
        elif isinstance(state_file, str):
            state_csv = pd.read_csv(state_file)
            self.set_data_attributes(state_csv.to_numpy()[:, 1:])
        else:
            raise ValueError(
                "you have to pass in either a ndarray with shape: (num_points, (1 + 2 + 2 + num_joints), last dimension: index, target pos, current_pos, state_angles"
            )

        if self._target_mode == TargetMode.ACTION:
            logging.info("create action file")
            self.actions = self._generate_actions()
            logging.info("done creating action file")

    def set_data_attributes(
        self, data: Union[ndarray, Tensor], truncation: float = 0
    ) -> None:
        """sets attributes like:
        - final_targets  wrt. global frame
        - intermediate_targets wrt. global frame
        - start_positions  wrt. global frame
        - state_angles

        Args:
            data (ndarray): data array consists of final_targets, start_positions, state_angles. Shape: (num_points, 4 + num_joints)
            truncation (float): If you want to approx. truncate the distribution around the origin. Defaults set to 0 -> No truncation
        """
        if isinstance(data, ndarray):
            data = torch.from_numpy(data)

        target_pos, current_pos, state_angels = split_state_information(data)
        self._final_targets = target_pos
        self._start_positions = current_pos
        self._state_angles = state_angels

        if self.std > 0:
            logging.info("start action constraining")
            self._intermediate_targets = self._preprocess_targets(data, truncation)
            logging.info("done action constraining")
        else:
            self._intermediate_targets = target_pos

    # TODO: make this function based on
    def _preprocess_targets(self, data: ndarray, truncation: float = 0) -> Tensor:
        target_positions, current_positions, _ = split_state_information(data)

        radius_noise = torch.normal(
            torch.zeros(len(current_positions)),
            torch.ones(len(current_positions)) * self.std,
        )

        # truncate radius noise
        if truncation > 0:
            logging.info(f"truncate radius noise to {truncation}")
            radius_noise = torch.fmod(radius_noise, truncation)
        elif truncation == 0:
            logging.info("no truncation")
        else:
            logging.warning(
                f"no truncation but truncation={truncation} is an invalid value"
            )

        radius_noise = torch.abs(radius_noise)[:, None]
        
        target_vector = target_positions - current_positions
        target_dists = (
            torch.linalg.norm(target_vector, axis=1) + 1e-15
        )  # to handle cases where target dists == 0 -> no div by 0
        target_dists = target_dists[:, None]
        # target_noise = (target_vector * np.repeat(radius_noise, 2, axis=1)) / np.repeat(
        #     target_dists, 2, axis=1
        # )
        target_noise = (target_vector * radius_noise.repeat(1, 2)) / target_dists.repeat(1, 2)

        intermediate_targets: Tensor = current_positions + torch.where(
            target_dists < radius_noise, target_vector, target_noise
        )

        return intermediate_targets

    def _generate_actions(self) -> Tensor:
        action_array = np.zeros_like(self._state_angles.numpy())
        bar = Bar("get actions for targets", max=len(self))
        for state_idx in range(len(self)):
            new_target = np.zeros(3)
            new_target[0:2] = self._final_targets[state_idx].numpy()
            # solve IK for this new position
            label_angles, _, _, _ = IK(
                new_target,
                np.rad2deg(self._state_angles[state_idx].numpy()).copy(),
                np.ones_like(self._state_angles[state_idx].numpy()),
                err_min=0.001,
            )
            label_angles = np.deg2rad(label_angles)
            label_angles = (
                np.cumsum(label_angles) - self._state_angles[state_idx].numpy()
            )
            action_array[state_idx] = label_angles
            bar.next()
        bar.finish()

        actions = torch.from_numpy(action_array)
        return actions

    def __len__(self):
        return len(self._final_targets)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = torch.cat(
            [
                self._intermediate_targets[idx] - self._start_positions[idx],
                self._start_positions[idx],
                self._state_angles[idx],
            ]
        ).float()

        x = Variable(x, requires_grad=True)

        # TODO: move this chunk of if to a pre-build y selection function
        if self._target_mode == TargetMode.ACTION:
            y = Tensor(self.actions[idx]).float()
        elif self._target_mode == TargetMode.INTERMEDIATE_POSITION:
            y = self._intermediate_targets[idx].float()
        elif self._target_mode == TargetMode.FINAL_POSITION:
            y = self._final_targets[idx].float()
        elif self._target_mode == TargetMode.POSITION:
            logging.warning(
                "No positional behavior defined. Resume to fall back solution -> final target"
            )
            y = self._final_targets[idx].float()
        else:
            modes = TargetMode._member_names_
            modes.remove("UNDEFINED")
            raise ValueError(
                f"No appropriate target mode behavior defined take one of {modes}"
            )

        return x, y

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

    @property
    def states(self) -> torch.Tensor:
        states = torch.cat(
            [
                self._final_targets,
                self._intermediate_targets,
                self._start_positions,
                self._state_angles,
            ],
            dim=1,
        )
        return states


def get_datasets(
    feature_source: str, num_joints: int, batch_size: int, action_radius: float
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataloader = None
    val_dataloader = None
    test_dataloader = None

    action_radius = get_action_radius(action_radius, num_joints)
    if feature_source == "state":
        train_data = ActionStateDataset(
            action_file=f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
            state_file=f"./datasets/{num_joints}/train/state_IK_random_start.csv",
            action_constrain_radius=action_radius,
        )
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_data = ActionStateDataset(
            action_file=f"./datasets/{num_joints}/val/actions_IK_random_start.csv",
            state_file=f"./datasets/{num_joints}/val/state_IK_random_start.csv",
            action_constrain_radius=action_radius,
        )
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        test_data = ActionStateDataset(
            action_file=f"./datasets/{num_joints}/test/actions_IK_random_start.csv",
            state_file=f"./datasets/{num_joints}/test/state_IK_random_start.csv",
            action_constrain_radius=action_radius,
        )
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

    elif feature_source == "targets":
        train_data = ActionTargetDataset(
            action_file=f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
            target_file=f"./datasets/{num_joints}/train/targets_IK_random_start.csv",
        )
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_data = ActionTargetDataset(
            action_file=f"./datasets/{num_joints}/val/actions_IK_random_start.csv",
            target_file=f"./datasets/{num_joints}/val/targets_IK_random_start.csv",
        )
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        test_data = ActionTargetDataset(
            action_file=f"./datasets/{num_joints}/test/actions_IK_random_start.csv",
            target_file=f"./datasets/{num_joints}/test/targets_IK_random_start.csv",
        )
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

    elif feature_source == "gaussian_target":
        train_data = TargetGaussianDataset(
            state_file=f"./datasets/{num_joints}/train/state_IK_random_start.csv",
            std=action_radius,
            target_mode=TargetMode.INTERMEDIATE_POSITION,
        )
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_data = TargetGaussianDataset(
            state_file=f"./datasets/{num_joints}/val/state_IK_random_start.csv",
            std=action_radius,
            target_mode=TargetMode.INTERMEDIATE_POSITION,
        )
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        test_data = TargetGaussianDataset(
            state_file=f"./datasets/{num_joints}/test/state_IK_random_start.csv",
            std=action_radius,
            target_mode=TargetMode.INTERMEDIATE_POSITION,
        )
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

    else:
        logging.error(
            f"feature source has to be either 'targets' or 'state', you chose: {feature_source}"
        )

    return train_dataloader, val_dataloader, test_dataloader


def check_action_constrain_dataset(
    num_joints: int,
    random: bool = False,
    action_radius: float = None,
    num_samples: int = 1,
):
    dataset = ActionStateDataset(
        action_file=f"./datasets/{num_joints}/test/actions_IK_random_start.csv",
        state_file=f"./datasets/{num_joints}/test/state_IK_random_start.csv",
        action_constrain_radius=action_radius,
    )
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=random)

    features, labels = next(iter(dataloader))
    target_position, state_position, state_angels = split_state_information(features)

    target_angles = state_angels + labels

    # plot
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.add_patch(plt.Circle((0, 0), num_joints, fill=False))
    if action_radius is not None:
        ax.add_patch(plt.Circle(state_position[0], action_radius, fill=False))
    ax.scatter(target_position[:, 0], target_position[:, 1], c="b", s=1)
    ax.scatter(state_position[:, 0], state_position[:, 1], c="g", s=1)
    for position_sequence in forward_kinematics(state_angels):
        ax.plot(
            position_sequence[:, 0],
            position_sequence[:, 1],
            color="k",
            marker=".",
            alpha=1 / 10,
        )
    for position_sequence in forward_kinematics(target_angles):
        ax.plot(
            position_sequence[:, 0],
            position_sequence[:, 1],
            color="k",
            marker=".",
            alpha=1 / 10,
        )

    plt.show()


def get_action_radius(configuration: Any, num_joints: int):
    rescale_factor = 4.0  # this value defines how much the action radius is smaller than the whole action space
    # the value was chosen arbitrarily but be aware by decreasing the value that the actor may need more extreme
    # actions and you should consider adapting the min_action and max_action of your post_processor
    if isinstance(configuration, str):
        if configuration == "auto":
            return num_joints / rescale_factor
        else:
            raise RuntimeError("only configuration string = auto is allowed")
    elif isinstance(configuration, float) or isinstance(
        configuration, int
    ):  # the value is numeric
        if configuration == 0:
            return None
        else:
            return configuration
    else:
        raise RuntimeError("you chose the wrong type for configuration")


if __name__ == "__main__":
    # check_action_constrain_dataset(2, random=True, action_radius=0.5, num_samples=1)

    TargetGaussianDataset("datasets/2/train/state_IK_random_start.csv", 0.2)
