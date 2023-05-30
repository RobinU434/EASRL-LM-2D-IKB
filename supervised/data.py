import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from progress.bar import Bar
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from envs.robots.ccd import IK
from supervised.utils import forward_kinematics, split_state_information
from vae.data.data_set import YMode


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
        features = torch.tensor(self.target_csv.iloc[idx, 1:3]).float()  # 2D target position
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
        target_positions, current_positions, state_angles = split_state_information(self.state_csv.to_numpy().copy()[:, 1:])
        target_positions = target_positions.squeeze()
        current_positions = current_positions.squeeze()
        state_angles = state_angles.squeeze()
        
        # get distance from current_pos to target_pos
        target_vectors = target_positions - current_positions
        target_dists = np.sqrt(np.sum(np.square(target_vectors), axis=1))
        
        action_array = np.zeros_like(self.action_csv.to_numpy())
        # add index to action array 
        action_array[:, 0] = np.array(range(len(self)))
        
        bar = Bar("constraining actions", max = len(self))
        for state_idx in range(len(self)):
            if target_dists[state_idx] <= constrain_radius:
                action_array[state_idx] = self.action_csv.to_numpy().copy()[state_idx]

            # shrink target_vector to action_radius
            target_vector = np.where(
                target_dists[state_idx] == 0,
                np.zeros(2),
                target_vectors[state_idx] / target_dists[state_idx] * constrain_radius
                )
            new_target = np.zeros(3)
            new_target[0:2] = current_positions[state_idx] + target_vector
            # solve IK for this new position
            label_angles, _, _, _ = IK(
                new_target,
                np.rad2deg(state_angles[state_idx]).copy(),
                np.ones_like(state_angles[state_idx]),
                err_min=0.001)
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
        features = torch.tensor(self.state_csv.iloc[idx, 1:].to_numpy()).float()  # state information
        label_angles = torch.tensor(self.action_csv.iloc[idx, 1:].to_numpy()).float()
        
        return features, label_angles


class TargetGaussianDataset(Dataset):
    def __init__(self, state_file, std) -> None:
        super().__init__()
        self.state_csv = pd.read_csv(state_file)
        self.action_csv = None

        self.y_mode = YMode.ACTION
        
        self.std = std if std is not None else 0
        if self.std > 0:
            logging.info("start action constraining")
            self.state_csv = self.preprocess_targets()
            logging.info("done action constraining")
        logging.info("finished setting up conditional action target dataset")

        if self.y_mode == YMode.ACTION:
            logging.info("create action file")
            self.action_csv = self.generate_actions()
            logging.info("done creating action file")

    def preprocess_targets(self):
        index = np.array(range(len(self.state_csv)))
        index = np.expand_dims(index, axis=1)
        _, current_positions, state_angles = split_state_information(self.state_csv.to_numpy().copy()   [:, 1:])
        noise = np.random.normal(np.zeros_like(current_positions), np.ones_like(current_positions) * self.std)
        target_positions = current_positions + noise

        state = np.concatenate([index, target_positions, current_positions, state_angles], axis=1)
        state_df = pd.DataFrame(state)
        return state_df

    def generate_actions(self):
        index = np.array(range(len(self.state_csv)))
        index = np.expand_dims(index, axis=1)
        target_position, _, state_angles = split_state_information(self.state_csv.to_numpy().copy()[:, 1:])
        
        action_array = np.zeros_like(state_angles)
        bar = Bar("get actions for targets", max = len(self))
        for state_idx in range(len(self)):
            new_target = np.zeros(3)
            new_target[0:2] = target_position[state_idx]
            # solve IK for this new position
            label_angles, _, _, _ = IK(
                new_target,
                np.rad2deg(state_angles[state_idx]).copy(),
                np.ones_like(state_angles[state_idx]),
                err_min=0.001)
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
        state = torch.tensor(self.state_csv.iloc[idx, 1:].to_numpy()).float()  # we work in a two dimensional space
        state = state.unsqueeze(dim=0)
        target_position, current_position, current_angles = split_state_information(state)

        x = torch.cat([target_position - current_position, current_position, current_angles], dim=1).squeeze()
        x = Variable(x, requires_grad=True)

        if self.y_mode == YMode.ACTION:
            y = torch.tensor(self.action_csv.iloc[idx, 1:].to_numpy()).float()
        elif self.y_model == YMode.POSITION:
            y = target_position.squeeze()

        return x, y



def get_datasets(feature_source: str, num_joints: int, batch_size: int, action_radius: float) -> Tuple[DataLoader, DataLoader]:
    action_radius = get_action_radius(action_radius, num_joints) 
    if feature_source == "state":
        train_data = ActionStateDataset(
            action_file=f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
            state_file=f"./datasets/{num_joints}/train/state_IK_random_start.csv",
            action_constrain_radius=action_radius)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_data = ActionStateDataset(
            action_file=f"./datasets/{num_joints}/val/actions_IK_random_start.csv",
            state_file=f"./datasets/{num_joints}/val/state_IK_random_start.csv",
            action_constrain_radius=action_radius
        )
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        
    elif feature_source == "targets":
        train_data = ActionTargetDataset(
            action_file=f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
            target_file=f"./datasets/{num_joints}/train/targets_IK_random_start.csv"
            )
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_data = ActionTargetDataset(
            action_file=f"./datasets/{num_joints}/val/actions_IK_random_start.csv",
            target_file=f"./datasets/{num_joints}/val/targets_IK_random_start.csv"
            )
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
    elif feature_source == "gaussian_target":
        train_data = TargetGaussianDataset(
            state_file=f"./datasets/{num_joints}/train/state_IK_random_start.csv",
            std=action_radius
        )
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_data = TargetGaussianDataset(
            state_file=f"./datasets/{num_joints}/val/state_IK_random_start.csv",
            std=action_radius
        )
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        
    else: 
        logging.error(f"feature source has to be either 'targets' or 'state', you chose: {feature_source}")

    return train_dataloader, val_dataloader


def check_action_constrain_dataset(num_joints: int, random: bool = False, action_radius: float = None, num_samples: int = 1):
    dataset = ActionStateDataset(
        action_file=f"./datasets/{num_joints}/test/actions_IK_random_start.csv",
        state_file=f"./datasets/{num_joints}/test/state_IK_random_start.csv",
        action_constrain_radius=action_radius)
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
        ax.plot(position_sequence[:, 0], position_sequence[:, 1], color="k", marker=".", alpha=1/10)
    for position_sequence in forward_kinematics(target_angles):
        ax.plot(position_sequence[:, 0], position_sequence[:, 1], color="k", marker=".", alpha=1/10)
    
    plt.show()


def get_action_radius(configuration: Any, num_joints: int):
    rescale_factor = 4.0  # this value defines how much the action radius is smaller than the whole action space
    # the value was chosen arbitrarily but be aware by decreasing the value that the actor may need more extreme 
    # actions and you should consider adapting the min_action and max_action of your post_processor 
    if isinstance(configuration, str):
        if configuration == 'auto':
            return num_joints / rescale_factor
        else:
            raise RuntimeError('only configuration string = auto is allowed')
    elif isinstance(configuration, float) or isinstance(configuration, int):  # the value is numeric
        if configuration == 0:
            return None
        else:
            return configuration
    else:
        raise RuntimeError('you chose the wrong type for configuration') 


if __name__ == "__main__":
    check_action_constrain_dataset(2, random=True, action_radius=0.5, num_samples=1)

