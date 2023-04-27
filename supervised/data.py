import torch
import logging
import numpy as np
import pandas as pd

from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from progress.bar import Bar

from envs.robots.ccd import IK
from supervised.utils import split_state_information


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

        self.action_radius = action_constrain_radius 
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
                (state_angles[state_idx] / np.pi * 180).copy(),
                np.ones_like(state_angles[state_idx]),
                err_min=0.001)
            label_angles = label_angles / 180 * np.pi  # convert to rad
            label_angles = np.cumsum(label_angles) - state_angles[state_idx]
            action_array[state_idx, 1:] = label_angles

            bar.next()
        bar.finish()
        
        action_df = pd.DataFrame(action_array)  # cut indices
        return action_df
    
    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        features = torch.tensor(self.state_csv.iloc[idx, 1:]).float()  # state information
        label_angles = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        
        return features, label_angles


def get_datasets(feature_source: str, num_joints: int, batch_size: int, action_radius: float) -> Tuple[DataLoader, DataLoader]:
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
    else: 
        logging.error(f"feature source has to be either 'targets' or 'state', you chose: {feature_source}")

    return train_dataloader, val_dataloader