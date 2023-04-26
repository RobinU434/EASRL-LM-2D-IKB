from typing import Tuple
import numpy as np
import torch
import logging
import pandas as pd

from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, action_file, state_file, action_radius: float = 0.5):   
        self.action_csv = pd.read_csv(action_file)
        self.state_csv = pd.read_csv(state_file)

        self.action_radius = action_radius 

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        features = torch.tensor(self.state_csv.iloc[idx, 1:]).float()  # state information
        label_angles = torch.tensor(self.action_csv.iloc[idx, 1:]).float()

        if self.action_radius is None:
            return features, label_angles
        
        target_pos, current_pos, start_angles = split_state_information(features.unsqueeze(dim=0).clone())
        target_pos = target_pos.squeeze()
        current_pos = current_pos.squeeze()
        start_angles = start_angles.squeeze()
        # get distance from current_pos to target_pos
        target_vector = target_pos - current_pos
        target_dist = torch.linalg.norm(target_vector)
        # return action if the target position is very close to the current position
        if target_dist <= self.action_radius:
            return features, label_angles
        
        # shrink target_vector to action_radius
        target_vector = target_vector / target_dist * self.action_radius
        new_target = torch.zeros(3)
        new_target[0:2] = current_pos + target_vector
        # solve IK for this new position
        label_angles, _, _, _ = IK(
            new_target.numpy(),
            (start_angles / torch.pi * 180).numpy().copy(),
            torch.ones_like(start_angles).numpy(),
            err_min=0.001)
        label_angles = label_angles / 180 * np.pi  # convert to rad
        label_angles = np.cumsum(label_angles) - start_angles.numpy()
        label_angles = torch.tensor(label_angles)

        return features, label_angles

def get_datasets(feature_source: str, num_joints: int, batch_size: int, action_radius: float) -> Tuple[DataLoader, DataLoader]:
    if feature_source == "state":
        train_data = ActionStateDataset(
            action_file=f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
            state_file=f"./datasets/{num_joints}/train/state_IK_random_start.csv",
            action_radius=action_radius)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_data = ActionStateDataset(
            action_file=f"./datasets/{num_joints}/val/actions_IK_random_start.csv",
            state_file=f"./datasets/{num_joints}/val/state_IK_random_start.csv",
            action_radius=action_radius
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