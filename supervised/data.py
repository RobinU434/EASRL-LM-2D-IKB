from typing import Tuple
import torch
import logging
import pandas as pd

from torch.utils.data import Dataset, DataLoader


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
    def __init__(self, action_file, state_file):   
        self.action_csv = pd.read_csv(action_file)
        self.state_csv = pd.read_csv(state_file)

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        features = torch.tensor(self.state_csv.iloc[idx, 1:]).float()  # state information
        label = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        return features, label
    

def get_datasets(feature_source: str, num_joints: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    if feature_source == "state":
        train_data = ActionStateDataset(
            action_file=f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
            state_file=f"./datasets/{num_joints}/train/state_IK_random_start.csv"
            )
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_data = ActionStateDataset(
            action_file=f"./datasets/{num_joints}/val/actions_IK_random_start.csv",
            state_file=f"./datasets/{num_joints}/val/state_IK_random_start.csv"
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