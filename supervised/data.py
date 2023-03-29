import torch
import pandas as pd

from torch.utils.data import Dataset


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
        features = torch.tensor(self.target_csv.iloc[idx, 1:3]).float()  # we work in a two dimensional space
        label = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        return features, label