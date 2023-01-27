import torch
import pandas as pd

from torch.utils.data import Dataset


class ActionDataset(Dataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    Therefore are features and labels the same!
    """
    def __init__(self, annotations_file):   
        self.csv = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        label = torch.tensor(self.csv.iloc[idx, 1:]).float()
        features = label
        return features, label


if __name__ == "__main__":
    dataset = ActionDataset("./datasets/10/actions_1674557585.csv")
    dataset[0]