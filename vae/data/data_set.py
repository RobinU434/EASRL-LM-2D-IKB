import torch
import pandas as pd

from torch.utils.data import Dataset
from torch.autograd import Variable 


class ActionDataset(Dataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    Therefore we need no label which is in this case an empty tensor
    """
    def __init__(self, annotations_file: str, normalize: bool = False):   
        self.csv = pd.read_csv(annotations_file)
        self.normalize = normalize

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        features = torch.tensor(self.csv.iloc[idx, 1:]).float()
        features = Variable(features, requires_grad=True)

        if self.normalize:
            features = (features / torch.pi) - 1
        
        label = torch.tensor([])    
        label = Variable(label, requires_grad=True)
        return features, label


class ActionTargetDatasetV1(Dataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    In this version we enhance the latent space with the label
    """
    def __init__(self, action_file, target_file, normalize: bool = False):   
        self.action_csv = pd.read_csv(action_file)
        self.target_csv = pd.read_csv(target_file)

        self.normalize = normalize

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        features = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        features = Variable(features, requires_grad=True)
        
        if self.normalize:
            features = (features / torch.pi) - 1

        label = torch.tensor(self.target_csv.iloc[idx, 1:3]).float()  # we work in a two dimensional space
        label = Variable(label, requires_grad=True)
        return features, label


class ActionTargetDatasetV2(Dataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    In this version we concatenate the action with the target position and feed it in the VAE 
    """
    def __init__(self, action_file, target_file, normalize: bool = False):   
        self.action_csv = pd.read_csv(action_file)
        self.target_csv = pd.read_csv(target_file)

        self.normalize = normalize

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        action = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        target = torch.tensor(self.target_csv.iloc[idx, 1:3]).float()  # we work in a two dimensional space
        
        if self.normalize:
            action = (action / torch.pi) - 1
        
        features = torch.cat([action, target])
        features = Variable(features, requires_grad=True)
        
        label = torch.tensor([])
        label = Variable(label, requires_grad=True)
        return features, label


class ConditionalActionTargetDataset(Dataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    In this version we concatenate the action with the target position and feed it in the VAE 
    """
    def __init__(self, action_file, target_file, normalize: bool = False):   
        self.action_csv = pd.read_csv(action_file)
        self.state_csv = pd.read_csv(target_file)

        self.normalize = normalize

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        action = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        state = torch.tensor(self.state_csv.iloc[idx, 1:]).float()  # we work in a two dimensional space
        
        if self.normalize:
            action = (action / torch.pi) - 1
        
        # has to be concatenated because the will be feed directly into the encoder
        features = torch.cat([action, state])
        features = Variable(features, requires_grad=True)
        
        label = state
        label = Variable(label, requires_grad=True)
        return features, label


if __name__ == "__main__":
    # dataset = ActionDataset("./datasets/10/actions_1674557585.csv")
    # dataset[0]

    dataset = ConditionalActionTargetDataset("./datasets/2/test/actions_IK_random_start.csv", "./datasets/2/test/state_IK_random_start.csv", normalize=True)
    print(dataset[0])