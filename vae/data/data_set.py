import torch
import logging
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.autograd import Variable
from progress.bar import Bar

from envs.robots.ccd import IK
from vae.helper.extract_angles_and_position import split_conditional_info, split_state_information 
from vae.helper.loss import forward_kinematics


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
    def __init__(self, action_file, target_file, action_constrain_radius: float = None):   
        self.action_csv = pd.read_csv(action_file)
        self.target_csv = pd.read_csv(target_file)

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        features = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        features = Variable(features, requires_grad=True)
        
        label = torch.tensor(self.target_csv.iloc[idx, 1:3]).float()  # we work in a two dimensional space
        label = Variable(label, requires_grad=True)
        return features, label


class ActionTargetDatasetV2(Dataset):
    """
    This class is the dataset class for the VAE to encode actions into a latent space.
    In this version we concatenate the action with the target position and feed it in the VAE 
    """
    def __init__(self, action_file, target_file, action_constrain_radius: float = None):   
        self.action_csv = pd.read_csv(action_file)
        self.target_csv = pd.read_csv(target_file)

    def __len__(self):
        return len(self.action_csv)

    def __getitem__(self, idx):
        action = torch.tensor(self.action_csv.iloc[idx, 1:]).float()
        target = torch.tensor(self.target_csv.iloc[idx, 1:3]).float()  # we work in a two dimensional space
        
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
    def __init__(self, action_file, target_file, action_constrain_radius: float = None):   
        self.action_csv = pd.read_csv(action_file)
        self.state_csv = pd.read_csv(target_file)

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
        action = torch.tensor(self.action_csv.iloc[idx, 1:].to_numpy()).float()
        state = torch.tensor(self.state_csv.iloc[idx, 1:].to_numpy()).float()  # we work in a two dimensional space
        
        # has to be concatenated because the will be feed directly into the encoder
        # features = torch.cat([action, state])
        features = Variable(action, requires_grad=True)
        
        label = state
        label = Variable(label, requires_grad=True)
        return features, label


def check_action_constrain():
    constrain_radius = None 
    dataset = ConditionalActionTargetDataset(
        "./datasets/2/test/actions_IK_random_start.csv",
        "./datasets/2/test/state_IK_random_start.csv",
        normalize=False,
        action_constrain_radius=constrain_radius)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot()
    # find action distribution
    actions = []
    
    for idx, (target_action, state) in enumerate(dataset):
        target_position, current_position, state_angles = split_state_information(state.unsqueeze(dim=0))
        # for position_sequence in forward_kinematics(state_angles).detach().numpy():
        #     ax.plot(position_sequence[:, 0], position_sequence[:, 1], color="k", marker=".", alpha=1/10)

        target_position = target_position.detach().numpy()
        current_position = current_position.detach().numpy()
        ax.scatter(target_position[:, 0], target_position[:, 1], c="b")
        # ax.scatter(current_position[:, 0], current_position[:, 1], c="g")
        if constrain_radius is not None:
            ax.add_patch(plt.Circle(current_position[0], constrain_radius, fill=False))
        arm_positions = forward_kinematics(state_angles + target_action).detach().numpy()
        pred_position = arm_positions[:, -1, :]
        for position_sequence in arm_positions:
            ax.plot(position_sequence[:, 0], position_sequence[:, 1], color="k", marker=".", alpha=1/10)

        ax.scatter(pred_position[:, 0], pred_position[:, 1], c="orange")
        
        actions.append(target_action)

        if idx > 4:
            break
    actions = torch.cat(actions)
    # plt.hist(actions.detach().numpy())
    plt.show()


def plot_action_constrain_radius():
    constrain_radius = 1
    num_joints = 20
    num_points = 100

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1)
    
    # sample start
    theta = np.random.uniform(0,  np.pi)
    radius = np.random.uniform(0, num_joints)
    start = np.zeros(3)
    start[0] = np.cos(theta) * (radius - constrain_radius)
    start[1] = np.sin(theta) * (radius - constrain_radius)

    # sample targets
    theta = np.linspace(0, 2* np.pi, num_points)
    targets = np.zeros((num_points, 3))
    targets[:, 0] = np.cos(theta) * constrain_radius + start[0]
    targets[:, 1] = np.sin(theta) * constrain_radius + start[1]

    # get start config
    link = np.ones(num_joints)
    start_angles = np.zeros(num_joints)
    start_angles, _, _, _ = IK(start, start_angles.copy(), link, err_min=0.001)

    # calculate IK actions for each target
    target_angels = []
    bar = Bar("solve IK", max=num_points)
    for target in targets:
        target_action, _, _, _ = IK(target, start_angles.copy(), link, err_min=0.001)
        target_angels.append(target_action)
        bar.next()
    bar.finish()

    target_angels = np.stack(target_angels)
    target_angels = target_angels / 180 * np.pi  # convert to rad

    # make forward pass to get arm positions
    target_actions = np.cumsum(target_angels, axis=1)
    arm_positions = forward_kinematics(torch.tensor(target_actions)).numpy() 

    # plot
    axs[0].scatter(start[0], start[1], c="b", s=1)
    axs[0].add_patch(plt.Circle(start[0:2], constrain_radius, fill=False))
    axs[0].add_patch(plt.Circle((0, 0), num_joints, fill=False))
    for position_sequence in arm_positions:
        axs[0].plot(position_sequence[:, 0], position_sequence[:, 1], color="k", marker=".", alpha=1/10)
    for i in range(num_joints):
        axs[1].plot(target_angels[:, i])
    plt.show()


def plot_action_state_distribution():
    constrain_radius = 0.1
    num_joints = 2
    dataset = ConditionalActionTargetDataset(
        f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
        f"./datasets/{num_joints}/train/state_IK_random_start.csv",
        normalize=False,
        action_constrain_radius=constrain_radius)
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1)
    # find action distribution
    actions = []
    state_angles = []
    for target_action, state in dataset:
        _, _, state_angle = split_state_information(state.unsqueeze(dim=0))
        actions.append(target_action)
        state_angles.append(state_angle.squeeze())

    actions = torch.cat(actions)
    state_angles = torch.cat(state_angles)

    axs[0].hist(actions.detach().numpy(), bins=50)
    axs[1].hist(state_angles.detach().numpy(), bins=50)
    abs_angles = (actions.reshape((len(dataset), num_joints)) + state_angles.reshape((len(dataset), num_joints))).flatten()
    axs[2].hist(abs_angles.detach().numpy(), bins=50) 
    plt.show()

if __name__ == "__main__":
    plot_action_state_distribution()
    # check_action_constrain()
    # plot_action_constrain_radius()
    
