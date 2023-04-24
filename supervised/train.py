import time
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from supervised.data import ActionTargetDataset
from supervised.model import Regressor


def train(
        model, 
        train_data,
        val_data,
        n_epochs,
        logger,
        val_interval,
):

    for epoch_idx in range(n_epochs):
        losses = torch.tensor([])
        grads = []
        c = 0

        for x, y in train_data:
            x_hat = model.forward(x)

            # loss = imitation_loss(y, x_hat)
            loss = distance_loss(y, x_hat)
            losses = torch.cat([losses, torch.tensor([loss])])

            model.train(loss)

        if epoch_idx % val_interval == 0: 
            val_losses = torch.tensor([])
            imitation_losses = torch.tensor([])
            for x, y in val_data:
                x_hat = model.forward(x)

                # loss = imitation_loss(y, x_hat)
                loss = distance_loss(y, x_hat) + torch.square(x_hat).mean()
                imitation_losses = torch.cat([imitation_losses, torch.tensor([imitation_loss(y, x_hat)])])
                val_losses = torch.cat([val_losses, torch.tensor([loss])])

                target_pos = forward_kinematics(y[0][None, :])[:, 2]
                real_pos = forward_kinematics(x_hat[0][None, :])[:, 2]
                print(target_pos, real_pos)

                # print((y[0] - torch.pi) / torch.pi, x_hat[0])

            print(f"epoch: {epoch_idx}  train_loss: {losses.mean()} val_loss: {val_losses.mean()}, imi_loss: {imitation_losses.mean()}")   
            logger.add_scalar("supervised/train_loss", losses.mean(), epoch_idx)
            logger.add_scalar("supervised/val_loss", val_losses.mean(), epoch_idx)


def imitation_loss(y, x_hat):
    # squash y
    y = (y - torch.pi) / torch.pi
    # MSE
    loss = torch.float_power(angle_diff(y, x_hat), 2).mean()

    return loss


def distance_loss(y, x_hat):
    # expand x_hat to the original space
    x_hat = (x_hat * torch.pi) + torch.pi
    target_pos = forward_kinematics(y)[:, 2]
    real_pos = forward_kinematics(x_hat)[:, 2]

    loss = torch.float_power(target_pos - real_pos, 2).mean()

    return loss

def angle_diff(a : torch.tensor, b: torch.tensor):
        # source: https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
        dif = a - b
        return (dif + torch.pi) % (2 * torch.pi) - torch.pi 


def forward_kinematics(angles: torch.tensor):
    """_summary_

    Args:
        angles (np.array): shape (num_arms, num_joints)

    Returns:
        _type_: _description_
    """
    num_arms, num_joints = angles.size()
    positions = torch.zeros((num_arms, num_joints + 1, 2))

    for idx in range(num_joints):
        origin = positions[:, idx]

        # new position
        new_pos = torch.zeros((num_arms, 2))
        new_pos[:, 0] = torch.cos(angles[:, idx])
        new_pos[:, 1] = torch.sin(angles[:, idx])
        
        # translate position
        new_pos += origin

        positions[:, idx + 1] = new_pos

    return positions


if __name__ == "__main__":
    num_joints = 2
    n_epochs = 201
    batch_size = 64
    val_interval = 5
    
    model = Regressor(2, num_joints)
    train_data = ActionTargetDataset(
        f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
        f"./datasets/{num_joints}/train/state_IK_random_start.csv"
        )
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_data = ActionTargetDataset(
        f"./datasets/{num_joints}/val/actions_IK_random_start.csv",
        f"./datasets/{num_joints}/val/state_IK_random_start.csv"
        )
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    path = f"results/supervised/{num_joints}_{int(time.time())}"
    logger = SummaryWriter(path)

    train(
        model=model,
        train_data=train_dataloader,
        val_data=val_dataloader,
        n_epochs=n_epochs,
        logger=logger,
        val_interval=val_interval
    )
    