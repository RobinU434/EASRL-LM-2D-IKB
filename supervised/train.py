import logging
import yaml
import time
import torch
import numpy as np

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supervised.data import get_datasets
from supervised.model import Regressor, build_model


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "subdir",
        type=str,
        default="test",
        help="specifies in which subdirectory to store the results")
    parser.add_argument(
        "device",
        type=str,
        default="cpu",
        help=f"GPU or CPU, current avialable GPU index: {torch.cuda.device_count() - 1}")
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="if set to true -> debug mode is activated"
    )
    
    return parser


def load_config() -> dict:
    with open("config/base_supervised.yaml") as f:
        config = yaml.safe_load(f)

    return config


def train(
        model: Regressor,
        train_data: DataLoader,
        val_data: DataLoader,
        n_epochs: int,
        logger: SummaryWriter,
        val_interval: int,
        device: str,
        path: str,
        ) -> None:

    for epoch_idx in range(n_epochs):
        losses = torch.tensor([])
        
        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)
            
            x_hat = model.forward(x)

            loss = imitation_loss(y, x_hat)
            # loss = distance_loss(y, x_hat)  # + torch.square(x_hat).mean()
            losses = torch.cat([losses, torch.tensor([loss])])

            model.train(loss)

        if epoch_idx % val_interval == 0: 
            val_losses = torch.tensor([])
            imitation_losses = torch.tensor([])
            for x, y in val_data:
                x = x.to(device)
                y = y.to(device)
                
                x_hat = model.forward(x)

                loss = imitation_loss(y, x_hat)
                # loss = distance_loss(y, x_hat)
                imitation_losses = torch.cat([imitation_losses, torch.tensor([imitation_loss(y, x_hat)])])
                val_losses = torch.cat([val_losses, torch.tensor([loss])])

            print(f"epoch: {epoch_idx}  train_loss: {losses.mean()} val_loss: {val_losses.mean()}, imi_loss: {imitation_losses.mean()}")   
            logger.add_scalar("supervised/train_loss", losses.mean(), epoch_idx)
            logger.add_scalar("supervised/val_loss", val_losses.mean(), epoch_idx)

            # save model
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'loss': val_losses.mean(),
            }, path + f"/model_{epoch_idx}_val_loss_{float(val_losses.mean()):.4f}.pt")     


def imitation_loss(y, x_hat):
    # squash y
    y = (y / torch.pi) - 1
    # MSE
    loss = torch.square(angle_diff(y, x_hat)).mean()

    return loss


def distance_loss(y, x_hat):
    # expand x_hat to the original space
    x_hat = (x_hat + 1) * torch.pi
    target_pos = forward_kinematics(y)[:, 2]
    real_pos = forward_kinematics(x_hat)[:, 2]

    dist_loss = torch.square(target_pos - real_pos).mean()
    return dist_loss


def get_relative_angels(abs_angles: torch.tensor) -> torch.tensor:
    rel_angles = abs_angles.copy()
    rel_angles[1:] -= rel_angles[:-1].copy()
    return rel_angles


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
    config = load_config()

    parser = setup_parser(ArgumentParser())
    args = parser.parse_args()

    # set logging level
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    model = build_model(config["feature_source"], config["num_joints"], config["learning_rate"]).to(args.device)

    train_dataloader, val_dataloader = get_datasets(feature_source=config["feature_source"],
                                                    num_joints=config["num_joints"],
                                                    batch_size=config["batch_size"],
                                                    action_radius=config["action_radius"])

    path = f"results/supervised/{args.subdir}/{config['num_joints']}_{int(time.time())}"
    logger = SummaryWriter(path)

    # store config
    with open(path + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file)

    train(
        model=model,
        train_data=train_dataloader,
        val_data=val_dataloader,
        n_epochs=config["n_epochs"],
        logger=logger,
        val_interval=config["val_interval"],
        device=args.device,
        path=path
    )
