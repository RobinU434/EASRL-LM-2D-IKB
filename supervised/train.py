import json
import logging
import yaml
import time
import torch
import numpy as np

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supervised.data import get_datasets
from supervised.loss import DistanceLoss, ImitationLoss, PointDistanceLoss, get_loss_func
from supervised.model import Regressor, build_model
from supervised.utils import split_state_information
from vae.utils.post_processing import PostProcessor


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
        help=f"GPU or CPU, current available GPU index: {torch.cuda.device_count() - 1}")
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="if set to true -> debug mode is activated"
    )
    parser.add_argument(
        "--print_config",
        action='store_true',
        help='command just prints config and returns'
    )
    return parser


def load_config() -> dict:
    with open("config/base_supervised.yaml") as f:
        config = yaml.safe_load(f)

    return config



def run_model(model: Regressor,
              data: DataLoader,
              criterion,
              train: bool = False,
              device: str = "cpu",):

    imitation_loss_func = ImitationLoss()
    distance_loss_func = DistanceLoss()

    metrics = []
    metrics_dt = [
        ("loss", np.float32),
        ("imitation_loss", np.float32),
        ("distance_loss", np.float32)
        ]

    for x, y in data:
        x = x.to(device)
        y = y.to(device)

        _, _, state_angles = split_state_information(x)

        x_hat = model.forward(x)

        if isinstance(loss_func, PointDistanceLoss): 
            predicted_target_action = state_angles + x_hat
            loss = criterion(y, predicted_target_action)
            distance_loss = distance_loss_func(y, x_hat)
            imitation_loss = imitation_loss_func(torch.zeros_like(x_hat), x_hat)
        else:
            predicted_target_action = state_angles + x_hat
            target_action = state_angles + y
            loss = loss_func(target_action, predicted_target_action)
            distance_loss = distance_loss_func(y, x_hat)
            imitation_loss = imitation_loss_func(y, x_hat)

        if train:
            model.train(loss)

        metrics.append(
            np.array([
                loss.item(),
                imitation_loss.item(),
                distance_loss.item(),
            ])
        )
    
    metrics = np.stack(metrics)
    metrics = np.rec.fromarrays(metrics.T, dtype=metrics_dt)

    return metrics


def train(
        regressor: Regressor,
        train_data: DataLoader,
        val_data: DataLoader,
        loss_func, 
        n_epochs: int,
        logger: SummaryWriter,
        val_interval: int,
        device: str,
        path: str,
        ) -> None:

    for epoch_idx in range(n_epochs):
        train_metrics = run_model(model=regressor,
                                  data=train_data,
                                  criterion=loss_func,
                                  train=True,
                                  device=device)
        
        if epoch_idx % val_interval == 0: 
            val_metrics = run_model(model=regressor,
                                    data=val_data,
                                    criterion=loss_func,
                                    train=False,
                                    device=device)

            print(f"epoch: {epoch_idx}  train_loss: {train_metrics['loss'].mean()} val_loss: {val_metrics['loss'].mean()}")   
            
            logger.add_scalar("supervised/train_loss", train_metrics['loss'].mean(), epoch_idx)
            logger.add_scalar("supervised/train_imiation_loss", train_metrics['imitation_loss'].mean(), epoch_idx)
            logger.add_scalar("supervised/train_distance_loss", train_metrics['distance_loss'].mean(), epoch_idx)
            
            logger.add_scalar("supervised/val_loss", val_metrics['loss'].mean(), epoch_idx)
            logger.add_scalar("supervised/val_imiation_loss", val_metrics['imitation_loss'].mean(), epoch_idx)
            logger.add_scalar("supervised/val_distance_loss", val_metrics['distance_loss'].mean(), epoch_idx)
            
            # save model
            regressor.save(
                path=path + f"/model_{epoch_idx}_val_loss_{float(val_metrics['loss'].mean()):.4f}.pt",
                epoch_idx=epoch_idx,
                metrics=val_metrics
            )


if __name__ == "__main__":
    config = load_config()

    parser = setup_parser(ArgumentParser())
    args = parser.parse_args()

    if args.print_config:
        print(json.dumps(config, sort_keys=True, indent=4))
        exit()

    # set logging level
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    post_processor_config = config['post_processor']
    model = build_model(
        feature_source=config["feature_source"],
        num_joints=config["num_joints"],
        learning_rate=config["learning_rate"],
        post_processor_config=post_processor_config).to(args.device)

    train_dataloader, val_dataloader = get_datasets(feature_source=config["feature_source"],
                                                    num_joints=config["num_joints"],
                                                    batch_size=config["batch_size"],
                                                    action_radius=config["action_radius"])

    loss_func = get_loss_func(config["loss_func"], args.device, train_dataloader.dataset.y_mode )

    path = f"results/supervised/{args.subdir}/{config['loss_func']}/{config['num_joints']}_{int(time.time())}"
    logger = SummaryWriter(path)

    # store config
    with open(path + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file)

    train(
        regressor=model,
        train_data=train_dataloader,
        val_data=val_dataloader,
        loss_func=loss_func,
        n_epochs=config["n_epochs"],
        logger=logger,
        val_interval=config["val_interval"],
        device=args.device,
        path=path
    )
