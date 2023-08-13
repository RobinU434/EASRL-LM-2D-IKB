import glob
import yaml
import torch
import logging

from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn

from algorithms.sac.actor.base_actor import Actor, TrainMode
from supervised.loss import IKLoss
from supervised.model import Regressor

from typing import Any, List, Tuple, Literal
from supervised.train import run_model
from vae.data.data_set import YMode

from vae.utils.post_processing import PostProcessor


def load_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def extract_loss(path: str) -> float:
    """extracts the loss from model checkpoint with name pattern:
    'model_<x>_<loss description>_<y>.<z>.pt'

    Args:
        path (str): path to checkpoint

    Returns:
        loss: validation loss
    """
    checkpoint_name = path.split("/")[-1]
    loss = float(checkpoint_name.split("_")[-1][:-3])
    return loss


def load_best_checkpoint(
    vae_results_dir: str, output_dim: int
) -> Tuple[str, Any, dict]:
    """_summary_

    Args:
        vae_results_dir (str): _description_
        output_dim (int): _description_

    Returns:
        Tuple[str, Any, dict]: checkpoint file name, checkpoint, config dict from model
    """
    paths = glob.glob(vae_results_dir + f"/{output_dim}_*/*.pt")
    if len(paths) == 0:
        logging.error(
            f"there is not supervised model trained with output_dim dim: {output_dim}"
        )
    losses = map(extract_loss, paths)
    d = dict(zip(losses, paths))
    path = d[min(d)]
    print(f"use checkpoint for supervised at {path}")
    file_name = path.split("/")[-1]
    config = load_config("/".join(path.split("/")[:-1] + ["config.yaml"]))
    return file_name, load_checkpoint(path), config


class SuperActor(nn.Module):
    def __init__(
        self,
        device,
        input_dim: int,
        output_dim: int,
        learning_rate: float,
        architecture: List[int] = [128, 128],
        super_learning_mode: Literal[TrainMode] = TrainMode.STATIC,
        checkpoint_dir: str = "results/supervised",
        log_dir: str = "",  # for saving the loaded checkpoint
    ) -> None:
        super().__init__()

        self.device = device

        self.actor = Actor(
            input_dim=input_dim,
            output_dim=2,  # we want to predict a relative target position in 2D space
            learning_rate=learning_rate,
            architecture=architecture,
        )

        self.supervised_model = Regressor(
            input_dim=input_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            # important to disable because the is an additional tanh function in policy net
            post_processor=PostProcessor(enabled=False),
        ).to(self.device)

        self.super_criterion = IKLoss(0, 1, 0, target_mode=YMode.POSITION)

        self.super_learning_mode = super_learning_mode
        if self.super_learning_mode == TrainMode.STATIC.value:
            # load model and perform NO tuning at all
            file_name, checkpoint, supervised_config = load_best_checkpoint(
                checkpoint_dir, output_dim
            )
            self.supervised_model.load_state_dict(checkpoint["model_state_dict"])
            self.supervised_config = supervised_config
            self.supervised_model.save(
                log_dir + "/" + file_name,
                checkpoint["epoch"],
                {"loss": checkpoint["loss"]},
            )

        elif self.super_learning_mode == TrainMode.FINE_TUNING.value:
            # load model and perform fine tuning on that checkpoint
            file_name, checkpoint, supervised_config = load_best_checkpoint(
                checkpoint_dir, output_dim
            )
            self.supervised_model.load_state_dict(checkpoint["model_state_dict"])
            self.supervised_config = supervised_config
            self.supervised_model.save(
                log_dir + "/" + file_name,
                checkpoint["epoch"],
                {"loss": checkpoint["loss"]},
            )

        elif self.super_learning_mode == TrainMode.FROM_SCRATCH.value:
            # load NO checkpoint and train the supervised learning model on the fly
            raise NotImplementedError
        else:
            raise ValueError(
                f"You picked the wrong super_learning status: {self.super_learning_mode}"
            )

    def forward(self, x: Tensor):
        mu, std = self.actor.forward(x)

        return mu, std

    def train(self, loss: Tensor):
        self.actor.train(loss)

    def train_supervised(self, data: DataLoader, train_iterations: int):
        loss = []
        for _ in range(train_iterations):
            metrics = run_model(
                self.supervised_model,
                data,
                self.super_criterion,
                train=True,
                device=self.device,
            )
            loss.append(metrics["loss"])

        return np.mean(loss)
    
    @property
    def optimizer(self):
        return self.actor.optimizer
