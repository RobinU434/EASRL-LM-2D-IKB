import glob
import numpy as np
import yaml
import torch
import logging

from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from latent.datasets.utils import split_state_information
from latent.model.regressor import Regressor
from latent.model.utils.post_processor import PostProcessor

from rl.algorithms.sac.actor.base_actor import Actor
from rl.algorithms.sac.actor.utils import TrainMode

from typing import Any, List, Tuple, Literal, Union

from utils.model.neural_network import NeuralNetwork


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
    supervised_results_dir: str, output_dim: int
) -> Tuple[str, Any, dict]:
    """_summary_

    Args:
        vae_results_dir (str): _description_
        output_dim (int): _description_

    Returns:
        Tuple[str, Any, dict]: checkpoint file name, checkpoint, config dict from model
    """
    paths = glob.glob(supervised_results_dir + f"/{output_dim}_*/*.pt")
    if len(paths) == 0:
        raise ValueError(
            f"there is not supervised model trained with output_dim dim: {output_dim}"
        )
    losses = map(extract_loss, paths)
    d = dict(zip(losses, paths))
    path = d[min(d)]
    print(f"use checkpoint for supervised at {path}")
    file_name = path.split("/")[-1]
    config = load_config("/".join(path.split("/")[:-1] + ["config.yaml"]))
    return file_name, load_checkpoint(path), config


class SuperActor(NeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float,
        device: str = "cpu",
        architecture: List[int] = [128, 128],
        activation_function: str = "ReLU",
        super_learning_mode: Union[
            Literal[TrainMode.STATIC], Literal[TrainMode.FINE_TUNING]
        ] = TrainMode.STATIC,
        latent_checkpoint_dir: str = "results/supervised",
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, learning_rate, device, **kwargs)

        self._latent_checkpoint_dir = latent_checkpoint_dir
        self._regressor_learning_mode = super_learning_mode
        self._regressor = self._build_regressor()

        self.actor = Actor(
            input_dim=input_dim,
            output_dim=2,  # we want to predict a relative target position in 2D space
            learning_rate=learning_rate,
            architecture=architecture,
            activation_function=activation_function,
            device=device
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        action, log_prob = self.actor.forward(x)
        _, current_position, state_angles = split_state_information(x)
        latent = torch.cat([action,  current_position, state_angles], dim=1)
        regressor_action = self._regressor.forward(latent)
        return regressor_action, log_prob

    def train(self, loss: Tensor):
        self.actor.train(loss)

    def _build_regressor(self) -> Regressor:
        _, regressor_checkpoint, regressor_config = load_best_checkpoint(self._latent_checkpoint_dir, self._output_dim)
        regressor_config["post_processor"]["enabled"] = False
        regressor = Regressor.from_config(regressor_config)
        regressor.load_state_dict(regressor_checkpoint["model_state_dict"])
        return regressor
    
    @property
    def optimizer(self):
        return self.actor._optimizer
