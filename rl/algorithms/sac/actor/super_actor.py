import glob
import os
import numpy as np
import yaml
import torch
import logging

from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from latent.datasets.utils import split_state_information
from latent.model.regressor import Regressor
from latent.model.utils.post_processor import PostProcessor

from rl.algorithms.sac.actor.base_actor import Actor
from rl.algorithms.sac.actor.utils import TrainMode

from typing import Any, Dict, List, Tuple, Literal, Union
from utils.file_system import write_yaml
from utils.metrics import Metrics

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
    logging.debug(f"Use config at: {config}")
    return file_name, load_checkpoint(path), config


class SuperActor(NeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float,
        architecture: List[int] = [128, 128],
        activation_function: str = "ReLU",
        super_learning_mode: Union[
            Literal[TrainMode.STATIC], Literal[TrainMode.FINE_TUNING]
        ] = TrainMode.STATIC,
        latent_checkpoint_dir: str = "results/supervised",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, learning_rate, device, **kwargs)

        self._latent_checkpoint_dir = latent_checkpoint_dir
        self._regressor_learning_mode = super_learning_mode
        self._regressor = self._build_regressor()

        self._actor = Actor(
            input_dim=input_dim,
            output_dim=2,  # we want to predict a relative target position in 2D space
            learning_rate=learning_rate,
            architecture=architecture,
            activation_function=activation_function,
            device=device
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        action, log_prob = self._actor.forward(x)
        _, current_position, state_angles = split_state_information(x)
        latent = torch.cat([action,  current_position, state_angles], dim=1)
        regressor_action = self._regressor.forward(latent)
        return regressor_action, log_prob

    def train(self, loss: Tensor):
        self._actor.train(loss)

    def _build_regressor(self) -> Regressor:
        checkpoint_path, self._regressor_checkpoint, self._regressor_config = load_best_checkpoint(self._latent_checkpoint_dir, self._output_dim)
        self._regressor_checkpoint_name = checkpoint_path.split("/")[-1]
        self._regressor_config["post_processor"]["enabled"] = False
        regressor = Regressor.from_config(self._regressor_config)
        regressor.load_state_dict(self._regressor_checkpoint["model_state_dict"])
        return regressor

    def save(self, path: str, metrics: Metrics = ..., epoch_idx: int = 0):
        self._actor.save(path, metrics, epoch_idx)
        path = "/".join(path.split("/")[:-1])
        regressor_path = path + "/" + self._regressor_checkpoint_name
        if not os.path.isfile(regressor_path):
            logging.debug("save regressor checkpoint in save directory of experiment")
            torch.save(self._regressor_checkpoint, regressor_path)
        config_path = path + "/regressor_config.yaml"
        if not os.path.isfile(config_path):
            write_yaml(config_path, self._regressor_config)

    def load_checkpoint(self, path: str):
        """loads checkpoint from filesystem

        Args:
            path (str): path to actor checkpoint
        """ 
        self._actor.load_checkpoint(path)

        # find vae_path
        # assume the vae checkpoint is one directory up the tree
        path = "/".join(path.split("/")[:-2])
        regressor_checkpoint = glob.glob(path + "/Regressor_*.pt")[0]
        logging.debug(f"{regressor_checkpoint}")
        self._regressor.load_checkpoint(regressor_checkpoint)
    
    @property
    def optimizer(self) -> optim.Optimizer:
        return self._actor._optimizer
    
    @property
    def hparams(self) -> Dict[str, Union[str, int, float]]:
        hparams = {"learning_rate": self._learning_rate}
        return hparams # type: ignore
