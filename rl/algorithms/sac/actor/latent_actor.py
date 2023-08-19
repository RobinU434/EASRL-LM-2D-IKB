import glob
import logging
import os
import yaml
import torch
import torch.optim as optim

from typing import Any, Dict, List, Literal, Tuple, Union
from torch import Tensor
from latent.datasets.utils import split_state_information

from rl.algorithms.sac.actor.base_actor import Actor
from rl.algorithms.sac.actor.utils import TrainMode
from utils.file_system import write_yaml
from utils.metrics import Metrics
from utils.model.neural_network import NeuralNetwork
from latent.model.vae import VAE


def load_checkpoint(checkpoint_path: str):
    logging.debug(torch.cuda.memory_allocated(6))
    checkpoint = torch.load(checkpoint_path)
    logging.debug(torch.cuda.memory_allocated(6))

    return checkpoint


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def store_vae_config(config, path: str):
    with open(path + "/vae_config.yaml", "w") as config_file:
        yaml.dump(config, config_file)


def extract_loss(path: str):
    checkpoint_name = path.split("/")[-1]
    loss = float(checkpoint_name.split("_")[-1][:-3])
    return loss


def load_best_checkpoint(
    vae_results_dir: str, output_dim: int, latent_dim: int
) -> Tuple[str, Tensor, Dict[str, Any]]:
    """searches for model with least error according to naming schema

    Args:
        vae_results_dir (str): where to look for the lowest loss
        output_dim (int): number of joints
        latent_dim (int): how many values in the latent space

    Returns:
        Tuple[str, Tensor, Dict[str, Any]]:
         - path to checkpoint
         - loaded checkpoint
         - config of loaded model
    """
    paths = glob.glob(vae_results_dir + f"/{output_dim}_{latent_dim}*/*.pt")
    if len(paths) == 0:
        raise ValueError(
            f"there is not VAE trained output dim: {output_dim} and latent dim: {latent_dim}"
        )
    losses = map(extract_loss, paths)
    d = dict(zip(losses, paths))
    path = d[min(d)]
    print(f"use checkpoint for VAE at {path}")
    logging.debug(torch.cuda.memory_allocated(6))
    file_name = path.split("/")[-1]
    config = load_config("/".join(path.split("/")[:-1] + ["config.yaml"]))
    return file_name, load_checkpoint(path), config


class LatentActor(NeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        learning_rate: float,
        architecture: List[int] = [128, 128],
        activation_function: str = "ReLU",
        vae_learning_mode: Union[
            Literal[TrainMode.STATIC], Literal[TrainMode.FINE_TUNING]
        ] = TrainMode.STATIC,
        latent_checkpoint_dir: str = "results/vae",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, learning_rate, device,**kwargs)

        self._device = device

        self._vae_learning_mode = vae_learning_mode
        self._latent_dim = latent_dim
        self._latent_checkpoint_dir = latent_checkpoint_dir
        self._vae_checkpoint: Tensor
        self._vae_checkpoint_name: str
        self._vae_config: Dict[str, Any]
        self._vae = self._build_vae()
        
        self._actor = Actor(
            input_dim=input_dim,
            output_dim=self._vae.latent_dim,
            learning_rate=learning_rate,
            architecture=architecture,
            activation_function=activation_function,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        action, log_prob = self._actor.forward(x)
        """ideas
        - take just the action from the actor mapped from state space into latent space
        - sample latent x with output from actor = mu, std
        - evaluate all k nearest neighbors from a pre defined test set and take the one with the highest q value 
        (https://arxiv.org/pdf/1512.07679.pdf)
        - evaluate all actions from a pre defined test set by q which are in the ellipse around mu, axis of the ellipse are defined by std
        """
        _, current_position, state_angles = split_state_information(x)
        latent = torch.cat([action,  current_position, state_angles], dim=1)
        decoder_action = self._vae.decoder.forward(latent)
        return decoder_action, log_prob

    def train(self, loss: torch.Tensor):
        self._actor.train(loss)

    def _build_vae(self) -> VAE:
        checkpoint_path, self._vae_checkpoint, self._vae_config = load_best_checkpoint(
            self._latent_checkpoint_dir, self._output_dim, self._latent_dim
        )
        self._vae_checkpoint_name = checkpoint_path.split("/")[-1]
        # disable post processor
        self._vae_config["post_processor"]["enabled"] = False
        vae = VAE.from_config(self._vae_config)
        vae.load_state_dict(self._vae_checkpoint["model_state_dict"]) # type: ignore
        return vae

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._actor._optimizer
    
    @property
    def hparams(self) -> Dict[str, Union[str, int, float]]:
        hparams = {"learning_rate": self._learning_rate}
        return hparams # type: ignore
    
    def save(self, path: str, metrics: Metrics = ..., epoch_idx: int = 0):
        self._actor.save(path, metrics, epoch_idx)
        path = "/".join(path.split("/")[:-1])
        vae_path = path + "/" + self._vae_checkpoint_name
        if not os.path.isfile(vae_path):
            torch.save(self._vae_checkpoint, vae_path)
        config_path = path + "/vae_config.yaml"
        if not os.path.isfile(config_path):
            write_yaml(config_path, self._vae_config)

    def load_checkpoint(self, path: str):
        """loads checkpoint from filesystem

        Args:
            path (str): path to actor checkpoint
        """ 
        self._actor.load_checkpoint(path)

        # find vae_path
        # assume the vae checkpoint is one directory up the tree
        path = "/".join(path.split("/")[:-2])
        vae_checkpoint = glob.glob(path + "/VAE*.pt")[0]
        print(vae_checkpoint)
        self._vae.load_checkpoint(vae_checkpoint)