from abc import abstractmethod
import logging
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from torch import Tensor
from utils.cuda import dict_to_device

from utils.model.base_model import LearningModule
from utils.metrics import Metrics


class NeuralNetwork(LearningModule, nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, learning_rate: float, device: str = "cpu", **kwargs
    ) -> None:
        super().__init__()
        logging.debug(f"Init {type(self).__name__}")
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._learning_rate = learning_rate
        self._device = device
        logging.debug(f"{self._device=}")
        self._optimizer: torch.optim.Optimizer

    def train(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def save(self, path: str, metrics: Metrics = Metrics(), epoch_idx: int = 0):
        """saves model checkpoint in filesystem

        Args:
            path (str): path to directory where to store the model
            metrics (Metrics): metric object with at least public member loss
            epoch_idx (int): at which epoch this model is saved
        """
        path = self._create_save_path(path, epoch_idx, metrics)
        loss = metrics.loss if "loss" in vars(metrics).keys() else 0  # type: ignore
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": dict_to_device(self.state_dict(), "cpu"),
                "optimizer_state_dict": dict_to_device(self._optimizer.state_dict(), "cpu"),
                "loss": loss,
            },
            path,
        )

    def load_checkpoint(self, path):
        """loads checkpoint from filesystem

        Args:
            path (str): path to checkpoint
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])


    def _create_save_path(
        self, path: str, epoch_idx: int, metrics: Metrics = Metrics()
    ) -> str:
        """creates save path. 

        If epoch idx < 0 -> no epoch in path

        Args:
            path (str): _description_
            epoch_idx (int): _description_
            metrics (Metrics, optional): _description_. Defaults to Metrics().

        Returns:
            str: _description_
        """
        path += f"/{type(self).__name__}_"
        if epoch_idx >= 0:
            path += f"{epoch_idx}_"
        if "loss" in vars(metrics).keys():
            path += f"val_loss_{metrics.loss.mean().item():.4f}.pt"  # type: ignore
        else:
            path += ".pt"
        return path


class FeedForwardNetwork(NeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: List[int],
        activation_function: str,
        learning_rate: float,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, learning_rate, device)

        self._activation_function_type = getattr(nn, activation_function)
        self._linear = self._create_linear_unit(architecture)

    def _create_linear_unit(self, architecture: List[int]) -> nn.Sequential:
        """creates a linear unit specified with architecture and self._activation_function_type

        Args:
            architecture (List[int]): dimension of linear layers

        Returns:
            nn.Sequential: sequential linear unit
        """
        # input layer
        layers = [
            nn.Linear(self._input_dim, int(architecture[0])),
            self._activation_function_type(),
        ]
        # add hidden layers
        for idx in range(len(architecture) - 1):
            layers.extend(
                [
                    nn.Linear(int(architecture[idx]), int(architecture[idx + 1])),
                    self._activation_function_type(),
                ]
            )
        # output layer
        layers.append(nn.Linear(architecture[-1], self._output_dim))
        sequence = nn.Sequential(*layers)
        return sequence

    def forward(self, x: Tensor):
        return self._linear(x)

    @property
    def hparams(self) -> Dict[str, Union[str, int, float]]:
        return {}
