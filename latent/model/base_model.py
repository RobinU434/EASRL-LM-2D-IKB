from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch import Tensor

from latent.metrics.base_metrics import Metrics
from logger.base_logger import Logger


class LearningModule(ABC):
    def __init__(self) -> None:
        super().__init__()

    def log_internals(self, logger: List[Union[SummaryWriter, Logger]], epoch_idx: int):
        """log internal values like a latent distribution from a VAE

        Args:
            logger (SummaryWriter): logger to write the information into
            epoch_idx (int): at which epoch to log the metrics
        """
        pass

    @abstractmethod
    def save(self, path: str, epoch_idx: int, metrics: Metrics) -> None:
        pass

    @property
    @abstractmethod
    def hparams(self) -> Dict[str, Union[str, int, float]]:
        """returns the hyper parameter configuration of a model as a dictionary

        Returns:
            Dict[str, Union[str, int, float]]: key: hyperparameter name, value: hyperparameter value
        """
        pass


class NeuralNetwork(LearningModule, nn.Module):
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float) -> None:
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._learning_rate = learning_rate
        self._optimizer: torch.optim.Adam

    def train(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def save(self, path: str, metrics: Metrics, epoch_idx: int):
        """saves model checkpoint in filesystem

        Args:
            path (str): path to directory where to store the model
            metrics (Metrics): metric object with at least public member loss
            epoch_idx (int): at which epoch this model is saved
        """
        path = self._create_save_path(path, epoch_idx, metrics)
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": metrics.loss,
            },
            path,
        )

    def _create_save_path(self, path: str, epoch_idx: int, metrics: Metrics) -> str:
        return path + f"/{type(self).__name__}_{epoch_idx}_val_loss_{metrics.loss.mean().item():.4f}.pt"
        

class FeedForwardNetwork(NeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: List[int],
        activation_function: str,
        learning_rate: float,
    ) -> None:
        super().__init__(input_dim, output_dim, learning_rate)

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
