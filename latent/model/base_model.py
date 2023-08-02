from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

from latent.metrics.base_metrics import Metrics
from logger.base_logger import Logger


class LearningModule(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def save(self, path: str, epoch_idx: int, metrics: Metrics) -> None:
        pass

    @abstractmethod
    def log_internals(self, logger: List[Union[SummaryWriter, Logger]], epoch_idx: int):
        """log internal values like a latent distribution from a VAE

        Args:
            logger (SummaryWriter): logger to write the information into
            epoch_idx (int): at which epoch to log the metrics
        """
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

        self.input_dim = input_dim
        self.output_dim = output_dim
        self._learning_rate = learning_rate
        self._optimizer: torch.optim.Adam 

    def train(self, loss: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def save(self, *args, **kwargs) -> Any:
        pass