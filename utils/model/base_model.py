from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch import Tensor

from utils.metrics import Metrics
from logger.base_logger import Logger


class LearningModule(ABC):
    def __init__(self, *args, **kwargs) -> None:
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
