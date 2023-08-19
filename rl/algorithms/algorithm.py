from abc import ABC, abstractmethod
import logging
from typing import List, Union

from gym import Env
from torch.utils.tensorboard.writer import SummaryWriter

from logger.base_logger import Logger
from logger.fs_logger import FileSystemLogger


class RLAlgorithm(ABC):
    """base class for any rl algorithm
    Every rl algorithm should be capable of supporting following methods:
    - train
    - inference
    - load_checkpoint


    """

    def __init__(
        self,
        env: Env,
        logger: List[Union[Logger, SummaryWriter]],
        device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__()
        logging.debug("init RLAlgorithm")
        self._logger = logger
        self._env = env
        self._device = device
        logging.debug(f"{self._device=}")
        
    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def inference(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def print_model(self):
        raise NotImplementedError
    
    def _dump(self):
        for logger in self._logger:
            if isinstance(logger, FileSystemLogger):
                logger.dump(file_name="results.csv")