from abc import abstractmethod
import json
import logging
import time
from typing import List, Tuple, Union

import torch
from latent.criterion.base_criterion import Criterion
from latent.model.base_model import NeuralNetwork
from latent.trainer.base_trainer import Trainer
from logger.base_logger import Logger
from logger.fs_logger import FileSystemLogger

from process.base_process import Process

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from utils.decorator import not_implemented_warning


class LatentProcess(Process):
    def __init__(
        self, **kwargs
    ) -> None:
        self._base_config_path = f"config/base_{self._model_entity}.yaml"
        super().__init__(**kwargs)
        self._train_data: DataLoader
        self._val_data: DataLoader
        self._test_data: DataLoader
        self._criterion: Criterion
        self._model: NeuralNetwork
        self._trainer: Trainer

    def print_config(self) -> None:
        """prints config"""
        print(json.dumps(self._config, sort_keys=True, indent=4))

    def print_model(self) -> None:
        """prints model"""
        print(self._model)

    def build(self):
        """builds core component of process:

        - loads datasets
        - build model
        - build logger
        - build trainer
        - load criterion
        """
        self._logger = self._build_logger()

        datasets = self._load_datasets()
        self._train_data = datasets[0]
        self._val_data = datasets[1]
        self._test_data = datasets[2]

        self._criterion = self._load_criterion()
        self._model = self._build_model()
        self._trainer = self._build_trainer()

        # add additional information to config file
        self._config["input_dim"] = self._model.input_dim
        self._config["output_dim"] = self._model.output_dim

        self._initialized = True

    def load_checkpoint(self):
        """loads checkpoint from filesystem and passes the values into state dict of model
        """
        self._model = self._build_model_from_config()
        checkpoint = torch.load(self._checkpoint)
        self._model.load_state_dict(checkpoint["model_state_dict"])

    def train(self, *args, **kwargs) -> None:
        if self._initialized:
            self._init_train()
            self._trainer.fit()
        logging.error("Starting training was not possible because process was not build")

    @abstractmethod
    def feed_forward_inference(self) -> None:
        pass
    
    @abstractmethod
    def greedy_inference(self) -> None:
        pass
    def _init_train(self, *args, **kwargs) -> None:
        """initialized training process. Method to call right before a training process"""
        self._store_config(self._save_dir)

    def _build_logger(self) -> List[Union[SummaryWriter, Logger]]:
        """builds save_dir and logger

        builds:
         - FileSystemLogger
         - Tensorboard SummaryWriter
        Returns:
            List[Union[SummaryWriter, FileSystemLogger]]: list of logger which implement all the same interface
        """
        fs_logger = FileSystemLogger(self._save_dir)
        tb_logger = SummaryWriter(self._save_dir)

        return [tb_logger, fs_logger]

    def _get_config_path(self, **kwargs) -> str:
        """extract config path from maybe given checkpoint otherwise return base config path

        Returns:
            str: config path
        """
        if "checkpoint" in kwargs.keys():
            config_path = "/".join(kwargs["checkpoint"].split("/")[:-1]) + "/config.yaml"
            return config_path
        return self._base_config_path

    @abstractmethod
    def _load_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @abstractmethod
    def _load_criterion(self) -> Criterion:
        pass

    @abstractmethod
    def _build_model(self) -> NeuralNetwork:
        pass

    @abstractmethod
    def _build_model_from_config(self) -> NeuralNetwork:
        """in this function we rely only on information provided by the config file

        Returns:
            NeuralNetwork: build model
        """
        pass

    @abstractmethod
    def _build_trainer(self) -> Trainer:
        """build and return trainer object

        Returns:
            Trainer: trainer object which is build
        """
        pass
