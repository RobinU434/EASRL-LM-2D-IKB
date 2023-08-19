from abc import abstractmethod
from copy import copy
import json
import logging
import time
from typing import List, Tuple, Union

import torch
import latent.datasets as datasets
from latent.datasets.latent_dataset import LatentDataset
from latent.criterion.base_criterion import Criterion
from latent.datasets.load_dataset import load_data
from utils.model.neural_network import NeuralNetwork
from latent.trainer.base_trainer import Trainer
from logger.base_logger import Logger
from logger.fs_logger import FileSystemLogger

from utils.learning_process import LearningProcess

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from utils.decorator import not_implemented_warning


class LatentProcess(LearningProcess):
    def __init__(self, **kwargs) -> None:
        self._base_config_path = f"config/base_{self._model_entity_name}.yaml"
        super().__init__(**kwargs)
        self._train_data: DataLoader[LatentDataset]
        self._val_data: DataLoader[LatentDataset]
        self._test_data: DataLoader[LatentDataset]
        
        self._criterion: Criterion
        self._model: NeuralNetwork
        self._trainer: Trainer

    def print_config(self) -> None:
        """prints config"""
        print(json.dumps(self._config, sort_keys=True, indent=4))

    def print_model(self) -> None:
        """prints model"""
        print(self._model)

    def _build(self, no_logger: bool = False):
        """builds core component of process:

        - loads datasets
        - build model
        - build logger
        - build trainer
        - load criterion

        Args:
            no_logger (bool): prevents build of logger
        """
        if no_logger:
            self._logger = []
        else:
            self._logger = self._build_logger()

        self._set_data()

        self._criterion = self._load_criterion()
        self._model = self._build_model()
        self._trainer = self._build_trainer()

        # add additional information to config file
        self._config["input_dim"] = self._model._input_dim
        self._config["output_dim"] = self._model._output_dim

    def load_checkpoint(self):
        """builds model based on checkpoint config and loads checkpoint from filesystem and passes the values into state dict of model"""
        self._model = self._build_model_from_config()
        checkpoint = torch.load(self._checkpoint)
        self._model.load_state_dict(checkpoint["model_state_dict"])

    def train(self, *args, **kwargs) -> None:
        logging.info("Start training")
        if self._initialized:
            self._init_train()
            self._trainer.fit()
        logging.error(
            "Starting training was not possible because process was not build"
        )

    @abstractmethod
    def feed_forward_inference(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def greedy_inference(self) -> None:
        raise NotImplementedError

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
            config_path = (
                "/".join(kwargs["checkpoint"].split("/")[:-1]) + "/config.yaml"
            )
            return config_path
        return self._base_config_path

    def _set_data(self):
        """sets internal train_data, val_data and test data"""
        dataset_config = copy(self._config["dataset"])
        dataset_config["type"] = getattr(datasets, dataset_config["type"])
        self._train_data = load_data(
            n_joints=self._config["n_joints"],
            data_entity="train",
            **dataset_config,
        )
        self._val_data = load_data(
            n_joints=self._config["n_joints"],
            data_entity="val",
            **dataset_config,
        )
        self._test_data = load_data(
            n_joints=self._config["n_joints"],
            data_entity="test",
            **dataset_config,
        )

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
