from abc import ABC, abstractmethod
import logging
import random
import time
from typing import Any
import numpy as np

import torch

from utils.file_system import load_yaml, write_yaml


class LearningProcess(ABC):
    def __init__(self, device: str = "cpu", **kwargs) -> None:
        super().__init__()
        self._model_entity_name: str
        """str: what kind of model you want to train. Value gets used for logging training results, loading config, building save_directory"""
        self._base_config_path: str
        """str: path to base config. Used as fallback solution if no explicit path was provided. 'config/base_<model_entity>.yaml"""
        self._config = load_yaml(self._get_config_path(**kwargs))
        """Dict[str, Any]: config for model to run process on"""
        self._device = device
        """str: on which device should you run the process. cpu, cuda: 0, ..."""
        self._subdir: str = self._extract_from_kwargs("subdir", **kwargs)
        """str: subdir where to store checkpoints"""
        self._save_dir: str
        """str: path to directory where all metrics and files are stored during the process. Has rough structure of 'results/<model_entity>/<subdir>/<x>_<time_stamp>"""
        self._checkpoint: str = self._extract_from_kwargs("checkpoint", **kwargs)
        """str: Variable contains path to checkpoint if provided. Else: '' """
        self._sample_size: int = self._extract_from_kwargs("sample_size", **kwargs)
        """int: parameter stores sample size for inference step"""

        # set seed
        self._random_seed = self._extract_from_kwargs("random_seed", **kwargs)
        """value has to be set on build time
        """

        self._initialized = False

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """start learning process"""
        pass

    @abstractmethod
    def inference(self, *args, **kwargs) -> None:
        """executes inference"""
        pass

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        """builds more complex or memory / computing intense components of Process like a model, datasets, environments"""
        pass

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs) -> None:
        """loads checkpoint from filesystem"""

    def _store_config(self, path: str):
        """save config in file system

        Args:
            path (str): where to store config
        """
        write_yaml(path + "/config.yaml", self._config)

    def _build_save_dir_path(self) -> str:
        """function builds path to save directory

        Args:
            model_entity (str): what kind of model we want to train

        Returns:
            str: path to save directory
        """
        if self._checkpoint is not None:
            save_dir = "/".join(self._checkpoint.split("/")[:-1])
        elif self._subdir is not None:
            save_dir = (
                f"results/{self._model_entity_name}/{self._subdir}/{self._config['n_joints']}_"
            )
            # add latent dim if possible
            try:
                save_dir += f"{self._config['model']['latent_dim']}_"
            except KeyError:
                pass
            save_dir += f"{int(time.time())}"
        else:
            logging.warning("No save directory set")
            save_dir = "."
        return save_dir

    def _get_config_path(self, **kwargs) -> str:
        """extract config path from maybe given checkpoint otherwise return base config path

        Returns:
            str: config path
        """
        if "checkpoint" in kwargs.keys():
            print("/".join(kwargs["checkpoint"].split("/")[:-1]))
            config_path = (
                "/".join(kwargs["checkpoint"].split("/")[:-1])
                + f"/{self._model_entity_name.upper()}_config.yaml"
            )
            return config_path
        return self._base_config_path

    @staticmethod
    def _set_random_seed(random_seed: int):
        """sets random seed inside random number generators

        Args:
            random_seed (int):

        """
        if random_seed is None:
            random_seed = int(time.time())
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    @staticmethod
    def _extract_from_kwargs(key: str, **kwargs) -> Any:
        """extract data of key from kwargs

        Args:
            key (str): key in kwargs

        Returns:
            Any: value behind kwargs[key]
        """

        if key in kwargs.keys():
            return kwargs[key]
