import logging
from copy import copy
from typing import Tuple
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

import numpy as np
from progress.bar import Bar
import torch
from torch.utils.data import DataLoader

from envs.plane_robot_env.ikrlenv.robots.ccd import ccd
from latent.criterion.base_criterion import Criterion
from latent.criterion.ik_criterion import IKLoss
from latent.datasets.utils import TargetMode
from latent.model.regressor import Regressor
from latent.model.utils.post_processor import PostProcessor
from latent.process.latent_process import LatentProcess
from latent.trainer.base_trainer import Trainer
from latent.trainer.supervised_trainer import SupervisedTrainer
from utils.kinematics.kinematics import forward_kinematics
from utils.sampling import sample_target


class SupervisedProcess(LatentProcess):
    def __init__(self, **kwargs) -> None:
        self._model_entity_name = "supervised"
        super().__init__(**kwargs)

        self._model: Regressor

    def inference(self, *args, **kwargs) -> None:
        pass

    def feed_forward_inference(self) -> None:
        pass

    def greedy_inference(self) -> None:
        pass

    def _load_criterion(self) -> Criterion:
        loss_config = copy(self._config["loss_func"])
        loss_config.pop("type")
        criterion = IKLoss(
            **loss_config, target_mode=self._train_data.dataset.target_mode
        )

        return criterion

    def _build_model(self) -> Regressor:
        post_processor = PostProcessor(**copy(self._config["post_processor_config"]))
        feature_source = self._config["feature_source"]
        n_joints = self._config["n_joints"]
        learning_rate = self._config["learning_rate"]
        action_radius = self._config["action_radius"]

        if feature_source == "state" or feature_source == "gaussian_target":
            logging.info("use 'state' as feature source")
            model = Regressor(
                4 + n_joints,
                n_joints,
                learning_rate,
                post_processor,
                action_radius,
            )
        elif feature_source == "targets":
            logging.info("use 'targets' as feature source")
            model = Regressor(
                2, n_joints, learning_rate, post_processor, action_radius
            )
        else:
            logging.error(
                f"feature source has to be either 'targets' or 'state', you chose: {feature_source}"
            )
            raise NotImplementedError

        return model.to(self._device)

    def _build_model_from_config(self) -> Regressor:
        """in this function we rely only on information provided by the config file

        Returns:
            Regressor: _description_
        """
        return self._build_model()

    def _build_trainer(self) -> Trainer:
        trainer = SupervisedTrainer(
            model=self._model,
            train_data=self._train_data,
            val_data=self._val_data,
            test_data=self._test_data,
            n_epochs=self._config["n_epochs"],
            criterion=self._criterion,
            logger=self._logger,
            val_interval=5,
            device=self._device,
            results_path=self._save_dir,
        )

        return trainer
