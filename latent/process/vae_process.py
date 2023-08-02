from argparse import ArgumentParser
from copy import copy
import json
import time
from typing import Any, Callable, Tuple

from latent.criterion.elbo_criterion import ELBO, InvKinELBO
from latent.data.vae_dataset import VAEDataset
from latent.data.load_vae_dataset import load_action_dataset, load_action_target_dataset, load_target_dataset
from latent.model.base_model import NeuralNetwork
from latent.model.utils.post_processor import PostProcessor
from latent.model.vae import VariationalAutoencoder
from latent.process.latent_process import LatentProcess
from latent.trainer.base_trainer import Trainer
from latent.trainer.vae_trainer import VAETrainer
from process.base_process import Process
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


class VAEProcess(LatentProcess):
    def __init__(self, **kwargs) -> None:
        self._model_entity = "vae"
        super().__init__(**kwargs)
        
        self._criterion: ELBO
        self._model: VariationalAutoencoder

        self._train_data: DataLoader[VAEDataset]
        self._val_data: DataLoader[VAEDataset]
        self._val_data: DataLoader[VAEDataset]

    def inference(self, *args, **kwargs) -> None:
        return super().inference(*args, **kwargs)
    
    def feed_forward_inference(self) -> None:
        return super().feed_forward_inference()
    
    def greedy_inference(self) -> None:
        return super().greedy_inference()

    def _load_datasets(
        self,
    ) -> Tuple[DataLoader[VAEDataset], DataLoader[VAEDataset], DataLoader[VAEDataset]]:
        """loads datasets from filesystem and puts them into DataLoader objects

        Raises:
            ValueError: if the configured dataset is not implemented

        Returns:
            Tuple[DataLoader[VAEDataset], DataLoader[VAEDataset], DataLoader[VAEDataset]]: train val and test data
        """
        if self._config["dataset"] == "action":
            train_loader, val_loader, test_loader = load_action_dataset(self._config)

        elif self._config["dataset"] == "action_target_v1":
            train_loader, val_loader, test_loader = load_action_target_dataset(
                self._config
            )

        elif self._config["dataset"] == "action_target_v2":
            train_loader, val_loader, test_loader = load_action_target_dataset(
                self._config
            )

        elif self._config["dataset"] == "conditional_action_target":
            train_loader, val_loader, test_loader = load_action_target_dataset(
                self._config
            )

        elif self._config["dataset"] == "target_gaussian":
            train_loader, val_loader, test_loader = load_target_dataset(self._config)
        else:
            raise ValueError("you have not selected the right dataset in your config")

        return train_loader, val_loader, test_loader

    def _load_criterion(self) -> ELBO:
        """initializes loss function

        Returns:
            ELBO: Evidence Lower Bound Criterion
        """
        loss_config = copy(self._config["loss_func"])
        loss_func = InvKinELBO(
            **loss_config,
            target_mode=self._train_data.dataset.target_mode,
            device=self._device,
        )

        return loss_func

    def _build_model(self) -> VariationalAutoencoder:
        autoencoder = VariationalAutoencoder(
            input_dim=self._train_data.dataset.input_dim,
            latent_dim=self._config["latent_dim"],
            output_dim=self._config["num_joints"],
            conditional_info_dim=self._train_data.dataset.conditional_dim,
            learning_rate=self._config["learning_rate"],
            post_processor=PostProcessor(**self._config["post_processor"]),
            store_history=True,
            device=self._device,
            verbose=True,
        ).to(self._device)

        # save additional information in config file 
        self._config["conditional_info_dim"] = self._train_data.dataset.conditional_dim

        return autoencoder

    def _build_model_from_config(self) -> VariationalAutoencoder:
        autoencoder = VariationalAutoencoder(
            input_dim=self._config["input_dim"],
            latent_dim=self._config["latent_dim"],
            output_dim=self._config["num_joints"],
            conditional_info_dim=self._config["conditional_dim"],
            learning_rate=self._config["learning_rate"],
            post_processor=PostProcessor(**self._config["post_processor"]),
            store_history=True,
            device=self._device,
            verbose=True,
        ).to(self._device)
        return autoencoder


    def _build_trainer(self) -> Trainer:

        trainer = VAETrainer(
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
