from argparse import ArgumentParser
from copy import copy

from latent.criterion.elbo_criterion import ELBO, InvKinELBO
from latent.datasets.vae_dataset import VAEDataset
from latent.model.utils.post_processor import PostProcessor
from latent.model.vae import VAE
from latent.process.latent_process import LatentProcess
from latent.trainer.base_trainer import Trainer
from latent.trainer.vae_trainer import VAETrainer
from torch.utils.data import DataLoader


class VAEProcess(LatentProcess):
    def __init__(self, **kwargs) -> None:
        self._model_entity_name = "vae"
        super().__init__(**kwargs)
        
        self._criterion: ELBO
        self._model: VAE

        self._train_data: DataLoader[VAEDataset]
        self._val_data: DataLoader[VAEDataset]
        self._val_data: DataLoader[VAEDataset]

    def inference(self, *args, **kwargs) -> None:
        return super().inference(*args, **kwargs)
    
    def feed_forward_inference(self) -> None:
        return super().feed_forward_inference()
    
    def greedy_inference(self) -> None:
        self.load_checkpoint()
        
    
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

    def _build_model(self) -> VAE:
        model_config = copy(self._config["model"])
        autoencoder = VAE(
            input_dim=self._train_data.dataset.input_dim,
            latent_dim=model_config["latent_dim"],
            output_dim=self._config["n_joints"],
            conditional_info_dim=self._train_data.dataset.conditional_dim,
            encoder_config=copy(model_config["encoder"]),
            decoder_config=copy(model_config["decoder"]),
            post_processor=PostProcessor(**self._config["post_processor"]),
            learning_rate=self._config["learning_rate"],
            store_history=True,
            device=self._device,
            verbose=True,
        )

        # save additional information in config file 
        self._config["model"]["conditional_info_dim"] = list(self._train_data.dataset.conditional_dim)
        self._config["model"]["input_dim"] = autoencoder._input_dim

        return autoencoder

    def _build_model_from_config(self) -> VAE:
        model_config = copy(self._config)
        autoencoder = VAE(
            input_dim=model_config["input_dim"],
            latent_dim=model_config["latent_dim"],
            output_dim=self._config["n_joints"],
            conditional_info_dim=self._config["conditional_dim"],
            encoder_config=copy(model_config["encoder"]),
            decoder_config=copy(model_config["decoder"]),
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
