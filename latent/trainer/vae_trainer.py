import logging
from typing import List, Union
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from latent.criterion.elbo_criterion import ELBO, InvKinELBO
from latent.datasets.vae_dataset import VAEDataset
from latent.datasets.utils import TargetMode
from latent.metrics.vae_metrics import VAEInvKinMetrics
from latent.model.vae import VAE
from latent.trainer.base_trainer import Trainer
from logger.base_logger import Logger


class VAETrainer(Trainer):
    def __init__(
        self,
        model: VAE,
        train_data: DataLoader,
        val_data: DataLoader,
        test_data: DataLoader,
        n_epochs: int,
        criterion: ELBO,
        logger: List[Union[SummaryWriter, Logger]] = [],
        val_interval: int = 5,
        device: str = "cpu",
        results_path: str = "results",
    ) -> None:
        super().__init__(
            model,
            train_data,
            val_data,
            test_data,
            n_epochs,
            criterion,
            logger,
            val_interval,
            device,
            results_path,
        )

        self._criterion: InvKinELBO
        self._model: VAE

    def _run_model(self, data: DataLoader, train: bool = False) -> VAEInvKinMetrics:
        """runs the given autoencoder on the given dataset

        Args:
            autoencoder (VariationalAutoencoder): model to run the data on
            data (DataLoader): dataloader to draw data from. It is recommended that data.dataset inherit from VAEDataset
            criterion (IKLoss): loss function
            device (str): _description_
            train (bool, optional): _description_. Defaults to False.

        Returns:
            np.ndarray: structured array with keys:
            - reconstruction_loss
            - kl_loss
            - total_loss
            - std
            - imitation_loss
            - distance_loss
        """
        self._model.reset_history()

        if not isinstance(data.dataset, VAEDataset):
            logging.warning(
                "given dataset does not inherit from VAEDataset. As long it returns (x, c_enc, c_dec, y) on __getitem__ it is fine"
            )
        # prepare logging
        log_metrics_array = []
        log_metrics_names = [
            "reconstruction_loss",
            "kl_loss",
            "loss",
            "log_std",
            "imitation_loss",
            "distance_loss",
        ]
        for x, c_enc, c_dec, y in data:
            x = x.to(self._device)
            c_enc = c_enc.to(self._device)
            c_dec = c_dec.to(self._device)
            y = y.to(self._device)

            x_hat, mu, log_std = self._model.forward(
                x, c_enc, c_dec
            )  # out shape: (batch_size, number of joints)

            # TODO: it is a bit hacky and has to be adapted for other datasets where the current angles are not inside c_dec
            current_angles = c_dec[:, -self._model._output_dim :]
            action = x_hat + current_angles
            if self._criterion._target_mode == TargetMode.ACTION:
                # y is expected to be the target action we want to encode
                y = y + current_angles
            loss = self._criterion(y=y, x_hat=action, mu=mu, log_std=log_std)

            if train:
                self._model.train(loss)

            # log metrics in the correct order
            log_metrics_array.append(
                np.array(
                    [
                        self._criterion.reconstruction_loss.cpu().item(),  # reconstruction_loss
                        self._criterion.kl_loss.cpu().item(),  # kl loss
                        loss.cpu().item(),  # total loss
                        log_std.mean().cpu().item(),  # std
                        self._criterion.imitation_loss.cpu().item(),  # imitation_loss
                        self._criterion.distance_loss.cpu().item(),  # distance_loss
                    ]
                )
            )
        # make structured array
        log_metrics_array = np.stack(log_metrics_array)
        log_metrics_dict = dict(zip(log_metrics_names, log_metrics_array.T))
        log_metrics = VAEInvKinMetrics.from_dict(log_metrics_dict)
        return log_metrics

    def _print_status(self, entity: str, metrics: VAEInvKinMetrics) -> None:
        print(
            f"{entity}: loss: {metrics.reconstruction_loss.mean():.5f} kl_div: {metrics.kl_loss.mean():.5f}"
        )
