from typing import Any, List, Union

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from latent.criterion.base_criterion import Criterion
from latent.criterion.ik_criterion import IKLoss
from latent.datasets.utils import split_state_information
from latent.metrics.base_metrics import Metrics
from latent.model.base_model import NeuralNetwork
from latent.trainer.base_trainer import Trainer
from logger.base_logger import Logger


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        model: NeuralNetwork,
        train_data: DataLoader,
        val_data: DataLoader,
        test_data: DataLoader,
        n_epochs: int,
        criterion: Criterion,
        logger: List[Union[SummaryWriter, Logger]] = None,
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
        self._criterion: IKLoss
        self._metric_names = ["loss", "imitation_loss", "distance_loss"]

    def _run_model(self, data: DataLoader, train: bool = False) -> Metrics:
        metrics = []
        for x, y in data:
            x = x.to(self._device)
            y = y.to(self._device)

            x_hat = self._model.forward(x)

            _, _, state_angles = split_state_information(x)
            loss = self._criterion(y, state_angles + x_hat)

            if train:
                self._model.train(loss)

            metrics.append(
                np.array(
                    [
                        loss.item(),
                        self._criterion.imitation_loss.item(),
                        self._criterion.distance_loss.item(),
                    ]
                )
            )

        metrics = np.stack(metrics).T
        metrics = Metrics.build_from_dict(dict(zip(self._metric_names, metrics)))
        return metrics

    def _print_status(self, entity: str, metrics: Metrics) -> None:
        print(f"{entity}_loss: {metrics.loss.mean()}")

    def predict(self, data: DataLoader) -> Any:
        return self._model.forward(data)
