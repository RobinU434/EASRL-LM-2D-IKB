from abc import ABC, abstractmethod
from typing import List, Union

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from latent.metrics.base_metrics import Metrics
from latent.criterion.base_criterion import Criterion
from latent.model.base_model import NeuralNetwork
from logger.base_logger import Logger


class Trainer(ABC):
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
        super().__init__()
        self._model = model
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data
        self._n_epochs = n_epochs
        self._criterion = criterion

        self._logger = logger
        self._val_interval = val_interval
        self._device = device
        self._results_path = results_path

    def fit(self):
        # init storage usage
        train_metrics: Metrics
        val_metrics: Metrics
        test_metrics: Metrics

        for epoch_idx in range(self._n_epochs):
            train_metrics = self._run_model(data=self._train_data, train=True)

            if epoch_idx % self._val_interval == 0:
                val_metrics = self._run_model(data=self._val_data)

                self._model.save(
                    path=self._results_path
                    + f"/model_{epoch_idx}_val_loss_{val_metrics.loss.mean().item():.4f}.pt",
                    epoch_idx=epoch_idx,
                    metrics=val_metrics,
                )

            if self._logger is not None and epoch_idx % self._val_interval == 0:
                print(f"============== Epoch: {epoch_idx} ==============")
                self._print_status("train", train_metrics)
                self._print_status("val", val_metrics)

                self._log_scalar_metrics("train", train_metrics, epoch_idx)
                self._log_scalar_metrics("val", val_metrics, epoch_idx)

                self._model.log_internals(self._logger, epoch_idx)

        print("============== Finished Training ==============")
        test_metrics = self._run_model(data=self._test_data)
        self._log_scalar_metrics("test", test_metrics, self._n_epochs)
        self._print_status("test", test_metrics)

    def _log_scalar_metrics(self, entity: str, metrics: Metrics, epoch_idx: int) -> None:
        """send the incoming data to the summary writer

        Args:
            entity (str): is either train, val or test
            metrics (Metrics): metrics data class object
            epoch_idx (int): at which epoch the data was recorded
        """
        for logger in self._logger:
            for metric_name, metric_value in metrics.mean().items():
                logger.add_scalar(
                    type(self._model).__name__ + "_" + entity + "/" + metric_name,
                    metric_value,
                    epoch_idx,
                )

    def _log_model_hparams(self):
        """loggs hparams dict"""
        for logger in self._logger:
            logger.add_hparams(hparam_dict=self._model.hparams, metric_dict={})

    @abstractmethod
    def _run_model(self, data: DataLoader, train: bool = False) -> Metrics:
        """how the model is executed on data and how the logging metrics are recorded"""
        pass

    @abstractmethod
    def _print_status(self, entity: str, metrics: Metrics) -> None:
        """method gets called every val_interval.
        Its purpose is to print the current training status

        Args:
            entity (str): is either "train", "val" or "test"
            metrics (Metrics): metrics to print metrics
        """
        pass
