from typing import Any, Dict, List, Union
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from latent.metrics.supervised_metrics import SupervisedIKMetrics
from utils.model.neural_network import FeedForwardNetwork, NeuralNetwork
from latent.model.utils.post_processor import PostProcessor
import torch.nn as nn
from torch import Tensor

from logger.base_logger import Logger


class Regressor(FeedForwardNetwork):
    """Model will produce action to go from a fixed start position to a defined goal position
    input:
    target position

    output:
    corresponding angels to got to the requested target position

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: List[int],
        activation_function: str,
        learning_rate: float,
        post_processor: PostProcessor,
        device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            input_dim,
            output_dim,
            architecture,
            activation_function,
            learning_rate,
            device,
            **kwargs
        )
        self._post_processor = post_processor

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Regressor":
        input_dim = config["n_joints"] + 4
        output_dim = config["n_joints"]
        post_processor = PostProcessor(**config["post_processor"])
        regressor = cls(
            input_dim=input_dim,
            output_dim=output_dim,
            post_processor=post_processor,
            **config["model"]
        )
        return regressor

    def forward(self, x: Tensor):
        out = self._linear.forward(x)
        out = self._post_processor(out)
        return out

    def save(self, path: str, epoch_idx: int, metrics: SupervisedIKMetrics):
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": metrics.loss.mean(),
            },
            path,
        )

    def log_internals(self, logger: List[Union[SummaryWriter, Logger]], epoch_idx: int):
        pass

    @property
    def hparams(self) -> Dict[str, Union[str, int, float]]:
        return {"learning_rate": self._learning_rate}

    @property
    def post_processor(self) -> PostProcessor:
        return self._post_processor
