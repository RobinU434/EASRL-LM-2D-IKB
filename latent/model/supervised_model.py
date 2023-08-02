from typing import Dict, List, Union
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from latent.metrics.supervised_metrics import SupervisedIKMetrics
from latent.model.base_model import NeuralNetwork
from latent.model.utils.post_processor import PostProcessor
import torch.nn as nn
from torch import Tensor

from logger.base_logger import Logger


class Regressor(NeuralNetwork):
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
        learning_rate: float,
        post_processor: PostProcessor,
        action_radius: float = 0,  # 0 -> arm will be trained move without distance restriction. value > 0 -> trained move restriction around start position
        **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim, learning_rate)
        self._action_radius = action_radius

        self._model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            # nn.Linear(256, 256),
            # nn.LeakyReLU(),
            nn.Linear(256, output_dim),
        )
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)

        self._post_processor = post_processor

    def forward(self, x: Tensor):
        out = self._model.forward(x)
        out = self._post_processor(out)

        return out

    def train(self, loss: Tensor):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

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