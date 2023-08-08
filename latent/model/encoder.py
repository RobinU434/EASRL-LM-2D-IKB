from typing import List
import torch.nn as nn

from utils.model.neural_network import FeedForwardNetwork


class VariationalEncoder(FeedForwardNetwork):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: List[int],
        activation_function: str,
        learning_rate: float,
        **kwargs
    ) -> None:
        super().__init__(
            input_dim, output_dim, architecture, activation_function, learning_rate
        )
        # overwrite linear unit created by parent
        self._linear_mu = nn.Linear(architecture[-1], output_dim)
        self._linear_std = nn.Linear(architecture[-1], output_dim)

    def _create_linear_unit(self, architecture: List[int]) -> nn.Sequential:
        """create linear layers but dont create the output layer

        Args:
            architecture (List[int]): description of architecture

        Returns:
            Sequential: sequential linear layers
        """
        layers = [
            nn.Linear(self._input_dim, int(architecture[0])),
            self._activation_function_type(),
        ]
        # add hidden layers
        for idx in range(len(architecture) - 1):
            layers.extend(
                [
                    nn.Linear(int(architecture[idx]), int(architecture[idx + 1])),
                    self._activation_function_type(),
                ]
            )
        sequence = nn.Sequential(*layers)
        return sequence

    def forward(self, x):
        x = self._linear(x)
        mu = self._linear_mu(x)
        # linear_std is calculating the log std
        log_std = self._linear_std(x)
        return mu, log_std

