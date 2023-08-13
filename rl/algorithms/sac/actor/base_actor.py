from enum import Enum
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Sequential
from torch.distributions import Normal, constraints
from utils.metrics import Metrics

from utils.model.neural_network import FeedForwardNetwork


class Actor(FeedForwardNetwork):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: List[int],
        activation_function: str,
        learning_rate: float,
    ) -> None:
        super().__init__(
            input_dim, output_dim, architecture, activation_function, learning_rate
        )
        # add input layers
        self._linear_mu = nn.Linear(architecture[-1], output_dim)
        self._linear_std = nn.Linear(architecture[-1], output_dim)

        self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def _create_linear_unit(self, architecture: List[int]) -> Sequential:
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
        # output layer
        sequence = nn.Sequential(*layers)
        return sequence

    def forward(self, x):
        x = self._linear(x)
        mu = self._linear_mu(x)
        std = F.softplus(self._linear_std(x))

        dist = Normal(
            mu, std + 1e-28, validate_args={"scale": constraints.greater_than_eq}
        )
        # dist = self.action_sampling_func(mu, std, self.action_sampling_mode, self.covariance_decay)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        # sum log prob TODO: Does this still hold with dependent log probabilities?
        log_prob = log_prob.sum(dim=1)
        # independence assumption between individual probabilities
        # log(p(a1, a2)) = log(p(a1) * p(a2)) = log(p(a1)) + log(p(a2))

        return mu, log_prob

    def train(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def save(self, path: str, metrics: Metrics = ..., epoch_idx: int = 0):
        path = self._create_save_path(path, epoch_idx, metrics)
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
            },
            path,
        )

    def _create_save_path(
        self, path: str, epoch_idx: int, metrics: Metrics = ...
    ) -> str:
        path += f"/{type(self).__name__}_{epoch_idx}"
        if "reward" in vars(metrics).keys():
            path += f"reward_{metrics.reward.mean().item():.4f}.pt"  # type: ignore
        else:
            path += ".pt"
        return path

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer
