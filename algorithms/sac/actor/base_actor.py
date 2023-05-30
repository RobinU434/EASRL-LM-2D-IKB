import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from enum import Enum

from typing import List

class TrainMode(Enum):
    STATIC = 0  # no training is permitted
    FINE_TUNING = 1  # load a model and perform fine tuning
    FROM_SCRATCH = 2


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate, architecture: List[int] = [128, 128]) -> None:
        super().__init__()
        
        # add input layers
        layers = [
            nn.Linear(input_dim, architecture[0]),
            nn.ReLU(),
        ]
        
        # add hidden layers
        for idx in range(len(architecture) - 1):
            layers.extend([
                nn.Linear(architecture[idx], architecture[idx + 1]),
                nn.ReLU(),
                ]
            )
        self.linear = nn.Sequential(*layers)

        self.linear_mu = nn.Linear(architecture[-1], output_dim)
        self.linear_std = nn.Linear(architecture[-1], output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.linear(x)
        mu = self.linear_mu(x)
        std = F.softplus(self.linear_std(x))
        return mu, std

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # grad_tensor = torch.tensor([])
        # for param in self.parameters():
        #     grad_tensor = torch.cat([grad_tensor, param.grad.flatten()])
        # print("____")
        # print(loss)
        # print(grad_tensor.mean())
        self.optimizer.step()
