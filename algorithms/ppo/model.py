import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Module(nn.Module):
    def __init__(self, learning_rate, input_size, output_size) -> None:
        super(Module, self).__init__()

        self.fc1   = nn.Linear(input_size, 128)
        self.fc_mu = nn.Linear(128, output_size)
        self.fc_std  = nn.Linear(128, output_size)
        self.fc_v = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def value(self, state):
        hidden = F.relu(self.fc1(state))
        value = self.fc_v(hidden)
        return value

    def pi(self, state, softmax_dim = 0):
        hidden = F.relu(self.fc1(state))

        mu = torch.tanh(self.fc_mu(hidden))
        std = F.softplus(self.fc_std(hidden))

        if torch.isnan(std).any():
            # access point for debugging if the weights become nan
            # why is this happening?????!!!!
            print("found mal value")

        return mu, std

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.optimization_step += 1