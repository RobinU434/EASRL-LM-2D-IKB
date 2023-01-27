from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim


class Actor(nn.Module):
    def __init__(self, input_size, output_size, learning_rate) -> None:
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU()
        )

        self.linear_mu = nn.Linear(128, output_size)
        self.linear_std = nn.Linear(128, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.linear(x)
        mu = self.linear_mu(x)
        std = F.softplus(self.linear_std(x))

        return mu, std

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TripleAgent(nn.Module):
    def __init__(self, input_size, output_size, learning_rate) -> None:
        super().__init__()
        
        # test if the required output size an integer multiple of 3
        assert output_size % 3 == 0
        agent_output_size = output_size // 3

        self.agent1 = Actor(input_size, agent_output_size, learning_rate)
        self.agent2 = Actor(input_size, agent_output_size, learning_rate)
        self.agent3 = Actor(input_size, agent_output_size, learning_rate)
        
    def forward(self, x):
        mu1, std1 = self.agent1.forward(x)
        mu2, std2 = self.agent2.forward(x)
        mu3, std3 = self.agent3.forward(x)

        mu = torch.cat([mu1, mu2, mu3])
        std = torch.cat([std1, std2, std3])

        return mu, std

    def train(self, loss):
        self.agent1.train(loss)
        self.agent2.train(loss)
        self.agent3.train(loss)


class MultiAgent(nn.Module):
    def __init__(self, input_size, output_size, learning_rate, num_agents) -> None:
        super().__init__()

        # test if the required output size an integer multiple of num agents
        assert output_size % num_agents == 0
        agent_output_size = output_size // 3

        self.agents = []
        for _ in range(num_agents):
            self.agents.append(Actor(input_size, agent_output_size, learning_rate))

    def forward(self, x) -> Tuple[torch.tensor, torch.tensor]:
        mu_list = []
        std_list = []

        for agent in self.agents:
            mu, std = agent.forward(x)
            mu_list.append(mu)
            std_list.append(std)

        mu = torch.cat(mu_list)
        std = torch.cat(std_list)

        return mu, std

    def train(self, loss) -> None:
        for agent in self.agents:
            agent.train(loss)
