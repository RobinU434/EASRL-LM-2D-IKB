import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MuNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float):
        super(MuNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, self.output_dim)

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu
    
    def train(self, loss: torch.tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()