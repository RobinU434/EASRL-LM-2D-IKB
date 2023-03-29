import torch
import torch.nn as nn
import torch.optim as optim

class Regressor(nn.Module):
    """Model will produce action to go from a fixed start position to a defined goal position
    input: 
    target position

    output:
    corresponding angels to got to the requested target position

    """
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh(),
        )

        self.optimizer = optim.Adam(self.parameters()) 

    def forward(self, x):
        out = self.model.forward(x)
        # out = (out * 2 * torch.pi) # % (2 * torch.pi)
        # print(out.mean())

        return out

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Leverage(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * 2 * torch.pi