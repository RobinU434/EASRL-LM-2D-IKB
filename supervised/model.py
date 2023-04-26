import logging
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
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate) 

    def forward(self, x):
        out = self.model.forward(x)
        return out

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def build_model(feature_source: str, num_joints: int, learning_rate: float) -> Regressor:
    if feature_source == "state":
        logging.info("use 'state' as feature source")
        model = Regressor(4 + num_joints, num_joints, learning_rate)
    elif feature_source == "targets":
        logging.info("use 'targets' as feature source")
        model = Regressor(2, num_joints, learning_rate)
    else:
        logging.error(f"feature source has to be either 'targets' or 'state', you chose: {feature_source}")
    return model 
