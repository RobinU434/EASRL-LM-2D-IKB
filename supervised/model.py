import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from vae.utils.post_processing import PostProcessor


class Regressor(nn.Module):
    """Model will produce action to go from a fixed start position to a defined goal position
    input: 
    target position

    output:
    corresponding angels to got to the requested target position

    """
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float, post_processor: PostProcessor) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        self.post_processor = post_processor

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate) 

    def forward(self, x):
        out = self.model.forward(x)
        out = self.post_processor(out)

        return out

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def build_model(feature_source: str, num_joints: int, learning_rate: float, post_processor_config: Dict) -> Regressor:
    post_processor = PostProcessor(**post_processor_config)
    if feature_source == "state" or feature_source == "noisy_targets":
        logging.info("use 'state' as feature source")
        model = Regressor(4 + num_joints, num_joints, learning_rate, post_processor)
    elif feature_source == "targets":
        logging.info("use 'targets' as feature source")
        model = Regressor(2, num_joints, learning_rate, post_processor)
    else:
        logging.error(f"feature source has to be either 'targets' or 'state', you chose: {feature_source}")
    return model 
