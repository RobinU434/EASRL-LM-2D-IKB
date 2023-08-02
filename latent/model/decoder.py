import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
       
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, z):
        z = self.linear(z)
        return z 
