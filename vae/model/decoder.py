import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        # self.linear1 = nn.Linear(latent_dim, 512)
        # self.linear2 = nn.Linear(512, output_dim)
        
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 512),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, z):
        # z = F.relu(self.linear1(z))
        # z = torch.sigmoid(self.linear2(z))
        z = self.linear(z)
        # z = z * torch.pi / 1.8342796626648048
        return z 
