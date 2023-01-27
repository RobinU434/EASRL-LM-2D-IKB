import torch
import torch.nn as nn

from vae.model.decoder import Decoder
from vae.model.encoder import Encoder


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc # .cuda() hack to get sampling on the GPU
        # self.N.scale = self.N.scale # .cuda()

    def forward(self, x):
        mu, log_std = self.encoder(x)
        
        # sample the latent space
        sigma = torch.exp(log_std)
        z = mu + sigma * self.N.sample(mu.shape)

        return self.decoder(z), mu, log_std
