# https://avandekleut.github.io/vae/

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from vae.model.decoder import Decoder
from vae.model.encoder import VariationalEncoder


class VariationalAutoencoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 latent_dim: int,
                 output_dim: int,
                 learning_rate: float,
                 logger: SummaryWriter, 
                 conditional_info_dim: int = 0,
                 store_history: bool = False,
                 device: str = "cpu"):
        super(VariationalAutoencoder, self).__init__()
        self.conditional_info_dim = conditional_info_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.encoder = VariationalEncoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim + self.conditional_info_dim, output_dim)

        self.N = torch.distributions.Normal(0, 1)
        # self.N = torch.distributions.Normal(torch.tensor(0).to(device), torch.tensor(1).to(device))

        if "cuda" in device:
            self.N.loc = self.N.loc.to(device)  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.to(device)

        self.logger: SummaryWriter = logger

        self.store_history = store_history
        self.decoder_history = torch.tensor([])
        self.z_grad_history = torch.tensor([])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, conditional_information: torch.tensor = torch.tensor([])):
        x = torch.cat([x, conditional_information], dim=1)
        mu, log_std = self.encoder(x)  # output dim (batch_size, latent_space)
        
        # sample the latent space
        sigma = torch.exp(log_std)
        z = mu + sigma * self.N.sample(mu.shape)

        # enhance latent space
        z = torch.cat([z, conditional_information], dim=1)
        z = Variable(z, requires_grad=True)
        # store for logging the gradient
        self.z = z

        decoder_out = self.decoder.forward(z) 
        if self.store_history:
            self.decoder_history = torch.cat([self.decoder_history, decoder_out.detach().flatten().cpu()])

        return decoder_out, mu, log_std

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        # log gradient from latent space
        if self.store_history:
            z_grad = self.z.grad.cpu()
            self.z_grad_history = torch.cat([self.z_grad_history, z_grad], dim=0)

        self.optimizer.step()

    def log_parameters(self, epoch_idx):
        # parameter histogram
        param_tensor = torch.tensor([])
        for param in self.parameters():
            param_tensor = torch.cat([param_tensor, param.cpu().detach().flatten()])
        self.logger.add_histogram("vae/param", param_tensor, epoch_idx)

    def log_gradients(self, epoch_idx):
        grad_tensor = torch.tensor([])
        for param in self.parameters():
            grad_tensor = torch.cat([grad_tensor, param.grad.cpu().flatten()])
        self.logger.add_histogram("vae/grad", grad_tensor, epoch_idx)

    def log_decoder_distr(self, epoch_idx):
        # print(self.decoder_history)
        self.logger.add_histogram("vae/decoder_distr", self.decoder_history, epoch_idx)

    def log_z_grad(self, epoch_idx):
        # taking the absolute
        z_grad_abs = torch.abs(self.z_grad_history)
        z_grad_abs = z_grad_abs.sum(dim=0)
        # normalize inputs
        z_grad_abs /= torch.norm(z_grad_abs)
        z_grad_abs = z_grad_abs.unsqueeze(dim=0)
        z_grad_abs = z_grad_abs.unsqueeze(dim=0)

        self.logger.add_image("vae/z_grad", z_grad_abs, epoch_idx)

    def store(self, path: str, epoch_idx: int, val_total_loss: float):
        torch.save({
                'epoch': epoch_idx,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)     


    def reset_history(self):
        """
        call this function before an epoch to ensure that there is only data from one epoch inside 
        """
        self.decoder_history = torch.tensor([])
        self.z_grad_history = torch.tensor([])