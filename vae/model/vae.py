#  0

import math
from typing import Any, Tuple, Union
import torch
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.tensorboard.writer import SummaryWriter
from latent.metrics.vae_metrics import VAEIKMetrics, VAEMetrics

from vae.model.decoder import Decoder
from vae.model.encoder import VariationalEncoder
from vae.utils.post_processing import PostProcessor


class VariationalAutoencoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 latent_dim: int,
                 output_dim: int,
                 conditional_info_dim: Tuple[int, int] = (0, 0),
                 post_processor: PostProcessor = PostProcessor(False),
                 learning_rate: float = 1e-3,
                 logger: SummaryWriter = None,
                 store_history: bool = False,
                 device: str = "cpu",
                 verbose: bool = False):
        super(VariationalAutoencoder, self).__init__()
        self.conditional_info_dim = conditional_info_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.encoder = VariationalEncoder(
            input_dim + self.conditional_info_dim[0], 
            latent_dim)
        self.decoder = Decoder(
            latent_dim + self.conditional_info_dim[1], 
            output_dim)

        self.N = torch.distributions.Normal(0, 1)
        
        if "cuda" in device:
            # hack to get sampling on the GPU
            self.N.loc = self.N.loc.to(device)  
            self.N.scale = self.N.scale.to(device)

        self.post_processor = post_processor
        self.logger: SummaryWriter = logger

        self.store_history = store_history
        self.decoder_history = torch.tensor([])
        self.z_grad_history = torch.tensor([])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if verbose:
            print("architecture")
            s = f"{input_dim} + {self.conditional_info_dim[0]} -> [Encoder] -> {latent_dim} + {conditional_info_dim[1]} -> [Decoder] -> {output_dim}"
            if self.post_processor.enabled:
                s += f" -> [PostProcessor (tanh) + [{self.post_processor.min_action, self.post_processor.min_action}]]"
            print(s)
   

    def forward(self, x: torch.Tensor, c_enc: torch.Tensor, c_dec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """performs forward pass into model and through the postprocessor

        Args:
            x (torch.Tensor): input for encoder
            c_enc (torch.Tensor): conditional information for encoder
            c_dec (torch.Tensor): conditional information for decoder

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (output after CVAE and postprocessor (x_hat), mu from encoder, log_str from encoder)
        """
        x_con = torch.cat([x, c_enc], dim=1) 
        mu, log_std = self.encoder(x_con)  # output dim (batch_size, latent_space)
        
        # sample the latent space
        sigma = torch.exp(log_std)
        sigma = torch.ones_like(sigma) * math.exp(-40)
        z = mu + sigma * self.N.sample(mu.shape)
        
        # enhance latent space
        z = torch.cat([z, c_dec], dim=1)
        z = Variable(z, requires_grad=True)
        # store for logging the gradient
        self.z = z

        decoder_out = self.decoder.forward(z)
        x_hat = self.post_processor(decoder_out)
        if self.store_history:
            self.decoder_history = torch.cat([self.decoder_history, x_hat.detach().flatten().cpu()])

        return x_hat, mu, log_std
    
    def predict(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

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
        if self.logger is not None:
            param_tensor = torch.tensor([])
            for param in self.parameters():
                param_tensor = torch.cat([param_tensor, param.cpu().detach().flatten()])
            self.logger.add_histogram("vae/param", param_tensor, epoch_idx)

    def log_gradients(self, epoch_idx):
        if self.logger is not None:
            grad_tensor = torch.tensor([])
            for param in self.parameters():
                grad_tensor = torch.cat([grad_tensor, param.grad.cpu().flatten()])
            self.logger.add_histogram("vae/grad", grad_tensor, epoch_idx)

    def log_decoder_distr(self, epoch_idx):
        # print(self.decoder_history)
        if self.logger is not None:
            self.logger.add_histogram("vae/decoder_distr", self.decoder_history, epoch_idx)

    def log_z_grad(self, epoch_idx):
        # taking the absolute
        if self.logger is not None:
            z_grad_abs = torch.abs(self.z_grad_history)
            z_grad_abs = z_grad_abs.sum(dim=0)
            # normalize inputs
            z_grad_abs /= torch.norm(z_grad_abs)
            z_grad_abs = z_grad_abs.unsqueeze(dim=0)
            z_grad_abs = z_grad_abs.unsqueeze(dim=0)

            self.logger.add_image("vae/z_grad", z_grad_abs, epoch_idx)

    def store(self, path: str, epoch_idx: int, metrics: VAEIKMetrics):
        torch.save({
                'epoch': epoch_idx,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'reconstruction_loss': metrics.reconstruction_loss.mean(),
                'kl_loss': metrics.kl_loss.mean(),
                'distance_loss': metrics.distance_loss.mean(),
                'imitation_loss': metrics.imitation_loss.mean(),
            }, path)


    def reset_history(self):
        """
        call this function before an epoch to ensure that there is only data from one epoch inside 
        """
        self.decoder_history = torch.tensor([])
        self.z_grad_history = torch.tensor([])


def build_model(config: dict, input_dim: int, conditional_info_dim: Tuple[int, int], logger: SummaryWriter,):  
    autoencoder = VariationalAutoencoder(
        input_dim=input_dim,
        latent_dim=config["latent_dim"],
        output_dim=config["num_joints"],
        conditional_info_dim=conditional_info_dim,
        learning_rate=config["learning_rate"],
        logger=logger,
        post_processor=PostProcessor(**config['post_processor']),
        store_history=True, 
        device=config['device'],
        verbose=True).to(config['device'])
    
    return autoencoder