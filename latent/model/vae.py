#  0

import math
from typing import Dict, List, Tuple, Union
import torch
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.tensorboard.writer import SummaryWriter
from latent.metrics.vae_metrics import VAEIKMetrics
from latent.model.base_model import LearningModule, NeuralNetwork

from latent.model.decoder import Decoder
from latent.model.encoder import VariationalEncoder
from latent.model.utils.post_processor import PostProcessor
from logger.base_logger import Logger


class VariationalAutoencoder(NeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        conditional_info_dim: Tuple[int, int] = (0, 0),
        post_processor: PostProcessor = PostProcessor(False),
        learning_rate: float = 1e-3,
        store_history: bool = False,
        device: str = "cpu",
        verbose: bool = False,
    ):
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, learning_rate=learning_rate,
        )
        self._conditional_info_dim = conditional_info_dim
        self._latent_dim = latent_dim
        
        self._encoder = VariationalEncoder(
            input_dim + self._conditional_info_dim[0], latent_dim
        )
        self._decoder = Decoder(latent_dim + self._conditional_info_dim[1], output_dim)

        self._optimizer = torch.optim.Adam(self.parameters(), lr=self._learning_rate)

        self._N = torch.distributions.Normal(0, 1)

        if "cuda" in device:
            # hack to get sampling on the GPU
            self._N.loc = self._N.loc.to(device)
            self._N.scale = self._N.scale.to(device)

        self._post_processor = post_processor

        self._store_history = store_history
        self._decoder_history = torch.tensor([])
        self._z_grad_history = torch.tensor([])

        self._learning_rate = learning_rate
        
        if verbose:
            print("architecture")
            s = f"{input_dim} + {self._conditional_info_dim[0]} -> [Encoder] -> {latent_dim} + {conditional_info_dim[1]} -> [Decoder] -> {output_dim}"
            if self._post_processor.enabled:
                s += f" -> [PostProcessor (tanh) + [{self._post_processor.min_action, self._post_processor.min_action}]]"
            print(s)

    def forward(
        self, x: torch.Tensor, c_enc: torch.Tensor, c_dec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """performs forward pass into model and through the postprocessor

        Args:
            x (torch.Tensor): input for encoder
            c_enc (torch.Tensor): conditional information for encoder
            c_dec (torch.Tensor): conditional information for decoder

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (output after CVAE and postprocessor (x_hat), mu from encoder, log_str from encoder)
        """
        x_con = torch.cat([x, c_enc], dim=1)
        mu, log_std = self._encoder(x_con)  # output dim (batch_size, latent_space)

        # sample the latent space
        sigma = torch.exp(log_std)
        z = mu + sigma * self._N.sample(mu.shape)

        # enhance latent space
        z = torch.cat([z, c_dec], dim=1)
        # z = Variable(z, requires_grad=True)
        # store for logging the gradient
        self.z = z

        decoder_out = self._decoder.forward(z)
        x_hat = self._post_processor(decoder_out)
        if self._store_history:
            self._decoder_history = torch.cat(
                [self._decoder_history, x_hat.detach().flatten().cpu()]
            )

        return x_hat, mu, log_std

    def train(self, loss):
        self._optimizer.zero_grad()
        loss.backward()

        # log gradient from latent space
        # if self._store_history:
        #     z_grad = self.z.grad.cpu()
        #     self._z_grad_history = torch.cat([self._z_grad_history, z_grad], dim=0)

        self._optimizer.step()

    def log_internals(self, logger: List[Union[SummaryWriter, Logger]], epoch_idx) -> None:
        if logger is None:
            return
        for single_logger in logger:
            self._log_gradients(single_logger, epoch_idx)
            self._log_parameters(single_logger, epoch_idx)
            # self._log_z_grad(single_logger, epoch_idx)

    def _log_parameters(self, logger: Union[SummaryWriter, Logger], epoch_idx: int):
        # parameter histogram
        param_tensor = []
        for param in self.parameters():
            param_tensor.append(param.cpu().detach().flatten().numpy())
        param_tensor = np.concatenate(param_tensor)
        logger.add_histogram("vae/param", param_tensor, epoch_idx)

    def _log_gradients(self, logger: Union[SummaryWriter, Logger], epoch_idx: int):
        grad_tensor = []
        for param in self.parameters():
            grad_tensor.append(param.grad.cpu().flatten().numpy())
        grad_tensor = np.concatenate(grad_tensor)
        logger.add_histogram("vae/grad", grad_tensor, epoch_idx)

    def log_decoder_distr(self, logger: Union[SummaryWriter, Logger], epoch_idx: int):
        # print(self.decoder_history)
        logger.add_histogram("vae/decoder_distr", self._decoder_history, epoch_idx)

    def _log_z_grad(self, logger: Union[SummaryWriter, Logger], epoch_idx: int):
        # taking the absolute
        z_grad_abs = torch.abs(self._z_grad_history)
        z_grad_abs = z_grad_abs.sum(dim=0)
        # normalize inputs
        z_grad_abs /= torch.norm(z_grad_abs)
        z_grad_abs = z_grad_abs.unsqueeze(dim=0)
        z_grad_abs = z_grad_abs.unsqueeze(dim=0)

        logger.add_image("vae/z_grad", z_grad_abs, epoch_idx)

    def save(self, path: str, epoch_idx: int, metrics: VAEIKMetrics):
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "reconstruction_loss": metrics.reconstruction_loss.mean(),
                "kl_loss": metrics.kl_loss.mean(),
                "distance_loss": metrics.distance_loss.mean(),
                "imitation_loss": metrics.imitation_loss.mean(),
            },
            path,
        )

    def reset_history(self):
        """
        call this function before an epoch to ensure that there is only data from one epoch inside
        """
        self._decoder_history = torch.tensor([])
        self._z_grad_history = torch.tensor([])

    def hparams(self) -> Dict[str, Union[str, int, float]]:
        hparams = {
            "learning_rate": self._learning_rate,
            "input_dim": self._input_dim,
            "latent_dim": self._latent_dim,
        }
        return hparams
