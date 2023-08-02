from vae.model.decoder import Decoder
from vae.model.encoder import Encoder
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from torch.autograd import Variable

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, learning_rate: float, logger: SummaryWriter):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

        self.logger: SummaryWriter = logger

        self.decoder_history = torch.tensor([])
        self.z_grad_history = torch.tensor([])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, latent_enhancer: torch.tensor = torch.tensor([])):
        z = self.encoder(x)  # output dim (batch_size, latent_space)
        self.z = z

        decoder_out = self.decoder.forward(z) 
        self.decoder_history = torch.cat([self.decoder_history, decoder_out.flatten()])

        # return the two Nones to be conform with the VariationalAutoencoder class
        return decoder_out, None, None

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        # log gradient from latent space
        z_grad = self.z.grad
        self.z_grad_history = torch.cat([self.z_grad_history, z_grad], dim=0)

        self.optimizer.step()

    def log_parameters(self, epoch_idx):
        # parameter histogram
        param_tensor = torch.tensor([])
        for param in self.parameters():
            param_tensor = torch.cat([param_tensor, param.flatten()])
        self.logger.add_histogram("vae/param", param_tensor, epoch_idx)

    def log_gradients(self, epoch_idx):
        grad_tensor = torch.tensor([])
        for param in self.parameters():
            grad_tensor = torch.cat([grad_tensor, param.grad.flatten()])
        self.logger.add_histogram("vae/grad", grad_tensor, epoch_idx)

    def log_decoder_distr(self, epoch_idx):
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

    def reset_history(self):
        """
        call this function before an epoch to ensure that there is only data from one epoch inside 
        """
        self.decoder_history = torch.tensor([])
        self.z_grad_history = torch.tensor([])