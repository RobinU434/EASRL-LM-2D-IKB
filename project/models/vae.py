from typing import Dict, Tuple
from torch import Tensor, nn
import torch
from torch.distributions import Normal
from pytorch_lightning import LightningModule


def generate_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
    activation_func: str = "ReLU",
    n_layers: int = 1,
) -> nn.Module:
    if n_layers == 1:
        return nn.Linear(input_dim, output_dim)

    activation_func = getattr(nn, activation_func)
    layers = [nn.Linear(input_dim, hidden_dim), activation_func()]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation_func())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(layers)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        conditional_dim: int = 0,
        hidden_dim: int = 128,
        n_layers: int = 1,
        activation_func: str = "ReLU",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.conditional_dim = conditional_dim
        self.n_layers = n_layers
        self.activation_func = activation_func

        self.encoder = generate_mlp(
            self.input_dim + self.conditional_dim,
            2 * self.latent_dim,
            self.hidden_dim,
            self.activation_func,
            self.n_layers,
        )
        self.decoder = generate_mlp(
            self.latent_dim + self.conditional_dim,
            self.input_dim,
            self.hidden_dim,
            self.activation_func,
            self.n_layers,
        )
        self.latent_distr = Normal(0, 1)

    def encode(self, x: Tensor, c: Tensor = None) -> Tuple[Tensor, Tensor]:
        if c is not None:
            assert (
                c.shape[-1] == self.conditional_dim
            ), f"Given conditional has an unexpected feature dimension {c.shape[-1]} != {self.conditional_dim=}"
            x = torch.cat([x, c], dim=-1)
        h = self.encoder.forward(x)
        mu = h[..., : self.latent_dim // 2]
        std = h[..., self.latent_dim // 2 :]
        std = nn.functional.softplus(std)  # ensure always positive
        return mu, std

    def sample(self, mu: Tensor, std: Tensor) -> Tensor:
        z = self.latent_distr.rsample(mu.shape)
        z = z * std
        z = z + mu
        return z

    def decode(self, z: Tensor, c: Tensor = None) -> Tensor:
        if c is not None:
            assert (
                c.shape[-1] == self.conditional_dim
            ), f"Given conditional has an unexpected feature dimension {c.shape[-1]} != {self.conditional_dim=}"
            z = torch.cat([z, c], dim=-1)
        return self.decoder.forward(z)

    def forward(self, x: Tensor, c: Tensor = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        mu, std = self.encode(x, c)
        z = self.sample(mu, std)
        x_hat = self.decode(z, c)
        return x_hat, {"z": z, "std": std, "mu": mu}
