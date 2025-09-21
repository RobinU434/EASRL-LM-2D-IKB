from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


# ---------- Model: Affine coupling + simple flip permutation ----------


class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2 * 2),
        )

    def forward(self, x, reverse=False):
        x_a, x_b = x.chunk(2, dim=1)
        params = self.net(x_a)
        s, t = params.chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            y_b = x_b * torch.exp(s) + t
            y = torch.cat([x_a, y_b], dim=1)
            log_det = s.sum(dim=1)
            return y, log_det
        else:
            y_b = (x_b - t) * torch.exp(-s)
            y = torch.cat([x_a, y_b], dim=1)
            log_det = -s.sum(dim=1)
            return y, log_det


class Flip(nn.Module):
    def forward(self, x: torch.Tensor):
        torch.flip
        return x.flip(dims=[1])


class RealNVP(nn.Module):
    def __init__(
        self,
        dim=2,
        hidden_dim=128,
        n_couplings=6,
        device="cpu",
        prior: torch.distributions.Distribution = None,
    ):
        super().__init__()
        self.dim = dim
        self.device = device
        if prior is None:
            self.prior = MultivariateNormal(
                torch.zeros(dim).to(device), torch.eye(dim).to(device)
            )
        else:
            self.prior = prior
        layers = []
        for _ in range(n_couplings):
            layers.append(AffineCoupling(dim, hidden_dim))
            layers.append(Flip())
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # maps x -> z, returns z and log|det dz/dx|
        log_det_sum = x.new_zeros(x.shape[0]).to(self.device)
        z = x
        for layer in self.layers:
            if isinstance(layer, AffineCoupling):
                z, ld = layer(z, reverse=False)
                log_det_sum = log_det_sum + ld
            else:
                z = layer(z)
        return z, log_det_sum

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        # maps z -> x, returns x and log|det dx/dz|
        log_det_sum = z.new_zeros(z.shape[0]).to(self.device)
        x = z
        for layer in reversed(self.layers):
            if isinstance(layer, AffineCoupling):
                x, ld = layer(x, reverse=True)
                log_det_sum = log_det_sum + ld
            else:
                x = layer(x)
        return x, log_det_sum

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        return self.prior.log_prob(z) + log_det


# ---------- Conditional Flow: conditional affine coupling + ConditionalRealNVP ----------


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Linear(dim // 2 + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2 * 2),
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the conditional affine coupling layer.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditional tensor.
            reverse (bool, optional): Whether to reverse the flow. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and log determinant.
        """
        # x: [B, D], c: [B, cond_dim]
        x_a, x_b = x.chunk(2, dim=1)
        inp = torch.cat([x_a, c], dim=1)
        params = self.net(inp)
        s, t = params.chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            y_b = x_b * torch.exp(s) + t
            y = torch.cat([x_a, y_b], dim=1)
            log_det = s.sum(dim=1)
            return y, log_det
        else:
            y_b = (x_b - t) * torch.exp(-s)
            y = torch.cat([x_a, y_b], dim=1)
            log_det = -s.sum(dim=1)
            return y, log_det


class ConditionalRealNVP(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        cond_dim: int = 2,
        hidden_dim: int = 128,
        n_couplings: int = 6,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.device = device
        self.cond_dim = cond_dim
        self.prior = MultivariateNormal(
            torch.zeros(dim).to(device), torch.eye(dim).to(self.device)
        )
        layers = []
        for _ in range(n_couplings):
            layers.append(ConditionalAffineCoupling(dim, cond_dim, hidden_dim))
            layers.append(Flip())
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the flow model.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditional tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and log determinant.
        """
        log_det_sum = x.new_zeros(x.shape[0]).to(self.device)
        z = x
        for layer in self.layers:
            if isinstance(layer, ConditionalAffineCoupling):
                z, ld = layer(z, c, reverse=False)
                log_det_sum = log_det_sum + ld
            else:
                log_det_sum = x.new_zeros(x.shape[0]).to(self.device)
        return z, log_det_sum

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass through the flow model.

        Args:
            z (torch.Tensor): Latent tensor.
            c (torch.Tensor): Conditional tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and log determinant.
        """
        log_det_sum = z.new_zeros(z.shape[0]).to(self.device)
        x = z
        for layer in reversed(self.layers):
            if isinstance(layer, ConditionalAffineCoupling):
                x, ld = layer(x, c, reverse=True)
                log_det_sum = log_det_sum + ld
            else:
                log_det_sum = z.new_zeros(z.shape[0]).to(self.device)
        return x, log_det_sum

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Log probability density function.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditional tensor.

        Returns:
            torch.Tensor: Log probability density.
        """
        z, log_det = self.forward(x, c)
        return self.prior.log_prob(z) + log_det