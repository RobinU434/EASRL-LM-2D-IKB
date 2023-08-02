from abc import ABC, abstractmethod
import logging
from typing import Any, Literal

import torch
from torch import Tensor
from latent.criterion.base_criterion import Criterion
from latent.criterion.distance_criterion import EuclideanDistance
from latent.criterion.ik_criterion import IKLoss
from latent.criterion.imitation_criterion import ImitationLoss
from latent.data.utils import TargetMode
from supervised.utils import forward_kinematics


def kl_divergence(mu: Tensor, log_std: Tensor) -> Tensor:
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # D_KL(N(mu, sigma)|N(0, 1))
    kl = (
        torch.exp(2 * log_std) / 2 + torch.float_power(mu, 2) / 2 - log_std - 1 / 2
    ).sum()

    return kl


class ELBO(Criterion, ABC):
    """Abstract base class for Evidence Lower Bound

    Attributes:
        reconstruction_loss (float): float value for reconstruction loss
        kl_loss (float): float value for kl loss
    """

    def __init__(
        self,
        kl_loss_weight: float,
        reconstruction_loss_weight: float,
        device: str = "cpu",
    ) -> None:
        super().__init__(device)
        self.reconstruction_loss: Tensor = torch.zeros(1)
        self.kl_loss: Tensor = torch.zeros(1)

        self._kl_loss_weight = kl_loss_weight
        self._reconstruction_loss_weight = reconstruction_loss_weight

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """make the weighted sum of reconstruction loss and kl loss

        Returns:
            Tensor: final loss value
        """
        self.loss = self._kl_loss_weight * self._kl_loss_func(
            *args, **kwargs
        ) + self._reconstruction_loss_weight * self._reconstruction_loss_func(
            *args, **kwargs
        )
        return self.loss

    @abstractmethod
    def _reconstruction_loss_func(self, *args, **kwargs) -> Tensor:
        """implements the reconstruction loss for an ELBO criterion

        Returns:
            float: reconstruction loss value
        """
        pass

    @abstractmethod
    def _kl_loss_func(self, *args, **kwargs) -> Tensor:
        """implements the kl loss for an ELBO criterion

        Returns:
            float: reconstruction loss value
        """
        pass


class InvKinELBO(ELBO):
    """
    InvKinELBO is a loss function specifically shaped for the application of a
    inverse kinematics problem in a Variational Autoencoder setting. Here you can tune your algorithm with the
    tradeoff between an imitation task with a given expert action and the
    action outcome (distance between target and predicted action)

    This class also contains the traditional weights for kl loss and
    reconstruction loss
    """

    def __init__(
        self,
        kl_loss_weight: float = 1,
        reconstruction_loss_weight: float = 1,
        imitation_loss_weight: float = 1,
        distance_loss_weight: float = 1,
        target_mode: Literal[
            TargetMode.ACTION,
            TargetMode.POSITION,
            TargetMode.UNDEFINED,
            TargetMode.FINAL_POSITION,
            TargetMode.INTERMEDIATE_POSITION,
        ] = TargetMode.UNDEFINED,
        device: str = "cpu",
    ) -> None:
        super().__init__(kl_loss_weight, reconstruction_loss_weight, device)

        self._target_mode = target_mode

        self._ik_loss = IKLoss(
            imitation_loss_weight=imitation_loss_weight,
            distance_loss_weight=distance_loss_weight,
            regularizer_weight=1,
            target_mode=target_mode,
            device=self._device
        )

        if self._reconstruction_loss_weight == 0:
            logging.warning("reconstruction loss is disabled")
        if self._kl_loss_weight == 0:
            logging.warning("kl loss is disabled")

    def _reconstruction_loss_func(self, y: Tensor, x_hat: Tensor) -> Tensor:
        self.reconstruction_loss = self._ik_loss(y, x_hat)
        return self.reconstruction_loss

    def _kl_loss_func(self, mu: Tensor, log_std: Tensor) -> Tensor:
        self.kl_loss = kl_divergence(mu, log_std)
        return self.kl_loss

    def __call__(self, y: Tensor, x_hat: Tensor, mu: Tensor, log_std: Tensor) -> Tensor:
        """computes inverse kinematics loss on autoencoder. Final loss is a weighted sum of
        - distance_loss
        - imitation_loss
        - kl_divergence

        Function stores all intermediate values in their own variables

        Args:
            y (torch.Tensor): target. Is either an target action (self.target_mode == TargetMode.ACTION) or a position (self.target_mode == TargetMode.POSITION)
            x_hat (torch.Tensor): predicted action from the model + state_angles
            mu (torch.Tensor): partial output from encoder for latent sampling
            log_std (torch.Tensor): partial output from encoder for latent sampling

        Returns:
            torch.Tensor: weighted sum
        """
        self.loss = self._reconstruction_loss_weight * self._reconstruction_loss_func(
            y, x_hat
        ) + self._kl_loss_weight * self._kl_loss_func(mu, log_std)
        return self.loss

    def __str__(self) -> str:
        imitation_enabled = isinstance(
            self._ik_loss.imitation_loss_func, ImitationLoss
        )
        distance_enabled = isinstance(
            self._ik_loss.distance_loss_func, EuclideanDistance
        )
        s = f"""Use: {type(self).__name__}
        kl_loss_weight: {self._kl_loss_weight}
        reconstruction_loss_weight: {self._reconstruction_loss_weight}
        target_mode: {self._target_mode}
        imitation_enabled: {imitation_enabled}
        imitation_weight: {self.imitation_loss_weight}
        distance_enabled: {distance_enabled}
        distance_weight: {self.distance_loss_weight}"""
        return s

    @property
    def imitation_loss(self) -> Tensor:
        return self._ik_loss.imitation_loss

    @property
    def distance_loss(self) -> Tensor:
        return self._ik_loss.distance_loss

    @property
    def imitation_loss_weight(self) -> float:
        return self._ik_loss.imitation_loss_weight

    @property
    def distance_loss_weight(self) -> float:
        return self._ik_loss.distance_loss_weight