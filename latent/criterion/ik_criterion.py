import logging

import torch
from latent.criterion.base_criterion import Criterion
from latent.criterion.distance_criterion import EuclideanDistance
from latent.criterion.imitation_criterion import ImitationLoss
from latent.datasets.utils import TargetMode
from utils.kinematics import forward_kinematics


class IKLoss(Criterion):
    def __init__(
        self,
        imitation_loss_weight: float = 1,
        distance_loss_weight: float = 1,
        regularizer_weight: float = 1,
        target_mode: TargetMode = TargetMode.UNDEFINED,
        device: str = "cpu",
    ):
        super().__init__(device)
        self.imitation_loss_weight = imitation_loss_weight
        self.distance_loss_weight = distance_loss_weight

        self.imitation_loss = torch.zeros(1)
        self.distance_loss = torch.zeros(1)
        self.loss = 0

        self.target_mode = target_mode
        if self.target_mode == TargetMode.UNDEFINED:
            raise ValueError(
                f"You have to chose a proper target mode. You can chose from {list(TargetMode)}. Normally you get this value from your loaded dataset member variable y_mode"
            )

        elif self.target_mode == TargetMode.ACTION:
            logging.info("DistanceLoss and ImitationLoss are enabled")
            logging.info(
                f"corresponding weights: distance: {self.distance_loss_weight}, imitation: {self.imitation_loss_weight}"
            )
            self.imitation_loss_func = ImitationLoss()
            self.distance_loss_func = EuclideanDistance()

            if self.distance_loss_weight == 0 and self.imitation_loss_weight == 0:
                logging.error(
                    "distance loss weight and imitation loss weight is 0, therefor it is not possible to create any gradient"
                )

        elif (
            self.target_mode == TargetMode.POSITION
            or self.target_mode == TargetMode.INTERMEDIATE_POSITION
            or self.target_mode == TargetMode.FINAL_POSITION
        ):
            logging.warning(
                f"Only DistanceLoss is enabled ImitationLoss is disabled. The corresponding weight {self.imitation_loss_weight} will not be taken into account"
            )
            self.imitation_loss_func = lambda *x: torch.zeros(1)
            self.distance_loss_func = EuclideanDistance()
            # self.distance_loss_func = torch.nn.HuberLoss()

            if self.distance_loss_weight == 0:
                logging.error(
                    "distance loss weight is zero and imitation loss is disabled, therefor it is not possible to create any gradient"
                )

        else:
            raise ValueError(
                f"Unrecognized target mode {self.target_mode}. You can choose from {list(TargetMode)}"
            )

    def __call__(self, y: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """calculates the loss for an inverse kinematics problem consisting of
        an imitation loss if y is an action and a distance loss

        Args:
            y (torch.Tensor): is either a position in 2D space (y_mode == 2) or an angle (y_Mode == 1) with which we
                can compute also an imitation loss
            x_hat (torch.Tensor): action from the network and its post-processor

        Returns:
            torch.Tensor: tensor with only one element -> the loss
        """
        self.imitation_loss = self.imitation_loss_func(y, x_hat)
        predicted_end_effector = forward_kinematics(x_hat)[:, -1]
        if (
            self.target_mode == TargetMode.POSITION
            or self.target_mode == TargetMode.INTERMEDIATE_POSITION
            or self.target_mode == TargetMode.FINAL_POSITION
        ):
            self.distance_loss = self.distance_loss_func(
                y.cpu(), predicted_end_effector
            )
        elif self.target_mode == TargetMode.ACTION:
            real_end_effector = forward_kinematics(y)[:, -1]
            self.distance_loss = self.distance_loss_func(
                predicted_end_effector, real_end_effector
            )
        else:
            logging.warning("No update on distance loss")

        self.loss = (
            self.imitation_loss_weight * self.imitation_loss
            + self.distance_loss_weight * self.distance_loss
        )
        
        return self.loss.to(self._device)

    def __str__(self) -> str:
        s = f"""Use: {type(self).__name__}
        target_mode: {self.target_mode}
        imitation_loss_func: {type(self.imitation_loss_func).__name__}
        imitation_weight: {self.imitation_loss_weight}
        distance_loss_func: {type(self.distance_loss_func).__name__}
        distance_weight: {self.distance_loss_weight}"""
        return s
