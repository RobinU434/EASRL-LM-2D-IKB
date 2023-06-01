import logging
from typing import Any
import torch

from supervised.utils import forward_kinematics

from vae.data.data_set import YMode



class ImitationLoss:
    def __init__(self,) -> None:
        pass

    def __call__(self, y: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        # MSE
        loss = torch.square(angle_diff(y, x_hat)).mean()
        return loss


class DistanceLoss:
    def __init__(self) -> None:
        pass

    def __call__(self, y: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """y and x_hat are stacked 2D positions
        Args:
            y (torch.Tensor): point in 2D space
            x_hat (torch.Tensor): point in 2D space

        Returns:
            torch.Tensor: mean distance between all the 2D vectors
        """
        dist_loss = torch.linalg.norm(target_pos - real_pos, dim=1).mean()
        return dist_loss
    

class PointDistanceLoss:
    def __init__(self, device) -> None:
        self.device = device

    def __call__(self, y: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """y is a point in 2D space
        x_hat is the action that should be close to y 

        Args:
            y (torch.Tensor): point in 2D space
            x_hat (torch.Tensor): predicted action

        Returns:
            torch.Tensor: _description_
        """
        real_pos = forward_kinematics(x_hat)[:, -1].to(self.device)

        dist_loss = torch.linalg.norm(y - real_pos, dim=1).mean()
        return dist_loss


class MergeLoss:
    def __init__(self, imitation_loss_weight: float = 1 , distance_loss_weight: float = 1) -> None:
        self.imitation_loss_weight = imitation_loss_weight
        self.distance_loss_weight = distance_loss_weight

        self.imitation_loss_func = ImitationLoss()
        self.distance_loss_func = DistanceLoss()

    def __call__(self, y: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        imitation_loss = self.imitation_loss_func(y, x_hat)
        distance_loss = self.distance_loss_func(y, x_hat)

        return self.imitation_loss_weight * imitation_loss + self.distance_loss_weight * distance_loss


class IKLoss:
    def __init__(
        self,
        imitation_loss_weight: float = 1,
        distance_loss_weight: float = 1,
        regularizer_weight: float = 1,
        target_mode: int = YMode.UNDEFINED
        ):

        self.imitation_loss_weight = imitation_loss_weight
        self.distance_loss_weight = distance_loss_weight

        self.imitation_loss = 0
        self.distance_loss = 0
        self.loss = 0

        self.target_mode = target_mode
        if self.target_mode == YMode.UNDEFINED:
            raise ValueError(f"You have to chose a proper target mode. You can chose from {list(YMode)}. Normally you get this value from your loaded dataset member variable y_mode")

        elif self.target_mode == YMode.ACTION:
            logging.info("DistanceLoss and ImitationLoss are enabled")
            logging.info(f"corresponding weights: distance: {self.distance_loss_weight}, imitation: {self.imitation_loss_weight}")
            self.imitation_loss_func = ImitationLoss()
            self.distance_loss_func = DistanceLoss()

            if self.distance_loss_weight == 0 and self.imitation_loss_weight == 0:
                logging.error("distance loss weight and imitation loss weight is 0, therefor it is not possible to create any gradient")

        elif self.target_mode == YMode.POSITION:
            logging.warning(f"Only DistanceLoss is enabled ImitationLoss is disabled. The corresponding weight {self.imitation_loss_weight} will not be taken into account")
            self.imitation_loss_func = lambda *x: torch.zeros(1).to(device)
            self.distance_loss_func = DistanceLoss()

            if self.distance_loss_weight == 0:
                logging.error("distance loss weight is zero and imitation loss is disabled, therefor it is not possible to create any gradient")

        else:
            raise ValueError(f"Unrecognized target mode {self.target_mode}. You can choose from {list(YMode)}")
    
    def __call__(self, y: torch.Tensor, x_hat: torch.tensor):
        self.imitation_loss = self.imitation_loss_func(y, x_hat)
        if self.target_mode == YMode.POSITION:
            predicted_end_effector = forward_kinematics(x_hat)[:, -1]
            self.distance_loss = self.distance_loss_func(y, predicted_end_effector)
        elif self.target_mode == YMode.ACTION:
            predicted_end_effector = forward_kinematics(x_hat)[:, -1]
            real_end_effector = forward_kinematics(y)[:, -1]
            self.distance_loss = self.distance_loss_func(predicted_end_effector, real_end_effector)
        else:
            logging.warning("No update on distance loss")

        self.loss = self.imitation_loss_weight * self.imitation_loss + self.distance_loss_weight * self.distance_loss
        return self.loss


def angle_diff(a : torch.Tensor, b: torch.Tensor):
    # source: https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
    dif = a - b
    return (dif + torch.pi) % (2 * torch.pi) - torch.pi 


def get_loss_func(loss_func_name: str, device: str, y_mode: int = YMode.UNDEFINED):
    loss_func_names = ["ImitationLoss", "DistanceLoss", "MergeLoss", "PointDistanceLoss"]

    if loss_func_name == "ImitationLoss":
        loss_func = ImitationLoss()
    elif loss_func_name == "DistanceLoss":
        loss_func =  DistanceLoss()
    elif loss_func_name == "MergeLoss":
        loss_func = MergeLoss(
            imitation_loss_weight=0.1,
            distance_loss_weight=1
        )
    elif loss_func_name == "PointDistanceLoss":
        loss_func = PointDistanceLoss(device)
    elif loss_func_name == "IKLoss":
        loss_func = IKLoss(
            imitation_loss_weight=1,
            distance_loss_weight=1,
            regularizer_weight=0.1,
            target_mode=y_mode,
        )
    else:
        logging.error(f"chose a loss func from: {loss_func_names}, but you chose: ", loss_func_name)
        raise ValueError(f"chose a loss func from: {loss_func_names}, but you chose: ", loss_func_name)
    
    print("Use: ", type(loss_func).__name__)
    return loss_func