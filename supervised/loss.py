import logging
from typing import Any
import torch

from supervised.utils import forward_kinematics


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
        # expand x_hat to the original space
        target_pos = forward_kinematics(y)[:, -1]
        real_pos = forward_kinematics(x_hat)[:, -1]

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


def angle_diff(a : torch.Tensor, b: torch.Tensor):
    # source: https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
    dif = a - b
    return (dif + torch.pi) % (2 * torch.pi) - torch.pi 


def get_loss_func(loss_func_name: str, device):
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
    else:
        logging.error(f"chose a loss func from: {loss_func_names}, but you chose: ", loss_func_name)
        raise ValueError(f"chose a loss func from: {loss_func_names}, but you chose: ", loss_func_name)
    
    print("Use: ", loss_func_name)
    return loss_func