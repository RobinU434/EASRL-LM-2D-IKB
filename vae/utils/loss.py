from typing import Any, Tuple
import torch
import logging

from vae.data.data_set import YMode
from vae.utils.fk import forward_kinematics



def kl_divergence(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # D_KL(N(mu, sigma)|N(0, 1))
    kl = (torch.exp(2 * log_std) / 2 + torch.float_power(mu, 2) / 2 - log_std - 1/2).sum()

    return kl


def cyclic_mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    mse = mse(angle_diff(x, x_hat))
    return mse


def mse(diff: torch.tensor):
    # mean squared error
    # sum over the number of joints and mean for the batch
    return torch.float_power(diff, 2).mean()


def angle_diff(a : torch.Tensor, b: torch.Tensor, kappa: torch.Tensor = torch.pi):
        # source: https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
        dif = a - b
        return (dif + kappa) % (2 * kappa) - kappa


class ImitationLoss:
    def __init__(self,) -> None:
        pass

    def __call__(self, y: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        # MSE
        loss = torch.square(angle_diff(y, x_hat)).mean()
        return loss


class DistanceLoss:
    def __init__(self, norm: float = 1) -> None:
        self.norm = 1

    def __call__(self, y: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        x: desired target position
        x_hat: is the predicted target position

        Args:
            x (torch.Tensor): ground truth
            x_hat (torch.Tensor): prediction

        Returns:
            torch.Tensor: distance between y and x_hat
        """
        dist_loss = torch.linalg.norm(y - x_hat, axis=1).mean()
        return dist_loss


class VAELoss:
    def __init__(self, kl_loss_weight: float = 1, reconstruction_loss_weight: float = 1) -> None:
        self.kl_loss_weight = kl_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
    
    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor):
        r_loss = mse(x - x_hat)
        self.r_loss = r_loss
        return r_loss
    
    def kl_loss(self, mu: torch.Tensor, log_std: torch.Tensor):
        kl = kl_divergence(mu, log_std)
        self.kl_div = kl
        return kl
    
    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        loss = self.reconstruction_loss_weight * self.reconstruction_loss(x, x_hat) + self.kl_loss_weight * self.kl_loss(mu, log_std)
        self.loss = loss
        return loss


class CyclicVAELoss(VAELoss):
    def __init__(self, kl_loss_weight: float = 1, reconstruction_loss_weight: float = 1) -> None:
        super().__init__(kl_loss_weight, reconstruction_loss_weight)

        self.kappa = torch.pi

    def angle_diff(self, a: torch.tensor, b: torch.tensor):
        return angle_diff(a, b, self.kappa)

    def reconstruction_loss(self, x: torch.tensor, x_hat: torch.tensor):
        r_loss = mse(self.angle_diff(x, x_hat))
        self.r_loss = r_loss
        return r_loss      


class DistVAELoss(VAELoss):
    def __init__(self, kl_loss_weight: float = 1, reconstruction_loss_weight: float = 1) -> None:
        super().__init__(kl_loss_weight, reconstruction_loss_weight)

        self.loss_func = DistanceLoss()

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): ground truth action current angles from state
            x_hat (torch.Tensor): predicted action + current angles from state

        Returns:
            torch.Tensor: single float
        """
        dist_mean = self.loss_func(x, x_hat)
        self.r_loss = dist_mean
        return dist_mean


class PointDistanceLoss(VAELoss):
    """very similar to DistanceLoss but you pass in two 2D points and calculate the mean distance between those two 
    """
    def __init__(self,  kl_loss_weight: float = 1, reconstruction_loss_weight: float = 1) -> None:
        super().__init__(kl_loss_weight, reconstruction_loss_weight)

    def reconstruction_loss(self, y: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """needs two arrays of 2D points and returns the mean distance between those two

        Args:
            y (torch.Tensor): ground truth position
            x_hat (torch.Tensor): prediction position

        Returns:
            torch.Tensor: mean distance
        """
        self.r_loss = torch.linalg.norm(y - x_hat, dim=1).mean() 
        return self.r_loss
    

class IKLoss:
    """
    IK loss is a loss function specifically shaped for the application of a
    inverse kinematics problem. Here you can tune your algorithm with the 
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
            target_mode: int = YMode.UNDEFINED,
            device: str = "cpu") -> None:
        
        self.kl_loss_weight = kl_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.imitation_loss_weight = imitation_loss_weight
        self.distance_loss_weight = distance_loss_weight

        self.imitation_loss = 0
        self.distance_loss = 0
        self.r_loss = 0
        self.kl_loss = 0
        self.loss = 0

        self.target_mode = target_mode
        if self.target_mode == YMode.UNDEFINED:
            raise ValueError(f"You have to chose a proper target mode. You can chose from {list(YMode)}. Normally you get this value from your loaded dataset member variable y_mode")
        
        elif self.target_mode == YMode.ACTION:
            logging.info("DistanceLoss and ImitationLoss are enabled")
            logging.info(f"corresponding weights: distance: {self.distance_loss_weight}, imitation: {self.imitation_loss_weight}")
            self.imitation_loss_func = ImitationLoss()
            self.distance_loss_func = DistanceLoss()

        elif self.target_mode == YMode.POSITION:
            logging.warning(f"Only DistanceLoss is enabled ImitationLoss is disabled. The corresponding weight {self.imitation_loss_weight} will not be taken into account")
            self.imitation_loss_func = lambda x, y: torch.zeros(1).to(device)
            self.distance_loss_func = DistanceLoss()

        else:
            raise ValueError(f"Unrecognized target mode {self.target_mode}. You can choose from {list(YMode)}")

        self.device = device

    def reconstruction_loss(
            self,
            x: Tuple[torch.Tensor, torch.Tensor],
            x_hat: torch.Tensor):
        
        self.imitation_loss = self.imitation_loss_func(x, x_hat)

        predicted_position = forward_kinematics(x_hat)[:, -1].to(self.device)
        if self.target_mode == YMode.POSITION:
            target_position = x
        elif self.target_mode == YMode.ACTION:
            target_position = forward_kinematics(x)[:, -1]
        self.distance_loss = self.distance_loss_func(target_position, predicted_position)
        self.r_loss = self.imitation_loss_weight * self.imitation_loss + \
            self.distance_loss_weight * self.distance_loss
        
        return self.r_loss
    
    def kl_divergence(self, mu: torch.Tensor, log_std: torch.Tensor):
        kl = kl_divergence(mu, log_std)
        self.kl_div = kl
        return kl
    
    def __call__(
            self,
            y: Tuple[torch.Tensor, torch.Tensor],
            x_hat: torch.Tensor,
            mu: torch.Tensor,
            log_std: torch.Tensor) -> torch.Tensor:
        self.loss = self.reconstruction_loss_weight * self.reconstruction_loss(y, x_hat) + \
            self.kl_loss_weight * self.kl_divergence(mu, log_std)
        return self.loss
    
    def __str__(self) -> str:
        imitation_enabled = True if type(self.imitation_loss_func) == ImitationLoss else False
        distance_enabled = True if type(self.distance_loss_func) == DistanceLoss else False
        s =  f""" Use: {type(self).__name__}
        kl_loss_weight: {self.kl_loss_weight}
        reconstruction_loss_weight: {self.reconstruction_loss_weight}
        target_mode: {self.target_mode}
        imitation_enabled: {imitation_enabled}
        distance_enabled: {distance_enabled}  """
        return s


def get_loss_func(loss_config: dict, device: str) -> IKLoss:
    '''loss_func_name = config["loss_func"]
    # catch case where we need a specific loss function for the conditional 
    if config["dataset"] == "conditional_target":
        return PointDistanceLoss(
            kl_loss_weight=config["kl_loss_weight"],
            reconstruction_loss_weight=config["kl_loss_weight"]
        )

    loss_func = None
    if loss_func_name == VAELoss.__name__:
        loss_func = VAELoss(
            kl_loss_weight=config["kl_loss_weight"],
            reconstruction_loss_weight=config["kl_loss_weight"]
        )
    elif loss_func_name == CyclicVAELoss.__name__:
        loss_func = CyclicVAELoss(
            kl_loss_weight=config["kl_loss_weight"],
            reconstruction_loss_weight=config["reconstruction_loss_weight"],
            # normalization=config["normalize"]
        )
    elif loss_func_name == DistVAELoss.__name__:
        loss_func = DistVAELoss(
            kl_loss_weight=config["kl_loss_weight"],
            reconstruction_loss_weight=config["reconstruction_loss_weight"],
            # normalization=config["normalize"]
        )
    else:
        logging.error(f"available loss functions: {VAELoss.__name__}, {CyclicVAELoss.__name__}, {DistVAELoss.__name__}, but you chose {loss_func_name}")
    '''

    loss_func = IKLoss(**loss_config, device=device)
    print(loss_func)
    
    return loss_func