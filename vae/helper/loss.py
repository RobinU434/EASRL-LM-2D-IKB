import torch
import logging


def kl_loss(mu: torch.tensor, log_std: torch.tensor) -> torch.tensor:
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # D_KL(N(mu, sigma)|N(0, 1))
    kl = (torch.exp(2 * log_std) / 2 + torch.float_power(mu, 2) / 2 - log_std - 1/2).sum()

    return kl


def cyclic_mse_loss(x: torch.tensor, x_hat: torch.tensor) -> torch.tensor:
    mse = mse(angle_diff(x, x_hat))
    return mse


def mse(diff: torch.tensor):
    # mean squared error
    # sum over the number of joints and mean for the batch
    return torch.float_power(diff, 2).mean()


def angle_diff(a : torch.tensor, b: torch.tensor, kappa: torch.tensor = torch.pi):
        # source: https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
        dif = a - b
        return (dif + kappa) % (2 * kappa) - kappa


def forward_kinematics(angles: torch.tensor):
    """_summary_

    Args:
        angles (np.array): shape (num_arms, num_joints)

    Returns:
        _type_: _description_
    """
    num_arms, num_joints = angles.size()
    positions = torch.zeros((num_arms, num_joints + 1, 2))

    for idx in range(num_joints):
        origin = positions[:, idx]

        # new position
        new_pos = torch.zeros((num_arms, 2))
        new_pos[:, 0] = torch.cos(angles[:, idx])
        new_pos[:, 1] = torch.sin(angles[:, idx])
        
        # translate position
        new_pos += origin

        positions[:, idx + 1] = new_pos

    return positions


class VAELoss:
    def __init__(self, kl_loss_weight: float = 1, reconstruction_loss_weight: float = 1) -> None:
        self.kl_loss_weight = kl_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
    
    def reconstruction_loss(self, x: torch.tensor, x_hat: torch.tensor):
        r_loss = mse(x - x_hat)
        self.r_loss = r_loss
        return r_loss
    
    def kl_loss(self, mu: torch.tensor, log_std: torch.tensor):
        kl = kl_loss(mu, log_std)
        self.kl_div = kl
        return kl
    
    def __call__(self, x: torch.tensor, x_hat: torch.tensor, mu: torch.tensor, log_std: torch.tensor) -> torch.tensor:
        loss = self.reconstruction_loss_weight * self.reconstruction_loss(x, x_hat) + self.kl_loss_weight * self.kl_loss(mu, log_std)
        self.loss = loss
        return loss


class CyclicVAELoss(VAELoss):
    def __init__(self, kl_loss_weight: float = 1, reconstruction_loss_weight: float = 1, normalization: bool = False) -> None:
        super().__init__(kl_loss_weight, reconstruction_loss_weight)

        self.normalization = normalization
        if self.normalization:
            self.kappa = 1
        else:
             self.kappa = torch.pi

    def angle_diff(self, a: torch.tensor, b: torch.tensor):
        return angle_diff(a, b, self.kappa)

    def reconstruction_loss(self, x: torch.tensor, x_hat: torch.tensor):
        r_loss = mse(self.angle_diff(x, x_hat))
        self.r_loss = r_loss
        return r_loss      


class DistVAELoss(VAELoss):
    def __init__(self, kl_loss_weight: float = 1, reconstruction_loss_weight: float = 1, normalization: bool = False) -> None:
        super().__init__(kl_loss_weight, reconstruction_loss_weight)

        self.normalization = normalization

    def reconstruction_loss(self, x: torch.tensor, x_hat: torch.tensor):
        # treat normalization -> state angles are in [0, 2pi] space and x_hat is in [-1, 1]
        pred_end_effector = forward_kinematics(x_hat)[:, -1, :]
        # move c to cpu ... very convinient way TODO: make this prettier
        x = x.to("cpu")
        # true end effector pose is in x, shape: (batch_size, 2)
        dist_mean = torch.sqrt(torch.sum(torch.float_power(pred_end_effector - x, 2), dim=1)).mean()
        self.r_loss = dist_mean
        return dist_mean


def get_loss_func(config: dict) -> VAELoss:
    loss_func_name = config["loss_func"]
    if loss_func_name == VAELoss.__name__:
        return VAELoss(
            kl_loss_weight=config["kl_loss_weight"],
            reconstruction_loss_weight=config["kl_loss_weight"]
        )
    elif loss_func_name == CyclicVAELoss.__name__:
        return CyclicVAELoss(
            kl_loss_weight=config["kl_loss_weight"],
            reconstruction_loss_weight=config["reconstruction_loss_weight"],
            normalization=config["normalize"]
        )
    elif loss_func_name == DistVAELoss.__name__:
        return DistVAELoss(
            kl_loss_weight=config["kl_loss_weight"],
            reconstruction_loss_weight=config["reconstruction_loss_weight"],
            normalization=config["normalize"]
        )
    else:
        logging.error(f"available loss functions: {VAELoss.__name__}, {CyclicVAELoss.__name__}, {DistVAELoss.__name__}, but you chose {loss_func_name}")