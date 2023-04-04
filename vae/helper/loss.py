import torch


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
    return torch.float_power(diff, 2).mean(dim=1).mean()

def angle_diff(a : torch.tensor, b: torch.tensor, kappa: torch.tensor = torch.pi):
        # source: https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
        dif = a - b
        return (dif + kappa) % (2 * kappa) - kappa


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
    