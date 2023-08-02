import torch
from torch import Tensor

from latent.criterion.base_criterion import Criterion

class EuclideanDistance(Criterion):
    def __init__(self, norm: float = 1, device: str = "cpu") -> None:
        super().__init__(device)
        self._norm = norm

    def __call__(self, y: Tensor, x_hat: Tensor) -> Tensor:
        """
        x: desired target position
        x_hat: is the predicted target position

        Args:
            x (torch.Tensor): ground truth
            x_hat (torch.Tensor): prediction

        Returns:
            torch.Tensor: distance between y and x_hat
        """
        self.loss = torch.linalg.norm(y - x_hat, axis=1).mean()
        return self.loss