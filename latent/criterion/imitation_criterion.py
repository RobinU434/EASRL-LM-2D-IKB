import torch
from torch import Tensor

from latent.criterion.base_criterion import Criterion


class ImitationLoss(Criterion):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)

    @staticmethod
    def _angle_diff(a : Tensor, b: Tensor, kappa: float = torch.pi) -> Tensor:
        # source: https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
        dif = a - b
        return (dif + kappa) % (2 * kappa) - kappa

    def __call__(self, y: Tensor, x_hat: Tensor) -> Tensor:
        # MSE
        self.loss = torch.square(self._angle_diff(y, x_hat)).mean()
        return self.loss