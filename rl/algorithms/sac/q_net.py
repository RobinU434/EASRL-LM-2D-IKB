from typing import Dict, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from utils.metrics import Metrics

from utils.model.neural_network import NeuralNetwork


class QNet(NeuralNetwork):
    def __init__(
        self,
        input_dim_state: int,
        input_dim_action: int,
        output_dim: int = 1,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__(
            input_dim_action + input_dim_state, output_dim, learning_rate, device
        )
        self._output_dim = output_dim
        self._learning_rate = learning_rate
        self._fc_s = nn.Linear(input_dim_state, 64)
        self._fc_a = nn.Linear(input_dim_action, 64)
        self._fc_cat1 = nn.Linear(128, 128)
        self._fc_cat2 = nn.Linear(128, 32)
        self._fc_out = nn.Linear(32, output_dim)
        self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        h1 = F.relu(self._fc_s(state))
        h2 = F.relu(self._fc_a(action))

        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self._fc_cat1(cat))
        q = F.relu(self._fc_cat2(q))
        q = self._fc_out(q)
        return q

    def train(
        self, target: Tensor, mini_batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tensor:
        s, a, r, s_prime, done = mini_batch
        q_val = self.forward(s.to(self._device), a.to(self._device))
        loss = F.smooth_l1_loss(q_val, target).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss

    def save(
        self,
        path: str,
        metrics: Metrics = ...,
        epoch_idx: int = 0,
        model_name: str = "",
    ):
        path = self._create_save_path(path, epoch_idx, metrics, model_name)
        loss = metrics.critic_loss if "critic_loss" in vars(metrics).keys() else 0  # type: ignore
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )

    def _create_save_path(
        self, path: str, epoch_idx: int, metrics: Metrics = ..., model_name: str = ""
    ) -> str:
        model_name = type(self).__name__ if len(model_name) == 0 else model_name
        path += f"/{model_name}_{epoch_idx}"
        if "critic_loss" in vars(metrics).keys():
            path += f"loss_{metrics.critic_loss.mean().item():.4f}.pt"  # type: ignore
        else:
            path += ".pt"
        return path

    def soft_update(self, net_target, tau):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    @property
    def hparams(self) -> Dict[str, Union[str, int, float]]:
        return {"learning_rate": self._learning_rate}

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer
