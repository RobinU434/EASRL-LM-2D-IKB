import logging
from typing import Any, Dict, Tuple, Union
import torch

from torch import nn
from torch import optim, Tensor

import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal
from torch.distributions import constraints
from algorithms.sac.actor.base_actor import Actor

from rl.algorithms.common.distributions import get_distribution
from rl.algorithms.sac.actor.latent_actor import LatentActor

# from rl.algorithms.sac.actor.super_actor import SuperActor

from rl.algorithms.sac.actor.super_actor import SuperActor
from utils.metrics import Metrics
from utils.model.neural_network import NeuralNetwork

class PolicyNet(NeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float,
        actor_config: Dict[str, Any],
        init_alpha: float,
        lr_alpha: float,
        action_magnitude: float = 1,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, learning_rate, device)

        if actor_config["type"] == Actor.__name__:
            self.actor = Actor(
                input_dim=input_dim,
                output_dim=output_dim,
                **actor_config,
            )
        elif actor_config["type"] == LatentActor.__name__:
            self.actor = LatentActor(
                input_dim=input_dim,
                output_dim=output_dim,
                device=device,
                **actor_config,
            )
        # elif actor_config["type"] == SuperActor.__name__:
        #     self.actor = SuperActor(
        #         device=actor_config["device"],
        #         input_dim=input_dim,
        #         output_dim=output_dim,
        #         learning_rate=learning_rate,
        #         super_learning_mode=actor_config["learning_mode"],
        #         checkpoint_dir=actor_config["checkpoint_dir"],
        #         log_dir=actor_config["log_dir"],
        #     )
        else:
            raise NotImplementedError(
                f"requested actor {actor_config['type']} is not implemented."
            )
        self.actor.to(self._device)
        logging.info(self.actor)
        
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self._learning_rate_alpha = lr_alpha
        self.log_alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=self._learning_rate_alpha
        )

        self.action_magnitude = action_magnitude

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """_summary_

        Args:
            x (Tensor): shape = (batch_size, input_size)

        Returns:
            Tuple(tensor):
        """
        action, log_prob = self.actor.forward(x.to(self._device))
        # real_action = (torch.tanh(action) + 1.0) * torch.pi  # multiply by pi in order to match the action space
        # TODO(RobunU434): add post processor functionality in here
        real_action = torch.tanh(action) * self.action_magnitude - (
            1 - self.action_magnitude
        )

        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is a bijection and differentiable
        # this equation can be found in the original paper as equation (21)
        real_log_prob = log_prob - torch.sum(
            torch.log(1 - torch.tanh(action).pow(2) + 1e-7), dim=-1
        )  # sum over action dimension

        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch, target_entropy):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        # minus: to change the sign from the log prob -> log prob is normally negative
        entropy = -self.log_alpha.exp() * log_prob
        # TODO: make env easier: at this point make sure there is no exploration only exploitation
        # entropy = - 0.0 # *  log_prob

        q1_val = q1(s.to(self._device), a.to(self._device))
        q2_val = q2(s.to(self._device), a.to(self._device))
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = (-min_q - entropy).mean()  # for gradient ascent
        self.actor.train(loss)

        # learn alpha parameter
        self.log_alpha_optimizer.zero_grad()
        # if log_prob + (-target_entropy) is positive -> make log_alpha as big as positive
        # if log_prob + (-target_entropy) is negative -> make log_alpha as small as positive
        alpha_loss = -(
            self.log_alpha.exp() * (log_prob + target_entropy).detach()
        ).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return entropy, loss, alpha_loss

    @property
    def optimizer(self) -> optim.Optimizer:
        return self.actor.optimizer # type: ignore

    @property
    def hparams(self) -> Dict[str, Union[str, int, float]]:
        return {
            "learning_rate": self.actor._learning_rate,  # type: ignore
            "learning_rate_alpha": self._learning_rate_alpha,
        }

    def save(self, path: str, metrics: Metrics = ..., epoch_idx: int = 0):
        self.actor.save(path, metrics, epoch_idx)

    def load_checkpoint(self, path: str):
        self.actor.load_checkpoint(path)
