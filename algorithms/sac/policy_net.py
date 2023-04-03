import torch

from torch import nn
from torch import optim

import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal
from torch.distributions import constraints

from algorithms.common.distributions import get_distribution
from algorithms.sac.actor.latent_actor import LatentActor
from algorithms.sac.actor.multi_actor import Actor, InformedMultiAgent, MultiAgent

class PolicyNet(nn.Module):
    def __init__(
        self,
        learning_rate,
        input_size,
        output_size,
        init_alpha,
        lr_alpha,
        action_sampling_mode: str = "independent",
        covariance_decay: float = 0.5,
        action_magnitude: float = 1,
        ):
        super(PolicyNet, self).__init__()
        # self.actor = Actor(input_size, output_size, learning_rate)
        # self.actor = InformedMultiAgent(input_size, output_size, learning_rate, 2)
        # self.actor = MultiAgent(input_size, output_size, learning_rate, 2)
        # self.actor = LatentActor(input_size, 5, output_size, learning_rate, kl_weight=0.01, vae_learning=True)
        # self.actor = LatentActor(input_size, 5, output_size, learning_rate, enhanced_latent_dim=2, vae_learning=True)
        self.actor = LatentActor("cpu", input_size, 2, output_size, learning_rate, enhanced_latent_dim=6, vae_learning=True)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.action_magnitude = action_magnitude

        self.covariance_decay = covariance_decay
        self.action_sampling_mode = action_sampling_mode
        self.action_sampling_func = get_distribution

    def forward(self, x):
        """_summary_

        Args:
            x (torch.tensor): shape = (batch_size, input_size)

        Returns:
            Tuple(torch.tensor): 
        """
        mu, std = self.actor.forward(x)
        # print(mu, std)

        dist = Normal(mu, std + 1e-28, validate_args={"scale": constraints.greater_than_eq})
        # dist = self.action_sampling_func(mu, std, self.action_sampling_mode, self.covariance_decay)
        action = dist.rsample()
        # shape action for buffer layout
        action = action.squeeze()
        log_prob = dist.log_prob(action)
        # sum log prob TODO: Does this still hold with dependent log probabilities?
        log_prob = log_prob.sum(dim=1) # independence assumption between individual probabilities
        # log(p(a1, a2)) = log(p(a1) * p(a2)) = log(p(a1)) + log(p(a2))

        if type(self.actor) == LatentActor:
            if self.actor.auto_encoder.conditional_info_dim == 0:
                action = self.actor.auto_encoder.decoder.forward(action)
            elif self.actor.auto_encoder.conditional_info_dim == 2:
                target_pos = x[:, :2].squeeze()
                latent_input = torch.cat([action, target_pos], dim=len(target_pos.size()) - 1)
                action = self.actor.auto_encoder.decoder.forward(latent_input)

        # TODO: implement function to maps from sampled vector into direct action -> latent decoder
         
        # real_action = (torch.tanh(action) + 1.0) * torch.pi  # multiply by pi in order to match the action space
        real_action = torch.tanh(action) * self.action_magnitude

        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is a bijection and differentiable
        # this equation can be found in the original paper as equation (21)
        real_log_prob = log_prob - torch.sum(torch.log(1 - torch.tanh(action).pow(2) + 1e-7))

        # print(real_log_prob)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch, target_entropy):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        # minus: to change the sign from the log prob -> log prob is normally negative
        entropy = -self.log_alpha.exp() * log_prob
        # TODO: make env easier: at this point make sure there is no exploration only exploitation
        # entropy = - 0.0 # *  log_prob
      

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        
        loss = (-min_q - entropy).mean() # for gradient ascent
        self.actor.train(loss)

        # learn alpha parameter
        self.log_alpha_optimizer.zero_grad()
        # if log_prob + (-target_entropy) is positive -> make log_alpha as big as positive
        # if log_prob + (-target_entropy) is negative -> make log_alpha as small as positive
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return entropy, loss, alpha_loss

    @property
    def optimizer(self):
        return self.actor.optimizer
