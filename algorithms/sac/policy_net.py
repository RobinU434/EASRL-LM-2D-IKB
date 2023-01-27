import torch

from torch import nn
from torch import optim

import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal
from torch.distributions import constraints

from algorithms.common.distributions import get_distribution
from algorithms.sac.actor import Actor

class PolicyNet(nn.Module):
    def __init__(
        self,
        learning_rate,
        input_size,
        output_size,
        init_alpha,
        lr_alpha,
        action_sampling_mode: str = "independent",
        covariance_decay: float = 0.5
        ):
        super(PolicyNet, self).__init__()
        self.actor = Actor(input_size, output_size, learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        self.covariance_decay = covariance_decay
        self.action_sampling_mode = action_sampling_mode
        self.action_sampling_func = get_distribution

    def forward(self, x):
        # squeeze input tensor from (x, 1) to (x, )
        x = np.squeeze(x)
        mu, std = self.actor.forward(x)
        
        # dist = Normal(mu, std, validate_args={"scale": constraints.greater_than_eq})
        dist = self.action_sampling_func(mu, std, self.action_sampling_mode, self.covariance_decay)
        action = dist.rsample()
        # shape action for buffer layout
        action = action.squeeze()
        log_prob = dist.log_prob(action)
        # sum log prob TODO: Does this still hold with dependent log probabilities?
        log_prob =  log_prob.sum() # independence assumption between individual probabilities
        # log(p(a1, a2)) = log(p(a1) * p(a2)) = log(p(a1)) + log(p(a2))

        real_action = torch.tanh(action)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is a bijection and differentiable
        real_log_prob = log_prob - torch.sum(torch.log(1 - torch.tanh(action).pow(2) + 1e-7))

        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch, target_entropy):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        
        loss = (-min_q - entropy).mean() # for gradient ascent
        self.actor.train(loss)

        # learn alpha parameter
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
