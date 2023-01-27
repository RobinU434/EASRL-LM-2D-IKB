import torch

from torch import nn
from torch import optim

import torch.nn.functional as F

class BaseQNet(nn.Module):
    def __init__(
        self,
        ):
        super(BaseQNet, self).__init__()
        
    def forward(self, s, a):
        raise NotImplementedError

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        q_val = self.forward(s, a)
        
        loss = F.smooth_l1_loss(q_val, target)
        print(loss.size(), loss.mean())
        self.optimizer.zero_grad()
        loss.mean().backward(retain_graph=True)
        self.optimizer.step()

    def soft_update(self, net_target, tau):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class QNet(nn.Module):
    def __init__(
        self,
        learning_rate,
        input_size_state: int,
        input_size_action: int,
        output_size: int = 1
        ):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(input_size_state, 64)
        self.fc_a = nn.Linear(input_size_action, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, s, a):
        h1 = F.relu(self.fc_s(s))
        h2 = F.relu(self.fc_a(a))
        
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        q_val = self.forward(s, a)
        loss = F.smooth_l1_loss(q_val, target)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def soft_update(self, net_target, tau):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
