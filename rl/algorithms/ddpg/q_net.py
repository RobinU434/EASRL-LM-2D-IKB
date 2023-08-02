import torch
import torch.nn as nn
import torch.nn.functional as  F
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, action_dim: int, state_dim: int, output_dim: int, learning_rate: float):
        super(QNet, self).__init__()
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._output_dim = output_dim
        self._learning_rate = learning_rate

        self.fc_s = nn.Linear(state_dim, 64)
        self.fc_a = nn.Linear(action_dim, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, output_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.tensor, a: torch.tensor) -> torch.tensor:
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)

        return q
    
    def train(self, loss: torch.tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
