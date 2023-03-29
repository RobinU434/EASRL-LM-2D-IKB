from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

from algorithms.sac.actor.base_actor import Actor

class MultiAgent(nn.Module):
    def __init__(self, input_size, output_size, learning_rate, num_agents, architecture: List[int] = [128]) -> None:
        super().__init__()

        # test if the required output size an integer multiple of num agents
        assert output_size % num_agents == 0
        agent_output_size = output_size // num_agents

        self.agents: List[Actor] = []
        for _ in range(num_agents):
            self.agents.append(Actor(input_size, agent_output_size, learning_rate, architecture))

    def forward(self, x) -> Tuple[torch.tensor, torch.tensor]:
        mu_list = []
        std_list = []

        for agent in self.agents:
            mu, std = agent.forward(x)
            mu_list.append(mu)
            std_list.append(std)
        
        if len(x.size()) > 1:
            # tis means x is batched and is coming from the buffer 
            mu = torch.cat(mu_list, dim=1)
            std = torch.cat(std_list, dim=1)
        else:
            mu = torch.cat(mu_list)
            std = torch.cat(std_list)

        print(mu.size())

        return mu, std

    def train(self, loss) -> None:
        for agent in self.agents:
            print(id(agent))
            agent.train(loss)


"""
This class is about sequential decision making
every agent controls a certain part of the arm
e.g.: 4 agents, every agent controls 3 joints (agent1: joint 0 to 2, ...)
input for every agent:
  - target_position
  - origin position from the predecessor / not really possible because we putting out a parameterized distribution -> input mu and std or calculating the 
  - angels from the joints that the agent is controlling (build a second version where the input space is way bigger)
output from every agent:
  - angels where to set joints
"""
class InformedMultiAgent(nn.Module):
    def __init__(self, input_size, output_size, learning_rate, num_agents, architecture: List[int] = [128]) -> None:
        super().__init__()

        # test if the required output size an integer multiple of num agents
        assert output_size % num_agents == 0
        agent_output_size = output_size // num_agents

        # input size depends linearly on the number of agents
        # target_pos, angles[idx], origin, origin_std
        input_size = 2 + agent_output_size + 2 + 2

        self.num_agents = num_agents
        self.agents: List[Actor] = []
        for _ in range(num_agents):
            self.agents.append(Actor(input_size, agent_output_size, learning_rate, architecture))

    def forward(self, x) -> Tuple[torch.tensor, torch.tensor]:
        target_pos = x[: 2]
        angles = x[4:]
        num_angles_per_agent = len(angles) // self.num_agents
        angles = torch.split(angles, num_angles_per_agent)

        mu_list = []
        std_list = []

        origin = torch.zeros(2)
        origin_std = torch.zeros(2)  # std deviation for sampled origin position
        
        for idx, agent in enumerate(self.agents):
            print(x.size())
            print(target_pos.size(), angles[idx].size(), origin.size(), origin_std.size())
            input = torch.cat([target_pos, angles[idx], origin, origin_std])

            mu, std = agent.forward(input)

            # TODO: look for closed form to calculate the expected position
            actions = self.sample_actions(mu, std, (10_000, num_angles_per_agent))
            forward_kinematics_pos = self.forward_kinematics(actions)
            origin += forward_kinematics_pos.mean(dim=0)
            origin_std = forward_kinematics_pos.std(dim=0)

            mu_list.append(mu)
            std_list.append(std)

        mu = torch.cat(mu_list)
        std = torch.cat(std_list)

        return mu, std

    def train(self, loss) -> None:
        for agent in self.agents:
            agent.train(loss)

    @staticmethod
    def sample_actions(mu: torch.tensor, std: torch.tensor, size: Tuple):
        # actions = torch.normal(mu, std, size)
        # TODO: make it work for torch
        mu = mu.detach().numpy()
        std = std.detach().numpy()
        actions = np.random.normal(mu, std, size)

        actions = torch.tensor(actions)
        return actions

    @staticmethod
    def forward_kinematics(angles: torch.tensor):
        """
        calculates a forward pass for kinematics
        !!!!assert segment length of 1!!!!
        Args:
            angles (torch.tensor): joint angles
        """

        angles = angles % (2 * torch.pi)
        # [x, y]
        num_arms, num_joints = angles.size()
        pos = torch.zeros((num_arms, 2))

        for angle_idx in range(num_joints):
            offset = torch.zeros((num_arms, 2))
            offset[:, 0] = torch.cos(angles[:, angle_idx])
            offset[:, 1] = torch.sin(angles[:, angle_idx])
            pos += offset

        return pos

