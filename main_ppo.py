import gym
import torch

from torch.utils.tensorboard import SummaryWriter

from algorithms.ppo.ppo import PPO
from envs.plane_robot_env import PlaneRobotEnv
from envs.task.reach_goal import ReachGoalTask

num_joints = 1

for i in range(1):
  seed = torch.seed()
  task = ReachGoalTask(epsilon=0.1)

  env = PlaneRobotEnv(
    n_joints=num_joints,
    segment_lenght=1,
    task=task)

  logger = SummaryWriter(f"results/ppo/10_000_000/{num_joints}_{seed}")

  ppo = PPO(
    env,
    logging_writer=logger
    )

  ppo.train(10000)