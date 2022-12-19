
import gym
import torch

from torch.utils.tensorboard import SummaryWriter

from algorithms.sac.sac import SAC
from envs.plane_robot_env import PlaneRobotEnv
from envs.task.reach_goal import ReachGoalTask


num_joints = 20

for i in range(10):
  seed = torch.seed()
  task = ReachGoalTask(epsilon=0.1)

  env = PlaneRobotEnv(
    n_joints=num_joints,
    segment_lenght=1,
    task=task)

  logger = SummaryWriter(f"results/sac/{num_joints}_{seed}")

  sac = SAC(
    env,
    logging_writer=logger
    )

  sac.train(1000)
