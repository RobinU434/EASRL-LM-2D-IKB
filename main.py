
import gym
from algorithms.sac import SAC
from envs.plane_robot_env import PlaneRobotEnv
from envs.task.reach_goal import ReachGoalTask


task = ReachGoalTask(epsilon=1)
env = PlaneRobotEnv(
  n_joints=500,
  segment_lenght=1,
  task=task)
# env = gym.make('Pendulum-v1')

sac = SAC(env)

sac.train(1000)
