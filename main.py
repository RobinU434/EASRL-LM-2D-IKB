
from algorithms.sac import SAC
from envs.plane_robot_env import PlaneRobotEnv


env = PlaneRobotEnv(4, 1)
sac = SAC(env)

sac.train(2)