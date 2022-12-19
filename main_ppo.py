import gym

from algorithms.ppo.ppo import PPO

env = gym.make('Pendulum-v1')

ppo = PPO(env, None)

ppo.train(1000)