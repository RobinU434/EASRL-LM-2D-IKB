import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from algorithms.ddpg.helper import soft_update
from algorithms.ddpg.mu_net import MuNet
from algorithms.ddpg.noise import OrnsteinUhlenbeckNoise
from algorithms.ddpg.q_net import QNet
from algorithms.ddpg.replay_buffer import ReplayBuffer
from algorithms.helper.helper import get_space_size
from logger.fs_logger import FileSystemLogger

#Hyperparameters

class DDPG:
    def __init__(self,
                 env: gym.Env,
                 logging_writer: SummaryWriter,
                 fs_logger: FileSystemLogger,
                 lr_mu: float = 0.0005,
                 lr_q: float = 0.001,
                 gamma: float = 0.99,
                 batch_size: int = 32,
                 buffer_limit: int = 50000,
                 tau: float = 0.005,
                 start_buffer_size: float = 2000,
                 train_iterations: float = 10,
                 ) -> None:
        
        self._env = env

        self._logger =  logging_writer
        self._fs_logger = fs_logger
    
        self._lr_mu        = lr_mu
        self._lr_q         = lr_q
        self._gamma        = gamma
        self._batch_size   = batch_size
        self._buffer_limit = buffer_limit
        self._tau          = tau # for target network soft update
        self._start_buffer_size = start_buffer_size
        self._train_iterations = train_iterations

        self._memory = ReplayBuffer(self._buffer_limit)

        action_dim = get_space_size(self._env.action_space.shape)
        observation_dim = get_space_size(self._env.observation_space.shape)

        self._q = QNet(action_dim=action_dim, state_dim=observation_dim, output_dim=1, learning_rate=self._lr_q)
        self._q_target = QNet(action_dim=action_dim, state_dim=observation_dim, output_dim=1, learning_rate=self._lr_q)
        self._q_target.load_state_dict(self._q.state_dict())

        self._mu = MuNet(input_dim=observation_dim, output_dim=action_dim, learning_rate=self._lr_mu)
        self._mu_target = MuNet(input_dim=observation_dim, output_dim=action_dim, learning_rate=self._lr_mu)
        self._mu_target.load_state_dict(self._mu.state_dict())

        self._ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_dim))

    def calc_target(self, r, s_prime, done_mask):
        target = r + self._gamma * self._q_target(s_prime, self._mu_target(s_prime)) * done_mask
        return target

    def train_q(self, s, a, target):
        zeros = torch.zeros_like(self._q(s, a), requires_grad=True)
        q_loss = F.smooth_l1_loss(zeros, target.detach())
        self._q.train(q_loss)
        return q_loss
    
    def train_mu(self, s):
        mu_loss = -self._q(s, self._mu(s)).mean() # That's all for the policy loss.
        self._mu.train(mu_loss)
        return mu_loss

    def train(self, n_epochs: int, print_interval: int = 20):
        score = 0.0
        num_steps = 0
        for epoch_idx in range(n_epochs):
            s = self._env.reset()
            done = False
            
            while not done:
                a = self._mu(torch.from_numpy(s).float()) 
                a = a + self._ou_noise()[0]
                s_prime, r, done, info = self._env.step(a.detach().numpy())
                self._memory.put((s, a, r, s_prime, done))
                score = score + r
                s = s_prime

            num_steps = num_steps + self._env.num_steps
            
            q_loss_list = []
            mu_loss_list = []
            if len(self._memory) > self._start_buffer_size:
                for i in range(self._train_iterations):
                    s, a, r, s_prime, done_mask  = self._memory.sample(self._batch_size)
                    target = self.calc_target(r, s_prime, done_mask)
                    q_loss = self.train_q(s, a, target)
                    q_loss_list.append(q_loss.detach())
                    mu_loss = self.train_mu(s)
                    mu_loss_list.append(mu_loss.detach())

                    # train(self._mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                    soft_update(self._mu, self._mu_target, self._tau)
                    soft_update(self._q,  self._q_target, self._tau)
            
            if epoch_idx % print_interval == 0 and epoch_idx != 0:
                avg_episode_len = num_steps / print_interval
                mean_reward = score / num_steps
                print("# of episode :{}, avg score : {:.1f}".format(epoch_idx, mean_reward))
                
                if self._logger is not None:
                    self._logger.add_scalar("stats/mean_reward", mean_reward, epoch_idx)
                    self._logger.add_scalar("stats/mean_episode_len", avg_episode_len, epoch_idx)
                    self._logger.add_scalar("ddpg/mu_loss", torch.tensor(mu_loss_list).mean(), epoch_idx)
                    self._logger.add_scalar("ddpg/q_loss", torch.tensor(q_loss_list).mean(), epoch_idx)

                if self._fs_logger is not None:
                    self._fs_logger.add_scalar("stats/mean_reward", mean_reward, epoch_idx)
                    self._fs_logger.add_scalar("stats/mean_episode_len", avg_episode_len, epoch_idx)
                    self._fs_logger.add_scalar("ddpg/mu_loss", torch.tensor(mu_loss_list).mean(), epoch_idx)
                    self._fs_logger.add_scalar("ddpg/q_loss", torch.tensor(q_loss_list).mean(), epoch_idx)
                
                torch.save({
                    "epoch": epoch_idx,
                    "mu_model_state": self._mu.state_dict(),
                    "mu_optimizer_state_dict": self._mu.optimizer.state_dict(),
                    "mu_target_model_state": self._mu_target.state_dict(),
                    "mu_target_optimizer_state_dict": self._mu_target.optimizer.state_dict(),
                    "q_state_dict": self._q.state_dict(),
                    "q_optimizer_state_dict": self._q.optimizer.state_dict(),
                    "q_target_state_dict": self._q_target.state_dict(),
                    "q_target_optimizer_state_dict": self._q_target.optimizer.state_dict(),
                    "reward": float(mean_reward),
                }, self._fs_logger.path + f"model_{epoch_idx}_reward_{mean_reward:.4f}.pt")

                score = 0.0
                num_steps = 0

        self.env.close()


