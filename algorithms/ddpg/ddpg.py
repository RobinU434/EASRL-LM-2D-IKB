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
        q_loss = F.smooth_l1_loss(self._q(s, a), target.detach())
        self._q.train(q_loss)
    
    def train_mu(self, s):
        mu_loss = -self._q(s, self._mu(s)).mean() # That's all for the policy loss.
        self._mu.train(mu_loss)

    def train(self, n_epochs: int, print_interval: int = 20):
        score = 0.0
       
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
                    
            if len(self._memory) > self._start_buffer_size:
                for i in range(self._train_iterations):
                    s, a, r, s_prime, done_mask  = self._memory.sample(self._batch_size)
                    target = self.calc_target(r, s_prime, done_mask)
                    self.train_q(s, a, target)
                    self.train_mu(s)

                    # train(self._mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                    soft_update(self._mu, self._mu_target, self._tau)
                    soft_update(self._q,  self._q_target, self._tau)
            
            if epoch_idx % print_interval == 0 and epoch_idx != 0:
                print("# of episode :{}, avg score : {:.1f}".format(epoch_idx, score/print_interval))
                score = 0.0

        self.env.close()


