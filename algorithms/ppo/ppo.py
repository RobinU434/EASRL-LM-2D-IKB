import gym
import numpy as np
import torch

import torch.nn.functional as F

from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from algorithms.ppo.buffer import RolloutBuffer
from algorithms.ppo.model import Module
from algorithms.helper.helper  import get_space_size
from algorithms.helper.kl_div import kl_divergence_from_weights

from logger.fs_logger import FileSystemLogger

class PPO:
    def __init__(
        self,
        env : gym.Env, 
        logging_writer: SummaryWriter,
        fs_logger: FileSystemLogger,
        learning_rate: float = 0.0003,
        gamma: float = 0.9,
        lmbda: float = 0.9,
        eps_clip: float = 0.2,
        K_epoch: float = 10,
        rollout_len: float = 3,
        buffer_size: float = 30,
        minibatch_size: float = 32,
        print_interval: float = 20,
        action_covariance_decay: float = 0.5,
        action_covariance_mode: str = "independent"
        ) -> None:
        
        self._env = env
        
        self._logger = logging_writer
        self._fs_logger = fs_logger
        
        self._memory = RolloutBuffer(buffer_size)

        self._learning_rate  = learning_rate
        self._gamma = gamma
        self._lmbda = lmbda
        self._eps_clip = eps_clip
        self._K_epoch = K_epoch
        self._rollout_len = rollout_len
        self._buffer_size = buffer_size
        self._minibatch_size = minibatch_size
        self._print_interval = print_interval
        self._action_covariance_decay = action_covariance_decay
        self._action_covariance_mode = action_covariance_mode

        # define model
        input_size = get_space_size(self._env.observation_space.shape)
        output_size = get_space_size(self._env.action_space.shape)

        self._model = Module(learning_rate, input_size, output_size)

        # logging parameters
        self._kl_div = []

    def calc_advantage(self, data):
        data_with_adv = []

        for mini_batch in data:
            # minibatch is a tuple of state, action, ... batches
            # each batch has the same length as the stored rollout
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch

            with torch.no_grad():
                td_target = r + self._gamma * self._model.value(s_prime) * done_mask
                delta = td_target - self._model.value(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self._gamma * self._lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

    def train_net(self):
        if len(self._memory) == self._minibatch_size * self._buffer_size:
            data = self._memory.make_batch(self._minibatch_size)
            data = self.calc_advantage(data)

            prev_weight_dist = self._model.get_weights()

            for i in range(self._K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self._model.pi(s, softmax_dim=1)
                    # dist = get_distribution(loc=mu, std=std, mode=self._action_covariance_mode, decay=self._action_covariance_decay)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self._eps_clip, 1 + self._eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self._model.value(s) , td_target)

                    self._model.train(loss)

            current_weight_dist = self._model.get_weights()

            _, _, kl_div = kl_divergence_from_weights(prev_weight_dist, current_weight_dist)
            self._kl_div.append(kl_div)

    def train(self, n_epochs):
        score = 0.0
        num_steps = 0
        rollout = []

        for epoch_idx in range(1, n_epochs + 1):  # plus 1 for logging
            # sample rollout 
            s = self._env.reset()
            done = False
            while not done:
                for t in range(self._rollout_len):
                    mu, std = self._model.pi(torch.from_numpy(s).float())
                    dist = Normal(mu, std)
                    a = dist.sample()
                    a = a.detach()
                    log_prob = dist.log_prob(a)
                    log_prob = torch.sum(log_prob)  # independence assumption between individual propabbilities
                    # log(p(a1, a2)) = log(p(a1) * p(a2)) = log(p(a1)) + log(p(a2))
                    s_prime, r, done, info = self._env.step(a.numpy())

                    rollout.append((s, a, r, s_prime, log_prob, done))
                    if len(rollout) == self._rollout_len:
                        self._memory.put(rollout)
                        rollout = []

                    s = s_prime
                    score += r
                    num_steps += 1
                    if done:
                        break

                self.train_net()

            if epoch_idx % self._print_interval == 0:
                avg_episode_len = num_steps / self._print_interval 
                mean_reward = score / num_steps
                print("# of episode :{}, mean reward / step : {:.1f}".format(epoch_idx, mean_reward))
                # log metrics
                # in tensorboard
                if self._logger is not None:
                    self._logger.add_scalar("stats/mean_reward", mean_reward, epoch_idx)
                    self._logger.add_scalar("stats/mean_episode_len", avg_episode_len, epoch_idx)
                    self._logger.add_scalar("ppo/kl_div", np.mean(self._kl_div))
                
                # log in filesystem 
                if self._fs_logger is not None:
                    self._fs_logger.add_scalar("stats/mean_reward", mean_reward, epoch_idx)
                    self._fs_logger.add_scalar("stats/mean_episode_len", avg_episode_len, epoch_idx)
                    self._fs_logger.add_scalar("ppo/kl_div", np.mean(self._kl_div))
                    
                score = 0.0
                num_steps = 0
                self._kl_div = []

        self._fs_logger.dump()

        self._env.close()

