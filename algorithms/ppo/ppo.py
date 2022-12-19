import gym
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from algorithms.ppo.buffer import RolloutBuffer
from algorithms.ppo.model import Module
from algorithms.helper.helper  import get_space_size

class PPO:
    def __init__(
        self,
        env : gym.Env, 
        logging_writer: SummaryWriter, 
        learning_rate = 0.0003,
        gamma = 0.9,
        lmbda = 0.9,
        eps_clip = 0.2,
        K_epoch = 10,
        rollout_len = 3,
        buffer_size = 30,
        minibatch_size = 32
        ) -> None:
        
        self._env = env
        
        self._logger = logging_writer

        self._memory = RolloutBuffer(buffer_size)

        self._learning_rate  = learning_rate
        self._gamma = gamma
        self._lmbda = lmbda
        self._eps_clip = eps_clip
        self._K_epoch = K_epoch
        self._rollout_len = rollout_len
        self._buffer_size = buffer_size
        self._minibatch_size = minibatch_size

        # define model
        input_size = get_space_size(self._env.observation_space.shape)
        output_size = get_space_size(self._env.action_space.shape)

        self._model = Module(learning_rate, input_size, output_size)

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

            for i in range(self._K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self._model.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self._eps_clip, 1 + self._eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self._model.value(s) , td_target)

                    self._model.train(loss)

    def train(self, n_epochs):
        score = 0.0
        num_steps = 0
        print_interval = 20
        rollout = []

        for epoch_idx in range(n_epochs + 1):  # plus 1 for logging
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
                    s_prime, r, done, info = self._env.step(a)

                    rollout.append((s, a, r/10.0, s_prime, log_prob, done))
                    if len(rollout) == self._rollout_len:
                        self._memory.put(rollout)
                        rollout = []

                    s = s_prime
                    score += r
                    num_steps += 1
                    if done:
                        break

                self.train_net()

            if epoch_idx % print_interval == 0 and epoch_idx != 0:
                avg_episode_len = num_steps / print_interval 
                mean_reward = score / num_steps
                print("# of episode :{}, mean reward / step : {:.1f}, opt step: {}".format(epoch_idx, mean_reward, self._model.optimization_step))
                self._logger.add_scalar("mean_reward", mean_reward, epoch_idx)
                self._logger.add_scalar("mean_episode_len", avg_episode_len, epoch_idx)
                
                score = 0.0
                num_steps = 0

        self._env.close()

