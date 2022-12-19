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
        logger: SummaryWriter, 
        learning_rate = 0.0003,
        gamma = 0.9,
        lmbda = 0.9,
        eps_clip = 0.2,
        K_epoch = 10,
        rollout_len = 3,
        buffer_size = 30,
        minibatch_size = 32
        ) -> None:
        
        self.env = env
        
        self.logger = logger

        self.memory = RolloutBuffer(buffer_size)

        self.learning_rate  = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.rollout_len = rollout_len
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size

        # define model
        input_size = get_space_size(self.env.observation_space.shape)
        output_size = get_space_size(self.env.action_space.shape)

        self.model = Module(learning_rate, input_size, output_size)

    def calc_advantage(self, data):
        data_with_adv = []

        for mini_batch in data:
            # minibatch is a tuple of state, action, ... batches
            # each batch has the same length as the stored rollout
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch

            with torch.no_grad():
                td_target = r + self.gamma * self.model.value(s_prime) * done_mask
                delta = td_target - self.model.value(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

    def train_net(self):
        if len(self.memory) == self.minibatch_size * self.buffer_size:
            print(len(self.memory))
            data = self.memory.make_batch(self.minibatch_size)
            data = self.calc_advantage(data)

            for i in range(self.K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.model.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.model.value(s) , td_target)

                    self.model.train(loss)

    def train(self, n_epochs):
        score = 0.0
        print_interval = 20
        rollout = []

        for n_epi in range(n_epochs):
            # sample rollout 
            s = self.env.reset()
            done = False
            while not done:
                for t in range(self.rollout_len):
                    mu, std = self.model.pi(torch.from_numpy(s).float())
                    dist = Normal(mu, std)
                    a = dist.sample()
                    log_prob = dist.log_prob(a)
                    s_prime, r, done, info = self.env.step([a.item()])

                    rollout.append((s, a, r/10.0, s_prime, log_prob.item(), done))
                    if len(rollout) == self.rollout_len:
                        self.memory.put(rollout)
                        rollout = []

                    s = s_prime
                    score += r
                    if done:
                        break

                self.train_net()

            if n_epi%print_interval==0 and n_epi!=0:
                print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, score/print_interval, self.model.optimization_step))
                score = 0.0

        self.env.close()

