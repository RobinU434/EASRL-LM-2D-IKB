# =====================================================================================================================
# This algorithm was adpated from: 
# https://github.com/seungeunrho/minimalRL/blob/master/sac.py 
# (date: 04.12.2022)
# =====================================================================================================================

import gym
import torch

from algorithms.buffer import ReplayBuffer
from algorithms.q_net import QNet
from algorithms.policy_net import PolicyNet

from envs.plane_robot_env import PlaneRobotEnv


class SAC:
    def __init__(
        self, 
        env: gym.Env,
        lr_pi = 0.0005, 
        lr_q = 0.001,
        init_alpha = 0.01,
        gamma = 0.98,
        batch_size = 32,
        buffer_limit = 50000,
        start_buffer_size = 1000,
        train_iterations = 20,
        tau = 0.01, # for target network soft update,
        target_entropy = -1.0, # for automated alpha update,
        lr_alpha = 0.001,  # for automated alpha update
        ) -> None:
        
        self._env = env 
        #Hyperparameters
        self._lr_pi             = lr_pi
        self._lr_q              = lr_q
        self._init_alpha        = init_alpha
        self._gamma             = gamma
        self._batch_size        = batch_size
        self._buffer_limit      = buffer_limit
        self._start_buffer_size = start_buffer_size
        self._train_iterations  = train_iterations
        self._tau               = tau # for target network soft update
        self._target_entropy    = target_entropy # for automated alpha update
        self._lr_alpha          = lr_alpha  # for automated alpha update

        # Replay-Buffer
        self._memory = ReplayBuffer(buffer_limit=buffer_limit)

        # Define networks
        input_size = env.observation_space.shape
        output_size = env.action_space.shape

        self._q1 = QNet(lr_q, input_size)
        self._q2 = QNet(lr_q, input_size)
        self._q1_target = QNet(lr_q, input_size)
        self._q2_target = QNet(lr_q, input_size)

        self._pi = PolicyNet(
            lr_pi,
            input_size,
            output_size,
            self._init_alpha,
            self._lr_alpha
            )

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch

        with torch.no_grad():
            a_prime, log_prob= self._pi(s_prime)
            entropy = -self._pi.log_alpha.exp() * log_prob
            
            q1_val = self.q1(s_prime,a_prime), 
            q2_val = self.q2(s_prime,a_prime)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            
            target = r + self._gamma * done * (min_q + entropy)

        return target

    def train(self, n_epochs):
        self._q1_target.load_state_dict(self._q1.state_dict())
        self._q2_target.load_state_dict(self._q2.state_dict())

        score = 0.0
        print_interval = 20

        for n_epi in range(n_epochs):
            s = self._env.reset()
            done = False

            while not done:
                a, log_prob = self._pi(torch.from_numpy(s).float())
                s_prime, r, done, info = self._env.step([2.0*a.item()])
                self._memory.put((s, a.item(), r / 10.0, s_prime, done))  # why is the reward divided by 10?????
                score +=r
                s = s_prime
                    
            if len(self._memory) > self._start_buffer_size:
                for _ in range(self._train_iterations):
                    mini_batch = self._memory.sample(self._batch_size)
                    td_target = self.calc_target(mini_batch)
                    
                    self._q1.train_net(td_target, mini_batch)
                    self._q2.train_net(td_target, mini_batch)
                    
                    entropy = self._pi.train_net(
                        self._q1, 
                        self._q2, 
                        mini_batch,
                        self._target_entropy)
                    
                    self._q1.soft_update(self._q1_target, self._tau)
                    self._q2.soft_update(self._q2_target, self._tau)
                    
            if n_epi%print_interval==0 and n_epi!=0:
                print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score/print_interval, self._pi.log_alpha.exp()))
                score = 0.0

        self._env.close()


if __name__ == '__main__':
    env = PlaneRobotEnv(4, 1)
    sac = SAC(env)