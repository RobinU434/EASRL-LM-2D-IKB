# =====================================================================================================================
# This algorithm was adapted from: 
# https://github.com/seungeunrho/minimalRL/blob/master/sac.py 
# (date: 04.12.2022)
# =====================================================================================================================

import torch
import random
import collections


class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n, dtype = torch.float):
        mini_batch = random.sample(self.buffer, n)

        # look at first element from minibatch to determine the shape of action, observation, ...
        s, a, r, s_prime, done = mini_batch[0]
        s_lst = torch.empty((n, *s.shape))
        a_lst = torch.empty((n, *a.shape))
        r_lst = torch.empty((n, 1))
        s_prime_lst = torch.empty((n, *s_prime.shape))
        done_mask_lst = torch.empty((n, 1))

        for idx, transition in enumerate(mini_batch):
            s, a, r, s_prime, done = transition
            s_lst[idx] = torch.tensor(s)
            a_lst[idx] = a
            r_lst[idx] = torch.tensor([r])
            s_prime_lst[idx] = torch.tensor(s_prime)
            done_mask = float(done)
            done_mask_lst[idx] = torch.tensor([done_mask])

        return  s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst 
    
    @property
    def size(self):
        return len(self.buffer)

    def __len__(self):
        return self.size