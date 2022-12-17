# =====================================================================================================================
# This algorithm was adpated from: 
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
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = float(done)
            done_mask_lst.append([done_mask])

        return  torch.tensor(s_lst, dtype=dtype), \
                torch.stack(a_lst), \
                torch.tensor(r_lst, dtype=dtype), \
                torch.tensor(s_prime_lst, dtype=dtype), \
                torch.tensor(done_mask_lst, dtype=dtype)
    
    @property
    def size(self):
        return len(self.buffer)

    def __len__(self):
        return self.size