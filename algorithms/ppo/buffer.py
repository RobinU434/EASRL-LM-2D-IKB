import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, buffer_size) -> None:
        # elementes in data are entire rollouts (sequences of transitions)
        self.data = []
        self.buffer_size = buffer_size

    def put(self, transition):
        self.data.append(transition)

    def make_batch(self, batch_size, dtype=torch.float):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for _ in range(self.buffer_size):
            for _ in range(batch_size):
                # get latest rollout from self.data
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                # unpack all transitions and pack them together into state, action, ... batches
                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append(a)
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = float(done)
                    done_lst.append([done_mask])

                a_lst = torch.stack(a_lst)

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                
            mini_batch = (
                torch.tensor(s_batch, dtype=dtype), 
                torch.stack(a_batch),
                torch.tensor(r_batch, dtype=dtype), 
                torch.tensor(s_prime_batch, dtype=dtype),
                torch.tensor(done_batch, dtype=dtype), 
                torch.tensor(prob_a_batch, dtype=dtype)
            )
            data.append(mini_batch)

        return data

    @property 
    def size(self):
        return len(self)

    def __len__(self):
        return len(self.data)