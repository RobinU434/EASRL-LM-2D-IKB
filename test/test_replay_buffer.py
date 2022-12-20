
import torch
import numpy as np

from algorithms.sac.buffer import ReplayBuffer


def test_put_buffer():
    buffer = ReplayBuffer(1)

    state = list(range(24))
    state = np.array(state)

    action = list(range(32))
    action = torch.tensor(action)

    reward = 0

    next_state = list(range(24))[::-1]
    next_state = np.array(next_state)

    done = False

    transition = (state, action, reward, next_state, done)

    buffer.put(transition)

    buffer_out = buffer.sample(1)
    
    expected_out = (
        torch.tensor(
            [
                [ 
                    0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.,
                    12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.
                ]
            ]
            ), 
        torch.tensor(
            [
                [ 
                    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                ]
            ]
            ), 
        torch.tensor([[0.]]), 
        torch.tensor(
            [
                [
                    23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13., 12., 
                    11., 10., 9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.,  0.
                ]
            ]), 
        torch.tensor([[0.]]))
    
    assert(torch.equal(buffer_out[0], expected_out[0]))
    assert(torch.equal(buffer_out[1], expected_out[1]))
    assert(torch.equal(buffer_out[2], expected_out[2]))
    assert(torch.equal(buffer_out[3], expected_out[3]))
    assert(torch.equal(buffer_out[4], expected_out[4]))

