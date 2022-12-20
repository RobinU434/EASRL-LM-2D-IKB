import torch
import numpy as np

from algorithms.ppo.buffer import RolloutBuffer

def test_rollout_buffer():
    num_mini_batches = 1
    buffer = RolloutBuffer(num_mini_batches)

    rollout_len = 4
    rollout = []

    state_dim = 6
    state = list(range(rollout_len * state_dim))
    state = np.array(state).reshape(rollout_len, state_dim)

    action_dim = 8
    action = list(range(rollout_len * action_dim))
    action = torch.tensor(action).reshape(rollout_len, action_dim)

    reward = list(range(rollout_len))

    next_state = list(range(rollout_len * state_dim))[::-1]
    next_state = np.array(next_state).reshape(rollout_len, state_dim)

    log_prob = list(range(rollout_len))[::-1]

    done = [False for _ in range(rollout_len)]

    for idx in range(rollout_len):
        rollout.append((state[idx], action[idx], reward[idx], next_state[idx], log_prob[idx], done[idx]))

    buffer.put(rollout)

    # test buffer_length
    assert 1 == len(buffer)

    data = buffer.make_batch(1)

    assert 0 == len(buffer)

    expected_out_one_dim = [
        (
            torch.tensor(
                [
                    [
                        [ 0.,  1.,  2.,  3.,  4.,  5.],
                        [ 6.,  7.,  8.,  9., 10., 11.],
                        [12., 13., 14., 15., 16., 17.],
                        [18., 19., 20., 21., 22., 23.]
                    ]
                ]
            ), 
            torch.tensor(
                [
                    [
                        [ 0,  1,  2,  3,  4,  5,  6,  7],
                        [ 8,  9, 10, 11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20, 21, 22, 23],
                        [24, 25, 26, 27, 28, 29, 30, 31]
                    ]
                ]
            ),
            torch.tensor([[[0.], [1.], [2.], [3.]]]), 
            torch.tensor(
                [
                    [
                        [23., 22., 21., 20., 19., 18.],
                        [17., 16., 15., 14., 13., 12.],
                        [11., 10.,  9.,  8.,  7.,  6.],
                        [ 5.,  4.,  3.,  2.,  1.,  0.]
                    ]
                ]
            ),
            torch.tensor([[[0.], [0.], [0.], [0.]]]), 
            torch.tensor([[[3.], [2.], [1.], [0.]]])
        )
    ]

    for expected, computed in zip(expected_out_one_dim, data):
        # e_ ... is short for  expected
        e_s, e_a, e_r, e_s_prime, e_done_mask, e_log_prob = expected
        s, a, r, s_prime, done_mask, log_prob = computed

        assert torch.equal(e_s, s)
        assert torch.equal(e_a, a)
        assert torch.equal(e_r, r)
        assert torch.equal(e_s_prime, s_prime)
        assert torch.equal(e_done_mask, done_mask)
        assert torch.equal(e_log_prob, log_prob)

    buffer.put(rollout)
    buffer.put(rollout)

    assert 2 == len(buffer)

    data = buffer.make_batch(2)

    assert 0 == len(buffer)

    expected_out_two_dim = [
        (
            torch.tensor(
                [
                    [
                        [ 0.,  1.,  2.,  3.,  4.,  5.],
                        [ 6.,  7.,  8.,  9., 10., 11.],
                        [12., 13., 14., 15., 16., 17.],
                        [18., 19., 20., 21., 22., 23.]
                    ],
                    [
                        [ 0.,  1.,  2.,  3.,  4.,  5.],
                        [ 6.,  7.,  8.,  9., 10., 11.],
                        [12., 13., 14., 15., 16., 17.],
                        [18., 19., 20., 21., 22., 23.]
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [ 0,  1,  2,  3,  4,  5,  6,  7],
                        [ 8,  9, 10, 11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20, 21, 22, 23],
                        [24, 25, 26, 27, 28, 29, 30, 31]
                    ],
                    [
                        [ 0,  1,  2,  3,  4,  5,  6,  7],
                        [ 8,  9, 10, 11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20, 21, 22, 23],
                        [24, 25, 26, 27, 28, 29, 30, 31]
                    ]
                ]
            ),
            torch.tensor([
                [
                    [0.], [1.], [2.], [3.]
                ], 
                [
                    [0.], [1.], [2.], [3.]
                ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [23., 22., 21., 20., 19., 18.],
                        [17., 16., 15., 14., 13., 12.],
                        [11., 10.,  9.,  8.,  7.,  6.],
                        [ 5.,  4.,  3.,  2.,  1.,  0.]
                    ],
                    [
                        [23., 22., 21., 20., 19., 18.],
                        [17., 16., 15., 14., 13., 12.],
                        [11., 10.,  9.,  8.,  7.,  6.],
                        [ 5.,  4.,  3.,  2.,  1.,  0.]
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [0.], [0.], [0.], [0.]],
                    [
                        [0.], [0.], [0.], [0.]]
                ]
            ),
            torch.tensor(
                [
                    [
                        [3.], [2.], [1.], [0.]],
                    [
                        [3.], [2.], [1.], [0.]]
                ]
            )
        )
    ]

    for expected, computed in zip(expected_out_two_dim, data):
        # e_ ... is short for  expected
        e_s, e_a, e_r, e_s_prime, e_done_mask, e_log_prob = expected
        s, a, r, s_prime, done_mask, log_prob = computed

        assert torch.equal(e_s, s)
        assert torch.equal(e_a, a)
        assert torch.equal(e_r, r)
        assert torch.equal(e_s_prime, s_prime)
        assert torch.equal(e_done_mask, done_mask)
        assert torch.equal(e_log_prob, log_prob)

    
    # create new buffer 
    num_mini_batches = 2
    buffer = RolloutBuffer(2)

    assert len(buffer) == 0

    buffer.put(rollout)
    buffer.put(rollout)

    assert len(buffer) == 2

    data = buffer.make_batch(1)

    expected_out = [expected_out_one_dim[0], expected_out_two_dim[0]]

    for expected, computed in zip(expected_out, data):
        # e_ ... is short for  expected
        e_s, e_a, e_r, e_s_prime, e_done_mask, e_log_prob = expected
        s, a, r, s_prime, done_mask, log_prob = computed

        assert torch.equal(e_s, s)
        assert torch.equal(e_a, a)
        assert torch.equal(e_r, r)
        assert torch.equal(e_s_prime, s_prime)
        assert torch.equal(e_done_mask, done_mask)
        assert torch.equal(e_log_prob, log_prob)
    