import numpy as np


class BaseTask:
    def __init__(self, epsilon: float = 0.01, n_time_steps: int = 200) -> None:
        self._epsilon = epsilon
        self._n_time_steps = n_time_steps

        self._step_counter = 0

    def update(self):
        self._step_counter += 1

    def reset(self):
         self._step_counter = 0

    def reward(self):
        raise NotImplementedError

    def done(self):
        """implementes exceeded time limit

        Returns:
            bool: if time limit was exceeded
        """
        if self._step_counter >= self._n_time_steps:
            return True
        return False