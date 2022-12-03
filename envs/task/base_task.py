import numpy as np


class BaseTask:
    def __init__(self, epsilon) -> None:
        self._epsilon = epsilon
        
    def reward(self):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError