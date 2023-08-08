

from numpy import ndarray
from utils.metrics import Metrics


class SACScalarMetrics(Metrics):
    """data class to provide and interface which metrics to store
    """
    def __init__(self) -> None:
        super().__init__()

        self.actor_loss: ndarray
        self.entropy: ndarray
        self.critic_loss: ndarray
        self.alpha_loss: ndarray
        self.log_prob: ndarray
        self.mean_score: ndarray
        self.episode_len: ndarray


class SACDistributionMetrics(Metrics):
    def __init__(self) -> None:
        super().__init__()

        self.actions: ndarray
