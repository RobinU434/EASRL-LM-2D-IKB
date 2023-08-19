from utils.metrics import Metrics
from numpy import ndarray

class IKMetrics(Metrics):
    def __init__(self) -> None:
        super().__init__()

        self.distance_loss: ndarray
        self.imitation_loss: ndarray