

from numpy import ndarray
from latent.metrics.base_metrics import IKMetrics, Metrics



class SupervisedMetrics(Metrics):
    def __init__(self) -> None:
        super().__init__()

        self.loss: ndarray


class SupervisedIKMetrics(SupervisedMetrics, IKMetrics):
    def __init__(self) -> None:
        super().__init__()