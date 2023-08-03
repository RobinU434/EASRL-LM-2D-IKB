

from numpy import ndarray
from latent.metrics.base_metrics import IKMetrics, Metrics


class VAEMetrics(Metrics):
    def __init__(self) -> None:
        super().__init__()
        self.loss: ndarray
        self.std: ndarray
        self.kl_loss: ndarray
        self.reconstruction_loss: ndarray


class VAEInvKinMetrics(VAEMetrics, IKMetrics):
    def __init__(self) -> None:
        super().__init__()