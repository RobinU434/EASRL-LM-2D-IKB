from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from numpy import ndarray
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from matplotlib.figure import Figure

class Logger(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def add_scalar(self, tag: str, scalar: float, global_step: int):
        pass

    @abstractmethod
    def add_histogram(self, tag: str, data: Union[Tensor, ndarray], global_step: int):
        pass
    
    @abstractmethod
    def add_figure(self, tag: str, figure: Figure, global_step: int):
        pass

    @abstractmethod
    def add_image(self, tag: str, data: Union[Tensor, ndarray], global_step: int):
        pass
    
    @abstractmethod
    def add_hparams(self, hparam_dict: Dict[str, Any],
    metric_dict: Dict[str, Any]):
        pass

if __name__ == "__main__":
    tb_logger = SummaryWriter(".")

    tb_logger.add_figure()