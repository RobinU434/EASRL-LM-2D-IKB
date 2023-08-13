import logging
import time
from typing import Any, Dict, Union
from matplotlib.figure import Figure
from numpy import ndarray 
from pandas import DataFrame
from torch import Tensor

from logger.base_logger import Logger
from utils.decorator import not_implemented_warning

class FileSystemLogger(Logger):
    def __init__(self, path) -> None:
        self._path = path

        self._data = []

    def add_scalar(self, entity, y, *argv):
        x_values = {}
        for idx, arg in enumerate(argv):
            x_values[f"x{idx}"] = arg
        
        self._data.append({
            "entity": entity,
            "y": y,
            **x_values,
            "time": time.time()
        })

    @not_implemented_warning
    def add_figure(self, tag: str, figure: Figure, global_step: int):
        pass

    @not_implemented_warning
    def add_image(self, tag: str, data: Union[Tensor, ndarray], global_step: int):
        pass

    @not_implemented_warning
    def add_histogram(self, tag: str, data: Union[Tensor, ndarray], global_step: int):
        pass

    @not_implemented_warning
    def add_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        pass
    
    def dump(self, file_name: str = "results.csv"):
        df = DataFrame(self._data)
        df.to_csv(self._path + "/" + file_name)

    @property
    def path(self):
        return self._path


if __name__ == "__main__":
    logger = FileSystemLogger(".")

    logger.add_scalar("a", 1, 1)
    logger.add_scalar("a", 2, 2)
    logger.add_scalar("a", 3, 3)
    
    logger.add_scalar("b", 4, 4)
    logger.add_scalar("b", 5, 5)
    logger.add_scalar("b", 6, 6)

    logger.add_scalar("c", 4, 4, 1)
    logger.add_scalar("c", 5, 5, 2)
    logger.add_scalar("c", 6, 6, 3)

    logger.add_scalar("d", [4, 1], 4, 1)
    logger.add_scalar("d", [5, 1], 5, 2)
    logger.add_scalar("d", [6, 1], 6, 3)

    df = DataFrame(logger._data)

    print(df)