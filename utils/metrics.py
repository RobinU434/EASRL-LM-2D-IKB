import numpy as np
import logging
from typing import Any, Callable, Dict

from numpy import ndarray


class Metrics:
    """is a data class which stores all metrics collected during a training process"""

    def __init__(self) -> None:
        """declare in descendent all member variables you want to have"""
        pass

    @classmethod
    def from_dict(cls, metrics_dict: Dict[str, Any]):
        """stores a given dictionary in the metrics class"""
        metrics = cls()
        if "loss" not in metrics_dict.keys():
            logging.debug("necessary loss metric is not set")
        for metric_name, metric_value in metrics_dict.items():
            setattr(metrics, metric_name, metric_value)
        return metrics

    def _aggregate(self, func: Callable[[ndarray], float]) -> Dict[str, float]:
        """aggregates all member variables with the given function and returns a dict with key = member function name and value = aggregated value

        Args:
            func (Callable[[ndarray], float]): how to aggregate a given numpy array

        Returns:
            Dict[str, float]: dictionary with key: member function name, value: aggregated value
        """
        result = {}
        for key, value in vars(self).items():
            result[key] = func(value)
        return result

    def mean(self) -> Dict[str, float]:
        """averages all member variables

        Returns:
            Dict[str, float]: dictionary with all member variables and their corresponding mean
        """
        return self._aggregate(np.mean)

    def std(self) -> Dict[str, float]:
        """calculates the std over all member variables

        Returns:
            Dict[str, float]: dictionary with all member variables and their corresponding std deviation
        """
        return self._aggregate(np.std)

    def append(self, metrics: "Metrics") -> None:
        """appends metric

        Args:
            metrics (Metrics): _description_
        """
        for key, value in vars(metrics).items():
            try:
                own_metric = getattr(self, key)
                extended_metric = np.concatenate([own_metric, value], axis=0)
            except AttributeError:
                extended_metric = value
            setattr(self, key, extended_metric)

    @property
    def num_metrics(self) -> int:
        """number of different metrics

        Returns:
            int: number of different metrics
        """
        return len(vars(self))
