

from typing import Any, Dict
import gym
from process.base_process import Process


class RLProcess(Process):
    def __init__(self, device: str = "cpu", **kwargs) -> None:
        super().__init__(device, **kwargs)

        self._env: gym.Env
        self._num_runs = self._extract_from_kwargs("num_runs", **kwargs)


    def train(self, *args, **kwargs) -> None:
        return super().train(*args, **kwargs)

    def inference(self, *args, **kwargs) -> None:
        return super().inference(*args, **kwargs)

    def build(self, *args, **kwargs) -> None:
        return super().build(*args, **kwargs)
    
    
