import json
from copy import deepcopy
from typing import Any, Dict, List, Type, Union

import gym
from gym import Env
from torch.utils.tensorboard.writer import SummaryWriter

from envs.plane_robot_env.ikrlenv.env.plane_robot_env import PlaneRobotEnv
from envs.plane_robot_env.ikrlenv.task.imitation_task import ImitationTask
from envs.plane_robot_env.ikrlenv.task.reach_goal_task import ReachGoalTask
from logger.fs_logger import FileSystemLogger
from rl.algorithms.algorithm import RLAlgorithm
from utils.file_system import load_yaml
from utils.learning_process import LearningProcess


class RLProcess(LearningProcess):
    def __init__(
        self,
        model_entity: Type[RLAlgorithm],
        env_type: Type[Env],
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self._model_entity = model_entity
        self._model_entity_name = model_entity.__name__.lower()
        # self._base_config_path = f"config/base_{self._model_entity_name}.yaml"
        self._base_config_path = f"config/env.yaml"
        super().__init__(device, **kwargs)

        self._env_type = env_type
        self._env: Env
        self._algorithm: RLAlgorithm
        self._num_runs = self._extract_from_kwargs("num_runs", **kwargs)

        self._env_config = self._load_env_config()
        self._algo_config = self._load_algo_config()

        self._logger: List[Union[SummaryWriter, FileSystemLogger]] = []

    def train(self, *args, **kwargs) -> None:
        return self._algorithm.train(*args, **kwargs)

    def inference(self, *args, **kwargs) -> None:
        return super().inference(*args, **kwargs)

    def build(self, *args, **kwargs) -> None:
        self._logger = [SummaryWriter(self._save_dir), FileSystemLogger(self._save_dir)]
        self._env = self._build_env()
        self._algorithm = self._build_algorithm()

    def print_model(self) -> None:
        self._env = self._build_env()
        self._algorithm = self._build_algorithm()
        self._algorithm.print_model()

    def print_config(self) -> None:
        print(json.dumps(self._algo_config, sort_keys=True, indent=4))
        print(json.dumps(self._env_config, sort_keys=True, indent=4))

    def _build_env(self, *args, **kwargs) -> Env:
        task_config = self._env_config["task"]
        if task_config["type"] == ReachGoalTask.__name__:
            task = ReachGoalTask(
                n_time_steps=self._env_config["n_time_steps"],
                n_joints=self._env_config["n_joints"],
                **task_config,
            )
        elif task_config["type"] == ImitationTask.__name__:
            task = ImitationTask(
                n_time_steps=self._env_config["n_time_steps"],
                n_joints=self._env_config["n_joints"],
                **task_config,
            )
        else:
            raise NotImplementedError(f"No such task as: {task_config['type']}")
        self._env_type: Type[PlaneRobotEnv]
        env_config = deepcopy(self._env_config)
        env_config.pop("task")
        env = self._env_type(task=task, **env_config)
        return env

    def _build_algorithm(self) -> RLAlgorithm:
        return self._model_entity(env=self._env, logger=self._logger, log_dir=self._save_dir, device=self._device, **self._algo_config)  # type: ignore

    def load_checkpoint(self, *args, **kwargs) -> None:
        return super().load_checkpoint(*args, **kwargs)

    def _load_env_config(self, checkpoint: str = "", **kwargs) -> Dict[str, Any]:
        base_config_path = "config/env.yaml"
        if len(checkpoint) == 0:
            return load_yaml(base_config_path)
        config_path = "/".join(checkpoint.split("/")[:-1]) + "/env.yaml"
        return load_yaml(config_path)

    def _load_algo_config(self, checkpoint: str = "", **kwargs) -> Dict[str, Any]:
        base_config_path = f"config/base_{self._model_entity_name.lower()}.yaml"
        if len(checkpoint) == 0:
            return load_yaml(base_config_path)
        config_path = (
            "/".join(checkpoint.split("/")[:-1]) + f"/{self._model_entity_name}"
        )
        return load_yaml(config_path)
