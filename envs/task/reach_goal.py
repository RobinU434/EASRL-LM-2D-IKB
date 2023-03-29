import numpy as np
from envs.task.base_task import BaseTask


class ReachGoalTask(BaseTask):
    def __init__(self, config) -> None:
        epsilon = config["target_epsilon"]
        n_time_steps = config["n_time_steps"]
        
        super().__init__(epsilon, n_time_steps)

    def _reward(self, arm_position, goal_position, normalize_factor: float = 1.0):
        reach_bonus = 10 if self.is_near_target(arm_position, goal_position) else 0
        # TODO: make env easier
        reach_bonus = 0

        target_distance = np.linalg.norm(arm_position - goal_position) * normalize_factor
       
        return - target_distance + reach_bonus

    def done(self, arm_position, goal_position):
        time_limit_reached = super().done()
        is_near_target = self.is_near_target(arm_position, goal_position)

        if is_near_target or time_limit_reached:
            return True
        else:
            return False

    def is_near_target(self, arm_position, goal_position):
        if np.linalg.norm(arm_position - goal_position) <= self._epsilon:
            return True
        else:
            return False
