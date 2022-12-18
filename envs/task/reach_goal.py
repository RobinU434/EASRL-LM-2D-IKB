import numpy as np
from envs.task.base_task import BaseTask


class ReachGoalTask(BaseTask):
    def __init__(self, epsilon: float = 0.01, n_time_steps: int = 200) -> None:
        super().__init__(epsilon, n_time_steps)

    def reward(self, arm_postion, goal_postion, normalize_factor: float = 1.0):
        super().update()
        reach_bonus = 10 if self.is_near_target(arm_postion, goal_postion) else 0

        target_distance = np.linalg.norm(arm_postion - goal_postion) * normalize_factor

        return - target_distance + reach_bonus

    def done(self, arm_postion, goal_postion):
        time_limit_reached = super().done()
        is_near_target = self.is_near_target(arm_postion, goal_postion)

        if is_near_target or time_limit_reached:
            return True
        else:
            return False

    def is_near_target(self, arm_postion, goal_postion):
        if np.linalg.norm(arm_postion - goal_postion) <= self._epsilon:
            return True
        else:
            return False
