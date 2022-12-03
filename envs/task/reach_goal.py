import numpy as np
from envs.task.base_task import BaseTask


class ReachGoalTask(BaseTask):
    def __init__(self, epsilon) -> None:
        super().__init__(epsilon)

    def reward(self, arm_postion, goal_postion):
        reach_bonus = 10 if self.done(arm_postion, goal_postion) else 0
        
        return np.linalg.norm(arm_postion - goal_postion) + reach_bonus

    def done(self, arm_postion, goal_postion):
        if np.linalg.norm(arm_postion - goal_postion) <= self._epsilon:
            return True
        else:
            return False