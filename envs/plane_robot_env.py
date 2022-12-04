from typing import Any, Dict, Tuple
from gym import spaces
import gym
import numpy as np

from PIL import Image, ImageDraw

from envs.robots.robot_arm import RobotArm
from envs.task.base_task import BaseTask


class PlaneRobotEnv(gym.Env):
    def __init__(
        self, 
        n_joints: int = 1,
        segment_lenght: float = 1,
        constraints: np.array = None,
        task: BaseTask = None,
        ) -> None:

        self._robot_arm = RobotArm(n_joints, segment_lenght)
        # init angles and other 
        self.reset()
        
        if constraints is None:
            # [joint_idx][0] lower constraint
            # [joint_idx][1] upper constraint
            self._constraints = np.zeros((self._robot_arm.n_joints, 2))
        self._constraints = constraints

        self._goal_postion = self.get_goal_position(self._robot_arm.arm_length)

        self._task = task
        self.set_action_space()
        self.set_observation_space()

    def set_action_space(self) -> None:
        """
        an action is either +1 degree, -1 degree or 0 degrees of rotation per joint
        Therefor is one action a tensor with the length equal to the number of joints. 
        """
        self.action_space = spaces.Box(-1, 1, (self._robot_arm.n_joints, 1))

    def set_observation_space(self) -> None:
        """
        observation space is a 4 dimensional tensor.
            - first two dimensions: the 2D postion of the goal position
            - second two dimensions: the 2D postion of the robot arm tip
        """
        self.observation_space = spaces.Box(-self._robot_arm.arm_length, self._robot_arm.arm_length, (4, 1))

    def get_goal_position(self, radius: float):
        """
        sample goal postion in a circular shape around the origin
        """
        # angle to sample from
        theta = np.random.uniform(0, 2 * np.pi)

        return radius * np.array([np.cos(theta), np.sin(theta)])

    def apply_action(self, action: np.array):
        """apply +-1 action on robot arm
        TODO: check for constraints

        Args:
            action (np.array): +-1 action
        """
        new_angles = self._robot_arm.angles + action

        self._robot_arm.set(new_angles)

    def _observe(self):
        return np.concatenate((self._goal_postion, self._robot_arm.end_postion))

    def reset(self) -> Any:
        self._robot_arm.reset()

        self._goal_postion = self.get_goal_position(self._robot_arm.arm_length)

        return self._observe()

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        """_summary_

        Args:
            action (np.array): 

        Returns:
            Tuple[np.array, float, bool, Dict[str, Any]]: _description_
        """

        self.apply_action(action)

        reward = self._task.reward(self._robot_arm.end_postion, self._goal_postion)

        obs = self._observe()

        done = self._task.done()

        return obs, reward, done
        
    def render(self, render_size: Tuple[int, int] = (500, 300), path: str = "test.png"):
        # origin is in the upper left corner
        img = Image.new("RGB", render_size, (256, 256, 256))
        draw = ImageDraw.Draw(img)

        # set origin to the center
        origin = (render_size[0] / 2, render_size[1] / 2)
        scale_factor = 20
        
        self.draw_goal(draw, origin + self._goal_postion * scale_factor)
        self.draw_segments(draw, origin, scale_factor)
        self.draw_joints(draw, origin, scale_factor)

        img.save(path)

    def draw_goal(self, draw, origin: np.array = np.zeros((2)), radius=4):
        x, y = origin
        # distances to origin
        # (left, upper, right, lower)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0), outline=(0, 0, 0))

    def draw_joints(self, draw, origin, scale_factor: float = 20):
         for postion in self._robot_arm._postions[:-1].copy():
            # scale
            postion *= scale_factor
            
            # move the segment
            postion += origin

            self.draw_joint(draw, postion)

    def draw_segments(self, draw, origin, scale_factor: float = 20):
        for idx in range(self._robot_arm.n_joints):
            start = self._robot_arm._postions[idx].copy()
            end =  self._robot_arm._postions[idx + 1].copy()

            # scale 
            start *= scale_factor
            end *= scale_factor

            # move the segment
            start += origin
            end += origin
        
            self.draw_segment(draw, start, end)


    @staticmethod
    def draw_joint(draw, origin: Tuple[float, float] = (0, 0), radius=4):
        """draws joint from robot arm as a grey circle

        Args:
            draw (_type_): _description_
            origin (tuple, optional): origin of joint. (x, y) Defaults to (0, 0).
            radius (int, optional): radius of joint. Defaults to 5.
        """
        x, y = origin
        # distances to origin
        # (left, upper, right, lower)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(100, 100, 100), outline=(0, 0, 0))

    @staticmethod
    def draw_segment(draw, start: Tuple[float, float] = (0, 0), end: Tuple[float, float]  = (1, 1), width:float = 3):
        """draw segment as yellow line

        Args:
            draw (_type_): draw object to draw in
            start (Tuple[float, float], optional): start postion from line: (x, y). Defaults to (0, 0).
            end (Tuple[float, float], optional): end postion from line: (x, y). Defaults to (1, 1).
            width (float, optional): width from line. Defaults to 2.
        """
        coord = [start[0], start[1], end[0], end[1]]
        draw.line(coord, fill=(255, 255, 0), width=width)

        

if __name__ == "__main__":
    env = PlaneRobotEnv(4, 1)

    env.render()