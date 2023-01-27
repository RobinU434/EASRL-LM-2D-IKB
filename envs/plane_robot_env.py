import gym
import torch
import numpy as np

from gym import spaces
from PIL import Image, ImageDraw
from typing import Any, Dict, Tuple
from envs.common.sample_target import sample_target

from envs.robots.robot_arm import RobotArm
from envs.task.base_task import BaseTask


class PlaneRobotEnv(gym.Env):
    def __init__(
        self, 
        n_joints: int = 1,
        segment_length: float = 1,
        constraints: np.array = None,
        task: BaseTask = None,
        discrete_mode: bool = False,
        ) -> None:

        self._task = task
        self._robot_arm: RobotArm = RobotArm(n_joints, segment_length)
        # init angles and other 
        self.reset()
        
        if constraints is None:
            # [joint_idx][0] lower constraint
            # [joint_idx][1] upper constraint
            self._constraints = np.zeros((self._robot_arm.n_joints, 2))
        self._constraints = constraints

        # discrete mode = True is for discrete actions (+-1 / +-0 degrees)
        # discrete mode = False is for continuous actions
        self._discrete_mode = discrete_mode

        self._target_position = self.get_target_position(self._robot_arm.arm_length)

        self.set_action_space()
        self.set_observation_space()

        self._step_counter = 0

    def set_action_space(self) -> None:
        """
        an action is either +1 degree, -1 degree or 0 degrees of rotation per joint
        Therefor is one action a tensor with the length equal to the number of joints. 
        """
        if self._discrete_mode:
            self.action_space = spaces.Box(-1, 1, (self._robot_arm.n_joints, 1))
        else:
            self.action_space = spaces.Box(0, 2 * np.pi, (self._robot_arm.n_joints, 1))

    def set_observation_space(self) -> None:
        """
        observation space is a 4 dimensional tensor.
            - first two dimensions: the 2D position of the goal position
            - second two dimensions: the 2D position of the robot arm tip
        """
        self.observation_space = spaces.Box(
            -self._robot_arm.arm_length,
            self._robot_arm.arm_length, 
            (2 + 2 + self._robot_arm.n_joints, 1)
            )

    @staticmethod
    def get_target_position(radius: float):
        return sample_target(radius)

    def _apply_action(self, action: np.array):
        """apply +-1 action on robot arm
        TODO: check for constraints

        Args:
            action (np.array): +-1 action
        """
        if type(action) == torch.Tensor:
            # detach if the action tensor requires grad = True
            action = action.detach().numpy()

        # with discrete actions the action is -1 +1 or 0 which will be added on top of the current angle
        # with continuous actions the action itself is the delta angle which will be also added on top of the current angle
        action = np.squeeze(action)
        action = self._robot_arm.angles + action

        self._robot_arm.set(action)

    def _observe(self, normalize: bool = True):
        if normalize:
            # normalize observations 
            target_position = self._target_position / self._robot_arm.arm_length
            arm_end_position = self._robot_arm.end_position / self._robot_arm.arm_length
        else:
            target_position = self._target_position
            arm_end_position = self._robot_arm.end_position

        obs = np.concatenate(
            (
                target_position, 
                arm_end_position,
                self._robot_arm.angles)
            )

        return obs

    def reset(self) -> Any:
        self._robot_arm.reset()
        self._task.reset()

        self._step_counter = 0

        self._target_position = self.get_target_position(self._robot_arm.arm_length)

        return self._observe()

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        """_summary_

        Args:
            action (np.array): 

        Returns:
            Tuple[np.array, float, bool, Dict[str, Any]]: _description_
        """
        self._step_counter += 1
        self._apply_action(action)
        
        # the target could be two times the arm length away from the end position
        normalize_factor = 1 / (2 * self._robot_arm.arm_length) 
        reward = self._task.reward(self._robot_arm.end_position, self._target_position, normalize_factor) 

        obs = self._observe()

        done = self._task.done(self._robot_arm.end_position, self._target_position, )

        return obs, reward, done, {}
        
    def render(self, path: str = "test.png"):
        render_size = (int(self._robot_arm.arm_length * 1.1), int(self._robot_arm.arm_length * 1.1))
        # origin is in the upper left corner
        img = Image.new("RGB", render_size, (256, 256, 256))
        draw = ImageDraw.Draw(img)

        # set origin to the center
        origin = (render_size[0] / 2, render_size[1] / 2)
        scale_factor = 20
        
        self._draw_goal(draw, origin + self._target_position * scale_factor)
        self._draw_segments(draw, origin, scale_factor)
        self._draw_joints(draw, origin, scale_factor)

        img.save(path)

    def _draw_goal(self, draw, origin: np.array = np.zeros((2)), radius=4):
        x, y = origin
        # distances to origin
        # (left, upper, right, lower)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0), outline=(0, 0, 0))

    def _draw_joints(self, draw, origin, scale_factor: float = 20):
         for position in self._robot_arm._positions[:-1].copy():
            # scale
            position *= scale_factor
            
            # move the segment
            position += origin

            self._draw_joint(draw, position)

    def _draw_segments(self, draw, origin, scale_factor: float = 20):
        for idx in range(self._robot_arm.n_joints):
            start = self._robot_arm._positions[idx].copy()
            end =  self._robot_arm._positions[idx + 1].copy()

            # scale 
            start *= scale_factor
            end *= scale_factor

            # move the segment
            start += origin
            end += origin
        
            self.draw_segment(draw, start, end)


    @staticmethod
    def _draw_joint(draw, origin: Tuple[float, float] = (0, 0), radius=4):
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
            start (Tuple[float, float], optional): start position from line: (x, y). Defaults to (0, 0).
            end (Tuple[float, float], optional): end position from line: (x, y). Defaults to (1, 1).
            width (float, optional): width from line. Defaults to 2.
        """
        coord = [start[0], start[1], end[0], end[1]]
        draw.line(coord, fill=(255, 255, 0), width=width)
        
    @property
    def num_steps(self):
        return self._step_counter
        

if __name__ == "__main__":
    env = PlaneRobotEnv(4, 1)

    env.render()