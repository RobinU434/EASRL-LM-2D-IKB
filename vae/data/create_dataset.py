import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame
from progress.bar import Bar
from argparse import ArgumentParser

from envs.robots.robot_arm import RobotArm
from envs.common.sample_target import sample_target


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "n_joints",
        type=int,
        help="number of joints for the robot arm")
    parser.add_argument(
        "n",
        type=int,
        help="number of data points to generate")
    parser.add_argument(
        "mode",
        type=str,
        choices=["IK_constant_start", "IK_random_start", "relative_uniform", "relative_tanh"]
    )
    
    return parser


def check_positive(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_value


def sample_start(n_joints: int, n: int, start_mode: str = "constant_start"):
    if start_mode == "constant_start":
        return np.zeros((n, n_joints))
    elif start_mode == "random_start":
        return np.random.uniform(0, 2 * np.pi, (n, n_joints))


def create_IK_dataset(n_joints: int , n: int = 10_000, start_mode: str = "constant start"):
    arm = RobotArm(n_joints)

    # array to write in the calculated method
    actions = np.zeros((n, n_joints))

    # expand dimensions because the IK algorithm needs a this dimension
    targets = np.zeros((n, 3))
    
    # set random seed to the current time
    time_stamp = int(time.time()) 
    np.random.seed(time_stamp)
    # assumption: every segment has length one
    targets[:, 0:2] = sample_target(radius=n_joints, n_points=n)
    start_angles = sample_start(n_joints, n, start_mode)
    
    bar = Bar('CCD on targets', max=n)
    for idx in range(n):
        # set start  position in arm
        target = targets[idx]
        
        arm.set(start_angles[idx])
        arm.IK(target)

        actions[idx] = arm.angles
        targets[idx, 0:2] = arm.end_position
        
        arm.reset()
        
        if idx % 10 == 0:
            bar.next(10)

    bar.finish()

    print("Write actions and corresponding positions into csv file")
    df_actions = DataFrame(actions)
    df_targets = DataFrame(targets)

    df_actions.to_csv(f"datasets/{n_joints}/actions_{time_stamp}.csv")
    df_targets.to_csv(f"datasets/{n_joints}/targets_{time_stamp}.csv")


def create_relative_dataset(n_joints: int, n: int = 10_000, mode: str = "relative_uniform"):
    time_stamp = int(time.time()) 
    np.random.seed(time_stamp)
    
    size = (n, n_joints)

    if mode == "relative_uniform":
        data = np.random.uniform(-1, 1, size)

    elif mode == "relative_tanh":
        data = np.tanh(np.random.normal(0, 1, size))
    
    df = DataFrame(data)
    df.to_csv(f"datasets/{n_joints}/actions_{mode}_{time_stamp}.csv")


def create_dataset(n_joints: int, n: int, mode: str):
    if "IK" in mode:
        return create_IK_dataset(n_joints, n, mode)
    elif "relative" in mode:
        return create_relative_dataset(n_joints, n, mode)


if __name__ == "__main__":
    parser = setup_parser(ArgumentParser())
    args = parser.parse_args()
    arg_dict = vars(args)
    create_dataset(**arg_dict)
