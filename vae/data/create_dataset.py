import logging
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame
from progress.bar import Bar
from argparse import ArgumentParser

from envs.robots.robot_arm import RobotArm
from envs.common.sample_target import sample_target

MODE_LIST = ["IK_constant_start", "IK_random_start", "noise_uniform", "noise_tanh"]

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
        choices=MODE_LIST,
        help="mode for what kind of data in the dataset"
    )
    parser.add_argument(
        "entity",
        type=str,
        choices=["train", "val", "test"],
        help="What kind of data set you want to create -> train data, ...."
    )


    return parser


def check_positive(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_value


def sample_start(n_joints: int, n: int, start_mode: str = "constant_start"):
    if start_mode == "IK_constant_start":
        return np.zeros((n, n_joints))
    elif start_mode == "IK_random_start":
        return np.random.uniform(0, 2 * np.pi, (n, n_joints))
    else:
        logging.error(f"you have to chose from {MODE_LIST} instead you chose: {start_mode}")

def create_IK_dataset(n_joints: int , n: int = 10_000, start_mode: str = "constant start", entity: str = "train"):
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
    state = np.zeros((n, 4 + n_joints))
    
    bar = Bar('CCD on targets', max=n)
    for idx in range(n):
        # set start  position in arm
        target = targets[idx]
        
        arm.set(start_angles[idx])
        start_end_position = arm.end_position.copy()
        arm.IK(target)

        actions[idx] = arm.angles - start_angles[idx]
        targets[idx, 0:2] = arm.end_position
        
        state[idx, :] = np.concatenate([arm.end_position, start_end_position, start_angles[idx]])

        arm.reset()
        
        if idx % 10 == 0:
            bar.next(10)

    bar.finish()

    print("Write actions and corresponding positions into csv file")
    df_actions = DataFrame(actions)
    df_targets = DataFrame(targets)
    df_state = DataFrame(state)

    df_actions.to_csv(f"datasets/{n_joints}/{entity}/actions_{start_mode}.csv")
    df_targets.to_csv(f"datasets/{n_joints}/{entity}/targets_{start_mode}.csv")
    df_state.to_csv(f"datasets/{n_joints}/{entity}/state_{start_mode}.csv")
    

def create_relative_dataset(n_joints: int, n: int = 10_000, mode: str = "relative_uniform", entity: str = "train"):
    time_stamp = int(time.time()) 
    np.random.seed(time_stamp)
    
    size = (n, n_joints)

    if mode == "noise_uniform":
        data = np.random.uniform(-1, 1, size)

    elif mode == "noise_tanh":
        data = np.tanh(np.random.normal(0, 1, size))
    
    df = DataFrame(data)
    df.to_csv(f"datasets/{n_joints}/{entity}/actions_{mode}.csv")


def create_dataset(n_joints: int, n: int, mode: str, entity: str):
    if "IK" in mode:
        return create_IK_dataset(n_joints, n, mode, entity)
    elif "noise" in mode:
        return create_relative_dataset(n_joints, n, mode, entity)


if __name__ == "__main__":
    parser = setup_parser(ArgumentParser())
    args = parser.parse_args()
    arg_dict = vars(args)
    create_dataset(**arg_dict)
