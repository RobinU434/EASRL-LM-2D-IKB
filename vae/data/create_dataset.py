import time
import argparse
import numpy as np

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
    return parser


def check_positive(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_value


def create_dataset(n_joints: int , n: int = 10_000):
    arm = RobotArm(n_joints)

    actions = np.zeros((n, n_joints))

    # expand dimensions because the IK algorithm needs a this dimension
    targets = np.zeros((n, 3))
    # set random seed to the current time
    time_stamp = int(time.time()) 
    np.random.seed(time_stamp)
    # assumption: every segment has length one
    targets[:, 0:2] = sample_target(radius=n_joints, n_points=n)
    
    bar = Bar('CCD on targets', max=n)
    for idx, target in enumerate(targets):
        arm.IK(target)
        actions[idx] = arm.angles
        arm.reset()

        if idx % 10 == 0:
            bar.next(10)

    bar.finish()

    print("Write actions and corresponding positions into csv file")
    df_actions = DataFrame(actions)
    df_targets = DataFrame(targets)

    df_actions.to_csv(f"datasets/{n_joints}/actions_{time_stamp}.csv")
    df_targets.to_csv(f"datasets/{n_joints}/targets_{time_stamp}.csv")


if __name__ == "__main__":
    parser = setup_parser(ArgumentParser())
    args = parser.parse_args()
    arg_dict = vars(args)
    create_dataset(**arg_dict)
