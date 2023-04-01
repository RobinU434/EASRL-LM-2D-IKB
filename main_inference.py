import yaml
import numpy as np

from typing import List
from argparse import ArgumentParser

from algorithms.sac.sac import SAC
from envs.plane_robot_env import PlaneRobotEnv
from envs.task.reach_goal import ReachGoalTask
from envs.task.imitation_learning import ImitationTask


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("model_path", type=str, help="path to model checkpoint")

    return parser


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main(sac_config: dict, model_path: str):
    if sac_config["task"] == ReachGoalTask.__name__:
        task = ReachGoalTask(config = sac_config)
    elif sac_config["task"] == ImitationTask.__name__:
        task = ImitationTask(config=sac_config)
    print(sac_config["n_joints"], " joints")

    env = PlaneRobotEnv(
        n_joints=sac_config["n_joints"],
        segment_length=sac_config["segment_length"],
        task=task, 
        action_mode=sac_config["action_mode"]
        )
    
    sac = SAC(
            env,
            logging_writer=None,
            fs_logger=None,
            lr_pi=sac_config["lr_pi"],
            lr_q=sac_config["lr_q"],
            init_alpha=sac_config["init_alpha"],
            gamma=sac_config["gamma"],
            batch_size=sac_config["batch_size"],
            buffer_limit=sac_config["buffer_limit"],
            start_buffer_size=sac_config["start_buffer_size"],
            train_iterations=sac_config["train_iterations"],
            tau=sac_config["tau"],
            target_entropy=sac_config["target_entropy"],
            lr_alpha=sac_config["lr_alpha"],
            action_covariance_decay = sac_config["action_covariance_decay"],
            action_covariance_mode = sac_config["action_covariance_mode"],
            action_magnitude=sac_config["action_magnitude"],
            )
    
    sac.load_checkpoint(model_path)
    target_positions = np.array([[1, 0]])
    sac.inference(target_positions)



if __name__ == "__main__":
    parser = setup_parser(ArgumentParser())

    args = parser.parse_args()

    # extract config path
    config_path = "/".join(args.model_path.split("/")[:-1]) + "/config.yaml"
    sac_config = load_config(config_path)

    # load common target positions

    args_dict = vars(args)
    args_dict["sac_config"] = sac_config

    main(**args_dict)    