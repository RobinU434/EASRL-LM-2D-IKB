import copy
import logging
from argparse import ArgumentParser
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from algorithms.sac.sac import SAC
from envs.plane_robot_env import PlaneRobotEnv
from envs.task.imitation_learning import ImitationTask
from envs.task.reach_goal import ReachGoalTask
from main import extract_actor_config
from vae.inference import sample_target

matplotlib.rcParams["figure.dpi"] = 300


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("model_path", type=str, help="path to model checkpoint")
    # parser.add_argument("--arm", type=bool, default=False, help="plot arm positions")

    return parser


def load_config(config_path: str) -> dict:
    logging.info(f"load config at {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["config_path"] = config_path
    return config


def main(sac_config: dict, model_path: str):
    # path for loading potentially latent actor
    log_dir = "/".join(sac_config["config_path"].split("/")[:-1])
    actor_config = extract_actor_config(copy.deepcopy(sac_config), log_dir=log_dir)
    # TODO: load supervised checkpoint or
    if sac_config["task"] == ReachGoalTask.__name__:
        task = ReachGoalTask(config=sac_config)
    elif sac_config["task"] == ImitationTask.__name__:
        task = ImitationTask(config=sac_config)
    else:
        logging.error(f"No task with name: {sac_config['task']} available")
        task = None

    print(sac_config["n_joints"], " joints")

    env = PlaneRobotEnv(
        n_joints=sac_config["n_joints"],
        segment_length=sac_config["segment_length"],
        task=task,
        action_mode=sac_config["action_mode"],
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
        action_covariance_decay=sac_config["action_covariance_decay"],
        action_covariance_mode=sac_config["action_covariance_mode"],
        action_magnitude=sac_config["action_magnitude"],
        actor_config=actor_config,
    )

    sac.load_checkpoint(model_path)
    target_positions = sample_target(1).numpy() * sac_config["n_joints"]
    print(target_positions)
    # target_positions = np.array([[0, 1]])
    trajectories = sac.inference(target_positions)
    distances = np.linalg.norm(trajectories[0, :, -1, :] - target_positions, axis=1)

    fig, axs = plt.subplots(2, 1)
    axs[0].set_xlim([-sac_config["n_joints"], sac_config["n_joints"]])
    axs[0].set_ylim([-sac_config["n_joints"], sac_config["n_joints"]])
    axs[0].axis("equal")
    axs[0].add_patch(plt.Circle((0, 0), sac_config["n_joints"], fill=False))
    for trajectory in trajectories[0]:
        axs[0].plot(
            trajectory[:, 0], trajectory[:, 1], color="orange", marker=".", alpha=2 / 5
        )

    axs[0].plot(trajectories[0, :, -1, 0], trajectories[0, :, -1, 1])
    axs[0].plot(target_positions[0, 0], target_positions[0, 1], color="r", marker=".")

    axs[1].set_ylim([-0.5, max(distances) + 0.5])
    axs[1].grid()
    axs[1].plot(distances)

    print(distances.mean())

    fig.savefig("results/sac_inference.png")


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
