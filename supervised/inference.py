from matplotlib import pyplot as plt
import numpy as np
import yaml
import torch

from argparse import ArgumentParser
from progress.bar import Bar

from envs.common.sample_target import sample_target
from envs.robots.ccd import IK
from supervised.model import Regressor, build_model
from supervised.utils import forward_kinematics


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "num_samples",
        type=int,
        help="how many points are sampled for the state"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="path to checkpoint"
    )
    parser.add_argument(
        "device",
        type=str,
        default="cpu",
        help="device on which the forward pass should happen"
    )
    return parser


def load_config(checkpoint_folder: str) -> dict:
    with open(checkpoint_folder) as f:
        config = yaml.safe_load(f)

    return config


def load_model(config: dict, checkpoint_path: str) -> Regressor:
    # build basic model
    model = build_model(config["feature_source"], config["num_joints"], config["learning_rate"])

    checkpoint = torch.load(checkpoint_path)
    # load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def inference(model: Regressor, num_samples: int, device: str, config: dict):
    # sample target
    targets = np.repeat(np.expand_dims(sample_target(model.output_dim, 1), axis=0), num_samples, axis=0)
    print("target: ", targets[0])

    # sample start position
    start_positions = np.zeros((num_samples, 3))
    start_position = sample_target(model.output_dim, 1)
    start_position = np.expand_dims(start_position, axis=0)
    noise = np.random.normal(np.zeros(model.output_dim), np.ones(model.output_dim) * 0.05, (num_samples, model.output_dim))
    start_positions[:, 0:2] = start_position + noise

    # solve IK for start positions
    state_angles = np.zeros((num_samples, model.output_dim))
    link = np.ones(model.output_dim)
    bar = Bar("solve IK", max=num_samples)
    for idx in range(num_samples):
        state_action, _, _, _ = IK(start_positions[idx], state_angles[idx].copy(), link, err_min=0.001)
        state_angles [idx] = state_action / 180 * np.pi  # convert to rad
        bar.next()
    bar.finish()

    # build state vector
    state = np.concatenate([targets, start_positions[:, 0:2], state_angles], axis=1)

    # make batches
    split_idx = np.arange(0, num_samples, config["batch_size"])[1:]
    batches = np.split(state, split_idx)
    actions = []
    # bar = Bar("network forward pass", max=len(split_idx))
    for batch in batches:
        batch = torch.tensor(batch).to(device).type(torch.float32)
        # model forward pass
        action = model(batch)
        actions.append(action)
        # bar.next()
    # bar.finish()
    
    actions = torch.stack(actions).detach().squeeze().numpy()
    target_actions = state_angles + actions
    arm_positions = forward_kinematics(torch.tensor(target_actions))

    # plot 
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.add_patch(plt.Circle((0, 0), 2, fill=False))
    if config["action_radius"] != 0:
        ax.add_patch(plt.Circle(start_position[0], config["action_radius"], fill=False))
    ax.scatter(start_positions[:, 0], start_positions[:, 1], c="g", s=1)
    ax.scatter(arm_positions[:, -1, 0], arm_positions[:, -1, 1], c="r", s=1)
    ax.scatter(targets[0, 0], targets[0, 1], c="b", s=1)

    fig.savefig("results/supervised_inference.png")


if __name__ == "__main__":
    parser = setup_parser(ArgumentParser())
    args = parser.parse_args()

    # load config
    config_path = "/".join(args.checkpoint.split("/")[:-1]) + "/config.yaml"
    config = load_config(config_path)
    
    model = load_model(config, args.checkpoint).to(args.device)

    inference(model, args.num_samples, args.device, config)
    