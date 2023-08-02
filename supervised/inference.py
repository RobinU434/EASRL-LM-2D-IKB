import json
from argparse import ArgumentParser

import matplotlib
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.pyplot import Circle
from progress.bar import Bar

from envs.common.sample_target import sample_target
from envs.robots.ccd import IK
from supervised.data import get_action_radius
from supervised.model import Regressor, build_model
from supervised.utils import forward_kinematics
from vae.utils.post_processing import PostProcessor
from supervised.train import build_state, run_greedy

matplotlib.rcParams["figure.dpi"] = 300


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "num_samples", type=int, help="how many points are sampled for the state"
    )
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument(
        "device",
        type=str,
        default="cpu",
        help="device on which the forward pass should happen",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="command just prints config and returns",
    )
    return parser


def load_config(checkpoint_folder: str) -> dict:
    with open(checkpoint_folder) as f:
        config = yaml.safe_load(f)

    return config


def load_model(config: dict, checkpoint_path: str) -> Regressor:
    # build basic model
    model = build_model(
        **config
    )

    checkpoint = torch.load(checkpoint_path)
    # load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def inference(model: Regressor, num_samples: int, device: str, config: dict):
    # sample target
    targets = np.repeat(
        np.expand_dims(sample_target(model.output_dim, 1), axis=0), num_samples, axis=0
    )
    print("target: ", targets[0])

    # sample start position
    start_positions = np.zeros((num_samples, 3))
    start_position = sample_target(model.output_dim, 1)
    start_position = np.expand_dims(start_position, axis=0)
    noise = np.random.normal(
        np.zeros(2), np.ones(2) * 0.05, (num_samples, 2)
    )  # 2D noise
    start_positions[:, 0:2] = start_position + noise

    # solve IK for start positions
    state_angles = np.zeros((num_samples, model.output_dim))
    link = np.ones(model.output_dim)
    bar = Bar("solve IK for state config", max=num_samples)
    for idx in range(num_samples):
        state_action, _, _, _ = IK(
            start_positions[idx], state_angles[idx].copy(), link, err_min=0.001
        )
        state_angles[idx] = np.deg2rad(state_action)  # convert to rad
        bar.next()
    bar.finish()

    # build state vector
    print("build state")
    state = np.concatenate([targets, start_positions[:, 0:2], state_angles], axis=1)

    # make batches
    print("run model")
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
    ax.add_patch(Circle((0, 0), model.output_dim, fill=False))
    if config["action_radius"] != 0:
        ax.add_patch(
            Circle(
                start_position[0],
                get_action_radius(config["action_radius"], model.output_dim),
                fill=False,
            )
        )
    ax.scatter(start_positions[:, 0], start_positions[:, 1], c="g", s=1)
    ax.scatter(arm_positions[:, -1, 0], arm_positions[:, -1, 1], c="r", s=1)
    ax.scatter(targets[0, 0], targets[0, 1], c="b", s=1)
    fig.savefig("results/supervised_inference.png")

    fig = plt.figure()
    ax = fig.add_subplot()
    if model.post_processor.enabled:
        # ax.set_xlim([model.post_processor.min_action, model.post_processor.max_action])
        bins = np.linspace(-1, 1, 200)
    else:
        bins = 50
    for idx, joint_actions in enumerate(actions.T):
        ax.hist(
            joint_actions,
            bins=bins,
            alpha=1 / np.sqrt(model.output_dim),
            label=str(idx),
        )
    fig.legend()
    fig.savefig("results/supervised_action_distribution.png")


def greedy_inference(
    model: Regressor, num_steps: int, device: str, plot_arms: bool = False
):
    print(f"perfom a maximum number of {num_steps}")
    # sample target
    target = torch.from_numpy(sample_target(model.output_dim, 1)).unsqueeze(dim=0)
    target = torch.tensor([[-5, 5]])
    print("target position: ", target.squeeze().tolist())

    # sample start config
    state_angles = torch.randn(model.output_dim) * 2 * np.pi
    state_angles = state_angles.unsqueeze(dim=0)
    # state_angles = torch.zeros_like(state_angles)
    current_position = forward_kinematics(state_angles)[:, -1]

    # build init state -> target position is not important because it will be replaced
    state = (
        torch.cat([torch.zeros(1, 2), current_position, state_angles], dim=1).to(device).float()
    )
    eps = 0.1  # parameter which controls the loop break condition
    metrics = run_greedy(
        model=model,
        target_pos=target.to(device),
        init_state=state,
        step_length=model.action_radius,
        max_steps=num_steps,
        eps=eps,
        device=device,
    )

    # plot results
    fig, axs = plt.subplots(2, 1)
    axs[0].add_patch(Circle((0, 0), model.output_dim, fill=False))
    axs[0].scatter(target[0, 0], target[0, 1], c="g", s=2, label="target", zorder=1)
    axs[0].scatter(
        metrics["trajectory"][0, 0],
        metrics["trajectory"][0, 1],
        c="b",
        s=2,
        label="start",
        zorder=1,
    )
    axs[0].plot(metrics["trajectory"][:, 0], metrics["trajectory"][:, 1], c="r")

    if plot_arms:
        for position_sequence in metrics["arms"][0 :: model.output_dim]:
            axs[0].plot(
                position_sequence[:, 0],
                position_sequence[:, 1],
                color="orange",
                marker=".",
                alpha=2 / 5,
            )

    axs[1].grid()
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("distance to target")
    dist_to_target = np.linalg.norm(metrics["trajectory"] - target.numpy(), axis=1)
    diff = np.diff(metrics["trajectory"], axis=0)
    step_sizes = np.linalg.norm(diff, axis=1)
    axs[1].plot(step_sizes)

    axs[0].legend()
    
    print(f"needed: {len(dist_to_target)} steps")
    print(f"distance: min={min(dist_to_target):.4f}, mean={np.mean(dist_to_target):.4f}")

    print(f"average step length: {step_sizes.mean():.4f} trained on: {model.action_radius}")

    fig.savefig("results/supervised_greedy_inference.png")

    return target[0], dist_to_target


def multiple_greedy_runs():
    pass


if __name__ == "__main__":
    parser = setup_parser(ArgumentParser())
    args = parser.parse_args()

    # load config
    config_path = "/".join(args.checkpoint.split("/")[:-1]) + "/config.yaml"
    config = load_config(config_path)

    if args.print_config:
        print(json.dumps(config, sort_keys=True, indent=4))
        exit()

    model = load_model(config, args.checkpoint).to(args.device)

    # inference(model, args.num_samples, args.device, config)
    greedy_inference(model, args.num_samples, args.device, False)
