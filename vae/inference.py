import logging
import math
import numpy as np
import yaml
import torch
import argparse
import matplotlib.pyplot as plt

from vae.model.vae import VariationalAutoencoder


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "sample_size",
        type=int,
        help="how many points should be sampled from the latent space"
    )
    
    parser.add_argument(
        "model_path",
        help="path to model checkpoint"
        )
    
    parser.add_argument(
        "-fp", "--fixed_position",
        action="store_true",
        help="if true -> sample from a fixed position, if false -> sample from different target positions"
    )

    parser.add_argument(
        "-g", "--greedy",
        action="store_true",
        help="if you set this argument the inference will try to reach a target position choosing greedily from VAE latent space"
    )
    
    return parser


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def load_model(path: str, config: dict) -> VariationalAutoencoder:
    if config["dataset"] == "action":
        input_dim = config["num_joints"]
        latent_dim = config["latent_dim"]
        output_dim  = config["num_joints"]
        enhance_dim = 0
    if config["dataset"] == "action_target_v1":
        input_dim = config["num_joints"]
        latent_dim = config["latent_dim"]
        output_dim  = config["num_joints"]
        enhance_dim = 2
    if config["dataset"] == "action_target_v2":
        input_dim = config["num_joints"] + 2
        latent_dim = config["latent_dim"]
        output_dim  = config["num_joints"]
        enhance_dim = 0
    if config["dataset"] == "conditional_action_target":
        if config["conditional_info"] == "state":
            conditional_info_dim = config["num_joints"]  + 4
        elif config["conditional_info"] == "targets":
            conditional_info_dim = 2

        input_dim = config["num_joints"] + conditional_info_dim
        latent_dim = config["latent_dim"]
        output_dim  = config["num_joints"]
        enhance_dim = conditional_info_dim
    
    model = VariationalAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        learning_rate=config["learning_rate"],
        logger=None,
        conditional_info_dim=enhance_dim
    )

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def sample_target(n: int) -> torch.tensor:
    theta = torch.rand(n) * 2 * torch.pi        
    radius = torch.rand(n)

    x = torch.cos(theta)
    y = torch.sin(theta)

    coords = torch.stack([x, y]) * radius
    coords = coords.T

    return coords


def sample_latent(n: int, mu: torch.tensor, std: torch.tensor):
    normal_distr = torch.distributions.Normal(0, 1)

    sampled = mu + std * normal_distr.sample((n, len(mu)))

    return sampled


def forward_kinematics(angles: torch.tensor) -> torch.tensor:
    """_summary_

    Args:
        angles (np.array): shape (num_arms, num_joints)

    Returns:
        torch.tensor: out shape: (num_arms, num_joints + 1, 2)
    """
    num_arms, num_joints = angles.shape
    positions = torch.zeros((num_arms, num_joints + 1, 2))

    for idx in range(num_joints):
        origin = positions[:, idx]

        # new position
        new_x = torch.cos(angles[:, idx])
        new_y = torch.sin(angles[:, idx])
        new_pos = torch.stack([new_x, new_y]).T
        
        # translate position
        new_pos += origin

        positions[:, idx + 1] = new_pos

    return positions


def greedy_inference(model: VariationalAutoencoder, sample_size: int, plot: bool = False):
    n_epochs = 200
    target = sample_target(1) * model.output_dim
    print(target[0])
    best_angles = torch.zeros((sample_size, model.output_dim))
    end_positions = forward_kinematics(best_angles)[:, -1, :]
    state = torch.cat([target.repeat((sample_size, 1)), end_positions, best_angles], dim=1)
    
    pos_tensor = torch.empty((n_epochs + 1, 2))
    pos_tensor[0] = end_positions[0, :]
    dist_tensor = torch.tensor([])

    runtime = 199
    for epoch_idx in range(n_epochs):
        latent_sample = sample_latent(sample_size, torch.zeros(model.latent_dim), torch.ones(model.latent_dim))
        latent_input = torch.cat([latent_sample, state], dim=1)
        actions = model.decoder(latent_input).detach()

        # find minium action
        end_positions = forward_kinematics(actions)[:, -1, :]
        dist = torch.sqrt(torch.sum(torch.float_power(target - end_positions, 2), dim=1))
        min_dist = torch.min(dist)
        dist_tensor = torch.cat([dist_tensor, torch.tensor([min_dist])])
        if min_dist <= 0.1:
            runtime = epoch_idx
            break

        # build new state
        best_end_position = end_positions[torch.argmin(dist)]
        pos_tensor[epoch_idx + 1] = best_end_position
        best_end_position = torch.unsqueeze(best_end_position, dim=0)
        best_angles = actions[torch.argmin(dist)]
        best_angles = torch.unsqueeze(best_angles, dim=0)
        state = torch.cat([target, best_end_position, best_angles], dim=1).repeat((sample_size, 1))
    
    # plot
    if plot:
        fig, axs = plt.subplots(2, 1)
        axs[0].set_xlim([-model.output_dim, model.output_dim])
        axs[0].set_ylim([-model.output_dim, model.output_dim])
        axs[0].plot(pos_tensor[:, 0], pos_tensor[:, 1])
        axs[0].plot(target[:, 0], target[:, 1], marker=".", color="g")

        axs[1].plot(dist_tensor)

        fig.savefig("results/vae_greedy_inference.png")
        plt.close(fig)

    return target[0], runtime


def multiple_greedy_runs(model: VariationalAutoencoder, sample_size: int, num_runs: int = 100):
    runtimes = torch.tensor([])
    targets = torch.empty((num_runs, 2))
    for k in range(num_runs):
        print(k)
        target, runtime = greedy_inference(model, sample_size, plot=False)
        runtimes = torch.cat([runtimes, torch.tensor([runtime])])
        targets[k] = target

    print(runtimes.mean(), runtimes.std())
    fig = plt.figure()
    axs = fig.add_subplot()
    axs.hist(runtimes)
    fig.savefig("results/vae_greedy_time.png")
    plt.close(fig)

    fig = plt.figure()
    axs = fig.add_subplot()
    distance = torch.sqrt(torch.sum(torch.float_power(targets, 2), dim=1))
    axs.scatter(runtimes, distance)
    fig.savefig("results/distance_vs_runtime.png")


def absolute_inference(model: VariationalAutoencoder, fixed_position: bool, sample_size: int):
    latent_sample = sample_latent(sample_size, torch.zeros(model.latent_dim), torch.ones(model.latent_dim))
    
    if model.conditional_info_dim == 2:
        if fixed_position:
            target = sample_target(1)
            target = target.repeat((sample_size, 1))
        else:
            target = sample_target(sample_size)
    
        target *= model.output_dim
        concat_sample = torch.cat([latent_sample, target], dim=1)

    if model.conditional_info_dim > 2:
        if fixed_position:
            target = sample_target(1)
            target = target.repeat((sample_size, 1))
            state = torch.zeros((sample_size, model.output_dim))
            current_position = torch.tensor([1, 0]).repeat((sample_size, 1))
            current_position = current_position
        else:
            raise NotImplementedError
        target *= model.output_dim
        concat_sample = torch.cat([latent_sample, target, current_position, state], dim=1)
    
        print("target position: ", target[0].tolist())

    print("forward pass")
    actions = model.decoder(concat_sample).detach()
    positions = forward_kinematics((actions + 1)  * torch.pi)
    print("done")

    print("mean distance to target: ", torch.sqrt(torch.float_power(target - positions[:, -1], 2).sum(dim=1)).mean().item())

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim([-model.output_dim, model.output_dim])
    ax.set_ylim([-model.output_dim, model.output_dim])

    # plot arms
    print("plot arms")
    for position_sequence in positions:
        ax.plot(position_sequence[:, 0], position_sequence[:, 1], color="k", marker=".", alpha=1/math.sqrt(len(positions)))
    print("done")
    
    # plot end positions
    print("scatter end positions")
    ax.scatter(positions[:, -1, 0], positions[:, -1, 1], color="r", s=1) 
    print("done")

    # plot target position
    print("plot target postion")
    ax.scatter(target[0, 0], target[0, 1], color="g", s=1, zorder=1)
    print("done")

    fig.savefig("results/inference.png")

    if model.latent_dim == 1:
        fig = plt.figure()
        axs = fig.subplots(2, 1)
        axs[0].scatter(latent_sample, positions[:, -1, 0], s=1)
        x_target = torch.ones_like(latent_sample) * target[0, 0]
        axs[0].plot(latent_sample, x_target, color="orange")
        axs[1].scatter(latent_sample, positions[:, -1, 1], s=1)
        y_target = torch.ones_like(latent_sample) * target[0, 1]
        axs[1].plot(latent_sample, y_target, color="orange")
        
        fig.savefig("results/invariance.png")


def relative_inference(model: VariationalAutoencoder, sample_size: int):
    latent_sample = sample_latent(sample_size, torch.zeros(model.latent_dim), 10 * torch.ones(model.latent_dim))

    actions = model.decoder(latent_sample).detach()

    fig, axs = plt.subplots(model.output_dim, model.latent_dim)
    
    if model.latent_dim > 1:
        for k in range(model.latent_dim):
            for i in range(model.output_dim):
                axs[i, k].scatter(latent_sample[:, k], actions[:, i])
    else:
        for i in range(model.output_dim):
            axs[i].scatter(latent_sample, actions[:, i])

    plt.show()


def inference(model: VariationalAutoencoder, fixed_position: bool, sample_size: int, config: dict, greedy: bool = False):
    if greedy:
        # multiple_greedy_runs(model, sample_size, num_runs=100)
        greedy_inference(model, sample_size, plot=True)
    elif config["dataset_mode"] in ["relative_uniform", "relative_tanh"]:
        relative_inference(model, sample_size)
    elif config["dataset_mode"] in ["IK_const_start", "IK_random_start"] :
        absolute_inference(model, fixed_position, sample_size)
    else:
        logging.error("please chose a valid dataset mode")


if __name__ == "__main__":
    parser = setup_parser(argparse.ArgumentParser())
    args = parser.parse_args()
    args_dict = vars(args)
    
    config_path = "/".join(args_dict["model_path"].split("/")[:-1]) + "/config.yaml"
    config = load_config(config_path)
    args_dict["config"] = config

    model = load_model(args_dict.pop("model_path"), config)
    args_dict["model"] = model
    
    inference(**args_dict)
