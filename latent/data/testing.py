import numpy as np
import torch
from latent.data.vae_dataset import ConditionalActionTargetDataset
from latent.data.utils import split_state_information
from utils.kinematics import forward_kinematics
from envs.robots.ccd import IK


def check_action_constrain():
    constrain_radius = None
    dataset = ConditionalActionTargetDataset(
        "./datasets/2/test/actions_IK_random_start.csv",
        "./datasets/2/test/state_IK_random_start.csv",
        action_constrain_radius=constrain_radius,
    )

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    # find action distribution
    actions = []

    for idx, (target_action, state) in enumerate(dataset):
        target_position, current_position, state_angles = split_state_information(
            state.unsqueeze(dim=0)
        )
        # for position_sequence in forward_kinematics(state_angles).detach().numpy():
        #     ax.plot(position_sequence[:, 0], position_sequence[:, 1], color="k", marker=".", alpha=1/10)

        target_position = target_position.detach().numpy()
        current_position = current_position.detach().numpy()
        ax.scatter(target_position[:, 0], target_position[:, 1], c="b")
        # ax.scatter(current_position[:, 0], current_position[:, 1], c="g")
        if constrain_radius is not None:
            ax.add_patch(plt.Circle(current_position[0], constrain_radius, fill=False))
        arm_positions = (
            forward_kinematics(state_angles + target_action).detach().numpy()
        )
        pred_position = arm_positions[:, -1, :]
        for position_sequence in arm_positions:
            ax.plot(
                position_sequence[:, 0],
                position_sequence[:, 1],
                color="k",
                marker=".",
                alpha=1 / 10,
            )

        ax.scatter(pred_position[:, 0], pred_position[:, 1], c="orange")

        actions.append(target_action)

        if idx > 4:
            break
    actions = torch.cat(actions)
    # plt.hist(actions.detach().numpy())
    plt.show()


def plot_action_constrain_radius():
    constrain_radius = 1
    num_joints = 20
    num_points = 100

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1)

    # sample start
    theta = np.random.uniform(0, np.pi)
    radius = np.random.uniform(0, num_joints)
    start = np.zeros(3)
    start[0] = np.cos(theta) * (radius - constrain_radius)
    start[1] = np.sin(theta) * (radius - constrain_radius)

    # sample targets
    theta = np.linspace(0, 2 * np.pi, num_points)
    targets = np.zeros((num_points, 3))
    targets[:, 0] = np.cos(theta) * constrain_radius + start[0]
    targets[:, 1] = np.sin(theta) * constrain_radius + start[1]

    # get start config
    link = np.ones(num_joints)
    start_angles = np.zeros(num_joints)
    start_angles, _, _, _ = IK(start, start_angles.copy(), link, err_min=0.001)

    # calculate IK actions for each target
    target_angels = []
    bar = Bar("solve IK", max=num_points)
    for target in targets:
        target_action, _, _, _ = IK(target, start_angles.copy(), link, err_min=0.001)
        target_angels.append(target_action)
        bar.next()
    bar.finish()

    target_angels = np.stack(target_angels)
    target_angels = target_angels / 180 * np.pi  # convert to rad

    # make forward pass to get arm positions
    target_actions = np.cumsum(target_angels, axis=1)
    arm_positions = forward_kinematics(torch.tensor(target_actions)).numpy()

    # plot
    axs[0].scatter(start[0], start[1], c="b", s=1)
    axs[0].add_patch(plt.Circle(start[0:2], constrain_radius, fill=False))
    axs[0].add_patch(plt.Circle((0, 0), num_joints, fill=False))
    for position_sequence in arm_positions:
        axs[0].plot(
            position_sequence[:, 0],
            position_sequence[:, 1],
            color="k",
            marker=".",
            alpha=1 / 10,
        )
    for i in range(num_joints):
        axs[1].plot(target_angels[:, i])
    plt.show()


def plot_action_state_distribution():
    raise DeprecationWarning
    constrain_radius = 0.1
    num_joints = 2
    dataset = ConditionalActionTargetDataset(
        f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
        f"./datasets/{num_joints}/train/state_IK_random_start.csv",
        action_constrain_radius=constrain_radius,
    )

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1)
    # find action distribution
    actions = []
    state_angles = []
    for target_action, state in dataset:
        _, _, state_angle = split_state_information(state.unsqueeze(dim=0))
        actions.append(target_action)
        state_angles.append(state_angle.squeeze())

    actions = torch.cat(actions)
    state_angles = torch.cat(state_angles)

    axs[0].hist(actions.detach().numpy(), bins=50)
    axs[1].hist(state_angles.detach().numpy(), bins=50)
    abs_angles = (
        actions.reshape((len(dataset), num_joints))
        + state_angles.reshape((len(dataset), num_joints))
    ).flatten()
    axs[2].hist(abs_angles.detach().numpy(), bins=50)
    plt.show()


if __name__ == "__main__":
    # plot_action_state_distribution()
    # check_action_constrain()
    # plot_action_constrain_radius()

    dataset = TargetGaussianDataset(
        f"./datasets/5/test/state_IK_random_start.csv", std=1
    )
    dataloader = DataLoader(dataset)

    for x, y, z in dataloader:
        print(x, y, z)
