
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from envs.robots.robot_arm import RobotArm

def scatter_end_points(x_end: np.array, y_end: np.array) -> Figure:
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.set_ylim([0, 1])
    
    radius = np.sqrt(np.power(x_end, 2) + np.power(y_end, 2))
    theta = np.arctan2(y_end, x_end)

    ax.scatter(theta, radius, alpha=0.3)

    return fig


if __name__ == "__main__":
    num_points = 1000
    num_joints = 10

    arm = RobotArm(num_joints)
    # sample actions
    actions = np.random.uniform(0, 2 * np.pi, size=(num_points, num_joints))
    # empty array for end-positions
    pos = np.empty((num_points, 2))
    for idx, action in enumerate(actions):
        # apply action
        arm.set(action)
        pos[idx] = arm.end_position

    print(pos)

    fig = scatter_end_points(pos[:, 0], pos[:, 1], max_radius=num_joints)

    fig.savefig("test_polar.png")
